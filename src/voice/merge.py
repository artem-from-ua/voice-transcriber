"""Combine ASR segments with pyannote diarization turns.

For each ASR segment, pick the pyannote speaker whose `exclusive_diarization`
turns have the most temporal overlap. Segments without any overlap with a
pyannote turn keep `speaker=None`.

Two boundary-crossing concerns live here (issue #125):

  - `merge()` records baseline telemetry that counts how many ASR segments
    cross a pyannote turn boundary by more than `threshold_ms` on the
    second speaker. Two thresholds are reported (300 ms, 500 ms) so the
    sensitivity to noisy pyannote boundaries is visible from one run.
  - `split_on_turn_boundary()` (optional preprocessing, behind a CLI flag)
    cuts a boundary-crossing AsrSegment along the diarization boundary —
    at word granularity when Whisper word_timestamps are present,
    falling back to a character-proportional split when they are not.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .types import AsrSegment, DiarTurn, Segment, Word


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _count_speakers_above_threshold(
    seg: AsrSegment, turns: list[DiarTurn], *, threshold_s: float,
) -> int:
    by_spk: dict[str, float] = defaultdict(float)
    for t in turns:
        ov = _overlap(seg.start, seg.end, t.start, t.end)
        if ov > 0:
            by_spk[t.speaker] += ov
    return sum(1 for v in by_spk.values() if v > threshold_s)


def merge(
    asr_segments: list[AsrSegment], turns: list[DiarTurn],
) -> tuple[list[Segment], dict[str, Any]]:
    """Assign one pyannote speaker per ASR segment by max temporal overlap.

    Returns `(merged_segments, telemetry)`. The telemetry dict counts how
    many ASR segments cross a turn boundary by more than 300 ms / 500 ms
    on a second speaker — the baseline for issue #125 research.
    """
    merged: list[Segment] = []
    crossings_300 = 0
    crossings_500 = 0
    for seg in asr_segments:
        content = seg.content.strip()
        by_spk: dict[str, float] = defaultdict(float)
        for t in turns:
            ov = _overlap(seg.start, seg.end, t.start, t.end)
            if ov > 0:
                by_spk[t.speaker] += ov

        # `speaker_hint` from `split_on_turn_boundary` overrides max-overlap
        # (the snap fix may shift a segment's time outside the owner's
        # pyannote turn; the hint preserves the correct attribution).
        if seg.speaker_hint is not None:
            best = seg.speaker_hint
        else:
            best = max(by_spk.items(), key=lambda kv: kv[1])[0] if by_spk else None
        merged.append(Segment(
            start=seg.start, end=seg.end,
            content=content, speaker=best,
        ))

        if _count_speakers_above_threshold(seg, turns, threshold_s=0.300) >= 2:
            crossings_300 += 1
        if _count_speakers_above_threshold(seg, turns, threshold_s=0.500) >= 2:
            crossings_500 += 1

    telemetry: dict[str, Any] = {
        "strategy": "max_overlap",
        "total_asr_segments": len(asr_segments),
        "crossings_300ms": crossings_300,
        "crossings_500ms": crossings_500,
    }
    return merged, telemetry


def _argmax_turn(
    word_start: float, word_end: float, turns: list[DiarTurn],
) -> str | None:
    """Pick the pyannote speaker whose turn has max overlap with [word_start, word_end].

    Returns None if no turn overlaps the word at all.
    """
    best_spk: str | None = None
    best_ov = 0.0
    for t in turns:
        ov = _overlap(word_start, word_end, t.start, t.end)
        if ov > best_ov:
            best_ov = ov
            best_spk = t.speaker
    return best_spk


def _turn_at_time(t_s: float, turns: list[DiarTurn]) -> str | None:
    """Return the speaker whose turn contains `t_s`, else None."""
    for t in turns:
        if t.start <= t_s < t.end:
            return t.speaker
    return None


# English short fillers (back-channel confirmations). Case-insensitive
# after stripping trailing punctuation. Used by the filler-bias step (#177);
# only consulted when `filler_bias_tie_ms > 0`. Ukrainian fillers would
# need a separate flag — left out until validated on a UA reference.
DEFAULT_FILLER_WORDS: frozenset[str] = frozenset({
    "yes", "yeah", "yep", "yup", "okay", "ok", "sure", "right",
    "mhm", "uh-huh", "mm", "hmm", "uh", "um",
})


def _normalize_filler(text: str) -> str:
    return text.strip(".,!?…").lower()


def _next_turn_after(t_s: float, turns: list[DiarTurn]) -> DiarTurn | None:
    """Return the first turn whose start >= t_s, else None.

    Used by the filler-bias step (A) to resolve cases where a filler
    word's end falls in a gap between turns — pick the downstream turn.
    """
    best: DiarTurn | None = None
    for t in turns:
        if t.start >= t_s and (best is None or t.start < best.start):
            best = t
    return best


def _nearest_turn_boundary(t_s: float, turns: list[DiarTurn]) -> float | None:
    """Return the time of the closest turn-start across all `turns` (excluding
    the very first turn's start at 0). None if no turn starts past time 0.

    Used by the deadband step (C) to decide whether a word's centre falls
    within ±deadband_ms of any pyannote boundary.
    """
    best: float | None = None
    best_dist = float("inf")
    for t in turns[1:]:
        d = abs(t_s - t.start)
        if d < best_dist:
            best_dist = d
            best = t.start
    return best


def _downstream_turn_at(boundary_t: float, turns: list[DiarTurn]) -> str | None:
    """Return the speaker of the turn that STARTS at `boundary_t`."""
    for t in turns:
        if abs(t.start - boundary_t) < 1e-6:
            return t.speaker
    return None


def _argmax_turn_with_bias(
    word: Word, turns: list[DiarTurn], *,
    filler_bias_tie_ms: int = 0,
    filler_words: frozenset[str] = frozenset(),
    deadband_ms: int = 0,
) -> str | None:
    """Pick a word's owner with optional filler-bias and deadband refinements.

    Defaults reproduce `_argmax_turn` exactly. When `filler_bias_tie_ms > 0`,
    a filler word whose two top overlaps differ by < tie_ms is reassigned
    to the turn containing its `end` time — captures back-channel
    confirmations whose Whisper timestamps straddle a turn boundary
    (#177). When `deadband_ms > 0`, ANY word whose centre lies within
    ±deadband_ms of a pyannote boundary is reassigned to the downstream
    turn — recovers leading words of a turn that pyannote starts late.

    When both refinements are active, filler-bias is tried first; if it
    does not flip the decision, deadband is consulted.
    """
    baseline = _argmax_turn(word.start, word.end, turns)

    # --- A: filler-bias by word.end -----------------------------------------
    if filler_bias_tie_ms > 0 and filler_words:
        text = _normalize_filler(word.content)
        if text in filler_words:
            by_spk: dict[str, float] = defaultdict(float)
            for t in turns:
                ov = _overlap(word.start, word.end, t.start, t.end)
                if ov > 0:
                    by_spk[t.speaker] += ov
            if len(by_spk) >= 2:
                ranked = sorted(by_spk.values(), reverse=True)
                if (ranked[0] - ranked[1]) * 1000.0 < filler_bias_tie_ms:
                    end_owner = _turn_at_time(word.end, turns)
                    if end_owner is None:
                        nxt = _next_turn_after(word.end, turns)
                        if nxt is not None:
                            return nxt.speaker
                    else:
                        return end_owner

    # --- C: deadband downstream --------------------------------------------
    if deadband_ms > 0:
        deadband_s = deadband_ms / 1000.0
        w_center = (word.start + word.end) / 2.0
        nearest = _nearest_turn_boundary(w_center, turns)
        if nearest is not None and abs(w_center - nearest) <= deadband_s:
            downstream = _downstream_turn_at(nearest, turns)
            if downstream is not None:
                return downstream

    return baseline


def _snap_split_index_to_low_prob(
    words: list[Word], naive_split_idx: int,
    *, window_ms: int, prob_threshold: float,
) -> int:
    """Snap a naive word-boundary split index to the nearest low-probability word.

    pyannote turn boundaries are imprecise by ±200-500 ms. When a naive
    split index sits at a pyannote boundary, the true speaker change can
    be a couple of words to either side. Whisper exposes a per-word
    `probability`; it dips sharply at acoustic ambiguity (silence padding,
    crosstalk, overlapping speakers), which correlates strongly with the
    true speaker change. Pick the lowest-probability word inside a
    ±`window_ms` window around the naive cut whose probability is below
    `prob_threshold`, and snap the split to cut BEFORE that word.

    Returns the new split index (a value in `range(1, len(words))`). When
    no in-window word has `probability < prob_threshold`, or a candidate
    word has no probability at all, that candidate is ignored. If nothing
    qualifies the naive index is returned unchanged.
    """
    if naive_split_idx <= 0 or naive_split_idx >= len(words):
        return naive_split_idx

    # The naive cut is between words[naive_split_idx-1] and words[naive_split_idx].
    boundary_time = (words[naive_split_idx - 1].end + words[naive_split_idx].start) / 2.0
    window_s = window_ms / 1000.0

    best_idx = naive_split_idx
    best_prob = prob_threshold  # only snap if strictly below threshold
    # Each candidate i means: cut BEFORE words[i] (so groups become
    # words[:i] / words[i:]). A candidate qualifies when ANY part of the
    # word's [start, end] interval falls inside [boundary - window,
    # boundary + window] — measuring distance from the nearest point of
    # the word, not its leading edge, so a long word that straddles the
    # window edge still counts.
    for i in range(1, len(words)):
        w = words[i]
        if w.probability is None:
            continue
        nearest_edge = min(abs(w.start - boundary_time), abs(w.end - boundary_time))
        # Word also qualifies if it straddles the boundary itself.
        straddles = w.start <= boundary_time <= w.end
        if not straddles and nearest_edge > window_s:
            continue
        if w.probability < best_prob:
            best_prob = w.probability
            best_idx = i

    return best_idx


def _split_segment_by_words(
    seg: AsrSegment, turns: list[DiarTurn], *,
    min_segment_ms: int,
    snap_window_ms: int = 0,
    snap_prob_threshold: float = 0.7,
    filler_bias_tie_ms: int = 0,
    filler_words: frozenset[str] = frozenset(),
    deadband_ms: int = 0,
) -> tuple[list[AsrSegment], int, int, int, int]:
    """Word-level split. Returns
    (new_segments, fragments_skipped_below_min, snaps_applied,
     filler_bias_applied, deadband_applied).

    The naive cut points come from argmax-overlap turn assignment per word
    (pyannote boundaries), optionally refined by filler-bias and deadband
    (issue #177) — see `_argmax_turn_with_bias`. When `snap_window_ms > 0`,
    each cut is then snapped within ±`snap_window_ms` to the lowest-
    probability word whose probability is below `snap_prob_threshold` —
    see `_snap_split_index_to_low_prob`. Pass `snap_window_ms=0` (default)
    to disable snapping and reproduce the pure pyannote-boundary behaviour.
    """
    assert seg.words is not None
    if not seg.words:
        return [seg], 0, 0, 0, 0

    # Per-word owner assignment with optional filler-bias / deadband
    # refinements. Count how many words each refinement actually shifts.
    word_owner: list[str | None] = []
    filler_bias_applied = 0
    deadband_applied = 0
    for w in seg.words:
        baseline = _argmax_turn(w.start, w.end, turns)
        refined = _argmax_turn_with_bias(
            w, turns,
            filler_bias_tie_ms=filler_bias_tie_ms,
            filler_words=filler_words,
            deadband_ms=deadband_ms,
        )
        if refined != baseline:
            # Attribute the flip to whichever refinement applied. Filler-
            # bias runs first inside _argmax_turn_with_bias, so check it
            # in the same order to keep the counter accurate.
            text_norm = _normalize_filler(w.content)
            if filler_bias_tie_ms > 0 and filler_words and text_norm in filler_words:
                filler_bias_applied += 1
            else:
                deadband_applied += 1
        word_owner.append(refined)

    # Find naive split indices: positions where consecutive owners differ.
    naive_cuts = [
        i for i in range(1, len(seg.words))
        if word_owner[i] != word_owner[i - 1]
    ]
    if not naive_cuts:
        return [seg], 0, 0, filler_bias_applied, deadband_applied

    # Snap each naive cut to a nearby low-probability word boundary.
    # IMPORTANT: snapping shifts a cut from index i to j; the words
    # between min(i,j) and max(i,j) change owner from the left side's
    # speaker to the right side's (or vice versa). We rewrite word_owner
    # so the subsequent grouping reflects the snapped attribution.
    snaps_applied = 0
    if snap_window_ms > 0:
        snapped_cuts: list[int] = []
        for naive_idx in naive_cuts:
            new_idx = _snap_split_index_to_low_prob(
                seg.words, naive_idx,
                window_ms=snap_window_ms,
                prob_threshold=snap_prob_threshold,
            )
            if new_idx != naive_idx:
                snaps_applied += 1
                left_owner = word_owner[naive_idx - 1]
                right_owner = word_owner[naive_idx]
                if new_idx < naive_idx:
                    # Cut moved earlier — words [new_idx..naive_idx-1] now go
                    # to the right-side speaker.
                    for k in range(new_idx, naive_idx):
                        word_owner[k] = right_owner
                else:
                    # Cut moved later — words [naive_idx..new_idx-1] now go
                    # to the left-side speaker.
                    for k in range(naive_idx, new_idx):
                        word_owner[k] = left_owner
            snapped_cuts.append(new_idx)
    else:
        snapped_cuts = naive_cuts[:]

    # Adjacent snaps may collapse onto the same word — dedup and re-sort.
    snapped_cuts = sorted(set(snapped_cuts))

    cut_points = [0] + snapped_cuts + [len(seg.words)]
    groups: list[list[Word]] = [
        seg.words[cut_points[i]:cut_points[i + 1]]
        for i in range(len(cut_points) - 1)
    ]

    if len(groups) <= 1:
        return [seg], 0, snaps_applied, filler_bias_applied, deadband_applied

    # Per-group owner: majority-vote across the group's word_owner values.
    # After snap, word_owner has been rewritten so words on the side of
    # the snapped cut carry the correct speaker; this picks that up.
    out: list[AsrSegment] = []
    skipped = 0
    for g_idx, g in enumerate(groups):
        if not g:
            continue
        start = g[0].start
        end = g[-1].end
        if (end - start) * 1000.0 < min_segment_ms:
            skipped += 1
            continue
        text = " ".join(w.content for w in g).strip()
        if not text:
            skipped += 1
            continue
        # Recover this group's owner indices in word_owner.
        lo = cut_points[g_idx]
        hi = cut_points[g_idx + 1]
        votes: dict[str, int] = defaultdict(int)
        for owner in word_owner[lo:hi]:
            if owner is not None:
                votes[owner] += 1
        speaker_hint = (
            max(votes.items(), key=lambda kv: kv[1])[0] if votes else None
        )
        out.append(AsrSegment(
            start=start, end=end, content=text, words=list(g),
            speaker_hint=speaker_hint,
        ))
    if not out:
        # Skipping all fragments would silently drop the segment; keep
        # the original rather than lose its text.
        return [seg], skipped, snaps_applied, filler_bias_applied, deadband_applied
    return out, skipped, snaps_applied, filler_bias_applied, deadband_applied


def _split_segment_by_chars(
    seg: AsrSegment, turn_cuts: list[tuple[float, str | None]],
    *, min_segment_ms: int,
) -> tuple[list[AsrSegment], int]:
    """Character-proportional fallback (when Whisper words are absent).

    `turn_cuts` is a list of (boundary_time, speaker_after_boundary)
    sorted by time; the segment is split at each boundary at the
    corresponding character offset proportional to time.

    Returns (new_segments, fragments_skipped_below_min).
    """
    if not turn_cuts:
        return [seg], 0

    duration = seg.end - seg.start
    if duration <= 0 or not seg.content:
        return [seg], 0

    text = seg.content
    n_chars = len(text)
    # Compose breakpoint list including segment ends.
    breakpoints = [(seg.start, None)] + turn_cuts + [(seg.end, None)]

    out: list[AsrSegment] = []
    skipped = 0
    for i in range(len(breakpoints) - 1):
        start_t = breakpoints[i][0]
        end_t = breakpoints[i + 1][0]
        if end_t <= start_t:
            continue
        char_start = int((start_t - seg.start) / duration * n_chars)
        char_end = int((end_t - seg.start) / duration * n_chars)
        if i == len(breakpoints) - 2:
            char_end = n_chars
        chunk_text = text[char_start:char_end].strip()
        if not chunk_text:
            skipped += 1
            continue
        if (end_t - start_t) * 1000.0 < min_segment_ms:
            skipped += 1
            continue
        out.append(AsrSegment(
            start=start_t, end=end_t, content=chunk_text, words=None,
        ))
    if not out:
        return [seg], skipped
    return out, skipped


def split_on_turn_boundary(
    asr_segments: list[AsrSegment],
    turns: list[DiarTurn],
    *,
    min_segment_ms: int = 200,
    threshold_ms: int = 300,
    snap_window_ms: int = 0,
    snap_prob_threshold: float = 0.7,
    filler_bias_tie_ms: int = 0,
    filler_words: frozenset[str] = frozenset(),
    deadband_ms: int = 0,
) -> tuple[list[AsrSegment], dict[str, Any]]:
    """Cut ASR segments that cross pyannote turn boundaries.

    For each AsrSegment with ≥ 2 speakers whose overlap exceeds
    `threshold_ms`, split the segment so each output segment lies inside
    one speaker's turn. Word-level cut when `seg.words is not None`,
    character-proportional fallback otherwise. Output fragments shorter
    than `min_segment_ms` are dropped. If splitting would drop every
    fragment, the original segment is kept untouched.

    When `snap_window_ms > 0`, each naive word-level cut is snapped to
    the nearest low-`probability` word within the window — see
    `_snap_split_index_to_low_prob`. Off by default.

    When `filler_bias_tie_ms > 0` AND `filler_words` is non-empty, a
    near-tied filler word is reassigned to the turn containing its `end`
    time (#177 A). When `deadband_ms > 0`, any word whose centre lies
    within ±deadband_ms of a pyannote boundary is reassigned to the
    downstream turn (#177 C). Both refinements run during per-word owner
    assignment, BEFORE snap — see `_argmax_turn_with_bias`. Off by default.

    Returns `(new_asr_segments, telemetry)`.
    """
    threshold_s = threshold_ms / 1000.0
    out: list[AsrSegment] = []
    segments_split = 0
    new_segments_produced = 0
    fallback_char_split = 0
    fragments_skipped_below_min = 0
    snaps_applied = 0
    filler_bias_applied = 0
    deadband_applied = 0

    for seg in asr_segments:
        n_speakers = _count_speakers_above_threshold(
            seg, turns, threshold_s=threshold_s,
        )
        if n_speakers < 2:
            out.append(seg)
            continue

        if seg.words is not None:
            new_segs, skipped, n_snaps, n_filler, n_deadband = _split_segment_by_words(
                seg, turns,
                min_segment_ms=min_segment_ms,
                snap_window_ms=snap_window_ms,
                snap_prob_threshold=snap_prob_threshold,
                filler_bias_tie_ms=filler_bias_tie_ms,
                filler_words=filler_words,
                deadband_ms=deadband_ms,
            )
            snaps_applied += n_snaps
            filler_bias_applied += n_filler
            deadband_applied += n_deadband
        else:
            # Build turn-cut breakpoints inside the segment's time range.
            cuts: list[tuple[float, str | None]] = []
            seen_speakers: set[str] = set()
            for t in sorted(turns, key=lambda x: x.start):
                if t.start <= seg.start or t.start >= seg.end:
                    continue
                cuts.append((t.start, t.speaker))
                seen_speakers.add(t.speaker)
            if not cuts:
                out.append(seg)
                continue
            new_segs, skipped = _split_segment_by_chars(
                seg, cuts, min_segment_ms=min_segment_ms,
            )
            fallback_char_split += 1

        fragments_skipped_below_min += skipped
        if len(new_segs) > 1:
            segments_split += 1
            new_segments_produced += len(new_segs)
            out.extend(new_segs)
        elif len(new_segs) == 1 and new_segs[0].speaker_hint is not None:
            # No structural split, but a per-word refinement (filler-bias
            # or deadband) yielded a single fragment with a non-default
            # owner hint — propagate the hint instead of dropping back to
            # the original segment (whose max-overlap label would ignore
            # the refinement).
            out.extend(new_segs)
        else:
            out.append(seg)

    telemetry: dict[str, Any] = {
        "strategy": "split_on_boundary",
        "input_asr_segments": len(asr_segments),
        "output_asr_segments": len(out),
        "segments_split": segments_split,
        "new_segments_produced": new_segments_produced,
        "fallback_char_split": fallback_char_split,
        "fragments_skipped_below_min": fragments_skipped_below_min,
        "min_segment_ms": min_segment_ms,
        "threshold_ms": threshold_ms,
        "snap_window_ms": snap_window_ms,
        "snap_prob_threshold": snap_prob_threshold,
        "snaps_applied": snaps_applied,
        "filler_bias_tie_ms": filler_bias_tie_ms,
        "filler_words_count": len(filler_words),
        "filler_bias_applied": filler_bias_applied,
        "deadband_ms": deadband_ms,
        "deadband_applied": deadband_applied,
    }
    return out, telemetry
