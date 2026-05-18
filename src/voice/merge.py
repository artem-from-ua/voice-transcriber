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


def _split_segment_by_words(
    seg: AsrSegment, turns: list[DiarTurn], *, min_segment_ms: int,
) -> tuple[list[AsrSegment], int]:
    """Word-level split. Returns (new_segments, fragments_skipped_below_min)."""
    assert seg.words is not None
    if not seg.words:
        return [seg], 0

    # Assign each word to its argmax-overlap turn (or "None" speaker).
    word_owner: list[tuple[Word, str | None]] = [
        (w, _argmax_turn(w.start, w.end, turns)) for w in seg.words
    ]

    # Group consecutive words with the same owner.
    groups: list[list[tuple[Word, str | None]]] = []
    for item in word_owner:
        if groups and groups[-1][-1][1] == item[1]:
            groups[-1].append(item)
        else:
            groups.append([item])

    if len(groups) <= 1:
        return [seg], 0

    out: list[AsrSegment] = []
    skipped = 0
    for g in groups:
        words = [w for w, _ in g]
        start = words[0].start
        end = words[-1].end
        if (end - start) * 1000.0 < min_segment_ms:
            skipped += 1
            continue
        text = " ".join(w.content for w in words).strip()
        if not text:
            skipped += 1
            continue
        out.append(AsrSegment(
            start=start, end=end, content=text, words=list(words),
        ))
    if not out:
        # Skipping all fragments would silently drop the segment; keep
        # the original rather than lose its text.
        return [seg], skipped
    return out, skipped


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
) -> tuple[list[AsrSegment], dict[str, Any]]:
    """Cut ASR segments that cross pyannote turn boundaries.

    For each AsrSegment with ≥ 2 speakers whose overlap exceeds
    `threshold_ms`, split the segment so each output segment lies inside
    one speaker's turn. Word-level cut when `seg.words is not None`,
    character-proportional fallback otherwise. Output fragments shorter
    than `min_segment_ms` are dropped. If splitting would drop every
    fragment, the original segment is kept untouched.

    Returns `(new_asr_segments, telemetry)`.
    """
    threshold_s = threshold_ms / 1000.0
    out: list[AsrSegment] = []
    segments_split = 0
    new_segments_produced = 0
    fallback_char_split = 0
    fragments_skipped_below_min = 0

    for seg in asr_segments:
        n_speakers = _count_speakers_above_threshold(
            seg, turns, threshold_s=threshold_s,
        )
        if n_speakers < 2:
            out.append(seg)
            continue

        if seg.words is not None:
            new_segs, skipped = _split_segment_by_words(
                seg, turns, min_segment_ms=min_segment_ms,
            )
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
    }
    return out, telemetry
