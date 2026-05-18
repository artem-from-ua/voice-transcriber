"""Offline tests for merge: ASR × pyannote → assigned speaker.

Also covers the boundary-crossing telemetry and `split_on_turn_boundary`
helper added for issue #125 research.
"""

from voice.merge import merge, split_on_turn_boundary
from voice.types import AsrSegment, DiarTurn, Word


def test_picks_speaker_with_largest_overlap():
    asr = [AsrSegment(start=1.0, end=4.0, content="hello there")]
    turns = [
        DiarTurn(start=0.0, end=1.5, speaker="SPEAKER_00"),  # 0.5s overlap
        DiarTurn(start=1.5, end=4.0, speaker="SPEAKER_01"),  # 2.5s overlap → wins
    ]
    out, _ = merge(asr, turns)
    assert out[0].speaker == "SPEAKER_01"


def test_no_overlap_yields_none_speaker():
    asr = [AsrSegment(start=10.0, end=11.0, content="orphan")]
    turns = [DiarTurn(start=0.0, end=2.0, speaker="SPEAKER_00")]
    out, _ = merge(asr, turns)
    assert out[0].speaker is None


# --- Telemetry --------------------------------------------------------------


def test_telemetry_records_totals_and_strategy():
    asr = [AsrSegment(start=0.0, end=1.0, content="a")]
    turns = [DiarTurn(start=0.0, end=1.0, speaker="SPEAKER_00")]
    _, tel = merge(asr, turns)
    assert tel["strategy"] == "max_overlap"
    assert tel["total_asr_segments"] == 1
    assert tel["crossings_300ms"] == 0
    assert tel["crossings_500ms"] == 0


def test_telemetry_counts_crossing_above_300ms_only():
    # Segment 0.0→4.0. Spk0 owns 0.0→3.65 (3.65s overlap),
    # Spk1 owns 3.65→4.0 (0.35s overlap → above 300 ms, below 500 ms).
    asr = [AsrSegment(start=0.0, end=4.0, content="x")]
    turns = [
        DiarTurn(start=0.0, end=3.65, speaker="SPEAKER_00"),
        DiarTurn(start=3.65, end=4.0, speaker="SPEAKER_01"),
    ]
    _, tel = merge(asr, turns)
    assert tel["crossings_300ms"] == 1
    assert tel["crossings_500ms"] == 0


def test_telemetry_counts_crossing_above_both_thresholds():
    # Spk1 overlap 0.6 s — above both 300 and 500.
    asr = [AsrSegment(start=0.0, end=4.0, content="x")]
    turns = [
        DiarTurn(start=0.0, end=3.4, speaker="SPEAKER_00"),
        DiarTurn(start=3.4, end=4.0, speaker="SPEAKER_01"),
    ]
    _, tel = merge(asr, turns)
    assert tel["crossings_300ms"] == 1
    assert tel["crossings_500ms"] == 1


# --- split_on_turn_boundary (word-level) ------------------------------------


def _make_words(words: list[tuple[float, float, str]]) -> list[Word]:
    return [Word(start=s, end=e, content=c) for s, e, c in words]


def test_split_word_level_two_speakers():
    seg = AsrSegment(
        start=10.0, end=15.0,
        content="hello there yeah sure okay",
        words=_make_words([
            (10.0, 10.5, "hello"),
            (10.6, 11.1, "there"),
            (11.3, 11.8, "yeah"),
            (12.0, 12.5, "sure"),
            (14.0, 14.8, "okay"),
        ]),
    )
    turns = [
        DiarTurn(start=10.0, end=11.5, speaker="SPK_A"),
        DiarTurn(start=11.5, end=15.0, speaker="SPK_B"),
    ]
    out, tel = split_on_turn_boundary([seg], turns)
    assert tel["segments_split"] == 1
    assert tel["fallback_char_split"] == 0
    assert tel["output_asr_segments"] == 2
    assert len(out) == 2
    assert out[0].content == "hello there"
    assert out[1].content.startswith("yeah sure")
    assert out[0].words is not None and len(out[0].words) == 2
    assert out[1].words is not None and len(out[1].words) >= 2


def test_split_word_level_skips_below_min_segment_ms():
    # Both turns have > 300 ms overlap with the segment, so the segment
    # qualifies for splitting. But the SPK_A word-group is only 100 ms
    # of speech (one short word) — that fragment is dropped because it
    # is below the 200 ms default.
    seg = AsrSegment(
        start=0.0, end=5.0, content="hi everyone today",
        words=_make_words([
            (0.6, 0.7, "hi"),       # 100 ms — inside SPK_A turn
            (1.0, 1.5, "everyone"),  # inside SPK_B turn
            (1.6, 2.0, "today"),     # inside SPK_B turn
        ]),
    )
    turns = [
        DiarTurn(start=0.0, end=0.9, speaker="SPK_A"),   # 900 ms overlap with seg
        DiarTurn(start=0.9, end=5.0, speaker="SPK_B"),   # 4.1 s overlap
    ]
    out, tel = split_on_turn_boundary([seg], turns)
    assert tel["fragments_skipped_below_min"] >= 1
    # The 100 ms "hi" fragment is dropped; one long fragment remains.
    long_frags = [s for s in out if s.end - s.start > 0.4]
    assert len(long_frags) == 1


def test_split_word_level_three_turns():
    seg = AsrSegment(
        start=0.0, end=6.0, content="a b c d e f",
        words=_make_words([
            (0.0, 0.5, "a"),
            (0.6, 1.1, "b"),
            (2.2, 2.7, "c"),
            (2.9, 3.4, "d"),
            (4.6, 5.1, "e"),
            (5.3, 5.8, "f"),
        ]),
    )
    turns = [
        DiarTurn(start=0.0, end=2.0, speaker="SPK_A"),
        DiarTurn(start=2.0, end=4.5, speaker="SPK_B"),
        DiarTurn(start=4.5, end=6.0, speaker="SPK_C"),
    ]
    out, tel = split_on_turn_boundary([seg], turns)
    assert tel["segments_split"] == 1
    assert len(out) == 3


def test_split_leaves_clean_segment_alone():
    seg = AsrSegment(
        start=0.0, end=2.0, content="clean speech",
        words=_make_words([(0.0, 0.5, "clean"), (0.6, 1.4, "speech")]),
    )
    turns = [DiarTurn(start=0.0, end=2.0, speaker="SPK_A")]
    out, tel = split_on_turn_boundary([seg], turns)
    assert out == [seg]
    assert tel["segments_split"] == 0


# --- split_on_turn_boundary (character fallback) ----------------------------


def test_split_char_fallback_no_words():
    # Even split: two halves by time → halves by chars.
    seg = AsrSegment(start=0.0, end=4.0, content="aaaaaaaabbbbbbbb")  # 16 chars
    turns = [
        DiarTurn(start=0.0, end=2.0, speaker="SPK_A"),
        DiarTurn(start=2.0, end=4.0, speaker="SPK_B"),
    ]
    out, tel = split_on_turn_boundary([seg], turns)
    assert tel["fallback_char_split"] == 1
    assert tel["segments_split"] == 1
    assert len(out) == 2
    # Proportional time-to-char split: each half ~8 chars.
    assert out[0].content == "aaaaaaaa"
    assert out[1].content == "bbbbbbbb"
    assert all(s.words is None for s in out)


# --- snap to low-probability word (#125 follow-up) --------------------------


def _words_with_prob(items: list[tuple[float, float, str, float]]) -> list[Word]:
    return [Word(start=s, end=e, content=c, probability=p) for s, e, c, p in items]


def test_snap_moves_cut_to_low_prob_word_within_window():
    # The same shape as the real "so yeah so you know the field pretty well yes yes ..."
    # case: pyannote boundary lands between "know" and "the" at t=1.16, but the
    # actual speaker change is one word earlier, at the low-prob "so" (p=0.45).
    seg = AsrSegment(
        start=0.0, end=2.0,
        content="so yeah so you know the field",
        words=_words_with_prob([
            (0.00, 0.40, "so",     0.99),
            (0.40, 1.04, "yeah",   0.99),
            (1.04, 1.16, "so",     0.45),  # low-prob — true cut here
            (1.16, 1.30, "you",    0.99),
            (1.30, 1.42, "know",   0.99),
            (1.42, 1.60, "the",    0.99),
            (1.60, 1.88, "field",  0.99),
        ]),
    )
    # pyannote draws the boundary AFTER "know" (t=1.42); naive cut is index 5.
    turns = [
        DiarTurn(start=0.0, end=1.42, speaker="SPK_A"),
        DiarTurn(start=1.42, end=2.0, speaker="SPK_B"),
    ]

    # Without snap: cut lands between "know" and "the" → "the field" goes to SPK_B.
    out_no_snap, tel_no = split_on_turn_boundary([seg], turns, snap_window_ms=0)
    assert tel_no["snaps_applied"] == 0
    assert out_no_snap[0].content == "so yeah so you know"
    assert out_no_snap[1].content == "the field"

    # With snap: cut snaps to the low-prob "so" (index 2) inside ±500 ms.
    out_snap, tel_snap = split_on_turn_boundary(
        [seg], turns, snap_window_ms=500, snap_prob_threshold=0.7,
    )
    assert tel_snap["snaps_applied"] == 1
    assert out_snap[0].content == "so yeah"
    assert out_snap[1].content == "so you know the field"


def test_snap_leaves_cut_alone_when_no_low_prob_in_window():
    # Boundary words all have high probability — snap is a no-op.
    seg = AsrSegment(
        start=0.0, end=2.0, content="alpha beta gamma delta",
        words=_words_with_prob([
            (0.00, 0.40, "alpha", 0.99),
            (0.40, 0.80, "beta",  0.99),
            (0.80, 1.20, "gamma", 0.99),
            (1.20, 1.60, "delta", 0.99),
        ]),
    )
    turns = [
        DiarTurn(start=0.0, end=0.80, speaker="SPK_A"),
        DiarTurn(start=0.80, end=2.0, speaker="SPK_B"),
    ]
    out, tel = split_on_turn_boundary(
        [seg], turns, snap_window_ms=500, snap_prob_threshold=0.7,
    )
    assert tel["snaps_applied"] == 0
    assert out[0].content == "alpha beta"
    assert out[1].content == "gamma delta"


def test_snap_ignores_low_prob_word_outside_window():
    # Low-prob word exists but sits far outside the ±100 ms window.
    seg = AsrSegment(
        start=0.0, end=4.0, content="a b c d e",
        words=_words_with_prob([
            (0.00, 0.20, "a", 0.99),
            (0.30, 0.45, "b", 0.30),  # low-prob, far from boundary 2.0
            (1.80, 2.00, "c", 0.99),
            (2.00, 2.20, "d", 0.99),
            (2.20, 2.40, "e", 0.99),
        ]),
    )
    turns = [
        DiarTurn(start=0.0, end=2.0, speaker="SPK_A"),
        DiarTurn(start=2.0, end=4.0, speaker="SPK_B"),
    ]
    out, tel = split_on_turn_boundary(
        [seg], turns, snap_window_ms=100, snap_prob_threshold=0.7,
    )
    assert tel["snaps_applied"] == 0
    # Pyannote boundary is at t=2.0 → cut between "c" (ends 2.00) and "d" (starts 2.00).
    assert out[0].content == "a b c"
    assert out[1].content == "d e"


def test_snap_emits_speaker_hint_that_merge_respects():
    # Same shape as the real failure case: snap moves the cut backwards
    # into the speaker-A owned region, but the resulting middle fragment's
    # *time span* now overlaps speaker-A's pyannote turn more than
    # speaker-B's. The snap-derived owner (B, by word membership) should
    # win via `speaker_hint`, not the time-overlap label (A).
    seg = AsrSegment(
        start=0.0, end=8.0, content="a b c d e f g h",
        words=_words_with_prob([
            (0.0, 1.0, "a", 0.99),
            (1.0, 2.0, "b", 0.99),
            (2.0, 3.0, "c", 0.40),  # low-prob — true speaker change here
            (3.0, 3.2, "d", 0.99),
            (3.2, 3.4, "e", 0.99),
            (3.4, 4.6, "f", 0.99),
            (4.6, 4.8, "g", 0.99),
            (4.8, 5.0, "h", 0.99),
        ]),
    )
    # pyannote draws boundary AFTER "f" at t=4.6 (mid-word naive cut idx=6).
    # Naive split: "a b c d e f" → SPK_A (overlap 4.6s); "g h" → SPK_B.
    # After snap to "c": left "a b" → SPK_A, right "c d e f g h" → SPK_B.
    # Time-overlap of right fragment (2.0-5.0) with SPK_A turn (0.0-4.6)
    # is 2.6 s vs SPK_B turn (4.6-8.0) only 0.4 s → max-overlap would say
    # SPK_A. speaker_hint must override to SPK_B.
    turns = [
        DiarTurn(start=0.0, end=4.6, speaker="SPK_A"),
        DiarTurn(start=4.6, end=8.0, speaker="SPK_B"),
    ]
    asr_split, tel = split_on_turn_boundary(
        [seg], turns, snap_window_ms=3000, snap_prob_threshold=0.7,
    )
    assert tel["snaps_applied"] == 1
    # The middle fragment got SPK_B as its speaker_hint despite being
    # time-overlapped with SPK_A turn.
    right = next(s for s in asr_split if s.content.startswith("c"))
    assert right.speaker_hint == "SPK_B"

    # And merge() honours the hint instead of recomputing max-overlap.
    out, _ = merge(asr_split, turns)
    right_seg = next(s for s in out if s.content.startswith("c"))
    assert right_seg.speaker == "SPK_B"


# --- A: filler-bias + C: deadband (issue #177) ------------------------------


from voice.merge import DEFAULT_FILLER_WORDS


def _seg_with(words: list[Word]) -> AsrSegment:
    return AsrSegment(start=words[0].start, end=words[-1].end,
                      content=" ".join(w.content for w in words), words=words)


def test_filler_bias_flips_straddling_yes_to_downstream():
    # Mirrors the canonical issue #177 case from the 12-min reference.
    seg = _seg_with(_words_with_prob([
        (185.66, 185.94, "pretty", 1.0),
        (185.94, 186.30, "well",   1.0),
        (186.30, 187.20, "yes",    0.998),  # straddles A→B boundary
        (187.20, 187.80, "yes",    0.999),
        (187.80, 188.16, "yes",    0.994),
    ]))
    turns = [
        DiarTurn(start=185.17, end=186.55, speaker="A"),
        DiarTurn(start=187.02, end=194.11, speaker="B"),
    ]
    out, tel = split_on_turn_boundary(
        [seg], turns,
        filler_bias_tie_ms=120,
        filler_words=DEFAULT_FILLER_WORDS,
    )
    # The straddling "yes" overlap: A=0.25s, B=0.18s → diff=70ms < 120ms.
    # word.end=187.20 → B's turn → owner=B.
    assert tel["filler_bias_applied"] >= 1
    # Fragments emitted: pretty+well (A), then three yes-i (B).
    speakers = [s.speaker_hint for s in out]
    assert speakers[0] == "A"   # pretty well
    assert "B" in speakers      # at least one B-segment with the "yes"-es


def test_filler_bias_off_preserves_argmax_behaviour():
    seg = _seg_with(_words_with_prob([
        (185.66, 185.94, "pretty", 1.0),
        (185.94, 186.30, "well",   1.0),
        (186.30, 187.20, "yes",    0.998),
        (187.20, 187.80, "yes",    0.999),
    ]))
    turns = [
        DiarTurn(start=185.17, end=186.55, speaker="A"),
        DiarTurn(start=187.02, end=194.11, speaker="B"),
    ]
    # filler_bias_tie_ms=0 → no-op; baseline argmax wins → straddling yes
    # has more overlap with A (0.25s) than B (0.18s) so it goes to A.
    out, tel = split_on_turn_boundary([seg], turns)
    assert tel["filler_bias_applied"] == 0
    speakers = [s.speaker_hint for s in out]
    # First fragment is "pretty well yes" (A), then "yes" (B).
    assert speakers[0] == "A"


def test_filler_bias_does_not_touch_non_filler():
    # "definitely" with same straddling shape; should NOT flip.
    seg = _seg_with(_words_with_prob([
        (185.66, 185.94, "pretty",     1.0),
        (185.94, 186.30, "well",       1.0),
        (186.30, 187.20, "definitely", 0.998),
        (187.20, 187.80, "yes",        0.999),
    ]))
    turns = [
        DiarTurn(start=185.17, end=186.55, speaker="A"),
        DiarTurn(start=187.02, end=194.11, speaker="B"),
    ]
    out, tel = split_on_turn_boundary(
        [seg], turns,
        filler_bias_tie_ms=120, filler_words=DEFAULT_FILLER_WORDS,
    )
    assert tel["filler_bias_applied"] == 0


def test_deadband_flips_word_near_boundary_to_downstream():
    # Word whose centre lies within 100ms of the A→B boundary at 185.17.
    seg = _seg_with(_words_with_prob([
        (184.84, 185.04, "you",  1.0),
        (185.04, 185.16, "know", 0.985),   # centre 185.10, boundary 185.17, dist 70ms
        (185.16, 185.38, "the",  1.0),
        (185.38, 185.66, "field", 1.0),
    ]))
    turns = [
        DiarTurn(start=180.00, end=185.17, speaker="A"),
        DiarTurn(start=185.17, end=190.00, speaker="B"),
    ]
    out, tel = split_on_turn_boundary([seg], turns, deadband_ms=100)
    assert tel["deadband_applied"] >= 1
    # 'know' should land on B's side (downstream).
    flat_words = [(w.content, s.speaker_hint) for s in out for w in (s.words or [])]
    know_owner = dict(flat_words)["know"]
    assert know_owner == "B"


def test_deadband_off_keeps_argmax():
    seg = _seg_with(_words_with_prob([
        (184.84, 185.04, "you",  1.0),
        (185.04, 185.16, "know", 0.985),
        (185.16, 185.38, "the",  1.0),
    ]))
    turns = [
        DiarTurn(start=180.00, end=185.17, speaker="A"),
        DiarTurn(start=185.17, end=190.00, speaker="B"),
    ]
    out, tel = split_on_turn_boundary([seg], turns)
    assert tel["deadband_applied"] == 0


def test_ac_combo_fixes_both_canonical_cases():
    # Both Case 1 ('know' near boundary) and Case 4 ('yes' straddling) on
    # one segment, the way they appear on the real reference.
    seg = _seg_with(_words_with_prob([
        (184.84, 185.04, "you",    1.0),
        (185.04, 185.16, "know",   0.985),   # → deadband flips to B
        (185.16, 185.38, "the",    1.0),
        (185.38, 185.66, "field",  1.0),
        (185.66, 185.94, "pretty", 1.0),
        (185.94, 186.30, "well",   1.0),
        (186.30, 187.20, "yes",    0.998),   # → filler-bias flips to C
        (187.20, 187.80, "yes",    0.999),
    ]))
    turns = [
        DiarTurn(start=180.00, end=185.17, speaker="A"),
        DiarTurn(start=185.17, end=186.55, speaker="B"),
        DiarTurn(start=187.02, end=194.11, speaker="C"),
    ]
    out, tel = split_on_turn_boundary(
        [seg], turns,
        filler_bias_tie_ms=120, filler_words=DEFAULT_FILLER_WORDS,
        deadband_ms=100,
    )
    assert tel["deadband_applied"] >= 1
    assert tel["filler_bias_applied"] >= 1
    flat = {w.content: s.speaker_hint for s in out for w in (s.words or [])}
    assert flat["know"] == "B"   # deadband recovered the leading word for B
    assert flat["yes"] == "C"    # filler-bias caught the back-channel
