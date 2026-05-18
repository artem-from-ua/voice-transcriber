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
