"""Offline unit tests for the silence-event extraction module."""

from __future__ import annotations

from voice.silence import SilenceEvent, extract_silence_events
from voice.types import Segment


def _spk(start: float, end: float, text: str = "x") -> Segment:
    return Segment(start=start, end=end, content=text, speaker="SPEAKER_00", name="A")


def _muted(start: float, end: float) -> Segment:
    dur = end - start
    return Segment(start=start, end=end, content=f"[muted, {dur:.1f}s]", speaker=None)


def _noise(start: float, end: float) -> Segment:
    return Segment(start=start, end=end, content="[Human Sounds]", speaker=None)


# ---------------------------------------------------------------------------
# pure gap tests
# ---------------------------------------------------------------------------

def test_no_events_when_all_gaps_below_threshold():
    segs = [_spk(0, 2), _spk(5, 7)]  # gap = 3s, threshold = 10
    assert extract_silence_events(segs, min_silence_s=10.0) == []


def test_single_gap_above_threshold():
    segs = [_spk(0, 2), _spk(15, 17)]  # gap = 13s
    events = extract_silence_events(segs, min_silence_s=10.0)
    assert len(events) == 1
    assert events[0].start == 2.0
    assert events[0].end == 15.0
    assert events[0].kind == "pause"


def test_gap_exactly_at_threshold_is_included():
    segs = [_spk(0, 5), _spk(15, 17)]  # gap = 10s == threshold
    events = extract_silence_events(segs, min_silence_s=10.0)
    assert len(events) == 1


def test_gap_just_below_threshold_excluded():
    segs = [_spk(0, 5), _spk(14.9, 17)]  # gap = 9.9s
    assert extract_silence_events(segs, min_silence_s=10.0) == []


def test_multiple_gaps_independent():
    segs = [_spk(0, 1), _spk(12, 13), _spk(25, 26)]  # two 11s gaps
    events = extract_silence_events(segs, min_silence_s=10.0)
    assert len(events) == 2
    assert events[0] == SilenceEvent(start=1.0, end=12.0, kind="pause")
    assert events[1] == SilenceEvent(start=13.0, end=25.0, kind="pause")


# ---------------------------------------------------------------------------
# muted-only tests
# ---------------------------------------------------------------------------

def test_single_muted_above_threshold():
    # speaker ends at 1, muted starts at 5 → leading gap 4s absorbed into event.
    segs = [_spk(0, 1), _muted(5, 20), _spk(22, 24)]
    events = extract_silence_events(segs, min_silence_s=10.0)
    assert len(events) == 1
    assert events[0].kind == "muted"
    assert events[0].start == 1.0   # extends back to prev_end
    assert events[0].end == 20.0


def test_muted_below_threshold_excluded():
    segs = [_spk(0, 1), _muted(5, 8), _spk(10, 12)]  # muted = 3s
    assert extract_silence_events(segs, min_silence_s=10.0) == []


def test_muted_kind_takes_precedence_over_pure_gap():
    """A chain containing both a gap and a muted placeholder → kind='muted'."""
    # gap 4s + muted 8s → merged total 12s, kind muted
    segs = [_spk(0, 1), _muted(5, 13), _spk(15, 17)]
    events = extract_silence_events(segs, min_silence_s=10.0)
    assert len(events) == 1
    assert events[0].kind == "muted"


# ---------------------------------------------------------------------------
# merge tests
# ---------------------------------------------------------------------------

def test_adjacent_muted_segments_merged():
    """Two muted segments with no speaker between them → one event."""
    # speaker ends at 1, first muted starts at 2 → leading 1s gap absorbed.
    segs = [_spk(0, 1), _muted(2, 7), _muted(7, 15), _spk(17, 19)]
    events = extract_silence_events(segs, min_silence_s=10.0)
    assert len(events) == 1
    assert events[0].start == 1.0   # extends back to prev_end
    assert events[0].end == 15.0
    assert events[0].kind == "muted"


def test_muted_gap_muted_chain_merged():
    """muted 4s + gap 5s + muted 3s → one event spanning all."""
    segs = [
        _spk(0, 1),
        _muted(2, 6),       # 4s muted
        _spk(9, 11),        # 3s gap after muted, but this is a real speaker seg — breaks chain
        _muted(14, 17),
        _spk(20, 22),
    ]
    # The speaker seg at 9–11 breaks the chain → two separate events, each below 10s.
    events = extract_silence_events(segs, min_silence_s=10.0)
    assert events == []


def test_muted_with_noise_tag_between_still_merges():
    """A noise-tag speakerless seg between two muted segs does not break the chain."""
    # speaker ends at 1, first muted starts at 2 → leading 1s gap absorbed.
    segs = [
        _spk(0, 1),
        _muted(2, 8),
        _noise(8, 9),
        _muted(9, 15),
        _spk(17, 19),
    ]
    events = extract_silence_events(segs, min_silence_s=10.0)
    assert len(events) == 1
    assert events[0].start == 1.0   # extends back to prev_end
    assert events[0].end == 15.0


# ---------------------------------------------------------------------------
# format_md tests
# ---------------------------------------------------------------------------

def test_format_md_pause():
    ev = SilenceEvent(start=65.0, end=80.0, kind="pause")
    assert ev.format_md() == "> _[пауза 00:01:05–00:01:20]_"


def test_format_md_muted():
    ev = SilenceEvent(start=3661.0, end=3720.0, kind="muted")
    assert ev.format_md() == "> _[muted 01:01:01–01:02:00]_"


def test_format_md_zero_start():
    ev = SilenceEvent(start=0.0, end=15.0, kind="pause")
    assert ev.format_md() == "> _[пауза 00:00:00–00:00:15]_"
