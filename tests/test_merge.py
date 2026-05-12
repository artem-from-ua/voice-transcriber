"""Offline tests for merge: ASR × pyannote → assigned speaker."""

from voice.merge import merge
from voice.types import AsrSegment, DiarTurn


def test_picks_speaker_with_largest_overlap():
    asr = [AsrSegment(start=1.0, end=4.0, content="hello there")]
    turns = [
        DiarTurn(start=0.0, end=1.5, speaker="SPEAKER_00"),  # 0.5s overlap
        DiarTurn(start=1.5, end=4.0, speaker="SPEAKER_01"),  # 2.5s overlap → wins
    ]
    out = merge(asr, turns)
    assert out[0].speaker == "SPEAKER_01"


def test_no_overlap_yields_none_speaker():
    asr = [AsrSegment(start=10.0, end=11.0, content="orphan")]
    turns = [DiarTurn(start=0.0, end=2.0, speaker="SPEAKER_00")]
    out = merge(asr, turns)
    assert out[0].speaker is None
