"""Offline tests for section structuring."""

from __future__ import annotations

from voice.llm import LLMError
from voice.structure import structure_dialog
from voice.types import Segment


class StubLLM:
    def __init__(self, payload=None, error=None):
        self.payload = payload
        self.error = error
        self.calls = 0

    def chat_json(self, *_args, **_kwargs):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.payload


def _seg(start, end, content, speaker="SPEAKER_00", name="Артем"):
    return Segment(start=start, end=end, content=content, speaker=speaker, name=name)


def test_no_llm_returns_single_section():
    segs = [_seg(0.0, 10.0, "hello")]
    sd = structure_dialog(segs, llm=None, log=lambda _s: None)
    assert len(sd.sections) == 1
    assert sd.sections[0].title == "Розмова"
    assert sd.sections[0].start_ms == 0
    assert sd.sections[0].end_ms == 10000


def test_no_speech_returns_empty_sections():
    segs = [Segment(start=0.0, end=2.0, content="[Human Sounds]", speaker=None)]
    sd = structure_dialog(segs, llm=None, log=lambda _s: None)
    assert sd.sections == []
    assert sd.segments == segs


def test_valid_layout_accepted():
    segs = [
        _seg(0.0, 30.0, "intro"),
        _seg(30.0, 60.0, "topic A"),
        _seg(60.0, 90.0, "topic B"),
    ]
    payload = {"sections": [
        {"title": "Привітання", "start_ms": 0, "end_ms": 30000},
        {"title": "Тема А", "start_ms": 30000, "end_ms": 60000},
        {"title": "Тема Б", "start_ms": 60000, "end_ms": 90000},
    ]}
    llm = StubLLM(payload=payload)
    sd = structure_dialog(segs, llm=llm, log=lambda _s: None)
    assert [s.title for s in sd.sections] == ["Привітання", "Тема А", "Тема Б"]


def test_non_contiguous_falls_back():
    segs = [_seg(0.0, 60.0, "x")]
    payload = {"sections": [
        {"title": "A", "start_ms": 0, "end_ms": 20000},
        {"title": "B", "start_ms": 30000, "end_ms": 60000},  # gap
    ]}
    sd = structure_dialog(segs, llm=StubLLM(payload=payload), log=lambda _s: None)
    assert len(sd.sections) == 1
    assert sd.sections[0].title == "Розмова"


def test_too_few_sections_falls_back():
    segs = [_seg(0.0, 60.0, "x")]
    payload = {"sections": [{"title": "Only", "start_ms": 0, "end_ms": 60000}]}
    sd = structure_dialog(segs, llm=StubLLM(payload=payload), log=lambda _s: None)
    assert sd.sections[0].title == "Розмова"


def test_too_many_sections_falls_back():
    segs = [_seg(0.0, 800.0, "x")]
    payload = {"sections": [
        {"title": f"S{i}", "start_ms": i * 100000, "end_ms": (i + 1) * 100000}
        for i in range(8)
    ]}
    sd = structure_dialog(segs, llm=StubLLM(payload=payload), log=lambda _s: None)
    assert sd.sections[0].title == "Розмова"


def test_invalid_json_shape_falls_back():
    segs = [_seg(0.0, 30.0, "x")]
    sd = structure_dialog(segs, llm=StubLLM(payload=["not", "a", "dict"]), log=lambda _s: None)
    assert sd.sections[0].title == "Розмова"


def test_llm_error_falls_back():
    segs = [_seg(0.0, 30.0, "x")]
    sd = structure_dialog(
        segs, llm=StubLLM(error=LLMError("down")), log=lambda _s: None,
    )
    assert sd.sections[0].title == "Розмова"


def test_english_fallback_title():
    segs = [_seg(0.0, 10.0, "hi")]
    sd = structure_dialog(segs, llm=None, language="en", log=lambda _s: None)
    assert sd.sections[0].title == "Conversation"


def test_bounds_must_match_total_dialogue():
    segs = [_seg(0.0, 60.0, "x")]
    payload = {"sections": [
        {"title": "A", "start_ms": 0, "end_ms": 30000},
        {"title": "B", "start_ms": 30000, "end_ms": 50000},  # ends short
    ]}
    sd = structure_dialog(segs, llm=StubLLM(payload=payload), log=lambda _s: None)
    assert sd.sections[0].title == "Розмова"
