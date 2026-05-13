"""Offline tests for the safe_speech redaction stage."""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import pytest

from voice.llm import LLMError
from voice.safe_speech import DEFAULT_TOPICS, Decision, redact_dialog
from voice.types import Section, Segment, StructuredDialog

FIXTURES = Path(__file__).parent / "fixtures"


@contextmanager
def _noop_session(*_a, **_k):
    yield


class StubLLM:
    """Minimal LLM stub: returns pre-canned chat_json responses in order."""

    def __init__(self, responses: list[dict], error_on: int | None = None):
        self._responses = responses
        self._error_on = error_on
        self._call_count = 0

    def chat_json(self, messages, *, schema=None, on_token=None, **_kwargs):
        idx = self._call_count
        self._call_count += 1
        if self._error_on is not None and idx == self._error_on:
            raise LLMError("stub error")
        return self._responses[idx % len(self._responses)]

    prompt_cache_session = staticmethod(_noop_session)


def _seg(start, end, content, speaker="SPEAKER_00", name="Артем"):
    return Segment(start=start, end=end, content=content, speaker=speaker, name=name)


def _dialog(*sections_and_segs):
    sections, segs = sections_and_segs[0], sections_and_segs[1]
    return StructuredDialog(sections=sections, segments=segs)


def _load_fixture(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _dialog_from_json(data: dict) -> StructuredDialog:
    sections = [Section(**s) for s in data["sections"]]
    segments = [
        Segment(
            start=s["start"], end=s["end"], content=s["content"],
            speaker=s["speaker"], name=s["name"],
        )
        for s in data["segments"]
    ]
    return StructuredDialog(sections=sections, segments=segments)


# ---------------------------------------------------------------------------
# Basic behaviour
# ---------------------------------------------------------------------------

def test_empty_topics_skips_llm():
    llm = StubLLM([{"redactions": [{"start_idx": 0, "end_idx": 0, "topic": "health", "rationale": "x"}]}])
    dialog = StructuredDialog(
        sections=[Section("A", 0, 5000)],
        segments=[_seg(0, 5, "sensitive content")],
    )
    result, decisions = redact_dialog(dialog, llm=llm, topics=[], log=lambda _: None)
    assert llm._call_count == 0
    assert decisions == []
    assert result.segments[0].content == "sensitive content"


def test_redact_single_segment_placeholder():
    llm = StubLLM([{"redactions": [{"start_idx": 0, "end_idx": 0, "topic": "health", "rationale": "x"}]}])
    dialog = StructuredDialog(
        sections=[Section("S", 0, 10000)],
        segments=[_seg(0.0, 3.4, "У мене діабет.")],
    )
    result, decisions = redact_dialog(
        dialog, llm=llm, topics=["health"], policy="placeholder", log=lambda _: None,
    )
    assert len(result.segments) == 1
    assert result.segments[0].content == "[muted, 3.4s]"
    assert result.segments[0].speaker is None
    assert len(decisions) == 1
    assert decisions[0].topic == "health"
    assert decisions[0].total_dur_s == pytest.approx(3.4, abs=0.05)


def test_redact_range_placeholder():
    llm = StubLLM([{"redactions": [{"start_idx": 1, "end_idx": 2, "topic": "alcohol", "rationale": "x"}]}])
    segs = [
        _seg(0.0, 5.0, "Перший нейтральний."),
        _seg(5.0, 13.4, "Вчора пив."),
        _seg(13.4, 18.7, "І знову пив."),
        _seg(18.7, 23.0, "Четвертий нейтральний."),
    ]
    dialog = StructuredDialog(sections=[Section("S", 0, 25000)], segments=segs)
    result, decisions = redact_dialog(
        dialog, llm=llm, topics=["alcohol"], policy="placeholder", log=lambda _: None,
    )
    assert len(result.segments) == 3
    assert result.segments[0].content == "Перший нейтральний."
    assert result.segments[1].content == "[muted, 13.7s]"
    assert result.segments[1].speaker is None
    assert result.segments[2].content == "Четвертий нейтральний."
    assert decisions[0].total_dur_s == pytest.approx(13.7, abs=0.05)


def test_redact_whole_section_placeholder():
    llm = StubLLM([{"redactions": [{"start_idx": 0, "end_idx": 2, "topic": "health", "rationale": "x"}]}])
    segs = [
        _seg(30.0, 34.4, "Маю діагноз."),
        _seg(34.4, 40.0, "Лікар казав..."),
        _seg(40.0, 43.3, "Дотримуюсь рекомендацій."),
    ]
    dialog = StructuredDialog(sections=[Section("Здоров'я", 30000, 45000)], segments=segs)
    result, decisions = redact_dialog(
        dialog, llm=llm, topics=["health"], policy="placeholder", log=lambda _: None,
    )
    assert len(result.segments) == 1
    assert result.segments[0].content == "[muted, 13.3s]"
    assert result.segments[0].speaker is None
    assert len(result.sections) == 1


def test_redact_range_drop():
    llm = StubLLM([{"redactions": [{"start_idx": 1, "end_idx": 2, "topic": "alcohol", "rationale": "x"}]}])
    segs = [
        _seg(0.0, 5.0, "Перший нейтральний."),
        _seg(5.0, 10.0, "Пив вчора."),
        _seg(10.0, 15.3, "І позавчора."),
        _seg(15.3, 20.0, "Нейтральний."),
    ]
    dialog = StructuredDialog(sections=[Section("S", 0, 22000)], segments=segs)
    result, decisions = redact_dialog(
        dialog, llm=llm, topics=["alcohol"], policy="drop", log=lambda _: None,
    )
    assert len(result.segments) == 2
    assert result.segments[0].content == "Перший нейтральний."
    assert result.segments[1].content == "Нейтральний."


def test_drop_empty_section_keeps_header():
    llm = StubLLM([{"redactions": [{"start_idx": 0, "end_idx": 0, "topic": "health", "rationale": "x"}]}])
    segs = [_seg(155.0, 159.0, "Відчуваю депресію.")]
    dialog = StructuredDialog(
        sections=[Section("Самотньо", 155000, 165000)],
        segments=segs,
    )
    result, decisions = redact_dialog(
        dialog, llm=llm, topics=["health"], policy="drop", log=lambda _: None,
    )
    assert len(result.sections) == 1
    assert result.sections[0].title == "Самотньо"
    assert len(result.segments) == 1
    assert result.segments[0].content == "[muted, 0.0s]"


def test_false_positive_guard():
    llm = StubLLM([{"redactions": []}])
    segs = [
        _seg(121.0, 130.0, "Читав про наркотики в Португалії."),
        _seg(131.0, 138.0, "Цікава тема."),
        _seg(139.0, 147.0, "Результати вражають."),
    ]
    dialog = StructuredDialog(sections=[Section("Новини", 120000, 150000)], segments=segs)
    result, decisions = redact_dialog(
        dialog, llm=llm, topics=["drugs"], policy="placeholder", log=lambda _: None,
    )
    assert decisions == []
    assert len(result.segments) == 3
    assert result.segments[0].content == "Читав про наркотики в Португалії."


def test_llm_error_one_section():
    responses = [
        {"redactions": [{"start_idx": 0, "end_idx": 0, "topic": "health", "rationale": "x"}]},
        {"redactions": []},
    ]
    llm = StubLLM(responses, error_on=0)
    segs = [
        _seg(0.0, 5.0, "Маю хворобу.", speaker="SPEAKER_00", name="Артем"),
        _seg(30.0, 35.0, "Нейтральний.", speaker="SPEAKER_01", name="Остап"),
    ]
    sections = [Section("A", 0, 25000), Section("B", 25000, 40000)]
    dialog = StructuredDialog(sections=sections, segments=segs)
    warnings = []
    result, decisions = redact_dialog(
        dialog, llm=llm, topics=["health"], policy="placeholder",
        log=lambda s: warnings.append(s),
    )
    assert any("LLM error" in w for w in warnings)
    assert result.segments[0].content == "Маю хворобу."
    assert decisions == []


def test_decisions_dataclass_shape():
    llm = StubLLM([{"redactions": [{"start_idx": 0, "end_idx": 0, "topic": "health", "rationale": "особисте"}]}])
    segs = [_seg(0.0, 5.0, "Маю хворобу.")]
    dialog = StructuredDialog(sections=[Section("Test", 0, 10000)], segments=segs)
    _, decisions = redact_dialog(
        dialog, llm=llm, topics=["health"], policy="placeholder", log=lambda _: None,
    )
    assert len(decisions) == 1
    d = decisions[0]
    assert d.section_title == "Test"
    assert d.start_idx == 0
    assert d.end_idx == 0
    assert d.topic == "health"
    assert d.rationale == "особисте"
    assert isinstance(d.total_dur_s, float)


# ---------------------------------------------------------------------------
# Golden tests against full fixture dialog
# ---------------------------------------------------------------------------

def _stub_from_fixture() -> StubLLM:
    responses = _load_fixture("safe_speech_llm_responses.json")
    return StubLLM(responses)


def _segs_content(dialog: StructuredDialog) -> list[str]:
    return [s.content for s in dialog.segments]


def test_golden_placeholder_full_dialog():
    fixture_dialog = _dialog_from_json(_load_fixture("safe_speech_dialog.json"))
    expected = _dialog_from_json(_load_fixture("safe_speech_expected_placeholder.json"))

    result, decisions = redact_dialog(
        fixture_dialog,
        llm=_stub_from_fixture(),
        topics=DEFAULT_TOPICS,
        policy="placeholder",
        log=lambda _: None,
    )

    assert _segs_content(result) == _segs_content(expected)
    # No original sensitive text should remain
    sensitive_phrases = ["цукровий діабет", "перебрав на вечірці", "відчуваю депресію"]
    all_content = " ".join(_segs_content(result))
    for phrase in sensitive_phrases:
        assert phrase.lower() not in all_content.lower(), f"Sensitive phrase survived: {phrase!r}"

    assert len(decisions) == 3


def test_golden_drop_full_dialog():
    fixture_dialog = _dialog_from_json(_load_fixture("safe_speech_dialog.json"))
    expected = _dialog_from_json(_load_fixture("safe_speech_expected_drop.json"))

    result, decisions = redact_dialog(
        fixture_dialog,
        llm=_stub_from_fixture(),
        topics=DEFAULT_TOPICS,
        policy="drop",
        log=lambda _: None,
    )

    assert _segs_content(result) == _segs_content(expected)
    # No section should be completely empty (sections with all segments dropped get a synthetic marker)
    for section in result.sections:
        section_segs = [
            s for s in result.segments
            if section.start_ms <= int(s.start * 1000) < section.end_ms
        ]
        assert len(section_segs) >= 1, f"Section {section.title!r} is completely empty"
