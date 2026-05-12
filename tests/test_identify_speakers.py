"""Offline tests for identify_speakers — no live LLM required.

Uses a stub LLM that returns canned chat_json responses keyed by snippet.
"""

from __future__ import annotations

from voice.identify_speakers import NamedAssignment, identify_speakers
from voice.types import Segment


class StubLLM:
    """Returns predetermined JSON payloads from a snippet → payload table.

    A `default` payload is used for snippets not in the table.
    """

    def __init__(self, table: dict[str, dict], default: dict | None = None):
        self.table = table
        self.default = default or {"name": None, "confidence": "high"}
        self.calls: list[str] = []

    def chat_json(self, messages, **_kwargs):
        user = next(m["content"] for m in messages if m["role"] == "user")
        self.calls.append(user)
        for key, payload in self.table.items():
            if key in user:
                return payload
        return self.default


def _seg(start, end, content, speaker):
    return Segment(start=start, end=end, content=content, speaker=speaker)


def test_names_override_skips_llm():
    segs = [_seg(0, 1, "x", "SPEAKER_01"), _seg(1, 2, "y", "SPEAKER_00")]
    llm = StubLLM({})
    mapping = identify_speakers(
        segs, llm=llm,
        names_override=["Artem", "Ostap"],
        log=lambda _s: None,
    )
    # Order is first-appearance in time, not label order: SPEAKER_01 first (start=0).
    assert mapping == {
        "SPEAKER_01": NamedAssignment(name="Artem", source="user-specified"),
        "SPEAKER_00": NamedAssignment(name="Ostap", source="user-specified"),
    }
    assert llm.calls == []


def test_extracts_name_from_self_intro():
    segs = [
        _seg(0.0, 3.0, "Привіт, я Артем, давай почнемо", "SPEAKER_00"),
        _seg(3.0, 6.0, "А я Остап", "SPEAKER_01"),
    ]
    llm = StubLLM({
        "Привіт, я Артем": {"name": "Артем", "confidence": "high"},
        "А я Остап":       {"name": "Остап", "confidence": "high"},
    })
    mapping = identify_speakers(
        segs, llm=llm, unknown_policy="keep",
        log=lambda _s: None,
    )
    assert mapping == {
        "SPEAKER_00": NamedAssignment(name="Артем", source="self-introduced"),
        "SPEAKER_01": NamedAssignment(name="Остап", source="self-introduced"),
    }


def test_no_intro_keeps_cluster_unmapped():
    segs = [_seg(0, 5, "Ну я думаю що ні", "SPEAKER_00")]
    llm = StubLLM({}, default={"name": None, "confidence": "high"})
    mapping = identify_speakers(
        segs, llm=llm, unknown_policy="keep",
        log=lambda _s: None,
    )
    assert mapping == {}


def test_low_confidence_is_rejected():
    segs = [_seg(0, 3, "сь ім'я тут", "SPEAKER_00")]
    llm = StubLLM({"сь ім": {"name": "Foo", "confidence": "low"}})
    mapping = identify_speakers(
        segs, llm=llm, unknown_policy="keep",
        log=lambda _s: None,
    )
    assert mapping == {}


def test_conflict_resolution_prefers_higher_confidence():
    segs = [
        _seg(0.0, 2.0, "intro-A", "SPEAKER_00"),
        _seg(2.0, 4.0, "intro-B", "SPEAKER_01"),
    ]
    llm = StubLLM({
        "intro-A": {"name": "Артем", "confidence": "medium"},
        "intro-B": {"name": "Артем", "confidence": "high"},
    })
    mapping = identify_speakers(
        segs, llm=llm, unknown_policy="keep",
        log=lambda _s: None,
    )
    assert mapping == {"SPEAKER_01": NamedAssignment(name="Артем", source="self-introduced")}


def test_ask_policy_uses_provided_input():
    segs = [_seg(0, 5, "ну так от", "SPEAKER_00")]
    llm = StubLLM({}, default={"name": None, "confidence": "high"})
    mapping = identify_speakers(
        segs, llm=llm, unknown_policy="ask",
        log=lambda _s: None,
        read_input=lambda _prompt: "Артем",
    )
    assert mapping == {"SPEAKER_00": NamedAssignment(name="Артем", source="interactive")}


def test_validates_name_rejects_lowercase():
    segs = [_seg(0, 3, "intro", "SPEAKER_00")]
    llm = StubLLM({"intro": {"name": "артем", "confidence": "high"}})  # lowercase
    mapping = identify_speakers(
        segs, llm=llm, unknown_policy="keep",
        log=lambda _s: None,
    )
    assert mapping == {}
