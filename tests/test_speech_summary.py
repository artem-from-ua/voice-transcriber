"""Offline tests for the recursive per-section TL;DR pipeline."""

from __future__ import annotations

from contextlib import contextmanager

import pytest

from voice import speech_summary as ss
from voice.llm import LLMError
from voice.speech_summary import generate_tldr
from voice.types import Section, Segment, StructuredDialog


class StubLLM:
    """Records every chat() call. Optional `replies` queue supplies per-call
    responses; `error_on` indexes into the call sequence to raise instead.
    Implements `prompt_cache_session` as a no-op context manager so callers
    that use it work unchanged."""

    def __init__(
        self,
        reply: str = "ok",
        replies: list[str] | None = None,
        error: Exception | None = None,
        error_on: set[int] | None = None,
    ) -> None:
        self._default = reply
        self._replies = list(replies) if replies else None
        self._error = error
        self._error_on = error_on or set()
        self.calls: list[dict] = []  # one entry per chat() call

    def chat(self, messages, **kwargs) -> str:
        idx = len(self.calls)
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user = next((m["content"] for m in messages if m["role"] == "user"), "")
        self.calls.append({"system": system, "user": user, "kwargs": kwargs})
        if self._error is not None:
            raise self._error
        if idx in self._error_on:
            raise LLMError(f"stub error on call {idx}")
        if self._replies is not None and idx < len(self._replies):
            return self._replies[idx]
        return self._default

    @contextmanager
    def prompt_cache_session(self, prefix_messages):
        yield


@pytest.fixture(autouse=True)
def _silence_mlx(monkeypatch):
    """Bypass the real mlx.core calls in speech_summary so tests don't need
    the MLX runtime. We also replace `mx.get_peak_memory()` etc. with
    stubs that return 0 so the log lines stay deterministic."""

    class _MxStub:
        @staticmethod
        def clear_cache():
            return None

        @staticmethod
        def get_active_memory():
            return 0

        @staticmethod
        def get_peak_memory():
            return 0

        @staticmethod
        def reset_peak_memory():
            return None

    monkeypatch.setattr(ss, "mx", _MxStub)


def _seg(start: float, end: float, content: str, speaker: str = "SPEAKER_00",
         name: str | None = None) -> Segment:
    return Segment(start=start, end=end, content=content, speaker=speaker, name=name)


def _dialog(segments: list[Segment], sections: list[Section]) -> StructuredDialog:
    return StructuredDialog(sections=sections, segments=segments)


# ---------------------------------------------------------------------------
# Happy path: 2 real sections -> 2 section calls + 1 final = 3 calls
# ---------------------------------------------------------------------------

def test_two_sections_produce_three_llm_calls():
    segs = [
        _seg(0.0, 1.0, "перше", name="Артем"),
        _seg(1.0, 2.0, "відповідь", name="Остап"),
        _seg(2.0, 3.0, "друге", name="Артем"),
        _seg(3.0, 4.0, "знов", name="Остап"),
    ]
    sections = [
        Section(title="Вступ", start_ms=0, end_ms=2000),
        Section(title="Висновок", start_ms=2000, end_ms=4000),
    ]
    llm = StubLLM(replies=["sec1 body", "sec2 body", "## Підсумок\n- one\n- two"])
    out = generate_tldr(_dialog(segs, sections), llm=llm, language="uk", log=lambda _s: None)

    assert len(llm.calls) == 3
    # First two calls use the section prompt; third uses the final prompt.
    assert "TL;DR саме цієї секції" in llm.calls[0]["system"]
    assert "TL;DR саме цієї секції" in llm.calls[1]["system"]
    assert "фінальний TL;DR" in llm.calls[2]["system"]
    # Final call's user content contains both section bodies.
    assert "sec1 body" in llm.calls[2]["user"]
    assert "sec2 body" in llm.calls[2]["user"]
    # Output is the final pass body.
    assert "Підсумок" in out


def test_single_real_section_runs_section_then_final():
    """One real (non-synthetic) section: 1 section call + 1 final call."""
    segs = [_seg(0.0, 1.0, "привіт", name="Артем")]
    sections = [Section(title="Привітання", start_ms=0, end_ms=1000)]
    llm = StubLLM(replies=["section body", "final body"])
    out = generate_tldr(_dialog(segs, sections), llm=llm, language="uk", log=lambda _s: None)

    assert len(llm.calls) == 2
    assert out == "final body"


# ---------------------------------------------------------------------------
# Skip paths
# ---------------------------------------------------------------------------

def test_no_segments_returns_empty_without_llm():
    llm = StubLLM()
    out = generate_tldr(_dialog([], []), llm=llm, language="uk", log=lambda _s: None)
    assert out == ""
    assert llm.calls == []


def test_synthetic_fallback_section_is_skipped_with_warning(capsys):
    """Single section spanning the full dialog with title 'Розмова' is the
    synthetic fallback produced by --no-structure. The stage must skip it."""
    segs = [
        _seg(0.0, 1.0, "hi", name="Артем"),
        _seg(10.0, 11.0, "bye", name="Артем"),
    ]
    # _fallback_section uses int(start*1000) and int(end*1000):
    synthetic = Section(title="Розмова", start_ms=0, end_ms=11000)
    llm = StubLLM()
    msgs: list[str] = []
    out = generate_tldr(
        _dialog(segs, [synthetic]),
        llm=llm, language="uk", log=msgs.append,
    )
    assert out == ""
    assert llm.calls == []
    assert any("skipping TL;DR" in m for m in msgs)


def test_synthetic_fallback_english_title_also_skipped():
    segs = [_seg(0.0, 1.0, "hi", name="Sam"), _seg(5.0, 6.0, "bye", name="Sam")]
    synthetic = Section(title="Conversation", start_ms=0, end_ms=6000)
    llm = StubLLM()
    msgs: list[str] = []
    out = generate_tldr(
        _dialog(segs, [synthetic]),
        llm=llm, language="en", log=msgs.append,
    )
    assert out == ""
    assert llm.calls == []
    assert any("skipping TL;DR" in m for m in msgs)


def test_real_one_section_with_fallback_title_is_not_skipped():
    """If the only section happens to be called 'Розмова' but does NOT span
    the full dialog, it's a real section (some segments filtered out) and
    should be summarised normally."""
    segs = [
        _seg(0.0, 1.0, "early", name="Артем"),     # before section starts
        _seg(5.0, 6.0, "in-section", name="Артем"),
    ]
    real = Section(title="Розмова", start_ms=5000, end_ms=6000)  # doesn't span
    llm = StubLLM(replies=["sec body", "final body"])
    out = generate_tldr(_dialog(segs, [real]), llm=llm, language="uk", log=lambda _s: None)
    assert out == "final body"
    assert len(llm.calls) == 2


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_section_llm_error_skips_that_section_only():
    """Section 1 fails, sections 0 and 2 succeed -> final pass sees 2 bodies."""
    segs = [
        _seg(0.0, 1.0, "a", name="A"),
        _seg(1.0, 2.0, "b", name="A"),
        _seg(2.0, 3.0, "c", name="A"),
    ]
    sections = [
        Section(title="S1", start_ms=0, end_ms=1000),
        Section(title="S2", start_ms=1000, end_ms=2000),
        Section(title="S3", start_ms=2000, end_ms=3000),
    ]
    llm = StubLLM(replies=["body1", "ignored", "body3", "final"], error_on={1})
    out = generate_tldr(_dialog(segs, sections), llm=llm, language="uk", log=lambda _s: None)

    # Calls: 3 section attempts (one fails) + 1 final = 4
    assert len(llm.calls) == 4
    final_user = llm.calls[3]["user"]
    assert "body1" in final_user
    assert "body3" in final_user
    assert "ignored" not in final_user
    assert out == "final"


def test_final_pass_llm_error_returns_empty():
    segs = [
        _seg(0.0, 1.0, "a", name="A"),
        _seg(1.0, 2.0, "b", name="A"),
    ]
    sections = [
        Section(title="S1", start_ms=0, end_ms=1000),
        Section(title="S2", start_ms=1000, end_ms=2000),
    ]
    llm = StubLLM(replies=["body1", "body2", "ignored"], error_on={2})
    out = generate_tldr(_dialog(segs, sections), llm=llm, language="uk", log=lambda _s: None)
    assert out == ""


def test_all_sections_fail_returns_empty_before_final():
    segs = [_seg(0.0, 1.0, "a", name="A"), _seg(1.0, 2.0, "b", name="A")]
    sections = [
        Section(title="S1", start_ms=0, end_ms=1000),
        Section(title="S2", start_ms=1000, end_ms=2000),
    ]
    llm = StubLLM(replies=["x", "y"], error_on={0, 1})
    out = generate_tldr(_dialog(segs, sections), llm=llm, language="uk", log=lambda _s: None)
    assert out == ""
    # No final call should have happened.
    assert len(llm.calls) == 2


# ---------------------------------------------------------------------------
# Language routing
# ---------------------------------------------------------------------------

def test_english_path_uses_english_prompts():
    segs = [_seg(0.0, 1.0, "hi", name="Sam"), _seg(1.0, 2.0, "bye", name="Sam")]
    sections = [
        Section(title="Greeting", start_ms=0, end_ms=1000),
        Section(title="Farewell", start_ms=1000, end_ms=2000),
    ]
    llm = StubLLM(replies=["s1", "s2", "final"])
    generate_tldr(_dialog(segs, sections), llm=llm, language="en", log=lambda _s: None)
    # English section prompt mentions "transcript of **one section**"
    assert "transcript of **one section**" in llm.calls[0]["system"]
    # English final prompt mentions "final Markdown TL;DR"
    assert "final Markdown TL;DR" in llm.calls[2]["system"]


# ---------------------------------------------------------------------------
# Recursive aggregate path
# ---------------------------------------------------------------------------

def test_recursive_path_with_lowered_fanout(monkeypatch):
    """With TLDR_FANOUT=2 and 5 sections, the recursion unfolds as:

      level 0: 5 section calls          -> blocks=[r0,r1,r2,r3,r4] (5)
      level 1: groups [r0,r1],[r2,r3],[r4]
               -> 2 aggregate calls + 1 carry-through (singleton)
                                       -> blocks=[a,a,r4] (3)
      level 2: groups [a,a],[r4]
               -> 1 aggregate + 1 carry -> blocks=[a,r4] (2)
      2 is not > FANOUT=2, exit loop.
      final: 1 call.

      Total LLM calls = 5 + 2 + 1 + 1 = 9.
    """
    monkeypatch.setattr(ss, "TLDR_FANOUT", 2)
    segs = [_seg(float(i), float(i + 1), f"text{i}", name="A") for i in range(5)]
    sections = [
        Section(title=f"S{i}", start_ms=i * 1000, end_ms=(i + 1) * 1000)
        for i in range(5)
    ]
    llm = StubLLM(replies=[f"r{i}" for i in range(20)])
    out = generate_tldr(_dialog(segs, sections), llm=llm, language="uk", log=lambda _s: None)

    assert len(llm.calls) == 9
    # Level 0 calls 0..4 use the section prompt.
    for i in range(5):
        assert "TL;DR саме цієї секції" in llm.calls[i]["system"]
    # Calls 5,6 are aggregate (level 1 groups of 2).
    for i in (5, 6):
        assert "вже стиснені TL;DR" in llm.calls[i]["system"]
    # Call 7 is aggregate (level 2 group of 2 aggregates).
    assert "вже стиснені TL;DR" in llm.calls[7]["system"]
    # Final call uses the final prompt.
    assert "фінальний TL;DR" in llm.calls[-1]["system"]
    assert out  # non-empty


def test_max_levels_cap_logs_warning(monkeypatch):
    """With TLDR_FANOUT=2 and TLDR_MAX_LEVELS=1, 5 sections produces 5
    section calls + 1 level-1 pass (2 aggregate calls, 1 carry) and then
    bumps into the cap with 3 blocks > FANOUT=2 -> warning + final."""
    monkeypatch.setattr(ss, "TLDR_FANOUT", 2)
    monkeypatch.setattr(ss, "TLDR_MAX_LEVELS", 1)
    segs = [_seg(float(i), float(i + 1), f"t{i}", name="A") for i in range(5)]
    sections = [
        Section(title=f"S{i}", start_ms=i * 1000, end_ms=(i + 1) * 1000)
        for i in range(5)
    ]
    msgs: list[str] = []
    llm = StubLLM(replies=[f"r{i}" for i in range(20)])
    generate_tldr(_dialog(segs, sections), llm=llm, language="uk", log=msgs.append)

    assert any("TLDR_MAX_LEVELS=1" in m for m in msgs)


# ---------------------------------------------------------------------------
# Leading-heading strip and silence injection (preserved from old behaviour)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("reply,expected_start", [
    ("## TL;DR\n\nReal content.", "Real content."),
    ("# TL;DR\nReal content.", "Real content."),
    ("**TL;DR**\n\nReal content.", "Real content."),
    ("Real content from the start.", "Real content from the start."),
])
def test_final_pass_strips_leading_tldr_heading(reply, expected_start):
    segs = [_seg(0.0, 1.0, "hi", name="A"), _seg(1.0, 2.0, "bye", name="A")]
    sections = [
        Section(title="S1", start_ms=0, end_ms=1000),
        Section(title="S2", start_ms=1000, end_ms=2000),
    ]
    llm = StubLLM(replies=["s1", "s2", reply])
    out = generate_tldr(_dialog(segs, sections), llm=llm, language="uk", log=lambda _s: None)
    assert out.startswith(expected_start)


def test_include_silence_injects_pause_into_section_prompt():
    """When include_silence=True, a long gap inside a section appears in
    that section's LLM user content. Silence is detected per-section now
    (each section is summarised independently), so the pause must live
    inside one section's segment list."""
    segs = [
        _seg(0.0, 1.0, "перше.", name="Артем"),
        _seg(15.0, 16.0, "друге.", name="Артем"),
        _seg(20.0, 21.0, "інша секція.", name="Артем"),
    ]
    sections = [
        Section(title="Перша", start_ms=0, end_ms=17000),
        Section(title="Друга", start_ms=20000, end_ms=21000),
    ]
    llm = StubLLM(replies=["s1", "s2", "final"])
    generate_tldr(
        _dialog(segs, sections), llm=llm, language="uk",
        include_silence=True, log=lambda _s: None,
    )
    # Section 1 contains both pre-pause and post-pause segments so the
    # silence marker is injected between them.
    assert "пауза" in llm.calls[0]["user"]


def test_uses_name_when_present_falls_back_to_label():
    segs = [
        _seg(0.0, 1.0, "hello", name="Артем"),
        _seg(1.0, 2.0, "again", name=None, speaker="SPEAKER_01"),
    ]
    sections = [Section(title="Розмова", start_ms=0, end_ms=2000)]
    # Sentinel-skip would trigger here if start_ms/end_ms matched the
    # full dialog AND title was 'Розмова'. Here end_ms (2000) matches
    # int(2.0*1000)=2000, so this IS the synthetic fallback shape.
    # To exercise the speaker-naming path we use a non-fallback title.
    sections = [Section(title="Привітання", start_ms=0, end_ms=2000)]
    llm = StubLLM(replies=["section body", "final body"])
    generate_tldr(_dialog(segs, sections), llm=llm, language="uk", log=lambda _s: None)
    assert "Артем: hello" in llm.calls[0]["user"]
    assert "SPEAKER_01: again" in llm.calls[0]["user"]
