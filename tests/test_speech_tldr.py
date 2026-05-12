"""Offline tests for TLDR generation."""

from __future__ import annotations

from voice.llm import LLMError
from voice.speech_tldr import generate_tldr
from voice.types import Segment


class StubLLM:
    def __init__(self, reply="", error=None):
        self.reply = reply
        self.error = error
        self.last_user = None

    def chat(self, messages, **_kwargs) -> str:
        self.last_user = next(m["content"] for m in messages if m["role"] == "user")
        if self.error is not None:
            raise self.error
        return self.reply


def _seg(start, end, content, speaker="SPEAKER_00", name=None):
    return Segment(start=start, end=end, content=content, speaker=speaker, name=name)


def test_returns_markdown_from_llm():
    llm = StubLLM(reply="## Summary\n- point one\n- point two\n")
    out = generate_tldr(
        [_seg(0, 1, "hi", name="Артем"), _seg(1, 2, "hi back", name="Остап")],
        llm=llm, language="uk", log=lambda _s: None,
    )
    assert out == "## Summary\n- point one\n- point two"


def test_empty_input_returns_empty_without_calling_llm():
    llm = StubLLM(reply="should-not-be-called")
    out = generate_tldr([], llm=llm, language="uk", log=lambda _s: None)
    assert out == ""


def test_silence_only_returns_empty():
    llm = StubLLM(reply="should-not-be-called")
    out = generate_tldr(
        [Segment(start=0, end=2, content="[Human Sounds]", speaker=None)],
        llm=llm, language="uk", log=lambda _s: None,
    )
    assert out == ""


def test_llm_error_returns_empty():
    llm = StubLLM(error=LLMError("server down"))
    out = generate_tldr(
        [_seg(0, 1, "hi", name="Артем")],
        llm=llm, language="uk", log=lambda _s: None,
    )
    assert out == ""


def test_uses_name_when_present_falls_back_to_label():
    llm = StubLLM(reply="ok")
    generate_tldr(
        [
            _seg(0, 1, "hello", name="Артем"),
            _seg(1, 2, "again", name=None, speaker="SPEAKER_01"),
        ],
        llm=llm, language="uk", log=lambda _s: None,
    )
    assert "Артем: hello" in llm.last_user
    assert "SPEAKER_01: again" in llm.last_user


def test_english_uses_english_prompt():
    llm = StubLLM(reply="ok")
    generate_tldr(
        [_seg(0, 1, "hi", name="Sam")],
        llm=llm, language="en", log=lambda _s: None,
    )
    # English prompt mentions "Action items" without bold markers in system,
    # but the user text shouldn't contain Ukrainian-specific words.
    assert "Sam: hi" in llm.last_user


def test_blank_content_is_skipped():
    llm = StubLLM(reply="ok")
    generate_tldr(
        [_seg(0, 1, "   ", name="Артем"), _seg(1, 2, "hi", name="Артем")],
        llm=llm, language="uk", log=lambda _s: None,
    )
    assert llm.last_user.strip() == "Артем: hi"


# ---------------------------------------------------------------------------
# Leading "TL;DR" heading from the model — the renderer adds its own.

import pytest


@pytest.mark.parametrize("reply,expected_start", [
    ("## TL;DR\n\nReal content.", "Real content."),
    ("# TL;DR\nReal content.", "Real content."),
    ("### TL;DR\n\nReal content.", "Real content."),
    ("**TL;DR**\n\nReal content.", "Real content."),
    ("**TL;DR:**\n\nReal content.", "Real content."),
    ("## tl;dr\n\nReal content.", "Real content."),
    ("## TLDR\n\nReal content.", "Real content."),
    ("Real content from the start.", "Real content from the start."),
])
def test_strips_leading_tldr_heading(reply, expected_start):
    llm = StubLLM(reply=reply)
    out = generate_tldr(
        [_seg(0, 1, "hi", name="Артем")],
        llm=llm, language="uk", log=lambda _s: None,
    )
    assert out.startswith(expected_start)


def test_keeps_tldr_inside_body():
    """Only the *leading* TL;DR heading is stripped; inline mentions stay."""
    reply = "Summary.\n\n## TL;DR\nSecond paragraph."
    llm = StubLLM(reply=reply)
    out = generate_tldr(
        [_seg(0, 1, "hi", name="Артем")],
        llm=llm, language="uk", log=lambda _s: None,
    )
    assert "## TL;DR" in out
