"""Offline tests for the per-segment ASR proof-reader."""

from __future__ import annotations

from contextlib import contextmanager

import pytest

from voice.llm import LLMError
from voice.proofread import fix_asr_errors, fix_segment
from voice.types import Segment


@contextmanager
def _noop_session(*_args, **_kwargs):
    """Stand-in for `MlxLLM.prompt_cache_session()` on test stubs."""
    yield


class StubLLM:
    def __init__(self, reply: str | None = None, error: Exception | None = None):
        self.reply = reply
        self.error = error
        self.calls = 0

    def chat(self, *_args, **_kwargs) -> str:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.reply or ""

    prompt_cache_session = staticmethod(_noop_session)


def _seg(content, start=0.0, end=1.0, speaker="SPEAKER_00"):
    return Segment(start=start, end=end, content=content, speaker=speaker)


def test_very_short_segment_skipped():
    llm = StubLLM(reply="should-not-be-called")
    out, called = fix_segment("Так.", llm=llm, language="uk")
    assert out == "Так."
    assert called is False
    assert llm.calls == 0


def test_safe_fix_is_applied():
    original = "Це додаток на Hugging Space у фреймворку градіо."
    fixed = "Це додаток на Hugging Face у фреймворку Gradio."
    llm = StubLLM(reply=fixed)
    out, called = fix_segment(original, llm=llm, language="uk")
    assert out == fixed
    assert called is True


def test_quotes_are_stripped():
    original = "коротка фраза для тесту"
    llm = StubLLM(reply=f'"{original}"')
    out, called = fix_segment(original, llm=llm, language="uk")
    assert out == original
    assert called is True


def test_text_prefix_artifact_is_stripped():
    original = "Це додаток на Hugging Space у фреймворку градіо."
    fixed = "Це додаток на Hugging Face у фреймворку Gradio."
    llm = StubLLM(reply=f"Text: {fixed}")
    out, called = fix_segment(original, llm=llm, language="uk")
    assert out == fixed
    assert called is True
    assert not out.startswith("Text:")


def test_llm_error_keeps_original():
    original = "якийсь нормальний текст для тесту"
    llm = StubLLM(error=LLMError("server down"))
    out, called = fix_segment(original, llm=llm, language="uk")
    assert out == original
    # LLM was actually invoked — the call just raised. Telemetry counts
    # attempted calls, so this is True.
    assert called is True


def test_rejects_reply_with_too_many_edits():
    original = "коротка фраза про тестування ASR-постобробки"
    # totally different sentence — should be rejected by edit-distance check.
    llm = StubLLM(reply="хтось десь колись щось зробив зовсім інакше")
    out, called = fix_segment(original, llm=llm, language="uk")
    assert out == original
    assert called is True


def test_rejects_reply_that_is_too_long():
    original = "коротка фраза про тестування"
    llm = StubLLM(reply=original + " " + original + " " + original + " " + original)
    out, called = fix_segment(original, llm=llm, language="uk")
    assert out == original
    assert called is True


def test_rejects_empty_reply():
    original = "достатньо довгий рядок для перевірки"
    llm = StubLLM(reply="")
    out, called = fix_segment(original, llm=llm, language="uk")
    assert out == original
    assert called is True


def test_fix_asr_errors_preserves_order_and_counts():
    segs = [
        _seg("ОК.", 0, 1),
        _seg("додаток на Hugging Space там стоїть", 1, 4),
        _seg("ще один сегмент текстовий нормальної довжини", 4, 7),
    ]
    replies = iter([
        "додаток на Hugging Face там стоїть",   # fix applied
        "ще один сегмент текстовий нормальної довжини",  # unchanged
    ])

    class SeqLLM:
        def chat(self, *_a, **_k):
            return next(replies)

        prompt_cache_session = staticmethod(_noop_session)

    out, telemetry = fix_asr_errors(segs, llm=SeqLLM(), language="uk", log=lambda _s: None)
    assert [s.content for s in out] == [
        "ОК.",  # too short for MIN_LEN_FOR_FIX, passed through
        "додаток на Hugging Face там стоїть",
        "ще один сегмент текстовий нормальної довжини",
    ]
    # Two segments were long enough to trigger the LLM (the first is below
    # MIN_LEN_FOR_FIX). Counter mirrors actual `llm.chat()` invocations.
    assert telemetry == {"llm_calls": 2}


def test_fix_asr_errors_telemetry_skips_count_zero_when_all_too_short():
    segs = [_seg("ОК.", 0, 1), _seg("Так.", 1, 2)]

    class NeverCalledLLM:
        def chat(self, *_a, **_k):
            raise AssertionError("should not be called for too-short segments")

        prompt_cache_session = staticmethod(_noop_session)

    out, telemetry = fix_asr_errors(
        segs, llm=NeverCalledLLM(), language="uk", log=lambda _s: None,
    )
    assert [s.content for s in out] == ["ОК.", "Так."]
    assert telemetry == {"llm_calls": 0}


@pytest.mark.parametrize("text", ["", "   ", "\n"])
def test_blank_text_is_pass_through(text):
    llm = StubLLM(reply="x")
    out, called = fix_segment(text, llm=llm, language="uk")
    assert out == text
    assert called is False
