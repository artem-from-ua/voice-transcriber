"""Offline tests for the per-segment ASR proof-reader."""

from __future__ import annotations

import pytest

from voice.llm import LLMError
from voice.proofread import fix_asr_errors, fix_segment
from voice.types import Segment


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


def _seg(content, start=0.0, end=1.0, speaker="SPEAKER_00"):
    return Segment(start=start, end=end, content=content, speaker=speaker)


def test_very_short_segment_skipped():
    llm = StubLLM(reply="should-not-be-called")
    out = fix_segment("Так.", llm=llm, language="uk")
    assert out == "Так."
    assert llm.calls == 0


def test_safe_fix_is_applied():
    original = "Це додаток на Hugging Space у фреймворку градіо."
    fixed = "Це додаток на Hugging Face у фреймворку Gradio."
    llm = StubLLM(reply=fixed)
    out = fix_segment(original, llm=llm, language="uk")
    assert out == fixed


def test_quotes_are_stripped():
    original = "коротка фраза для тесту"
    llm = StubLLM(reply=f'"{original}"')
    out = fix_segment(original, llm=llm, language="uk")
    assert out == original


def test_text_prefix_artifact_is_stripped():
    original = "Це додаток на Hugging Space у фреймворку градіо."
    fixed = "Це додаток на Hugging Face у фреймворку Gradio."
    llm = StubLLM(reply=f"Text: {fixed}")
    out = fix_segment(original, llm=llm, language="uk")
    assert out == fixed
    assert not out.startswith("Text:")


def test_llm_error_keeps_original():
    original = "якийсь нормальний текст для тесту"
    llm = StubLLM(error=LLMError("server down"))
    out = fix_segment(original, llm=llm, language="uk")
    assert out == original


def test_rejects_reply_with_too_many_edits():
    original = "коротка фраза про тестування ASR-постобробки"
    # totally different sentence — should be rejected by edit-distance check.
    llm = StubLLM(reply="хтось десь колись щось зробив зовсім інакше")
    out = fix_segment(original, llm=llm, language="uk")
    assert out == original


def test_rejects_reply_that_is_too_long():
    original = "коротка фраза про тестування"
    llm = StubLLM(reply=original + " " + original + " " + original + " " + original)
    out = fix_segment(original, llm=llm, language="uk")
    assert out == original


def test_rejects_empty_reply():
    original = "достатньо довгий рядок для перевірки"
    llm = StubLLM(reply="")
    out = fix_segment(original, llm=llm, language="uk")
    assert out == original


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

    out = fix_asr_errors(segs, llm=SeqLLM(), language="uk", log=lambda _s: None)
    assert [s.content for s in out] == [
        "ОК.",  # too short for MIN_LEN_FOR_FIX, passed through
        "додаток на Hugging Face там стоїть",
        "ще один сегмент текстовий нормальної довжини",
    ]


@pytest.mark.parametrize("text", ["", "   ", "\n"])
def test_blank_text_is_pass_through(text):
    llm = StubLLM(reply="x")
    assert fix_segment(text, llm=llm, language="uk") == text
