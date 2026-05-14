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
    assert telemetry["llm_calls"] == 2
    # Telemetry carries the n_context used for the run so the dump artefact
    # records the parameter; the exact value here equals the default.
    assert "n_context" in telemetry


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
    assert telemetry["llm_calls"] == 0
    assert "n_context" in telemetry


@pytest.mark.parametrize("text", ["", "   ", "\n"])
def test_blank_text_is_pass_through(text):
    llm = StubLLM(reply="x")
    out, called = fix_segment(text, llm=llm, language="uk")
    assert out == text
    assert called is False


# ---------------------------------------------------------------------------
# Context-aware behaviour (introduced in v0.31.0, see iteration 2 of
# docs/benchmarks/proofread-hit-rate.md).
# ---------------------------------------------------------------------------


def _recording_chat_llm(reply: str):
    """LLM stub that records the user message it last saw."""

    class RecordingLLM:
        def __init__(self) -> None:
            self.last_user: str | None = None
            self.last_system: str | None = None
            self.calls = 0

        def chat(self, messages, **_kwargs):  # noqa: D401, ANN001
            self.calls += 1
            for msg in messages:
                if msg["role"] == "system":
                    self.last_system = msg["content"]
                elif msg["role"] == "user":
                    self.last_user = msg["content"]
            return reply

        prompt_cache_session = staticmethod(_noop_session)

    return RecordingLLM()


def test_fix_segment_no_context_collapses_to_bare_text():
    """When both context blocks are empty, the user prompt is the segment
    text alone — same shape as the v0.29.0 baseline so the prefix cache
    stays maximally effective."""
    rec = _recording_chat_llm(reply="this is the corrected segment text")
    out, called = fix_segment(
        "this is the original segment text",
        llm=rec,
        language="uk",
    )
    assert called is True
    assert rec.last_user is not None
    # No CONTEXT BEFORE / CONTEXT AFTER framing — bare text.
    assert "CONTEXT BEFORE" not in rec.last_user
    assert "CONTEXT AFTER" not in rec.last_user
    assert "CURRENT SEGMENT" not in rec.last_user
    assert rec.last_user.strip() == "this is the original segment text"


def test_fix_segment_with_context_wraps_in_labelled_blocks():
    rec = _recording_chat_llm(reply="corrected segment with surrounding context")
    fix_segment(
        "current segment to fix with surrounding context",
        llm=rec,
        language="uk",
        context_before="[SPEAKER_00] earlier turn one\n[SPEAKER_01] earlier turn two",
        context_after="[SPEAKER_00] upcoming raw turn",
    )
    user = rec.last_user or ""
    # Both labels present in the right order.
    assert user.index("CONTEXT BEFORE") < user.index("CURRENT SEGMENT")
    assert user.index("CURRENT SEGMENT") < user.index("CONTEXT AFTER")
    # Speaker tags from neighbouring segments survive into the prompt.
    assert "[SPEAKER_00] earlier turn one" in user
    assert "[SPEAKER_00] upcoming raw turn" in user
    # The current segment text appears under the CURRENT SEGMENT label.
    assert "current segment to fix with surrounding context" in user


def test_fix_segment_with_only_before_context_omits_after_block():
    rec = _recording_chat_llm(reply="corrected this segment using context")
    fix_segment(
        "we are at the last segment so no context after exists",
        llm=rec,
        language="uk",
        context_before="[SPEAKER_00] earlier turn",
        context_after="",
    )
    user = rec.last_user or ""
    assert "CONTEXT BEFORE" in user
    assert "CONTEXT AFTER" not in user
    assert "CURRENT SEGMENT" in user


def test_fix_asr_errors_builds_context_from_outputs_and_raw_ahead():
    """The middle segment's user prompt must contain (a) the prior already-
    corrected output as CONTEXT BEFORE and (b) the raw upcoming segment as
    CONTEXT AFTER. This is the cross-segment plumbing test.

    The replies below differ from the originals by a single token swap so
    they pass `_looks_safe` (which rejects replies that diverge from the
    original by more than the configured length/edit thresholds)."""
    segs = [
        _seg("ASR розпізнав хагінг фейсі як термін", 0, 2),
        _seg("середній сегмент достатньо довгий", 2, 4),
        _seg("третій сегмент достатньо довгий", 4, 6),
    ]
    replies = iter([
        # Single-token correction — survives _looks_safe.
        "ASR розпізнав Hugging Face як термін",
        "середній сегмент достатньо довгий",
        "третій сегмент достатньо довгий",
    ])

    seen_user_messages: list[str] = []

    class RecordingLLM:
        def chat(self, messages, **_kwargs):  # noqa: ANN001
            for msg in messages:
                if msg["role"] == "user":
                    seen_user_messages.append(msg["content"])
            return next(replies)

        prompt_cache_session = staticmethod(_noop_session)

    out, _ = fix_asr_errors(
        segs, llm=RecordingLLM(), language="uk",
        log=lambda _s: None, n_context=2,
    )

    # The first segment was actually corrected (passed _looks_safe).
    assert out[0].content == "ASR розпізнав Hugging Face як термін"

    # Second segment: CONTEXT BEFORE must contain the prior CORRECTED text
    # (Hugging Face, not хагінг фейсі), CONTEXT AFTER must contain the raw
    # upcoming segment.
    second_user = seen_user_messages[1]
    assert "Hugging Face" in second_user, second_user
    assert "хагінг фейсі" not in second_user, second_user
    # Note: speaker tag prepended by _format_context_block.
    assert "[SPEAKER_00] третій сегмент достатньо довгий" in second_user
    # And the current segment in its own block.
    assert "середній сегмент достатньо довгий" in second_user

    # Final segment: CONTEXT BEFORE contains both prior corrected outputs,
    # CONTEXT AFTER is empty (no upcoming neighbour) — block omitted.
    last_user = seen_user_messages[2]
    assert "Hugging Face" in last_user
    assert "середній сегмент достатньо довгий" in last_user
    assert "CONTEXT AFTER" not in last_user


def test_fix_asr_errors_n_context_zero_reproduces_isolation_baseline():
    """With n_context=0 the user prompt is bare segment text — no context
    labels at all. This is the v0.29.0 wire-format we keep for the sweep
    baseline."""
    segs = [_seg("достатньо довгий сегмент один", 0, 2),
            _seg("достатньо довгий сегмент два", 2, 4)]

    seen: list[str] = []

    class RecordingLLM:
        def chat(self, messages, **_kwargs):  # noqa: ANN001
            for m in messages:
                if m["role"] == "user":
                    seen.append(m["content"])
            return "reply"

        prompt_cache_session = staticmethod(_noop_session)

    fix_asr_errors(
        segs, llm=RecordingLLM(), language="uk",
        log=lambda _s: None, n_context=0,
    )

    for user in seen:
        assert "CONTEXT BEFORE" not in user
        assert "CONTEXT AFTER" not in user
        assert "CURRENT SEGMENT" not in user


def test_fix_asr_errors_telemetry_records_n_context_used():
    segs = [_seg("достатньо довгий сегмент один", 0, 2)]

    class StubReturning:
        def chat(self, *_a, **_k):
            return "fix"

        prompt_cache_session = staticmethod(_noop_session)

    for value in (0, 1, 5):
        _out, telemetry = fix_asr_errors(
            segs, llm=StubReturning(), language="uk",
            log=lambda _s: None, n_context=value,
        )
        assert telemetry["n_context"] == value
