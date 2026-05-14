"""Per-segment ASR-error fixer.

Sends each segment's text to the LLM with a strict "fix obvious mishearings
only" prompt. Skips very short fragments and rejects replies that diverge
from the original by more than the configured safety thresholds (length
ratio, edit distance).

Operating per-segment is still the unit of work — the LLM rewrites one
segment at a time and outputs that segment alone. **Neighbouring
segments** are optionally shown to the LLM as read-only context
(``[CONTEXT BEFORE]`` from already-corrected outputs, ``[CONTEXT AFTER]``
from raw upcoming ASR) so the model can disambiguate tech terms and
slang without trying to summarise across segments. See ADR 0028 and
``docs/benchmarks/proofread-hit-rate.md`` iteration 2 for the rationale
and the context-size sweep that picked the default.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from typing import Callable, Iterable, Sequence

import Levenshtein

from ._progress import NullProgress, ProgressReporter
from ._prompts import call_kwargs, render as render_prompt
from .llm import LLMError, MlxLLM
from .types import Segment


MIN_LEN_FOR_FIX = 10
MAX_EDIT_RATIO = 0.5
MAX_LENGTH_RATIO = 2.0

# Default number of neighbouring segments shown to the LLM as context on
# each side of the current segment. Chosen empirically — see iteration 2
# of docs/benchmarks/proofread-hit-rate.md. Set to 0 to reproduce the
# v0.29.0 behaviour where each segment was fixed in isolation.
DEFAULT_N_CONTEXT = 3


def _looks_safe(original: str, fixed: str) -> bool:
    """True if `fixed` is close enough to `original` to trust the LLM."""
    if not fixed:
        return False
    n0, n1 = len(original), len(fixed)
    if n0 == 0:
        return False
    if n1 / n0 > MAX_LENGTH_RATIO or n0 / n1 > MAX_LENGTH_RATIO:
        return False
    distance = Levenshtein.distance(original, fixed)
    if distance / max(n0, n1) > MAX_EDIT_RATIO:
        return False
    return True


def _strip_artifacts(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in '"\'`':
        s = s[1:-1].strip()
    # Some models echo the "Text:" label from the user-turn template back.
    if s.startswith("Text: "):
        s = s[len("Text: "):]
    return s


def _format_context_block(segs: Sequence[Segment]) -> str:
    """Format neighbouring segments as one text block for the user prompt.

    Uses the speaker label when present (``[SPEAKER_00] text``) so the LLM
    can tell who said what. Empty input → empty string; the caller decides
    whether to include the surrounding ``[CONTEXT BEFORE]`` / ``[CONTEXT
    AFTER]`` header at all.
    """
    lines: list[str] = []
    for seg in segs:
        prefix = f"[{seg.speaker}] " if seg.speaker else ""
        lines.append(f"{prefix}{seg.content}")
    return "\n".join(lines)


def _build_user_content(
    *,
    text: str,
    context_before: str,
    context_after: str,
) -> str:
    """Compose the user-message body shown to the proofread LLM.

    When both contexts are empty we collapse to bare current text — the
    LLM then sees the same shape as the v0.29.0 baseline. When either
    context is non-empty we wrap the segment in labelled blocks so the
    model knows what to edit and what is reference-only.
    """
    if not context_before and not context_after:
        return text
    parts: list[str] = []
    if context_before:
        parts.append("[CONTEXT BEFORE — your prior corrections, final form]")
        parts.append(context_before)
        parts.append("")
    parts.append("[CURRENT SEGMENT — fix this one only]")
    parts.append(text)
    if context_after:
        parts.append("")
        parts.append("[CONTEXT AFTER — raw ASR, not yet proofread]")
        parts.append(context_after)
    return "\n".join(parts)


def fix_segment(
    text: str,
    *,
    llm: MlxLLM,
    language: str,
    context_before: str = "",
    context_after: str = "",
) -> tuple[str, bool]:
    """Return ``(corrected_text, called)``.

    ``called`` is True when the LLM was actually invoked (regardless of
    whether the reply was accepted or rejected on safety grounds). It is
    False for skipped-too-short segments. Used by ``fix_asr_errors`` to
    report telemetry to the pipeline (see #57 / 01-meta.json stages).

    ``context_before`` / ``context_after`` are optional pre-formatted text
    blocks (one neighbour segment per line, see ``_format_context_block``)
    shown to the LLM around the current segment. When both are empty the
    user prompt collapses to the segment alone — matches the v0.29.0
    isolation baseline and keeps the prefix cache hit-rate identical.
    """
    if len(text.strip()) < MIN_LEN_FOR_FIX:
        return text, False
    content = _build_user_content(
        text=text,
        context_before=context_before,
        context_after=context_after,
    )
    messages = [
        {"role": "system", "content": render_prompt("proofread_system", language=language)},
        {"role": "user", "content": render_prompt("proofread_user", content=content)},
    ]
    try:
        reply = llm.chat(messages, **call_kwargs("proofread_system"))
    except LLMError:
        return text, True
    candidate = _strip_artifacts(reply)
    if not _looks_safe(text, candidate):
        return text, True
    return candidate, True


def fix_asr_errors(
    segments: Iterable[Segment],
    *,
    llm: MlxLLM,
    language: str = "uk",
    log: Callable[[str], None] = lambda s: print(s, file=sys.stderr),
    progress: "ProgressReporter | None" = None,
    n_context: int = DEFAULT_N_CONTEXT,
) -> tuple[list[Segment], dict]:
    """Return ``(new_segments, telemetry)``.

    ``new_segments`` is a list with each ``content`` proof-read; segments
    are returned in the same order. Marker / short segments are passed
    through unchanged. The function never aborts on an LLM error — a
    failed segment falls back to its original text.

    ``n_context`` controls how many neighbouring segments are shown to the
    LLM as context on each side of the current segment. ``CONTEXT BEFORE``
    is built from already-proofread outputs (final form), ``CONTEXT AFTER``
    from the raw upcoming segments. Pass ``n_context=0`` to fix each
    segment in isolation (v0.29.0 behaviour). See iteration 2 of
    ``docs/benchmarks/proofread-hit-rate.md`` for how the default was
    picked.

    ``telemetry`` is a dict carrying per-stage stats for ``01-meta.json``
    (see #57):
        - ``llm_calls`` — number of segments that triggered an actual
          ``llm.chat()`` call (short / blank segments are skipped and
          do not count).
        - ``n_context`` — the value used for this run, recorded so the
          dump artefact carries the parameter that produced it.
    """
    segs = list(segments)
    fixed_count = 0
    llm_calls = 0
    out: list[Segment] = []

    reporter = progress if progress is not None else NullProgress()
    system_msg = {
        "role": "system",
        "content": render_prompt("proofread_system", language=language),
    }
    with llm.prompt_cache_session(prefix_messages=[system_msg]), reporter.task(
        "[8/13] Proofread", total=len(segs)
    ) as advance:
        for i, seg in enumerate(segs):
            if n_context > 0:
                # CONTEXT BEFORE: prior corrected outputs already in `out`.
                before = _format_context_block(out[max(0, i - n_context):])
                # CONTEXT AFTER: raw upcoming segments, not yet visited.
                after = _format_context_block(segs[i + 1:i + 1 + n_context])
            else:
                before = ""
                after = ""
            new_text, called = fix_segment(
                seg.content,
                llm=llm,
                language=language,
                context_before=before,
                context_after=after,
            )
            if called:
                llm_calls += 1
            if new_text != seg.content:
                fixed_count += 1
            out.append(replace(seg, content=new_text))
            advance(1, suffix=f"{fixed_count} fixed")
    log(
        f"proofread: {fixed_count}/{len(out)} segments adjusted "
        f"({llm_calls} LLM calls, n_context={n_context})"
    )
    return out, {"llm_calls": llm_calls, "n_context": n_context}
