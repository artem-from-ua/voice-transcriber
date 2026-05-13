"""Per-segment ASR-error fixer.

Sends each segment's text to the LLM with a strict "fix obvious mishearings
only" prompt. Skips very short fragments and rejects replies that diverge
from the original by more than the configured safety thresholds (length
ratio, edit distance).

Operating per-segment, not on the whole transcript, is deliberate: the LLM
cannot rewrite or summarise globally because it never sees more than one
segment at a time.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from typing import Callable, Iterable

import Levenshtein

from ._progress import NullProgress, ProgressReporter
from ._prompts import call_kwargs, render as render_prompt
from .llm import LLMError, MlxLLM
from .types import Segment


MIN_LEN_FOR_FIX = 10
MAX_EDIT_RATIO = 0.5
MAX_LENGTH_RATIO = 2.0


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


def fix_segment(
    text: str,
    *,
    llm: MlxLLM,
    language: str,
) -> tuple[str, bool]:
    """Return ``(corrected_text, called)``.

    ``called`` is True when the LLM was actually invoked (regardless of
    whether the reply was accepted or rejected on safety grounds). It is
    False for skipped-too-short segments. Used by ``fix_asr_errors`` to
    report telemetry to the pipeline (see #57 / 01-meta.json stages).
    """
    if len(text.strip()) < MIN_LEN_FOR_FIX:
        return text, False
    messages = [
        {"role": "system", "content": render_prompt("proofread_system", language=language)},
        {"role": "user", "content": render_prompt("proofread_user", text=text)},
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
) -> tuple[list[Segment], dict]:
    """Return ``(new_segments, telemetry)``.

    ``new_segments`` is a list with each ``content`` proof-read; segments
    are returned in the same order. Marker / short segments are passed
    through unchanged. The function never aborts on an LLM error — a
    failed segment falls back to its original text.

    ``telemetry`` is a dict carrying per-stage stats for ``01-meta.json``
    (see #57):
        - ``llm_calls`` — number of segments that triggered an actual
          ``llm.chat()`` call (short / blank segments are skipped and
          do not count).
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
        for seg in segs:
            new_text, called = fix_segment(seg.content, llm=llm, language=language)
            if called:
                llm_calls += 1
            if new_text != seg.content:
                fixed_count += 1
            out.append(replace(seg, content=new_text))
            advance(1, suffix=f"{fixed_count} fixed")
    log(f"proofread: {fixed_count}/{len(out)} segments adjusted ({llm_calls} LLM calls)")
    return out, {"llm_calls": llm_calls}
