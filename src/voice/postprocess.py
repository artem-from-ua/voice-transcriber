"""Per-segment ASR-error fixer.

Sends each segment's text to the LLM with a strict "fix obvious mishearings
only" prompt. Skips markers (`[Human Sounds]`), very short fragments, and
rejects replies that diverge from the original by more than the configured
safety thresholds (length ratio, edit distance).

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


def _is_marker(text: str) -> bool:
    t = text.strip()
    return t.startswith("[") and t.endswith("]")


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


def _strip_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in '"\'`':
        return s[1:-1].strip()
    return s


def fix_segment(
    text: str,
    *,
    llm: MlxLLM,
    language: str,
) -> str:
    """Return a corrected segment text, or the original if the LLM reply
    is unavailable / unsafe."""
    if _is_marker(text) or len(text.strip()) < MIN_LEN_FOR_FIX:
        return text
    messages = [
        {"role": "system", "content": render_prompt("postprocess_system", language=language)},
        {"role": "user", "content": render_prompt("postprocess_user", text=text)},
    ]
    try:
        reply = llm.chat(messages, **call_kwargs("postprocess_system"))
    except LLMError:
        return text
    candidate = _strip_quotes(reply)
    if not _looks_safe(text, candidate):
        return text
    return candidate


def fix_asr_errors(
    segments: Iterable[Segment],
    *,
    llm: MlxLLM,
    language: str = "uk",
    log: Callable[[str], None] = lambda s: print(s, file=sys.stderr),
    progress: "ProgressReporter | None" = None,
) -> list[Segment]:
    """Return new segments with each `content` proof-read.

    Segments are returned in the same order. Marker / short segments are
    passed through unchanged. The function never aborts on an LLM error —
    a failed segment falls back to its original text.
    """
    segs = list(segments)
    fixed_count = 0
    out: list[Segment] = []

    reporter = progress if progress is not None else NullProgress()
    with reporter.task("[7/10] ASR-постобробка", total=len(segs)) as advance:
        for seg in segs:
            new_text = fix_segment(seg.content, llm=llm, language=language)
            if new_text != seg.content:
                fixed_count += 1
            out.append(replace(seg, content=new_text))
            advance(1, suffix=f"{fixed_count} fixed")
    log(f"postprocess: {fixed_count}/{len(out)} segments adjusted")
    return out
