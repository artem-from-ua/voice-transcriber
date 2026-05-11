"""Generate a Markdown TL;DR for a dialogue.

The function flattens the dialogue into `Name: text` lines and asks the LLM
for a short summary + bullet list of key points + optional action items.
On LLM failure it returns an empty string so the pipeline can omit the
section gracefully.
"""

from __future__ import annotations

import sys
from typing import Callable, Iterable

from ._prompts import render as render_prompt
from .llm import LLMError, MlxLLM
from .types import Segment


def _pick_prompt(language: str) -> str:
    name = "tldr_system_en" if language.lower().startswith("en") else "tldr_system_uk"
    return render_prompt(name)


def _format_dialogue(segments: list[Segment]) -> str:
    lines: list[str] = []
    for seg in segments:
        if seg.speaker is None:
            continue
        text = seg.content.strip()
        if not text:
            continue
        speaker = seg.name or seg.speaker
        lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


def generate_tldr(
    segments: Iterable[Segment],
    *,
    llm: MlxLLM,
    language: str = "uk",
    log: Callable[[str], None] = lambda s: print(s, file=sys.stderr),
) -> str:
    """Return a Markdown TL;DR string, or empty string on failure."""
    transcript = _format_dialogue(list(segments))
    if not transcript:
        return ""

    messages = [
        {"role": "system", "content": _pick_prompt(language)},
        {"role": "user", "content": transcript},
    ]
    try:
        text = llm.chat(messages, temperature=0.3, max_tokens=1024)
    except LLMError as exc:
        log(f"tldr: LLM error — {exc}; skipping")
        return ""
    return text.strip()
