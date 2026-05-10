"""Generate a Markdown TL;DR for a dialogue.

The function flattens the dialogue into `Name: text` lines and asks the LLM
for a short summary + bullet list of key points + optional action items.
On LLM failure it returns an empty string so the pipeline can omit the
section gracefully.
"""

from __future__ import annotations

import sys
from typing import Callable, Iterable

from .llm import LLMClient, LLMError
from .types import Segment


_PROMPT_UK = """Згенеруй TL;DR розмови у форматі Markdown:

- 2–3 речення основного підсумку
- 3–5 ключових тез у вигляді буллет-списку
- розділ **Action items:** (якщо є явні плани, домовленості або задачі — інакше пропусти)

Не додавай заголовок "TL;DR" (його додасть інша частина системи).
Не вигадуй фактів, яких немає в тексті.
"""

_PROMPT_EN = """Generate a Markdown TL;DR of the conversation:

- 2–3 sentences of the main summary
- 3–5 key points as a bullet list
- a **Action items:** section if explicit plans, agreements, or tasks were mentioned (omit otherwise)

Do not add a "TL;DR" heading (the rest of the system adds it).
Do not invent facts that are not in the transcript.
"""


def _pick_prompt(language: str) -> str:
    if language.lower().startswith("en"):
        return _PROMPT_EN
    return _PROMPT_UK


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
    llm: LLMClient,
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
