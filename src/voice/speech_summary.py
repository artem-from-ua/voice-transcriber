"""Generate a Markdown TL;DR for a dialogue.

The function flattens the dialogue into `Name: text` lines and asks the LLM
for a short summary + bullet list of key points + optional action items.
On LLM failure it returns an empty string so the pipeline can omit the
section gracefully.
"""

from __future__ import annotations

import re
import sys
from typing import Callable, Iterable

from ._progress import NullProgress, ProgressReporter
from ._prompts import call_kwargs, render as render_prompt
from .llm import LLMError, MlxLLM
from .silence import extract_silence_events
from .types import Segment


def _pick_prompt_name(language: str) -> str:
    return "tldr_system_en" if language.lower().startswith("en") else "tldr_system_uk"


# The prompt asks the LLM not to emit a "TL;DR" heading because the renderer
# adds its own. The model regularly ignores that and starts the reply with
# "## TL;DR" / "# TL;DR" / "**TL;DR**". Strip it so we don't end up with two
# headings stacked in the output.
_LEADING_TLDR_RE = re.compile(
    r"\A\s*(?:#{1,6}\s*TL;?DR\s*\n+|\*\*TL;?DR:?\*\*\s*\n+)",
    flags=re.IGNORECASE,
)


def _strip_leading_tldr_heading(text: str) -> str:
    return _LEADING_TLDR_RE.sub("", text, count=1).lstrip("\n")


def _format_dialogue(segments: list[Segment], include_silence: bool = False) -> str:
    lines: list[str] = []

    if include_silence:
        # Build silence events once (use same threshold as render default).
        from .render import MIN_SILENCE_S
        silence_events = extract_silence_events(segments, MIN_SILENCE_S)
        # Map: silence event -> index of the first speaker-seg after it.
        silence_before: dict[int, str] = {}
        for ev in silence_events:
            for idx, seg in enumerate(segments):
                if seg.speaker is not None and seg.start >= ev.end:
                    silence_before[idx] = ev.format_md().lstrip("> _").rstrip("_")
                    break

    for i, seg in enumerate(segments):
        if include_silence and i in silence_before:
            lines.append(silence_before[i])
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
    include_silence: bool = False,
    log: Callable[[str], None] = lambda s: print(s, file=sys.stderr),
    progress: "ProgressReporter | None" = None,
) -> str:
    """Return a Markdown TL;DR string, or empty string on failure."""
    transcript = _format_dialogue(list(segments), include_silence=include_silence)
    if not transcript:
        return ""

    prompt_name = _pick_prompt_name(language)
    messages = [
        {"role": "system", "content": render_prompt(prompt_name)},
        {"role": "user", "content": transcript},
    ]
    reporter = progress if progress is not None else NullProgress()
    try:
        with reporter.token_counter("[12/13] TL;DR") as advance:
            text = llm.chat(
                messages, on_token=advance, **call_kwargs(prompt_name),
            )
    except LLMError as exc:
        log(f"tldr: LLM error — {exc}; skipping")
        return ""
    return _strip_leading_tldr_heading(text.strip())
