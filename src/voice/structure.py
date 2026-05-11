"""Split a dialogue into thematic sections using the LLM.

The LLM gets a "script" of utterances with timestamps and pause markers and
returns a JSON list of sections (title + start_ms..end_ms). The output is
validated: ranges in bounds, contiguous, covering the whole dialogue. If the
LLM fails or returns an invalid layout, the fallback is a single section
titled "Розмова" / "Conversation".
"""

from __future__ import annotations

import sys
from typing import Callable, Iterable

from ._progress import NullProgress, ProgressReporter
from ._prompts import call_kwargs, render as render_prompt
from .llm import LLMError, MlxLLM
from .types import Section, Segment, StructuredDialog


PAUSE_GAP_S = 3.0
MIN_SECTIONS = 2
MAX_SECTIONS = 7


def _format_timestamp(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def _build_script(segments: list[Segment], pause_gap_s: float) -> str:
    """Render utterances + pause markers in a compact form for the LLM."""
    lines: list[str] = []
    prev_end: float | None = None
    for seg in segments:
        if prev_end is not None and seg.start - prev_end >= pause_gap_s:
            lines.append(f"[{_format_timestamp(prev_end)} — pause {seg.start - prev_end:.0f}s]")
        speaker = seg.name or seg.speaker or "?"
        lines.append(f"[{_format_timestamp(seg.start)}] {speaker}: {seg.content}")
        prev_end = seg.end
    return "\n".join(lines)


def _fallback_section(segments: list[Segment], language: str) -> list[Section]:
    title = "Conversation" if language.startswith("en") else "Розмова"
    if not segments:
        return [Section(title=title, start_ms=0, end_ms=0)]
    start_ms = int(segments[0].start * 1000)
    end_ms = int(segments[-1].end * 1000)
    return [Section(title=title, start_ms=start_ms, end_ms=end_ms)]


def _validate(
    payload: object,
    *,
    total_start_ms: int,
    total_end_ms: int,
) -> list[Section] | None:
    if not isinstance(payload, dict):
        return None
    raw = payload.get("sections")
    if not isinstance(raw, list) or not (MIN_SECTIONS <= len(raw) <= MAX_SECTIONS):
        return None

    sections: list[Section] = []
    for item in raw:
        if not isinstance(item, dict):
            return None
        title = item.get("title")
        start = item.get("start_ms")
        end = item.get("end_ms")
        if not isinstance(title, str) or not title.strip():
            return None
        if not isinstance(start, int) or not isinstance(end, int):
            return None
        if end <= start:
            return None
        sections.append(Section(title=title.strip(), start_ms=start, end_ms=end))

    sections.sort(key=lambda s: s.start_ms)
    if sections[0].start_ms != total_start_ms or sections[-1].end_ms != total_end_ms:
        return None
    for prev, nxt in zip(sections, sections[1:]):
        if nxt.start_ms != prev.end_ms:
            return None
    return sections


def structure_dialog(
    segments: Iterable[Segment],
    *,
    llm: MlxLLM | None = None,
    language: str = "uk",
    log: Callable[[str], None] = lambda s: print(s, file=sys.stderr),
    progress: "ProgressReporter | None" = None,
) -> StructuredDialog:
    """Return a `StructuredDialog`. Falls back to a single section on error."""
    segs = list(segments)
    speech = [s for s in segs if s.speaker is not None]
    if not speech:
        return StructuredDialog(sections=[], segments=segs)

    total_start_ms = int(speech[0].start * 1000)
    total_end_ms = int(speech[-1].end * 1000)

    if llm is None:
        log("structure: no LLM provided; using single-section fallback")
        return StructuredDialog(sections=_fallback_section(segs, language), segments=segs)

    script = _build_script(speech, PAUSE_GAP_S)
    messages = [
        {
            "role": "system",
            "content": render_prompt("structure_system", language=language),
        },
        {
            "role": "user",
            "content": render_prompt(
                "structure_user",
                total_start_ms=total_start_ms,
                total_end_ms=total_end_ms,
                script=script,
            ),
        },
    ]

    reporter = progress if progress is not None else NullProgress()
    try:
        with reporter.token_counter("[9/10] Структурування на секції") as advance:
            payload = llm.chat_json(
                messages,
                on_token=advance,
                **call_kwargs("structure_system"),
            )
    except LLMError as exc:
        log(f"structure: LLM error — {exc}; falling back to single section")
        return StructuredDialog(sections=_fallback_section(segs, language), segments=segs)

    sections = _validate(
        payload,
        total_start_ms=total_start_ms,
        total_end_ms=total_end_ms,
    )
    if sections is None:
        log("structure: invalid layout — falling back to single section")
        return StructuredDialog(sections=_fallback_section(segs, language), segments=segs)

    log(f"structure: {len(sections)} section(s)")
    return StructuredDialog(sections=sections, segments=segs)
