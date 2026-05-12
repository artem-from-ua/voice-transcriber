"""Sensitive-content redaction stage.

Runs after speech_structure and before speech_summary so the TL;DR is
generated from the already-redacted dialogue — no separate TL;DR pass needed.

One LLM call per section (per-section strategy): the working set is bounded
by the longest section, not by the total recording length, which makes memory
requirements predictable for hour-long recordings.

Granularity is always a range of whole utterances (Segments), never
individual words. The LLM returns (start_idx, end_idx) ranges in local
0-based indexing within each section; the caller applies the chosen policy.
"""

from __future__ import annotations

import copy
import sys
from dataclasses import dataclass
from typing import Callable, Literal

from ._progress import NullProgress, ProgressReporter
from ._prompts import call_kwargs, render as render_prompt
from .llm import LLMError, MlxLLM
from .types import Segment, Section, StructuredDialog

DEFAULT_TOPICS: list[str] = ["health", "drugs", "alcohol"]

_REDACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "redactions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start_idx": {"type": "integer"},
                    "end_idx": {"type": "integer"},
                    "topic": {"type": "string"},
                    "rationale": {"type": "string"},
                },
                "required": ["start_idx", "end_idx", "topic", "rationale"],
            },
        }
    },
    "required": ["redactions"],
}


@dataclass
class Decision:
    section_title: str
    start_idx: int
    end_idx: int
    topic: str
    rationale: str
    total_dur_s: float


def _pick_prompt_name(language: str) -> str:
    return "safe_speech_system_en" if language.lower().startswith("en") else "safe_speech_system"


def _format_segments(section_segments: list[Segment]) -> str:
    lines: list[str] = []
    for idx, seg in enumerate(section_segments):
        speaker = seg.name or seg.speaker or "?"
        text = seg.content.strip()
        lines.append(f"{idx}: {speaker}: {text}")
    return "\n".join(lines)


def _topics_text(topics: list[str], language: str) -> str:
    if language.lower().startswith("en"):
        return ", ".join(topics)
    label_map = {
        "health": "здоров'я (діагнози, симптоми, медичні процедури, психічне здоров'я)",
        "drugs": "наркотики (вживання, придбання)",
        "alcohol": "алкоголь (звички пиття, анекдоти про сп'яніння)",
        "legal": "правові проблеми",
        "finance": "фінансові деталі (зарплати, борги)",
        "sexual": "сексуальний контент",
    }
    parts = [label_map.get(t, t) for t in topics]
    return "\n".join(f"- {p}" for p in parts)


def _section_segments(dialog: StructuredDialog, section: Section) -> list[Segment]:
    return [
        seg for seg in dialog.segments
        if section.start_ms <= int(seg.start * 1000) < section.end_ms
    ]


def _apply_placeholder(
    all_segments: list[Segment],
    section_segs: list[Segment],
    start_idx: int,
    end_idx: int,
) -> None:
    """Replace segment range [start_idx, end_idx] with a single [muted, Xs] segment."""
    range_segs = section_segs[start_idx : end_idx + 1]
    if not range_segs:
        return
    total_dur = sum(s.end - s.start for s in range_segs)
    placeholder = Segment(
        start=range_segs[0].start,
        end=range_segs[-1].end,
        content=f"[muted, {total_dur:.1f}s]",
        speaker=None,
        name=None,
    )
    # Locate range in all_segments (by identity) and replace
    first_global = next(
        (i for i, s in enumerate(all_segments) if s is range_segs[0]), None
    )
    last_global = next(
        (i for i, s in enumerate(all_segments) if s is range_segs[-1]), None
    )
    if first_global is None or last_global is None:
        return
    all_segments[first_global : last_global + 1] = [placeholder]


def _apply_drop(
    all_segments: list[Segment],
    section_segs: list[Segment],
    start_idx: int,
    end_idx: int,
    section: Section,
    is_whole_section: bool,
) -> None:
    """Remove segment range; if it covers the whole section, keep a synthetic marker."""
    range_segs = section_segs[start_idx : end_idx + 1]
    if not range_segs:
        return
    first_global = next(
        (i for i, s in enumerate(all_segments) if s is range_segs[0]), None
    )
    last_global = next(
        (i for i, s in enumerate(all_segments) if s is range_segs[-1]), None
    )
    if first_global is None or last_global is None:
        return
    if is_whole_section:
        # Keep section header visible: replace with one synthetic marker
        synthetic = Segment(
            start=section.start_ms / 1000.0,
            end=section.end_ms / 1000.0,
            content="[muted, 0.0s]",
            speaker=None,
            name=None,
        )
        all_segments[first_global : last_global + 1] = [synthetic]
    else:
        del all_segments[first_global : last_global + 1]


def redact_dialog(
    dialog: StructuredDialog,
    *,
    llm: MlxLLM,
    topics: list[str],
    policy: Literal["placeholder", "drop"] = "placeholder",
    language: str = "uk",
    log: Callable[[str], None] = lambda s: print(s, file=sys.stderr),
    progress: "ProgressReporter | None" = None,
) -> tuple[StructuredDialog, list[Decision]]:
    """Return a (redacted dialog copy, decisions list).

    If topics is empty, returns the original dialog unchanged with no LLM call.
    On LLMError for a section, that section is left untouched; other sections
    are still processed.
    """
    if not topics:
        return dialog, []

    reporter = progress if progress is not None else NullProgress()
    prompt_name = _pick_prompt_name(language)
    topics_text = _topics_text(topics, language)
    system_prompt = render_prompt(prompt_name, topics=topics_text)

    # Work on a deep copy so the caller's original is never mutated
    dialog = copy.deepcopy(dialog)

    all_decisions: list[Decision] = []
    sections = dialog.sections if dialog.sections else [
        Section(title="Розмова", start_ms=0, end_ms=int(dialog.segments[-1].end * 1000) if dialog.segments else 0)
    ]

    redacted_ranges: int = 0
    redacted_segs: int = 0
    topic_counts: dict[str, int] = {}

    for section in sections:
        section_segs = _section_segments(dialog, section)
        if not section_segs:
            continue

        segments_text = _format_segments(section_segs)
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": render_prompt(
                    "safe_speech_user",
                    section_title=section.title,
                    segments=segments_text,
                ),
            },
        ]

        try:
            with reporter.token_counter(f"safe_speech [{section.title}]") as advance:
                result = llm.chat_json(
                    messages,
                    schema=_REDACTION_SCHEMA,
                    on_token=advance,
                    **call_kwargs(prompt_name),
                )
        except LLMError as exc:
            log(f"safe_speech: LLM error in section {section.title!r} — {exc}; skipping section")
            continue

        raw_redactions = result.get("redactions", [])
        # Sort descending by start_idx so in-place mutations don't shift indices
        raw_redactions = sorted(raw_redactions, key=lambda r: r.get("start_idx", 0), reverse=True)

        for red in raw_redactions:
            start_idx = int(red.get("start_idx", 0))
            end_idx = int(red.get("end_idx", start_idx))
            topic = str(red.get("topic", ""))
            rationale = str(red.get("rationale", ""))

            # Clamp to valid range
            n = len(section_segs)
            start_idx = max(0, min(start_idx, n - 1))
            end_idx = max(start_idx, min(end_idx, n - 1))

            range_segs = section_segs[start_idx : end_idx + 1]
            total_dur = sum(s.end - s.start for s in range_segs)
            is_whole_section = (start_idx == 0 and end_idx == n - 1)

            # Refresh section_segs after each mutation (previous iterations may have changed all_segments)
            section_segs = _section_segments(dialog, section)

            if policy == "placeholder":
                _apply_placeholder(dialog.segments, section_segs, start_idx, end_idx)
            else:
                _apply_drop(dialog.segments, section_segs, start_idx, end_idx, section, is_whole_section)

            # Refresh again for the next iteration
            section_segs = _section_segments(dialog, section)

            redacted_ranges += 1
            seg_count = end_idx - start_idx + 1
            redacted_segs += seg_count
            topic_counts[topic] = topic_counts.get(topic, 0) + seg_count

            all_decisions.append(Decision(
                section_title=section.title,
                start_idx=start_idx,
                end_idx=end_idx,
                topic=topic,
                rationale=rationale,
                total_dur_s=round(total_dur, 1),
            ))

    if all_decisions:
        topic_summary = ", ".join(f"{t}={c}" for t, c in sorted(topic_counts.items()))
        log(f"safe_speech: redacted {redacted_ranges} ranges / {redacted_segs} segments ({topic_summary})")
    else:
        log("safe_speech: no sensitive content found")

    return dialog, all_decisions
