"""Render the final Markdown transcript.

Layout:
- H1 with the source basename
- a quoted metadata block (date, duration, models, language)
- optional TL;DR section
- one H2 section per `Section`, each with paragraphs of `**Emoji Name:** text`

A paragraph break inside a section happens when:
- the speaker changes, OR
- the gap between consecutive same-speaker utterances exceeds `PARAGRAPH_GAP_S`.

Marker segments (`[Human Sounds]`, `[Silence]`) and silent gaps with no
speaker are dropped unless they are long enough to warrant an explicit
`> _[пауза Nс]_` quote between paragraphs.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from functools import lru_cache
from importlib.resources import files
from pathlib import Path

from .speaker_emojis import assign_emojis
from .types import AudioMeta, Section, Segment, StructuredDialog


PARAGRAPH_GAP_S = 2.0
EXPLICIT_PAUSE_S = 3.0


@lru_cache(maxsize=1)
def _language_names() -> dict[str, str]:
    """Load friendly language names from the packaged JSON file.

    Keys are normalised to lowercase. The `_comment` field is dropped.
    Adding a new language only takes editing
    `src/voice/data/language_names.json` — no Python change.
    """
    raw = files("voice.data").joinpath("language_names.json").read_text(encoding="utf-8")
    return {
        k.lower(): v
        for k, v in json.loads(raw).items()
        if not k.startswith("_")
    }


def _format_duration(seconds: float) -> str:
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, s2 = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s2:02d}"
    return f"{m}:{s2:02d}"


def _format_started_at(iso: str) -> str:
    """ISO datetime → `YYYY-MM-DD HH:MM UTC` (always normalised to UTC)."""
    try:
        dt = datetime.fromisoformat(iso)
    except ValueError:
        return iso  # fall back to the raw string; better than crashing
    dt_utc = dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    return dt_utc.strftime("%Y-%m-%d %H:%M UTC")


def _language_name(code: str) -> str:
    """Friendly Ukrainian name for a language code; pass-through if unknown."""
    return _language_names().get(code.lower(), code)


def _speakers_in_order(segments: list[Segment]) -> list[str]:
    """Display names (or labels) in order of first appearance."""
    seen: list[str] = []
    for seg in segments:
        if seg.speaker is None:
            continue
        label = seg.name or seg.speaker
        if label not in seen:
            seen.append(label)
    return seen


def _segments_in_section(segments: list[Segment], section: Section) -> list[Segment]:
    """Segments whose midpoint lies within the section's range."""
    out = []
    for seg in segments:
        if seg.speaker is None:
            continue
        mid_ms = int(((seg.start + seg.end) / 2) * 1000)
        if section.start_ms <= mid_ms < section.end_ms:
            out.append(seg)
    return out


def _render_section_body(
    seg_in_section: list[Segment],
    emoji_for: dict[str, str],
) -> list[str]:
    """Return Markdown paragraphs for one section."""
    if not seg_in_section:
        return []

    blocks: list[str] = []
    cur_speaker: str | None = None
    cur_lines: list[str] = []
    prev_end: float | None = None

    def flush() -> None:
        if cur_lines and cur_speaker is not None:
            emoji = emoji_for.get(cur_speaker, "")
            label = f"{emoji} **{cur_speaker}:**".strip()
            blocks.append(f"{label} " + " ".join(cur_lines).strip())

    for seg in seg_in_section:
        label = seg.name or seg.speaker
        same_speaker = label == cur_speaker
        gap = (seg.start - prev_end) if prev_end is not None else 0.0

        if not same_speaker or gap >= PARAGRAPH_GAP_S:
            flush()
            cur_lines = []
            if gap >= EXPLICIT_PAUSE_S:
                blocks.append(f"> _[пауза {int(round(gap))}с]_")
            cur_speaker = label

        cur_lines.append(seg.content.strip())
        prev_end = seg.end

    flush()
    return blocks


def render_markdown(
    *,
    audio_meta: AudioMeta,
    dialog: StructuredDialog,
    tldr: str = "",
    language: str = "uk",
    asr_label: str = "VibeVoice-ASR",
) -> str:
    """Compose the full Markdown output."""
    basename = Path(audio_meta.path).name
    speakers = _speakers_in_order(dialog.segments)
    emoji_for = assign_emojis(speakers)

    lines: list[str] = []
    lines.append(f"# Транскрипт: {basename}")
    lines.append("")
    lines.append(f"> 📅 **Початок:** {_format_started_at(audio_meta.started_at)}")
    lines.append(f"> ⏱️ **Тривалість:** {_format_duration(audio_meta.duration_s)}")
    lines.append(f"> 🎙️ **Транскрипція:** {asr_label} + pyannote 3.1")
    lines.append(f"> 🌐 **Мова:** {_language_name(language)}")
    lines.append("")

    if tldr:
        lines.append("## TL;DR")
        lines.append("")
        lines.append(tldr.strip())
        lines.append("")
        lines.append("---")
        lines.append("")

    if not dialog.sections:
        return "\n".join(lines).rstrip() + "\n"

    for section in dialog.sections:
        in_section = _segments_in_section(dialog.segments, section)
        if not in_section:
            continue
        lines.append(f"## {section.title}")
        lines.append("")
        for block in _render_section_body(in_section, emoji_for):
            lines.append(block)
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"
