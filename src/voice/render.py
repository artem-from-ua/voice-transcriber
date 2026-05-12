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

from datetime import datetime, timezone
from pathlib import Path

from .speaker_emojis import assign_emojis
from .types import AudioMeta, Section, Segment, StructuredDialog


PARAGRAPH_GAP_S = 2.0
EXPLICIT_PAUSE_S = 3.0


def _format_duration(seconds: float) -> str:
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, s2 = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s2:02d}"
    return f"{m}:{s2:02d}"


def _format_compact_duration(seconds: float) -> str:
    """Compact human-readable duration: `12m5s`, `45s`, `1h3m20s`."""
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, s2 = divmod(rem, 60)
    if h:
        return f"{h}h{m}m{s2}s"
    if m:
        return f"{m}m{s2}s"
    return f"{s2}s"


def _format_started_at(iso: str) -> str:
    """ISO datetime → `YYYY-MM-DD HH:MM UTC` (always normalised to UTC)."""
    try:
        dt = datetime.fromisoformat(iso)
    except ValueError:
        return iso  # fall back to the raw string; better than crashing
    dt_utc = dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    return dt_utc.strftime("%Y-%m-%d %H:%M UTC")




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
    asr_label: str | None = None,  # noqa: ARG001 — kept for backward-compatible signature
    models: dict[str, str] | None = None,
    timings: dict[str, float] | None = None,
) -> str:
    """Compose the full Markdown output.

    `asr_label` is accepted but no longer rendered (the header no longer
    advertises the toolchain). Kept in the signature so existing callers
    that still pass it don't break.

    `models` is an optional mapping with the keys `diarize`, `asr`, and
    `llm` (a single string with the per-stage models — same path repeated
    when one model handles all four stages, otherwise a `proofread=… ·
    structure=…` listing). The header renders one bullet per non-empty
    key. Pass `None` to omit the toolchain section entirely.
    """
    basename = Path(audio_meta.path).name
    speakers = _speakers_in_order(dialog.segments)
    emoji_for = assign_emojis(speakers)

    # Each header item is its own blockquote line. CommonMark merges
    # consecutive `>` lines into one paragraph (so all the chips collapse
    # onto one rendered line); two trailing spaces force a hard line
    # break inside the paragraph, which every common renderer respects
    # (GitHub, VS Code preview, Obsidian, pandoc).
    header_items: list[str] = []
    header_items.append(f"📅 **Початок:** {_format_started_at(audio_meta.started_at)}")
    header_items.append(f"⏱️ **Тривалість:** {_format_duration(audio_meta.duration_s)}")
    header_items.append(f"🌐 **Мова:** {language}")
    if speakers:
        speaker_chips = ", ".join(
            f"{emoji_for.get(s, '')} {s}".strip() for s in speakers
        )
        header_items.append(f"👥 **Учасники:** {speaker_chips}")
    if models:
        if models.get("diarize"):
            header_items.append(f"🗣️ **Діаризація:** {models['diarize']}")
        if models.get("asr"):
            header_items.append(f"📝 **ASR:** {models['asr']}")
        if models.get("llm"):
            header_items.append(f"🤖 **LLM:** {models['llm']}")
    if timings:
        total = timings.get("total")
        if total is not None:
            header_items.append(
                f"⏲️ **Обробка:** {_format_compact_duration(total)}"
            )
        breakdown_order = (
            "diarize", "asr", "proofread", "identify", "structure", "tldr",
        )
        chips = [
            f"{stage}={_format_compact_duration(timings[stage])}"
            for stage in breakdown_order
            if stage in timings
        ]
        if chips:
            header_items.append("⏱️ **AI-стадії:** " + " · ".join(chips))

    lines: list[str] = []
    lines.append(f"# Транскрипт: {basename}")
    lines.append("")
    for idx, item in enumerate(header_items):
        suffix = "  " if idx < len(header_items) - 1 else ""
        lines.append(f"> {item}{suffix}")
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
