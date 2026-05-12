"""Render the final Markdown transcript.

Layout:
- H1 with the source basename
- a quoted metadata block (date, duration, models, language)
- optional TL;DR section
- one H2 section per `Section`, each with paragraphs of `**Emoji Name:** text`

A paragraph break inside a section happens when:
- the speaker changes, OR
- the gap between consecutive same-speaker utterances exceeds `PARAGRAPH_GAP_S`.

Silent gaps with no speaker are dropped unless they are long enough to
warrant an explicit `> _[пауза Nс]_` quote between paragraphs.
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


def _render_lang_line(language: str, lang_detect_info: dict | None) -> str:
    """Format the 🌐 Мова line with optional auto-detect confidence."""
    if lang_detect_info is None:
        return f"🌐 **Мова:** {language} (user-specified)"
    top_lang, top_p = lang_detect_info["top"]
    tag = f"auto-detected={top_p:.2f}"
    second = lang_detect_info.get("second")
    if second:
        tag += f", {second[0]}={second[1]:.2f}"
    return f"🌐 **Мова:** {top_lang} ({tag})"


def _render_participants(
    speakers: list[str],
    emoji_for: dict[str, str],
    name_sources: dict[str, str] | None,
    segments: list[Segment],
) -> list[str]:
    """Return one blockquote line per participant with source tag."""
    if not speakers:
        return []
    lines = ["👥 **Учасники:**"]
    for speaker in speakers:
        emoji = emoji_for.get(speaker, "")
        # name_sources keys are pyannote cluster labels; speakers here are
        # display names (seg.name or seg.speaker). Find matching cluster.
        source = ""
        if name_sources:
            # Direct lookup by display name first, then by cluster label.
            cluster = next(
                (seg.speaker for seg in segments if (seg.name or seg.speaker) == speaker),
                None,
            )
            if cluster and cluster in name_sources:
                source = f" ({name_sources[cluster]})"
        chip = f"{emoji} **{speaker}**".strip()
        lines.append(f"  • {chip}{source}")
    return lines


def _render_ai_models_table(
    stage_models: dict[str, str],
    stage_timings: dict[str, float],
    model_load_elapsed: dict[str, float],
) -> list[str]:
    """Render a Markdown table grouping stages by model.

    Columns: Model | Stage | Time
    The first row for each model is "model loaded" with its load elapsed.
    """
    # Collect rows in execution order, grouping by model.
    # stage_models insertion order = execution order.
    rows: list[tuple[str, str, str]] = []
    seen_models: list[str] = []

    for stage, model in stage_models.items():
        if model not in seen_models:
            seen_models.append(model)
            load_s = model_load_elapsed.get(model, 0.0)
            rows.append((model, "model loaded", _format_compact_duration(load_s)))

        timing_s = stage_timings.get(stage)
        if timing_s is not None:
            rows.append(("", stage, _format_compact_duration(timing_s)))

    if not rows:
        return []

    # Build table. Model cell repeats label only on first row; subsequent rows
    # for same model get an empty cell to visually group them.
    # Right-align the Час column via `--:` alignment marker.
    lines = [
        "|  |  |  |",
        "| --- | --- | --: |",
    ]
    prev_model = None
    for model, stage, duration in rows:
        display_model = f"`{model}`" if model and model != prev_model else ""
        if model:
            prev_model = model
        lines.append(f"| {display_model} | {stage} | {duration} |")

    return lines


def render_markdown(
    *,
    audio_meta: AudioMeta,
    dialog: StructuredDialog,
    tldr: str = "",
    language: str = "uk",
    stage_models: dict[str, str] | None = None,
    stage_timings: dict[str, float] | None = None,
    model_load_elapsed: dict[str, float] | None = None,
    name_sources: dict[str, str] | None = None,
    lang_detect_info: dict | None = None,
) -> str:
    """Compose the full Markdown output."""
    basename = Path(audio_meta.path).name
    speakers = _speakers_in_order(dialog.segments)
    emoji_for = assign_emojis(speakers)

    header_items: list[str] = []
    header_items.append(
        f"📅 **Початок:** {_format_started_at(audio_meta.started_at)}, "
        f"**тривалість:** {_format_duration(audio_meta.duration_s)}"
    )
    header_items.append(_render_lang_line(language, lang_detect_info))

    participant_lines = _render_participants(
        speakers, emoji_for, name_sources, dialog.segments
    )
    header_items.extend(participant_lines)

    total = stage_timings.get("total") if stage_timings else None
    if total is not None:
        pct = round(total / audio_meta.duration_s * 100) if audio_meta.duration_s else 0
        header_items.append(
            f"⚡️ **Час обробки:** {_format_compact_duration(total)} ({pct}% of duration)"
        )

    if stage_models and stage_timings is not None and model_load_elapsed is not None:
        table_lines = _render_ai_models_table(
            stage_models, stage_timings, model_load_elapsed
        )
        header_items.extend(table_lines)

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
