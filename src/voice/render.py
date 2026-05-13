"""Render the final Markdown transcript.

Layout:
- H1 with the source basename
- a quoted metadata block (date, duration, models, language)
- optional TL;DR section
- one H2 section per `Section`, each with paragraphs of `**Emoji Name:** text`

A paragraph break inside a section happens when:
- the speaker changes, OR
- the gap between consecutive same-speaker utterances exceeds `PARAGRAPH_GAP_S`.

Silence events (ASR gaps and [muted] placeholders) are shown as `---`
(horizontal rule) only when they appear between two speaker paragraphs.
Leading and trailing silences in a section are suppressed. Events shorter
than `MIN_SILENCE_S` seconds are dropped silently. Adjacent muted/gap chains
are merged into a single event.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from .silence import SilenceEvent, extract_silence_events
from .speaker_emojis import assign_emojis
from .types import AudioMeta, Section, Segment, StructuredDialog


PARAGRAPH_GAP_S = 2.0
MIN_SILENCE_S = 10.0

# Auto-threshold bounds: short sections get a low threshold (more markers),
# long sections get a high one (only major pauses shown).
_AUTO_SILENCE_MIN_S = 5.0   # threshold at section duration <= _AUTO_SHORT_S
_AUTO_SILENCE_MAX_S = 15.0  # threshold at section duration >= _AUTO_LONG_S
_AUTO_SHORT_S = 60.0        # section duration anchor for min threshold
_AUTO_LONG_S = 300.0        # section duration anchor for max threshold


def _auto_silence_threshold(section_duration_s: float) -> float:
    """Linear interpolation between 5 s (≤60 s section) and 15 s (≥300 s section)."""
    t = (section_duration_s - _AUTO_SHORT_S) / (_AUTO_LONG_S - _AUTO_SHORT_S)
    t = max(0.0, min(1.0, t))
    return _AUTO_SILENCE_MIN_S + t * (_AUTO_SILENCE_MAX_S - _AUTO_SILENCE_MIN_S)


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
        mid_ms = int(((seg.start + seg.end) / 2) * 1000)
        if section.start_ms <= mid_ms < section.end_ms:
            out.append(seg)
    return out


def _build_silence_anchor_map(
    seg_in_section: list[Segment],
    min_silence_s: float,
) -> dict[int, SilenceEvent]:
    """Map each speaker-segment index to the silence event that precedes it.

    Returns a dict {seg_index: SilenceEvent} so the render loop can emit the
    event block just before the speaker paragraph that follows the silence.
    Events that trail the last speaker segment (or a section with only
    speakerless content) are stored under key `len(seg_in_section)`.
    """
    events = extract_silence_events(seg_in_section, min_silence_s)
    if not events:
        return {}

    # For each event, find the first speaker-segment whose start >= event.end.
    anchor: dict[int, SilenceEvent] = {}
    for ev in events:
        found = False
        for idx, seg in enumerate(seg_in_section):
            if seg.speaker is not None and seg.start >= ev.end:
                anchor[idx] = ev
                found = True
                break
        if not found:
            anchor[len(seg_in_section)] = ev
    return anchor


def _render_section_body(
    seg_in_section: list[Segment],
    emoji_for: dict[str, str],
    min_silence_s: float = MIN_SILENCE_S,
) -> list[str]:
    """Return Markdown paragraphs for one section."""
    if not seg_in_section:
        return []

    silence_anchor = _build_silence_anchor_map(seg_in_section, min_silence_s)

    blocks: list[str] = []
    cur_speaker: str | None = None
    cur_lines: list[str] = []

    def flush() -> None:
        if cur_lines and cur_speaker is not None:
            emoji = emoji_for.get(cur_speaker, "")
            label = f"{emoji} **{cur_speaker}:**".strip()
            blocks.append(f"{label} " + " ".join(cur_lines).strip())

    prev_end: float | None = None

    for i, seg in enumerate(seg_in_section):
        if i in silence_anchor and (blocks or cur_lines):
            # Only emit silence when speaker content has already been seen —
            # suppresses leading silences before the first speaker paragraph.
            flush()
            cur_lines = []
            cur_speaker = None
            blocks.append("---")

        if seg.speaker is None:
            # Speakerless segments (muted, noise tags) are handled via silence
            # events — skip them here to avoid double-emission.
            continue

        label = seg.name or seg.speaker
        same_speaker = label == cur_speaker
        gap = (seg.start - prev_end) if prev_end is not None else 0.0

        if not same_speaker or gap >= PARAGRAPH_GAP_S:
            flush()
            cur_lines = []
            cur_speaker = label

        cur_lines.append(seg.content.strip())
        prev_end = seg.end

    # Trailing silence (anchored past last segment) is suppressed — we only
    # show silences that are sandwiched between two speaker paragraphs.

    flush()
    return blocks


def _render_lang_value(language: str, lang_detect_info: dict | None) -> str:
    """Format the language cell value."""
    if lang_detect_info is None:
        return f"{language} (user-specified)"
    top_lang, top_p = lang_detect_info["top"]
    tag = f"auto-detected={top_p:.2f}"
    second = lang_detect_info.get("second")
    if second:
        tag += f", {second[0]}={second[1]:.2f}"
    return f"{top_lang} ({tag})"


def _render_meta_table(
    audio_meta: AudioMeta,
    language: str,
    lang_detect_info: dict | None,
) -> list[str]:
    """Two-column table: label | value for date/duration and language."""
    started = _format_started_at(audio_meta.started_at)
    duration = _format_compact_duration(audio_meta.duration_s)
    lang_val = _render_lang_value(language, lang_detect_info)
    return [
        "| | |",
        "| --- | --- |",
        f"| 📅 **Початок (тривалість):** | {started} ({duration}) |",
        f"| 🌐 **Мова:** | {lang_val} |",
    ]


def _render_participants_table(
    speakers: list[str],
    emoji_for: dict[str, str],
    name_sources: dict[str, str] | None,
    segments: list[Segment],
) -> list[str]:
    """Two-column table: emoji+name | source tag."""
    if not speakers:
        return []
    lines = [
        "| 👥 **Співрозмовники:** | |",
        "| --- | --- |",
    ]
    for speaker in speakers:
        emoji = emoji_for.get(speaker, "")
        source = ""
        if name_sources:
            cluster = next(
                (seg.speaker for seg in segments if (seg.name or seg.speaker) == speaker),
                None,
            )
            if cluster and cluster in name_sources:
                source = f"*{name_sources[cluster]}*"
        name_cell = f"{emoji} **{speaker}**".strip()
        lines.append(f"| {name_cell} | {source} |")
    return lines


def _render_ai_models_table(
    stage_models: dict[str, str],
    stage_timings: dict[str, float],
    model_load_elapsed: dict[str, float],
    total: float,
    duration_s: float,
) -> list[str]:
    """Render a Markdown table grouping stages by model.

    Header row contains the ⚡ processing summary.
    Columns: Model | Stage | Time (right-aligned).
    """
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

    pct = round(total / duration_s * 100) if duration_s else 0
    processing = f"⚡ **Час обробки:** {_format_compact_duration(total)} ({pct}% of duration)"

    if not rows:
        return [f"| {processing} | | |", "| --- | --- | --: |"]

    lines = [
        f"| {processing} | | |",
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
    min_silence_s: float | None = None,
) -> str:
    """Compose the full Markdown output.

    When *min_silence_s* is None (default), the threshold is computed
    automatically per section based on its duration (see
    `_auto_silence_threshold`).  Pass an explicit value to override for all
    sections — e.g. from ``--render-min-silence-s``.
    """
    basename = Path(audio_meta.path).name
    speakers = _speakers_in_order(dialog.segments)
    emoji_for = assign_emojis(speakers)

    lines: list[str] = []
    lines.append(f"# Транскрипт: {basename}")
    lines.append("")

    # Table 1: date/language
    for row in _render_meta_table(audio_meta, language, lang_detect_info):
        lines.append(f"> {row}")
    lines.append("> ")

    # Table 2: participants (omitted when no speakers)
    participant_rows = _render_participants_table(
        speakers, emoji_for, name_sources, dialog.segments
    )
    if participant_rows:
        for row in participant_rows:
            lines.append(f"> {row}")
        lines.append("> ")

    # Table 3: AI models + processing time (omitted when no timings)
    total = stage_timings.get("total") if stage_timings else None
    if total is not None and stage_timings is not None and model_load_elapsed is not None:
        for row in _render_ai_models_table(
            stage_models or {}, stage_timings, model_load_elapsed,
            total, audio_meta.duration_s,
        ):
            lines.append(f"> {row}")

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
        if min_silence_s is None:
            section_dur = (section.end_ms - section.start_ms) / 1000.0
            threshold = _auto_silence_threshold(section_dur)
        else:
            threshold = min_silence_s
        lines.append(f"## {section.title}")
        lines.append("")
        for block in _render_section_body(in_section, emoji_for, threshold):
            lines.append(block)
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"
