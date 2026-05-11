"""Offline tests for emoji assignment and Markdown rendering."""

from __future__ import annotations

from voice.render import render_markdown
from voice.speaker_emojis import EMOJI_PALETTE, assign_emojis
from voice.types import AudioMeta, Section, Segment, StructuredDialog


def test_assign_emojis_in_order():
    out = assign_emojis(["Артем", "Остап", "Артем"])
    assert out["Артем"] == EMOJI_PALETTE[0]
    assert out["Остап"] == EMOJI_PALETTE[1]
    assert len(out) == 2  # repeats don't add entries


def test_assign_emojis_cycles_past_palette():
    speakers = [f"S{i}" for i in range(len(EMOJI_PALETTE) + 2)]
    out = assign_emojis(speakers)
    assert out[speakers[0]] == EMOJI_PALETTE[0]
    assert out[speakers[len(EMOJI_PALETTE)]] == EMOJI_PALETTE[0]  # wraps
    assert out[speakers[len(EMOJI_PALETTE) + 1]] == EMOJI_PALETTE[1]


def _meta() -> AudioMeta:
    return AudioMeta(
        path="/tmp/foo.m4a",
        started_at="2026-05-10T15:44:02+00:00",
        ended_at="2026-05-10T15:50:16+00:00",
        duration_s=374.0,
        source="ffprobe creation_time",
    )


def test_render_minimal_dialogue():
    segs = [
        Segment(start=0.0, end=2.0, content="Привіт!", speaker="SPEAKER_00", name="Артем"),
        Segment(start=2.0, end=4.0, content="Привіт-привіт.", speaker="SPEAKER_01", name="Остап"),
    ]
    sections = [Section(title="Привітання", start_ms=0, end_ms=4000)]
    dialog = StructuredDialog(sections=sections, segments=segs)

    out = render_markdown(audio_meta=_meta(), dialog=dialog, tldr="", language="uk")
    assert "# Транскрипт: foo.m4a" in out
    assert "📅 **Початок:** 2026-05-10 15:44 UTC" in out
    assert "⏱️ **Тривалість:** 6:14" in out
    assert "🌐 **Мова:** Українська" in out
    assert "🏁" not in out, "no 'Кінець' line expected"
    assert "## Привітання" in out
    assert "🔵 **Артем:** Привіт!" in out
    assert "🟢 **Остап:** Привіт-привіт." in out


def test_render_uses_friendly_language_name_with_fallback():
    """Known codes (uk, pt-br) render full names; unknowns pass through."""
    segs = [Segment(start=0, end=1, content="hi", speaker="A", name="Sam")]
    dialog = StructuredDialog(
        sections=[Section(title="Test", start_ms=0, end_ms=1000)],
        segments=segs,
    )
    out_uk = render_markdown(audio_meta=_meta(), dialog=dialog, tldr="", language="uk")
    out_ptbr = render_markdown(audio_meta=_meta(), dialog=dialog, tldr="", language="pt-br")
    out_xx = render_markdown(audio_meta=_meta(), dialog=dialog, tldr="", language="xx-YY")
    assert "🌐 **Мова:** Українська" in out_uk
    assert "🌐 **Мова:** Бразильська португальська" in out_ptbr
    assert "🌐 **Мова:** xx-YY" in out_xx


def test_render_normalises_started_at_with_offset():
    """A non-UTC ISO string is converted to UTC for display."""
    meta = AudioMeta(
        path="/tmp/foo.m4a",
        started_at="2026-05-10T17:44:02+02:00",  # +02 → 15:44 UTC
        ended_at="2026-05-10T17:50:16+02:00",
        duration_s=374.0,
        source="ffprobe creation_time",
    )
    dialog = StructuredDialog(
        sections=[Section(title="T", start_ms=0, end_ms=1000)],
        segments=[Segment(start=0, end=1, content="hi", speaker="A", name="Sam")],
    )
    out = render_markdown(audio_meta=meta, dialog=dialog, tldr="", language="uk")
    assert "📅 **Початок:** 2026-05-10 15:44 UTC" in out


def test_render_includes_tldr_section_when_provided():
    segs = [Segment(start=0, end=1, content="hi", speaker="A", name="Sam")]
    sections = [Section(title="Intro", start_ms=0, end_ms=1000)]
    dialog = StructuredDialog(sections=sections, segments=segs)
    out = render_markdown(
        audio_meta=_meta(), dialog=dialog,
        tldr="- point one\n- point two", language="uk",
    )
    assert "## TL;DR" in out
    assert "- point one" in out
    assert out.index("## TL;DR") < out.index("## Intro")


def test_render_merges_same_speaker_short_gap_into_one_paragraph():
    segs = [
        Segment(start=0.0, end=2.0, content="перше речення.", speaker="A", name="Артем"),
        Segment(start=2.5, end=4.0, content="друге речення.", speaker="A", name="Артем"),
    ]
    sections = [Section(title="Test", start_ms=0, end_ms=4000)]
    dialog = StructuredDialog(sections=sections, segments=segs)
    out = render_markdown(audio_meta=_meta(), dialog=dialog, tldr="", language="uk")
    # 1 paragraph for Артем
    assert out.count("**Артем:**") == 1
    assert "перше речення. друге речення." in out


def test_render_explicit_pause_marker_on_long_gap():
    segs = [
        Segment(start=0.0, end=2.0, content="перше.", speaker="A", name="Артем"),
        Segment(start=10.0, end=12.0, content="друге.", speaker="A", name="Артем"),
    ]
    sections = [Section(title="Test", start_ms=0, end_ms=12000)]
    dialog = StructuredDialog(sections=sections, segments=segs)
    out = render_markdown(audio_meta=_meta(), dialog=dialog, tldr="", language="uk")
    assert "пауза 8с" in out
    # two separate Артем blocks now
    assert out.count("**Артем:**") == 2


def test_render_skips_speakerless_segments():
    segs = [
        Segment(start=0.0, end=2.0, content="[Human Sounds]", speaker=None),
        Segment(start=2.0, end=4.0, content="Привіт.", speaker="A", name="Артем"),
    ]
    sections = [Section(title="Test", start_ms=0, end_ms=4000)]
    dialog = StructuredDialog(sections=sections, segments=segs)
    out = render_markdown(audio_meta=_meta(), dialog=dialog, tldr="", language="uk")
    assert "[Human Sounds]" not in out
    assert "**Артем:** Привіт." in out


def test_render_uses_label_when_name_missing():
    segs = [Segment(start=0, end=1, content="hi", speaker="SPEAKER_00")]
    sections = [Section(title="Test", start_ms=0, end_ms=1000)]
    dialog = StructuredDialog(sections=sections, segments=segs)
    out = render_markdown(audio_meta=_meta(), dialog=dialog, tldr="", language="uk")
    assert "**SPEAKER_00:**" in out
