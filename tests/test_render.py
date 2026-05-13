"""Offline tests for emoji assignment and Markdown rendering."""

from __future__ import annotations

from voice.render import render_markdown, _auto_silence_threshold
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
    assert "📅 **Початок (тривалість):**" in out
    assert "2026-05-10 15:44 UTC" in out
    assert "6m14s" in out
    assert "🌐 **Мова:** | uk (user-specified)" in out
    assert "🏁" not in out, "no 'Кінець' line expected"
    assert "👥 **Співрозмовники:**" in out
    assert "🔵 **Артем**" in out
    assert "🟢 **Остап**" in out
    assert "## Привітання" in out
    assert "🔵 **Артем:** Привіт!" in out
    assert "🟢 **Остап:** Привіт-привіт." in out


def test_render_participants_header_skipped_when_no_speakers():
    """Speakerless transcripts get no Учасники line."""
    segs = [Segment(start=0, end=2, content="[Human Sounds]", speaker=None)]
    sections = [Section(title="Test", start_ms=0, end_ms=2000)]
    dialog = StructuredDialog(sections=sections, segments=segs)
    out = render_markdown(audio_meta=_meta(), dialog=dialog, tldr="", language="uk")
    assert "👥" not in out


def test_render_participants_uses_pyannote_label_when_unnamed():
    """If --names wasn't passed and ask was declined, SPEAKER_XX shows up."""
    segs = [
        Segment(start=0, end=1, content="x", speaker="SPEAKER_00"),
        Segment(start=1, end=2, content="y", speaker="SPEAKER_01"),
    ]
    sections = [Section(title="Test", start_ms=0, end_ms=2000)]
    dialog = StructuredDialog(sections=sections, segments=segs)
    out = render_markdown(audio_meta=_meta(), dialog=dialog, tldr="", language="uk")
    assert "👥 **Співрозмовники:**" in out
    assert "🔵 **SPEAKER_00**" in out
    assert "🟢 **SPEAKER_01**" in out


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
    assert "📅 **Початок (тривалість):** | 2026-05-10 15:44 UTC" in out


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
    assert out.count("**Артем:**") == 1
    assert "перше речення. друге речення." in out


def test_render_short_pause_below_default_threshold_hidden():
    """Gap of 8s is below MIN_SILENCE_S=10.0 — no pause marker emitted."""
    segs = [
        Segment(start=0.0, end=2.0, content="перше.", speaker="A", name="Артем"),
        Segment(start=10.0, end=12.0, content="друге.", speaker="A", name="Артем"),
    ]
    sections = [Section(title="Test", start_ms=0, end_ms=12000)]
    dialog = StructuredDialog(sections=sections, segments=segs)
    out = render_markdown(audio_meta=_meta(), dialog=dialog, tldr="", language="uk")
    assert "пауза" not in out
    assert "muted" not in out


def test_render_long_pause_shows_hr():
    """Gap >= MIN_SILENCE_S between two speaker paragraphs renders as ---."""
    segs = [
        Segment(start=0.0, end=2.0, content="перше.", speaker="A", name="Артем"),
        Segment(start=17.0, end=19.0, content="друге.", speaker="A", name="Артем"),
    ]
    sections = [Section(title="Test", start_ms=0, end_ms=19000)]
    dialog = StructuredDialog(sections=sections, segments=segs)
    out = render_markdown(audio_meta=_meta(), dialog=dialog, tldr="", language="uk")
    assert "\n---\n" in out
    assert "пауза" not in out
    assert out.count("**Артем:**") == 2


def test_render_pause_custom_threshold():
    """With min_silence_s=3.0 the 8s gap from the first test becomes visible."""
    segs = [
        Segment(start=0.0, end=2.0, content="перше.", speaker="A", name="Артем"),
        Segment(start=10.0, end=12.0, content="друге.", speaker="A", name="Артем"),
    ]
    sections = [Section(title="Test", start_ms=0, end_ms=12000)]
    dialog = StructuredDialog(sections=sections, segments=segs)
    out = render_markdown(
        audio_meta=_meta(), dialog=dialog, tldr="", language="uk",
        min_silence_s=3.0,
    )
    assert "\n---\n" in out
    assert "пауза" not in out


def test_render_muted_shows_hr():
    """[muted, X.Xs] placeholder between two speakers renders as ---."""
    segs = [
        Segment(start=0.0, end=2.0, content="перше.", speaker="A", name="Артем"),
        Segment(start=2.0, end=16.0, content="[muted, 14.0s]", speaker=None),
        Segment(start=16.0, end=18.0, content="друге.", speaker="A", name="Артем"),
    ]
    sections = [Section(title="Test", start_ms=0, end_ms=18000)]
    dialog = StructuredDialog(sections=sections, segments=segs)
    out = render_markdown(audio_meta=_meta(), dialog=dialog, tldr="", language="uk", min_silence_s=10.0)
    assert "\n---\n" in out
    assert "muted" not in out
    assert "[muted, 14.0s]" not in out


def test_render_silence_suppressed_at_section_start():
    """Silence event before the first speaker paragraph is not rendered."""
    segs = [
        Segment(start=0.0, end=3.0, content="[muted, 3.0s]", speaker=None),
        Segment(start=3.0, end=5.0, content="Привіт.", speaker="A", name="Артем"),
        Segment(start=15.0, end=17.0, content="Поки.", speaker="A", name="Артем"),
    ]
    sections = [Section(title="Test", start_ms=0, end_ms=17000)]
    dialog = StructuredDialog(sections=sections, segments=segs)
    out = render_markdown(audio_meta=_meta(), dialog=dialog, tldr="", language="uk", min_silence_s=3.0)
    section_body = out.split("## Test")[1]
    assert section_body.strip().startswith("🔵 **Артем:**")


def test_render_silence_suppressed_at_section_end():
    """Silence event after the last speaker paragraph is not rendered."""
    segs = [
        Segment(start=0.0, end=2.0, content="Бувай.", speaker="A", name="Артем"),
        Segment(start=2.0, end=15.0, content="[muted, 13.0s]", speaker=None),
    ]
    sections = [Section(title="Test", start_ms=0, end_ms=15000)]
    dialog = StructuredDialog(sections=sections, segments=segs)
    out = render_markdown(audio_meta=_meta(), dialog=dialog, tldr="", language="uk", min_silence_s=10.0)
    section_body = out.split("## Test")[1]
    assert "---" not in section_body


def test_render_muted_below_threshold_hidden():
    """Short muted placeholder (< min_silence_s) is silently dropped."""
    segs = [
        Segment(start=0.0, end=2.0, content="перше.", speaker="A", name="Артем"),
        Segment(start=2.0, end=5.0, content="[muted, 3.0s]", speaker=None),
        Segment(start=5.0, end=7.0, content="друге.", speaker="A", name="Артем"),
    ]
    sections = [Section(title="Test", start_ms=0, end_ms=7000)]
    dialog = StructuredDialog(sections=sections, segments=segs)
    out = render_markdown(audio_meta=_meta(), dialog=dialog, tldr="", language="uk", min_silence_s=10.0)
    assert "muted" not in out
    assert "пауза" not in out


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


# -------------------------------------------------------- new header features


def test_render_lang_user_specified():
    """Without lang_detect_info → language is tagged as user-specified."""
    segs = [Segment(start=0, end=1, content="x", speaker="A", name="Sam")]
    dialog = StructuredDialog(
        sections=[Section(title="T", start_ms=0, end_ms=1000)], segments=segs
    )
    out = render_markdown(audio_meta=_meta(), dialog=dialog, language="uk")
    assert "🌐 **Мова:** | uk (user-specified)" in out


def test_render_lang_auto_detected_with_runner_up():
    """lang_detect_info with runner-up → probability shown."""
    segs = [Segment(start=0, end=1, content="x", speaker="A", name="Sam")]
    dialog = StructuredDialog(
        sections=[Section(title="T", start_ms=0, end_ms=1000)], segments=segs
    )
    out = render_markdown(
        audio_meta=_meta(), dialog=dialog, language="uk",
        lang_detect_info={"top": ("uk", 0.74), "second": ("ru", 0.24)},
    )
    assert "🌐 **Мова:** | uk (auto-detected=0.74, ru=0.24)" in out


def test_render_lang_auto_detected_no_runner_up():
    """lang_detect_info without 'second' key → no runner-up shown."""
    segs = [Segment(start=0, end=1, content="x", speaker="A", name="Sam")]
    dialog = StructuredDialog(
        sections=[Section(title="T", start_ms=0, end_ms=1000)], segments=segs
    )
    out = render_markdown(
        audio_meta=_meta(), dialog=dialog, language="uk",
        lang_detect_info={"top": ("uk", 0.95)},
    )
    assert "auto-detected=0.95" in out
    assert "ru=" not in out


def test_render_participants_with_source_tags():
    """Each participant line includes its source tag in parentheses."""
    segs = [
        Segment(start=0, end=1, content="x", speaker="SPEAKER_00", name="Остап"),
        Segment(start=1, end=2, content="y", speaker="SPEAKER_01", name="Артем"),
    ]
    sections = [Section(title="T", start_ms=0, end_ms=2000)]
    dialog = StructuredDialog(sections=sections, segments=segs)
    out = render_markdown(
        audio_meta=_meta(), dialog=dialog, language="uk",
        name_sources={
            "SPEAKER_00": "self-introduced",
            "SPEAKER_01": "user-specified",
        },
    )
    assert "| 🔵 **Остап** | *self-introduced*" in out
    assert "| 🟢 **Артем** | *user-specified*" in out


def test_render_ai_models_table_grouping_single_llm():
    """All LLM stages on one model → one group in the table."""
    segs = [Segment(start=0, end=1, content="x", speaker="A", name="Sam")]
    dialog = StructuredDialog(
        sections=[Section(title="T", start_ms=0, end_ms=1000)], segments=segs
    )
    qwen = "mlx-community/Qwen2.5-7B-Instruct-4bit"
    stage_models = {
        "diarize_speakers": "pyannote/speaker-diarization-3.1",
        "speech2text": "Whisper-large-v3-MLX",
        "proofread": qwen,
        "speech_structure": qwen,
    }
    stage_timings = {
        "total": 321.0,
        "diarize_speakers": 28.0,
        "speech2text": 55.0,
        "proofread": 138.0,
        "speech_structure": 49.0,
    }
    model_load_elapsed = {
        "pyannote/speaker-diarization-3.1": 5.0,
        "Whisper-large-v3-MLX": 32.0,
        qwen: 62.0,
    }
    out = render_markdown(
        audio_meta=_meta(), dialog=dialog, language="uk",
        stage_models=stage_models,
        stage_timings=stage_timings,
        model_load_elapsed=model_load_elapsed,
    )
    assert "| --- | --- | --: |" in out
    # Each model appears once with model loaded as first stage.
    assert "model loaded" in out
    assert "pyannote/speaker-diarization-3.1" in out
    assert "Whisper-large-v3-MLX" in out
    assert qwen in out
    # Qwen should appear only once in the table (grouped).
    assert out.count(qwen) == 1


def test_render_ai_models_skips_disabled_stages():
    """Stages not in stage_models don't appear in the table."""
    segs = [Segment(start=0, end=1, content="x", speaker="A", name="Sam")]
    dialog = StructuredDialog(
        sections=[Section(title="T", start_ms=0, end_ms=1000)], segments=segs
    )
    out = render_markdown(
        audio_meta=_meta(), dialog=dialog, language="uk",
        stage_models={"speech2text": "Whisper-large-v3-MLX"},
        stage_timings={"total": 60.0, "speech2text": 55.0},
        model_load_elapsed={"Whisper-large-v3-MLX": 30.0},
    )
    assert "speech_tldr" not in out
    assert "speech_summary" not in out


def test_render_processing_percent_of_duration():
    """⏲️ line shows percentage of audio duration."""
    segs = [Segment(start=0, end=1, content="x", speaker="A", name="Sam")]
    dialog = StructuredDialog(
        sections=[Section(title="T", start_ms=0, end_ms=1000)], segments=segs
    )
    meta = AudioMeta(
        path="/tmp/foo.m4a",
        started_at="2026-05-10T15:44:02+00:00",
        ended_at="2026-05-10T15:54:02+00:00",
        duration_s=600.0,
        source="test",
    )
    out = render_markdown(
        audio_meta=meta, dialog=dialog, language="uk",
        stage_timings={"total": 300.0},
        stage_models={},
        model_load_elapsed={},
    )
    assert "⚡ **Час обробки:** 5m0s (50% of duration)" in out


# -------------------------------------------------------- auto silence threshold


def test_auto_silence_threshold_short_section():
    assert _auto_silence_threshold(30.0) == 5.0


def test_auto_silence_threshold_at_lower_bound():
    assert _auto_silence_threshold(60.0) == 5.0


def test_auto_silence_threshold_midpoint():
    # 180 s is exactly halfway between 60 and 300 → midpoint of 5..15 = 10
    assert _auto_silence_threshold(180.0) == 10.0


def test_auto_silence_threshold_at_upper_bound():
    assert _auto_silence_threshold(300.0) == 15.0


def test_auto_silence_threshold_long_section():
    assert _auto_silence_threshold(600.0) == 15.0


def test_render_auto_threshold_used_when_not_overridden():
    """Default (no min_silence_s) uses auto threshold: short section → 5 s floor."""
    segs = [
        Segment(start=0.0, end=2.0, content="перше.", speaker="A", name="Артем"),
        Segment(start=8.0, end=10.0, content="друге.", speaker="A", name="Артем"),
    ]
    # Section duration = 10 s → auto threshold = 5 s; gap = 6 s → visible
    sections = [Section(title="Test", start_ms=0, end_ms=10000)]
    dialog = StructuredDialog(sections=sections, segments=segs)
    out = render_markdown(audio_meta=_meta(), dialog=dialog, tldr="", language="uk")
    assert "\n---\n" in out


def test_render_explicit_override_suppresses_auto():
    """Explicit min_silence_s overrides auto threshold for all sections."""
    segs = [
        Segment(start=0.0, end=2.0, content="перше.", speaker="A", name="Артем"),
        Segment(start=8.0, end=10.0, content="друге.", speaker="A", name="Артем"),
    ]
    # Same 6 s gap, but explicit threshold = 10 s → hidden
    sections = [Section(title="Test", start_ms=0, end_ms=10000)]
    dialog = StructuredDialog(sections=sections, segments=segs)
    out = render_markdown(
        audio_meta=_meta(), dialog=dialog, tldr="", language="uk",
        min_silence_s=10.0,
    )
    assert "\n---\n" not in out
