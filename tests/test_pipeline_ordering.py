"""Verify the diarize→normalize→ASR order and the --no-loudness-normalize
plumbing without loading real models.

Strategy: monkey-patch the pipeline's module-level references for ASR,
diarize, audio_preprocess, MlxLLM, ffprobe, and the LLM-dependent stages.
Each stub records what it was called with, and the test inspects the call
log after `pipeline.run()` returns.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from voice import pipeline as pipeline_module
from voice.pipeline import PipelineOptions, run
from voice.types import AsrSegment, AudioMeta, DiarTurn, Section, Segment, StructuredDialog


class _Recorder:
    """Records each call so the test can assert order + arguments."""

    def __init__(self) -> None:
        self.events: list[tuple[str, Any]] = []

    def __call__(self, name: str, payload: Any = None) -> None:
        self.events.append((name, payload))

    def names(self) -> list[str]:
        return [n for n, _ in self.events]

    def payload(self, name: str) -> Any:
        for n, p in self.events:
            if n == name:
                return p
        raise KeyError(name)


@pytest.fixture
def patched_pipeline(monkeypatch, tmp_path):
    rec = _Recorder()

    fake_audio = tmp_path / "input.m4a"
    fake_audio.write_bytes(b"not-a-real-audio-file")

    # ffmpeg: don't shell out, just create the expected output WAV.
    def fake_ffmpeg(src: Path, dst: Path, log) -> None:
        dst.write_bytes(b"RIFF\x00\x00\x00\x00fake")
        rec("ffmpeg", str(dst))

    monkeypatch.setattr(pipeline_module, "_to_wav_16k_mono", fake_ffmpeg)

    # ffprobe: return a minimal AudioMeta.
    fake_meta = AudioMeta(
        path=str(fake_audio),
        started_at="2026-05-10T15:44:02+00:00",
        ended_at="2026-05-10T15:44:05+00:00",
        duration_s=3.0,
        source="cli override",
    )
    monkeypatch.setattr(
        pipeline_module.ffprobe_module,
        "extract_metadata",
        lambda *a, **k: fake_meta,
    )

    # diarize: return two turns.
    fake_turns = [
        DiarTurn(start=0.0, end=1.5, speaker="SPEAKER_00"),
        DiarTurn(start=1.5, end=3.0, speaker="SPEAKER_01"),
    ]

    def fake_diarize(wav_path, *, log=print):
        rec("diarize", str(wav_path))
        return fake_turns

    monkeypatch.setattr(pipeline_module.diarize_module, "diarize", fake_diarize)

    # audio_preprocess: pretend to write a sibling normalized.wav and record
    # the args it was called with.
    def fake_normalize(wav_path, turns, **kwargs):
        out = Path(wav_path).with_suffix(".normalized.wav")
        out.write_bytes(b"RIFF\x00\x00\x00\x00fake-normalized")
        rec(
            "loudness_normalize",
            {
                "wav_path": str(wav_path),
                "turn_count": len(turns),
                "target_dbfs": kwargs.get("target_dbfs"),
                "max_gain_db": kwargs.get("max_gain_db"),
            },
        )
        return out

    monkeypatch.setattr(
        pipeline_module.audio_preprocess_module,
        "loudness_normalize",
        fake_normalize,
    )

    # ASR: record the wav path; return one minimal segment.
    def fake_transcribe(wav_path, **kwargs):
        rec("asr", str(wav_path))
        return [AsrSegment(start=0.0, end=3.0, content="hi", speaker_asr=0)]

    monkeypatch.setattr(pipeline_module.asr_module, "transcribe", fake_transcribe)

    # LLM: never actually instantiate one. The pipeline only uses LLM if
    # run_postprocess / run_tldr / run_structure are true; we'll disable
    # those for ordering tests. But the unknown_speaker="ask" branch + names
    # override path still pull in identify+structure, so monkey-patch those
    # too as light stubs.
    class _StubLLM:
        model_path = "/tmp/fake-model"

        def __init__(self, *a, **k):
            pass

        def load(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(pipeline_module, "MlxLLM", _StubLLM)

    monkeypatch.setattr(
        pipeline_module.identify_module,
        "identify_speakers",
        lambda *a, **k: {"SPEAKER_00": "A", "SPEAKER_01": "B"},
    )
    monkeypatch.setattr(
        pipeline_module.structure_module,
        "structure_dialog",
        lambda segs, **k: StructuredDialog(
            sections=[Section(title="t", start_ms=0, end_ms=3000)],
            segments=segs,
        ),
    )
    monkeypatch.setattr(
        pipeline_module.tldr_module,
        "generate_tldr",
        lambda *a, **k: "",
    )
    monkeypatch.setattr(
        pipeline_module,
        "render_markdown",
        lambda **k: "# fake\n",
    )

    return rec, fake_audio


def test_default_order_is_diarize_then_normalize_then_asr(patched_pipeline, tmp_path):
    rec, fake_audio = patched_pipeline
    out = tmp_path / "out.md"

    run(
        PipelineOptions(
            audio_path=str(fake_audio),
            output_path=str(out),
            run_postprocess=False,
            run_tldr=False,
            run_structure=False,
            names_override=["A", "B"],
            unknown_speaker="keep",
        )
    )

    names = rec.names()
    # ffprobe call uses extract_metadata directly (not recorded). The recorded
    # sequence we care about is ffmpeg → diarize → loudness_normalize → asr.
    i_diar = names.index("diarize")
    i_norm = names.index("loudness_normalize")
    i_asr = names.index("asr")
    assert i_diar < i_norm < i_asr, f"unexpected order: {names}"

    # ASR sees the normalised sibling, not the raw WAV.
    asr_path = rec.payload("asr")
    assert asr_path.endswith(".normalized.wav"), asr_path

    # Diarize sees the original raw WAV.
    diar_path = rec.payload("diarize")
    assert diar_path.endswith("audio.wav") and ".normalized" not in diar_path

    # Loudness stage got the right defaults.
    norm = rec.payload("loudness_normalize")
    assert norm["turn_count"] == 2
    assert norm["target_dbfs"] == -20.0
    assert norm["max_gain_db"] == 16.0


def test_no_loudness_normalize_keeps_raw_wav_for_asr(patched_pipeline, tmp_path):
    rec, fake_audio = patched_pipeline
    out = tmp_path / "out.md"

    run(
        PipelineOptions(
            audio_path=str(fake_audio),
            output_path=str(out),
            run_postprocess=False,
            run_tldr=False,
            run_structure=False,
            names_override=["A", "B"],
            unknown_speaker="keep",
            run_loudness_normalize=False,
        )
    )

    names = rec.names()
    assert "loudness_normalize" not in names, "stage must be skipped"

    # Diarize still runs before ASR even with the stage off.
    assert names.index("diarize") < names.index("asr")

    # ASR sees the same path diarize saw.
    assert rec.payload("asr") == rec.payload("diarize")
