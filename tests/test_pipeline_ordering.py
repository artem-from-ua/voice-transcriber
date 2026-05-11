"""Verify the diarize→clearspeech→ASR order and the AGC/bandpass toggles
plumbing without loading real models.

Strategy: monkey-patch the pipeline's module-level references for ASR,
diarize, clearspeech, MlxLLM, ffprobe, and the LLM-dependent stages.
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

    # clearspeech: pretend to write one sibling WAV per applied effect and
    # record what the dispatcher was asked to do.
    def fake_clearspeech(wav_path, chain, **kwargs):
        chain = tuple(chain)
        current = Path(wav_path)
        for idx, effect in enumerate(chain, start=1):
            # Mirror real clearspeech: each effect writes a sibling whose
            # name is `<previous-stem>.<effect>.wav`.
            current = current.with_suffix(f".{effect}.wav")
            current.write_bytes(f"RIFF\x00\x00\x00\x00fake-{effect}".encode())
            if kwargs.get("dump") is not None:
                kwargs["dump"](idx, effect, current)
        rec(
            "clearspeech",
            {
                "wav_path": str(wav_path),
                "chain": list(chain),
                "turn_count": len(kwargs.get("agc_turns") or []),
                "agc_target_dbfs": kwargs.get("agc_target_dbfs"),
                "agc_max_gain_db": kwargs.get("agc_max_gain_db"),
                "bandpass_low_hz": kwargs.get("bandpass_low_hz"),
                "bandpass_high_hz": kwargs.get("bandpass_high_hz"),
            },
        )
        config = {
            "any_enabled": bool(chain),
            "chain": list(chain),
            "steps": [{"name": e, "applied": True, "params": {}, "stats": {}} for e in chain],
        }
        return (current if chain else Path(wav_path)), config

    monkeypatch.setattr(
        pipeline_module.clearspeech_module,
        "clearspeech",
        fake_clearspeech,
    )

    # ASR: record the wav path; return one minimal segment.
    def fake_transcribe(wav_path, **kwargs):
        rec("asr", str(wav_path))
        return [AsrSegment(start=0.0, end=3.0, content="hi", speaker_asr=0)]

    monkeypatch.setattr(pipeline_module.asr_module, "transcribe", fake_transcribe)

    # LLM: never actually instantiate one. The pipeline only uses LLM if
    # run_proofread / run_tldr / run_structure are true; we'll disable
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


def test_default_chain_runs_agc_only(patched_pipeline, tmp_path):
    """0.13.0-equivalent defaults: AGC on, bandpass off."""
    rec, fake_audio = patched_pipeline
    out = tmp_path / "out.md"

    run(
        PipelineOptions(
            audio_path=str(fake_audio),
            output_path=str(out),
            run_proofread=False,
            run_tldr=False,
            run_structure=False,
            names_override=["A", "B"],
            unknown_speaker="keep",
        )
    )

    names = rec.names()
    i_diar = names.index("diarize")
    i_cs = names.index("clearspeech")
    i_asr = names.index("asr")
    assert i_diar < i_cs < i_asr, f"unexpected order: {names}"

    cs = rec.payload("clearspeech")
    assert cs["chain"] == ["agc"]
    assert cs["turn_count"] == 2
    assert cs["agc_target_dbfs"] == -20.0
    assert cs["agc_max_gain_db"] == 16.0

    # ASR sees the .agc.wav sibling, not the raw WAV.
    asr_path = rec.payload("asr")
    assert asr_path.endswith(".agc.wav"), asr_path

    # Diarize still sees the original raw WAV.
    diar_path = rec.payload("diarize")
    assert diar_path.endswith("audio.wav") and ".agc" not in diar_path


def test_bandpass_extends_chain(patched_pipeline, tmp_path):
    rec, fake_audio = patched_pipeline
    out = tmp_path / "out.md"

    run(
        PipelineOptions(
            audio_path=str(fake_audio),
            output_path=str(out),
            run_proofread=False,
            run_tldr=False,
            run_structure=False,
            names_override=["A", "B"],
            unknown_speaker="keep",
            clearspeech_bandpass=True,
        )
    )

    cs = rec.payload("clearspeech")
    assert cs["chain"] == ["agc", "bandpass"]
    assert cs["bandpass_low_hz"] == 80.0
    assert cs["bandpass_high_hz"] == 7_900.0

    # ASR sees the final sibling with both suffixes.
    assert rec.payload("asr").endswith(".agc.bandpass.wav")


def test_no_clearspeech_agc_passes_raw_wav_to_asr(patched_pipeline, tmp_path):
    """With both effects off, ASR sees the raw WAV directly."""
    rec, fake_audio = patched_pipeline
    out = tmp_path / "out.md"

    run(
        PipelineOptions(
            audio_path=str(fake_audio),
            output_path=str(out),
            run_proofread=False,
            run_tldr=False,
            run_structure=False,
            names_override=["A", "B"],
            unknown_speaker="keep",
            clearspeech_agc=False,
        )
    )

    cs = rec.payload("clearspeech")
    assert cs["chain"] == []

    # Diarize still runs before ASR even when the chain is empty.
    names = rec.names()
    assert names.index("diarize") < names.index("asr")

    # Empty chain means ASR sees the same path as diarize.
    assert rec.payload("asr") == rec.payload("diarize")
