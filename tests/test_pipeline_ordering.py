"""Verify the diarize→clearspeech→ASR order and the autogain/bandpass toggles
plumbing without loading real models.

Strategy: monkey-patch the pipeline's module-level references for
whisper_asr, diarize, clearspeech, MlxLLM, audiometa, and the
LLM-dependent stages. Each stub records what it was called with, and the
test inspects the call log after `pipeline.run()` returns.
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

    monkeypatch.setattr(
        pipeline_module.transcode_module, "transcode", fake_ffmpeg
    )

    # audiometa: return a minimal AudioMeta.
    fake_meta = AudioMeta(
        path=str(fake_audio),
        started_at="2026-05-10T15:44:02+00:00",
        ended_at="2026-05-10T15:44:05+00:00",
        duration_s=3.0,
        source="cli override",
    )
    monkeypatch.setattr(
        pipeline_module.audiometa_module,
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
                "turn_count": len(kwargs.get("autogain_turns") or []),
                "autogain_target_dbfs": kwargs.get("autogain_target_dbfs"),
                "autogain_max_gain_db": kwargs.get("autogain_max_gain_db"),
                "bandpass_low_hz": kwargs.get("bandpass_low_hz"),
                "bandpass_high_hz": kwargs.get("bandpass_high_hz"),
                "presence_center_hz": kwargs.get("presence_center_hz"),
                "presence_boost_db": kwargs.get("presence_boost_db"),
                "presence_q": kwargs.get("presence_q"),
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

    # Whisper backend: record the wav path; return one minimal segment.
    def fake_whisper_transcribe(wav_path, **kwargs):
        rec("asr", str(wav_path))
        return [AsrSegment(start=0.0, end=3.0, content="hi")]

    monkeypatch.setattr(
        pipeline_module.whisper_asr_module, "transcribe", fake_whisper_transcribe
    )

    # LLM: never actually instantiate one. The pipeline only uses LLM if
    # run_proofread / run_tldr / run_structure are true; we'll disable
    # those for ordering tests. But the unknown_speaker="ask" branch + names
    # override path still pull in identify+structure, so monkey-patch those
    # too as light stubs.
    class _StubLLM:
        instances: list["_StubLLM"] = []

        def __init__(self, *a, **k):
            self.model_path = k.get("model_path", "/tmp/fake-model-default")
            self.log_memory = k.get("log_memory", False)
            self.sampling_overrides = k.get("sampling_overrides", {})
            # Mirrors the real `MlxLLM._resolved_path` populated by `.load()`.
            # The pipeline compares it against the desired spec to decide
            # whether to swap models, so stubs must expose the attribute.
            self._resolved_path: str | None = None
            self.closed = False
            _StubLLM.instances.append(self)

        def load(self):
            self._resolved_path = self.model_path

        def close(self):
            self.closed = True
            self._resolved_path = None

    # Reset per-test so `instances` reflects only this run.
    _StubLLM.instances = []
    monkeypatch.setattr(pipeline_module, "MlxLLM", _StubLLM)
    # Expose the class so a test can inspect created instances.
    rec.stub_llm = _StubLLM

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


def test_default_chain_runs_autogain_only(patched_pipeline, tmp_path):
    """0.13.0-equivalent defaults: autogain on, bandpass off."""
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
    assert cs["chain"] == ["autogain"]
    assert cs["turn_count"] == 2
    assert cs["autogain_target_dbfs"] == -20.0
    assert cs["autogain_max_gain_db"] == 16.0

    # ASR sees the .autogain.wav sibling, not the raw WAV.
    asr_path = rec.payload("asr")
    assert asr_path.endswith(".autogain.wav"), asr_path

    # Diarize still sees the original raw WAV.
    diar_path = rec.payload("diarize")
    assert diar_path.endswith("audio.wav") and ".autogain" not in diar_path


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
            clearspeech_chain="autogain,bandpass",
        )
    )

    cs = rec.payload("clearspeech")
    assert cs["chain"] == ["autogain", "bandpass"]
    assert cs["bandpass_low_hz"] == 150.0
    assert cs["bandpass_high_hz"] == 5_500.0

    # ASR sees the final sibling with both suffixes.
    assert rec.payload("asr").endswith(".autogain.bandpass.wav")


def test_empty_chain_passes_raw_wav_to_asr(patched_pipeline, tmp_path):
    """Empty chain string disables all preprocessing — ASR reads raw WAV."""
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
            clearspeech_chain="",
        )
    )

    cs = rec.payload("clearspeech")
    assert cs["chain"] == []

    # Diarize still runs before ASR even when the chain is empty.
    names = rec.names()
    assert names.index("diarize") < names.index("asr")

    # Empty chain means ASR sees the same path as diarize.
    assert rec.payload("asr") == rec.payload("diarize")


def test_full_chain_autogain_bandpass_presence(patched_pipeline, tmp_path):
    """Three effects in canonical order; ASR sees triple-suffixed WAV."""
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
            clearspeech_chain="autogain,bandpass,presence",
        )
    )

    cs = rec.payload("clearspeech")
    assert cs["chain"] == ["autogain", "bandpass", "presence"]
    assert cs["presence_center_hz"] == 3_000.0
    assert cs["presence_boost_db"] == 6.0
    assert cs["presence_q"] == 1.0
    assert rec.payload("asr").endswith(".autogain.bandpass.presence.wav")


def test_reordered_chain_runs_in_given_order(patched_pipeline, tmp_path):
    """User-specified order overrides any canonical assumption."""
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
            clearspeech_chain="presence,autogain",
        )
    )

    cs = rec.payload("clearspeech")
    assert cs["chain"] == ["presence", "autogain"]
    assert rec.payload("asr").endswith(".presence.autogain.wav")


# ---------------------------------------------------------------- per-stage LLM


def test_single_llm_model_reused_across_all_stages(patched_pipeline, tmp_path):
    """One --llm-model → exactly one MlxLLM instance for the whole pipeline."""
    rec, fake_audio = patched_pipeline
    out = tmp_path / "out.md"

    run(
        PipelineOptions(
            audio_path=str(fake_audio),
            output_path=str(out),
            llm_model="/tmp/single-model",
            run_proofread=True,
            run_tldr=True,
            run_structure=True,
            names_override=["A", "B"],
            unknown_speaker="keep",
        )
    )

    instances = rec.stub_llm.instances
    assert len(instances) == 1, [i.model_path for i in instances]
    assert instances[0].model_path == "/tmp/single-model"
    assert instances[0].closed is True


def test_per_stage_models_swap_when_different(patched_pipeline, tmp_path):
    """Different paths per stage → cold-reload between stages."""
    rec, fake_audio = patched_pipeline
    out = tmp_path / "out.md"

    run(
        PipelineOptions(
            audio_path=str(fake_audio),
            output_path=str(out),
            llm_model="/tmp/big",
            llm_proofread_model="/tmp/small",
            llm_structure_model="/tmp/big",
            llm_tldr_model="/tmp/medium",
            run_proofread=True,
            run_tldr=True,
            run_structure=True,
            names_override=["A", "B"],
            unknown_speaker="keep",
        )
    )

    paths = [i.model_path for i in rec.stub_llm.instances]
    # proofread → /tmp/small, structure → /tmp/big, tldr → /tmp/medium
    assert paths == ["/tmp/small", "/tmp/big", "/tmp/medium"], paths
    # Each non-final instance must have been closed when the next swap occurred.
    for inst in rec.stub_llm.instances[:-1]:
        assert inst.closed is True
    # Last instance is closed by the final `finally` block.
    assert rec.stub_llm.instances[-1].closed is True


def test_same_path_across_stages_reuses_one_instance(patched_pipeline, tmp_path):
    """Per-stage flags all pointing at the same path → one instance, no swap."""
    rec, fake_audio = patched_pipeline
    out = tmp_path / "out.md"

    run(
        PipelineOptions(
            audio_path=str(fake_audio),
            output_path=str(out),
            llm_proofread_model="/tmp/same",
            llm_structure_model="/tmp/same",
            llm_tldr_model="/tmp/same",
            run_proofread=True,
            run_tldr=True,
            run_structure=True,
            names_override=["A", "B"],
            unknown_speaker="keep",
        )
    )

    instances = rec.stub_llm.instances
    assert len(instances) == 1, [i.model_path for i in instances]
    assert instances[0].model_path == "/tmp/same"


def test_per_stage_override_does_not_leak_to_unspecified_stages(
    patched_pipeline, tmp_path
):
    """When only proofread is overridden, structure/tldr must NOT reuse it.

    Regression for the bug where `_resolve_stage_model` returned None for
    unspecified stages and `_ensure_llm` then treated None as "keep the
    current model loaded" — silently inheriting the proofread override.
    """
    rec, fake_audio = patched_pipeline
    out = tmp_path / "out.md"

    run(
        PipelineOptions(
            audio_path=str(fake_audio),
            output_path=str(out),
            llm_proofread_model="/tmp/small",  # only proofread overridden
            run_proofread=True,
            run_tldr=True,
            run_structure=True,
            names_override=["A", "B"],
            unknown_speaker="keep",
        )
    )

    paths = [i.model_path for i in rec.stub_llm.instances]
    # First instance must be the override; the second (structure/tldr) must
    # be the built-in default path, not "/tmp/small".
    from voice.llm import DEFAULT_MODEL as default
    assert paths[0] == "/tmp/small", paths
    assert paths[1] == default, paths
    # Structure and TLDR share the default — they should reuse one instance.
    assert len(paths) == 2, paths


def test_no_llm_required_when_no_stages_and_names_override(patched_pipeline, tmp_path):
    """Disabled stages + names override + keep policy → no LLM ever loaded."""
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

    assert rec.stub_llm.instances == []
