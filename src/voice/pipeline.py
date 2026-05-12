"""End-to-end orchestration: audio file → Markdown transcript.

Stages run sequentially; intermediate results live in memory. The only
on-disk artefact apart from the input audio is the final Markdown file.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from . import audio_meta as audio_meta_module
from . import clear_speech as clear_speech_module
from . import diarize_speakers as diarize_speakers_module
from . import identify_speakers as identify_speakers_module
from . import lang_detect as lang_detect_module
from . import proofread as proofread_module
from . import whisper_asr as whisper_asr_module
from ._dump import StageDumper
from ._memory import free_mlx
from . import speech_structure as speech_structure_module
from . import speech_summary as speech_summary_module
from . import transcode as transcode_module
from ._progress import ProgressReporter
from .llm import DEFAULT_MODEL as LLM_DEFAULT_MODEL, LLMError, MlxLLM, _resolve_model_path
from .merge import merge
from .render import render_markdown
from .types import Segment


@dataclass
class PipelineOptions:
    audio_path: str
    output_path: str | None = None
    language: str | None = None    # None → detect on the longest pyannote turn ([4] lang_detect)
    unknown_speaker: str = "ask"           # "ask" | "keep"
    names_override: list[str] | None = None
    datetime_override: datetime | None = None
    llm_model: str | None = None
    llm_proofread_model: str | None = None
    llm_identify_model: str | None = None
    llm_structure_model: str | None = None
    llm_tldr_model: str | None = None
    # Defaults match the Qwen2.5-Instruct family's official recommendation
    # (the project default since v0.22.0 — see ADR 0020). Pass `--llm-...`
    # CLI flags to override per-run when using a different model.
    llm_temperature: float | None = 0.7
    llm_top_p: float | None = 0.8
    llm_top_k: int | None = 20
    llm_repetition_penalty: float | None = 1.05
    run_proofread: bool = True
    run_tldr: bool = True
    run_structure: bool = True
    clearspeech_chain: str = "autogain"
    clearspeech_autogain_target_dbfs: float = -20.0
    clearspeech_autogain_max_gain_db: float = 16.0
    clearspeech_bandpass_low_hz: float = 150.0
    clearspeech_bandpass_high_hz: float = 5_500.0
    clearspeech_presence_center_hz: float = 3_000.0
    clearspeech_presence_boost_db: float = 6.0
    clearspeech_presence_q: float = 1.0
    clearspeech_denoise_noise_floor_db: float = -25.0
    clearspeech_denoise_reduction_db: float = 12.0
    clearspeech_dereverb_rt60_floor_ms: float = 300.0
    clearspeech_dereverb_subtract_factor: float = 1.0
    clearspeech_dereverb_crossfade_ms: float = 50.0
    dump_stages_dir: str | None = None
    verbose: bool = False


def _log(verbose: bool) -> Callable[[str], None]:
    def emit(msg: str) -> None:
        if verbose:
            print(msg, file=sys.stderr, flush=True)
    return emit


def _default_output_path(audio_path: str) -> str:
    p = Path(audio_path)
    return str(p.with_suffix(".md"))


def _short_model_name(spec: str) -> str:
    """Return a human-readable short name for a model spec.

    For HF repo ids (`org/name`) keep the bare `name`. For filesystem
    paths take the directory basename. Used in the rendered Markdown
    header to keep the toolchain line short.
    """
    if "/" in spec and not spec.startswith((".", "/", "~")):
        return spec.split("/", 1)[1]
    return Path(os.path.expanduser(spec)).name


def _llm_summary(options: PipelineOptions) -> str:
    """One-line description of which model ran on which LLM stage.

    Collapses to a single bare model name when all enabled LLM stages use
    the same spec; otherwise lists `stage=name` chips per stage that ran.
    Disabled stages are omitted.
    """
    enabled: list[tuple[str, str]] = []
    if options.run_proofread:
        enabled.append(("proofread", _resolve_stage_model(options, "proofread")))
    needs_identify = (
        options.unknown_speaker == "ask" and options.names_override is None
    )
    if needs_identify:
        enabled.append(("identify_speakers", _resolve_stage_model(options, "identify_speakers")))
    if options.run_structure:
        enabled.append(("speech_structure", _resolve_stage_model(options, "speech_structure")))
    if options.run_tldr:
        enabled.append(("speech_summary", _resolve_stage_model(options, "speech_summary")))

    if not enabled:
        return ""
    unique_specs = {spec for _, spec in enabled}
    if len(unique_specs) == 1:
        return _short_model_name(next(iter(unique_specs)))
    return " · ".join(f"{stage}={_short_model_name(spec)}" for stage, spec in enabled)


def _resolve_model_path_safe(spec: str) -> str | None:
    """`_resolve_model_path` but swallows LLMError into None.

    Used by `_ensure_llm` to compare a desired model spec against the
    resident model's resolved path. A miss-resolved spec (cache miss) is
    not an error here — the actual load call will raise the user-visible
    error when the time comes.
    """
    try:
        return _resolve_model_path(spec)
    except LLMError:
        return None


def _resolve_stage_model(options: PipelineOptions, stage: str) -> str:
    """Return the model path for `stage`, resolving fallbacks.

    Stage values: "proofread", "identify", "structure", "tldr".

    Resolution order: per-stage override → `--llm-model` global override →
    `MlxLLM.DEFAULT_MODEL`. Resolving to a concrete path here (rather than
    leaving it as None) is load-bearing: `_ensure_llm` uses `model_path`
    equality to decide whether to reuse the resident model, so each stage
    must see the same concrete path or it will accidentally reuse the
    previous stage's override.
    """
    per_stage = {
        "proofread": options.llm_proofread_model,
        "identify_speakers": options.llm_identify_model,
        "speech_structure": options.llm_structure_model,
        "speech_summary": options.llm_tldr_model,
    }
    return per_stage[stage] or options.llm_model or LLM_DEFAULT_MODEL


def _ensure_llm(
    current: MlxLLM | None,
    want_path: str,
    *,
    log: Callable[[str], None],
    log_memory: bool,
    progress: ProgressReporter,
    sampling_overrides: dict | None = None,
) -> MlxLLM:
    """Return an MlxLLM ready for use at `want_path`, reusing `current` when
    the resident model already matches; otherwise unload the old one and
    load fresh.
    """
    if current is not None:
        # `model_path` is the user-supplied spec; `_resolved_path` is the
        # concrete on-disk directory after HF cache lookup. Compare both —
        # a repo id and its resolved snapshot path point to the same model.
        if (
            current.model_path == want_path
            or current._resolved_path == want_path
            or (
                current._resolved_path is not None
                and current._resolved_path == _resolve_model_path_safe(want_path)
            )
        ):
            return current
        with progress.spinner(f"Unloading LLM ({Path(current.model_path).name})"):
            current.close()
        free_mlx(log)

    new_llm = MlxLLM(
        model_path=want_path,
        log=log,
        log_memory=log_memory,
        sampling_overrides=sampling_overrides or {},
    )
    with progress.spinner(f"Loading LLM ({Path(new_llm.model_path).name})"):
        new_llm.load()
    return new_llm


def run(options: PipelineOptions) -> str:
    """Run the full pipeline. Returns the path to the produced Markdown file."""
    log = _log(options.verbose)
    audio = Path(options.audio_path)
    if not audio.is_file():
        raise FileNotFoundError(f"Audio not found: {audio}")

    out_path = Path(options.output_path or _default_output_path(str(audio)))

    needs_identify_llm = (
        options.unknown_speaker == "ask" and options.names_override is None
    )

    sampling_overrides: dict = {}
    if options.llm_temperature is not None:
        sampling_overrides["temperature"] = options.llm_temperature
    if options.llm_top_p is not None:
        sampling_overrides["top_p"] = options.llm_top_p
    if options.llm_top_k is not None:
        sampling_overrides["top_k"] = options.llm_top_k
    if options.llm_repetition_penalty is not None:
        sampling_overrides["repetition_penalty"] = options.llm_repetition_penalty

    tmpdir = Path(tempfile.mkdtemp(prefix="voice-pipeline-"))
    progress = ProgressReporter()
    # Stage timings used for the rendered Markdown header. Recorded in
    # insertion order so the rendered list mirrors execution order.
    stage_timings: dict[str, float] = {}

    @contextmanager
    def _timed(stage: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            stage_timings[stage] = stage_timings.get(stage, 0.0) + (
                time.perf_counter() - t0
            )

    pipeline_t0 = time.perf_counter()
    dumper = StageDumper(
        Path(options.dump_stages_dir).expanduser() if options.dump_stages_dir else None
    )
    if dumper.enabled():
        log(f"      Dumping stage artefacts to {dumper.target}")
    try:
        with progress:
            wav_path = tmpdir / "audio.wav"
            with progress.spinner("[1/11] transcode → WAV 16 kHz mono"):
                transcode_module.transcode(audio, wav_path, log)

            with progress.spinner("[2/11] audio_meta"):
                audio_meta = audio_meta_module.extract_metadata(
                    audio, override_started_at=options.datetime_override
                )
            log(f"      Початок: {audio_meta.started_at} ({audio_meta.source})")
            log(f"      Тривалість: {audio_meta.duration_s:.1f}s")
            dumper.write("01-meta.json", audio_meta)

            # Diarize first on the raw WAV — its per-turn boundaries are the
            # guard-rails the next stage needs for per-segment autogain. ASR then
            # runs on the normalised WAV. Each module loads its own model and
            # frees it on return; the LLM is loaded after both, so the three
            # are never co-resident.
            with progress.spinner("[3/11] Діаризація (pyannote 3.1)"), _timed("diarize_speakers"):
                turns = diarize_speakers_module.diarize(wav_path, log=log)
            log(f"      {len({t.speaker for t in turns})} мовців")
            dumper.write("02-diarize_speakers.json", turns)

            # [4] lang_detect — runs only when --language was not given.
            # Whisper-large-v3's first-30 s auto-detect is unreliable on quiet
            # or short opening segments (see ADR 0022); detecting on the
            # longest pyannote turn instead gives a much stronger signal.
            if options.language is None:
                with progress.spinner("[4/11] Визначення мови"), _timed("lang_detect"):
                    effective_language = lang_detect_module.detect_language_on_longest_turn(
                        wav_path, turns, log=log,
                    )
                origin = "auto-detect"
            else:
                effective_language = options.language
                origin = "cli override"
            log(f"      Мова: {effective_language} ({origin})")

            chain = tuple(
                part.strip() for part in options.clearspeech_chain.split(",")
                if part.strip()
            )
            chain_label = " → ".join(chain) if chain else "no-op"

            def _dump_step(idx: int, effect: str, path: Path) -> None:
                dumper.write_binary(f"02b-clear_speech-{idx}-{effect}.wav", path)

            with progress.spinner(f"[5/11] clear_speech ({chain_label})"):
                processed_wav_path, clearspeech_config = clear_speech_module.clearspeech(
                    wav_path,
                    chain=chain,
                    autogain_turns=turns,
                    autogain_target_dbfs=options.clearspeech_autogain_target_dbfs,
                    autogain_max_gain_db=options.clearspeech_autogain_max_gain_db,
                    bandpass_low_hz=options.clearspeech_bandpass_low_hz,
                    bandpass_high_hz=options.clearspeech_bandpass_high_hz,
                    presence_center_hz=options.clearspeech_presence_center_hz,
                    presence_boost_db=options.clearspeech_presence_boost_db,
                    presence_q=options.clearspeech_presence_q,
                    denoise_noise_floor_db=options.clearspeech_denoise_noise_floor_db,
                    denoise_reduction_db=options.clearspeech_denoise_reduction_db,
                    dereverb_turns=turns,
                    dereverb_rt60_floor_ms=options.clearspeech_dereverb_rt60_floor_ms,
                    dereverb_subtract_factor=options.clearspeech_dereverb_subtract_factor,
                    dereverb_crossfade_ms=options.clearspeech_dereverb_crossfade_ms,
                    log=log,
                    dump=_dump_step if dumper.enabled() else None,
                )
            dumper.write("02b-clear_speech-config.json", clearspeech_config)

            engine_label = "Whisper-large-v3-MLX"
            with progress.spinner(f"[6/11] speech2text ({engine_label})"), _timed("asr"):
                asr_segments = whisper_asr_module.transcribe(
                    processed_wav_path,
                    language=effective_language,
                    log=log,
                )
            log(f"      {len(asr_segments)} ASR-сегментів")
            dumper.write("03-asr.json", asr_segments)

            with progress.spinner("[7/11] Merge"):
                segments: list[Segment] = merge(asr_segments, turns)
            dumper.write("04-merge.json", segments)

            # LLM stages may use different models per stage (--llm-{stage}-model).
            # `_ensure_llm` reuses the resident model when the next stage wants
            # the same path, otherwise unloads + cold-reloads. close() drops the
            # model and clears MLX cache so render does not contend with weights.
            llm: MlxLLM | None = None
            try:
                if options.run_proofread:
                    llm = _ensure_llm(
                        llm, _resolve_stage_model(options, "proofread"),
                        log=log, log_memory=options.verbose, progress=progress,
                        sampling_overrides=sampling_overrides,
                    )
                    with _timed("proofread"):
                        segments = proofread_module.fix_asr_errors(
                            segments,
                            llm=llm,
                            language=effective_language,
                            log=log,
                            progress=progress,
                        )
                    dumper.write("05-proofread.json", segments)
                    free_mlx(log)
                else:
                    log(f"[8/11] Proofread пропущено")

                if needs_identify_llm:
                    llm = _ensure_llm(
                        llm, _resolve_stage_model(options, "identify_speakers"),
                        log=log, log_memory=options.verbose, progress=progress,
                        sampling_overrides=sampling_overrides,
                    )
                    identify_llm = llm
                else:
                    identify_llm = None
                with _timed("identify_speakers"):
                    name_map = identify_speakers_module.identify_speakers(
                        segments,
                        language=effective_language,
                        llm=identify_llm,
                        unknown_policy=options.unknown_speaker,  # type: ignore[arg-type]
                        names_override=options.names_override,
                        log=log,
                        progress=progress,
                    )
                for seg in segments:
                    if seg.speaker in name_map:
                        seg.name = name_map[seg.speaker]
                log(f"      {len(name_map)} мовців іменовано: {name_map}")
                dumper.write("06-identify_speakers.json", name_map)
                dumper.write("07-segments-named.json", segments)
                if identify_llm is not None:
                    free_mlx(log)

                if options.run_structure:
                    llm = _ensure_llm(
                        llm, _resolve_stage_model(options, "speech_structure"),
                        log=log, log_memory=options.verbose, progress=progress,
                        sampling_overrides=sampling_overrides,
                    )
                    with _timed("speech_structure"):
                        dialog = speech_structure_module.structure_dialog(
                            segments, llm=llm, language=effective_language,
                            log=log, progress=progress,
                        )
                    free_mlx(log)
                else:
                    log(f"[10/11] Структурування пропущене")
                    dialog = speech_structure_module.structure_dialog(
                        segments, llm=None, language=effective_language, log=log,
                    )
                dumper.write("08-speech_structure.json", dialog)

                if options.run_tldr:
                    llm = _ensure_llm(
                        llm, _resolve_stage_model(options, "speech_summary"),
                        log=log, log_memory=options.verbose, progress=progress,
                        sampling_overrides=sampling_overrides,
                    )
                    with _timed("speech_summary"):
                        tldr_text = speech_summary_module.generate_tldr(
                            segments, llm=llm, language=effective_language,
                            log=log, progress=progress,
                        )
                    free_mlx(log)
                else:
                    tldr_text = ""
                    log(f"[11/11] TL;DR пропущено")
                dumper.write("09-speech_summary.txt", tldr_text)
            finally:
                if llm is not None:
                    with progress.spinner("Unloading LLM"):
                        llm.close()

        models = {
            "diarize": "pyannote/speaker-diarization-3.1",
            "asr": engine_label,
            "llm": _llm_summary(options),
        }
        total_elapsed = time.perf_counter() - pipeline_t0
        timings = {"total": total_elapsed, **stage_timings}
        markdown = render_markdown(
            audio_meta=audio_meta,
            dialog=dialog,
            tldr=tldr_text,
            language=effective_language,
            models=models,
            timings=timings,
        )
        out_path.write_text(markdown, encoding="utf-8")
        log(f"      → {out_path}")
        return str(out_path)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
