"""End-to-end orchestration: audio file → Markdown transcript.

Stages run sequentially; intermediate results live in memory. The only
on-disk artefact apart from the input audio is the final Markdown file.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from . import asr as asr_module
from . import clearspeech as clearspeech_module
from . import diarize as diarize_module
from . import ffprobe as ffprobe_module
from . import identify as identify_module
from . import proofread as proofread_module
from ._dump import StageDumper
from . import structure as structure_module
from . import tldr as tldr_module
from ._progress import ProgressReporter
from .llm import MlxLLM
from .merge import merge
from .render import render_markdown
from .types import Segment


@dataclass
class PipelineOptions:
    audio_path: str
    output_path: str | None = None
    language: str = "uk"
    asr_bits: int = 6
    asr_chunk_duration: float | None = None
    asr_temperature: float | None = None
    unknown_speaker: str = "ask"           # "ask" | "keep"
    names_override: list[str] | None = None
    datetime_override: datetime | None = None
    llm_model: str | None = None
    run_proofread: bool = True
    run_tldr: bool = True
    run_structure: bool = True
    clearspeech_agc: bool = True
    clearspeech_agc_target_dbfs: float = -20.0
    clearspeech_agc_max_gain_db: float = 16.0
    clearspeech_bandpass: bool = False
    clearspeech_bandpass_low_hz: float = 150.0
    clearspeech_bandpass_high_hz: float = 5_500.0
    dump_stages_dir: str | None = None
    verbose: bool = False


def _log(verbose: bool) -> Callable[[str], None]:
    def emit(msg: str) -> None:
        if verbose:
            print(msg, file=sys.stderr, flush=True)
    return emit


def _to_wav_16k_mono(src: Path, dst: Path, log: Callable[[str], None]) -> None:
    """Convert any audio to 16 kHz mono PCM WAV via ffmpeg."""
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-i", str(src),
        "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(dst),
    ]
    log(f"[1/10] audiotranscode → WAV 16 kHz mono")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg not found in PATH. Install ffmpeg.") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg failed: {exc.stderr.strip()}") from exc


def _default_output_path(audio_path: str) -> str:
    p = Path(audio_path)
    return str(p.with_suffix(".md"))


def run(options: PipelineOptions) -> str:
    """Run the full pipeline. Returns the path to the produced Markdown file."""
    log = _log(options.verbose)
    audio = Path(options.audio_path)
    if not audio.is_file():
        raise FileNotFoundError(f"Audio not found: {audio}")

    out_path = Path(options.output_path or _default_output_path(str(audio)))

    llm_kwargs: dict = {}
    if options.llm_model:
        llm_kwargs["model_path"] = options.llm_model

    llm_required = options.run_proofread or options.run_tldr or options.run_structure or (
        options.unknown_speaker == "ask" or options.names_override is None
    )

    tmpdir = Path(tempfile.mkdtemp(prefix="voice-pipeline-"))
    progress = ProgressReporter()
    dumper = StageDumper(
        Path(options.dump_stages_dir).expanduser() if options.dump_stages_dir else None
    )
    if dumper.enabled():
        log(f"      Dumping stage artefacts to {dumper.target}")
    try:
        with progress:
            wav_path = tmpdir / "audio.wav"
            with progress.spinner("[1/10] audiotranscode → WAV 16 kHz mono"):
                _to_wav_16k_mono(audio, wav_path, log)

            with progress.spinner("[2/10] audiometa"):
                audio_meta = ffprobe_module.extract_metadata(
                    audio, override_started_at=options.datetime_override
                )
            log(f"      Початок: {audio_meta.started_at} ({audio_meta.source})")
            log(f"      Тривалість: {audio_meta.duration_s:.1f}s")
            dumper.write("01-meta.json", audio_meta)

            # Diarize first on the raw WAV — its per-turn boundaries are the
            # guard-rails the next stage needs for per-segment AGC. ASR then
            # runs on the normalised WAV. Each module loads its own model and
            # frees it on return; the LLM is loaded after both, so the three
            # are never co-resident.
            with progress.spinner("[3/10] Діаризація (pyannote 3.1)"):
                turns = diarize_module.diarize(wav_path, log=log)
            log(f"      {len({t.speaker for t in turns})} мовців")
            dumper.write("02-diarize.json", turns)

            chain = tuple(
                name for name, on in (
                    ("agc", options.clearspeech_agc),
                    ("bandpass", options.clearspeech_bandpass),
                ) if on
            )
            chain_label = " → ".join(chain) if chain else "no-op"

            def _dump_step(idx: int, effect: str, path: Path) -> None:
                dumper.write_binary(f"02b-clearspeech-{idx}-{effect}.wav", path)

            with progress.spinner(f"[4/10] Clearspeech ({chain_label})"):
                processed_wav_path, clearspeech_config = clearspeech_module.clearspeech(
                    wav_path,
                    chain=chain,
                    agc_turns=turns,
                    agc_target_dbfs=options.clearspeech_agc_target_dbfs,
                    agc_max_gain_db=options.clearspeech_agc_max_gain_db,
                    bandpass_low_hz=options.clearspeech_bandpass_low_hz,
                    bandpass_high_hz=options.clearspeech_bandpass_high_hz,
                    log=log,
                    dump=_dump_step if dumper.enabled() else None,
                )
            dumper.write("02b-clearspeech-config.json", clearspeech_config)

            with progress.spinner(
                f"[5/10] ASR (VibeVoice-{options.asr_bits}bit)"
            ):
                asr_segments = asr_module.transcribe(
                    processed_wav_path,
                    bitness=options.asr_bits,
                    language=options.language,
                    context=f"Розмова мовою {options.language}.",
                    chunk_duration=options.asr_chunk_duration,
                    temperature=options.asr_temperature,
                    log=log,
                )
            log(f"      {len(asr_segments)} ASR-сегментів")
            dumper.write("03-asr.json", asr_segments)

            with progress.spinner("[6/10] Merge"):
                segments: list[Segment] = merge(asr_segments, turns)
            dumper.write("04-merge.json", segments)

            # One LLM, four stages, in-process. close() drops the model and
            # clears MLX cache so the render step does not contend with weights.
            llm = MlxLLM(log=log, **llm_kwargs) if llm_required else None
            try:
                if llm is not None:
                    with progress.spinner(
                        f"Loading LLM ({Path(llm.model_path).name})"
                    ):
                        llm.load()

                if options.run_proofread and llm is not None:
                    segments = proofread_module.fix_asr_errors(
                        segments,
                        llm=llm,
                        language=options.language,
                        log=log,
                        progress=progress,
                    )
                    dumper.write("05-proofread.json", segments)
                else:
                    log(f"[7/10] Proofread пропущено")

                name_map = identify_module.identify_speakers(
                    segments,
                    language=options.language,
                    llm=llm,
                    unknown_policy=options.unknown_speaker,  # type: ignore[arg-type]
                    names_override=options.names_override,
                    log=log,
                    progress=progress,
                )
                for seg in segments:
                    if seg.speaker in name_map:
                        seg.name = name_map[seg.speaker]
                log(f"      {len(name_map)} мовців іменовано: {name_map}")
                dumper.write("06-identify.json", name_map)
                dumper.write("07-segments-named.json", segments)

                if options.run_structure and llm is not None:
                    dialog = structure_module.structure_dialog(
                        segments, llm=llm, language=options.language,
                        log=log, progress=progress,
                    )
                else:
                    log(f"[9/10] Структурування пропущене")
                    dialog = structure_module.structure_dialog(
                        segments, llm=None, language=options.language, log=log,
                    )
                dumper.write("08-structure.json", dialog)

                if options.run_tldr and llm is not None:
                    tldr_text = tldr_module.generate_tldr(
                        segments, llm=llm, language=options.language,
                        log=log, progress=progress,
                    )
                else:
                    tldr_text = ""
                    log(f"[10/10] TL;DR пропущено")
                dumper.write("09-tldr.txt", tldr_text)
            finally:
                if llm is not None:
                    with progress.spinner("Unloading LLM"):
                        llm.close()

        markdown = render_markdown(
            audio_meta=audio_meta,
            dialog=dialog,
            tldr=tldr_text,
            language=options.language,
            asr_label=f"VibeVoice-ASR-{options.asr_bits}bit",
        )
        out_path.write_text(markdown, encoding="utf-8")
        log(f"      → {out_path}")
        return str(out_path)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
