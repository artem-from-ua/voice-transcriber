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
from . import diarize as diarize_module
from . import ffprobe as ffprobe_module
from . import identify as identify_module
from . import postprocess as postprocess_module
from . import structure as structure_module
from . import tldr as tldr_module
from .llm import LLMClient
from .merge import merge
from .render import render_markdown
from .types import Segment


@dataclass
class PipelineOptions:
    audio_path: str
    output_path: str | None = None
    language: str = "uk"
    asr_bits: int = 6
    unknown_speaker: str = "ask"           # "ask" | "keep"
    names_override: list[str] | None = None
    datetime_override: datetime | None = None
    llm_model: str | None = None
    llm_base_url: str | None = None
    run_postprocess: bool = True
    run_tldr: bool = True
    run_structure: bool = True
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
    log(f"[1/9] Конвертування → WAV 16 kHz mono")
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
        llm_kwargs["model"] = options.llm_model
    if options.llm_base_url:
        llm_kwargs["base_url"] = options.llm_base_url

    llm_required = options.run_postprocess or options.run_tldr or options.run_structure or (
        options.unknown_speaker == "ask" or options.names_override is None
    )

    tmpdir = Path(tempfile.mkdtemp(prefix="voice-pipeline-"))
    try:
        wav_path = tmpdir / "audio.wav"
        _to_wav_16k_mono(audio, wav_path, log)

        log(f"[2/9] Метадані")
        audio_meta = ffprobe_module.extract_metadata(
            audio, override_started_at=options.datetime_override
        )
        log(f"      Початок: {audio_meta.started_at} ({audio_meta.source})")
        log(f"      Тривалість: {audio_meta.duration_s:.1f}s")

        log(f"[3/9] ASR (VibeVoice-{options.asr_bits}bit)")
        asr_segments = asr_module.transcribe(
            wav_path,
            bitness=options.asr_bits,
            language=options.language,
            context=f"Розмова мовою {options.language}.",
            log=log,
        )
        log(f"      {len(asr_segments)} ASR-сегментів")

        log(f"[4/9] Діаризація (pyannote 3.1)")
        turns = diarize_module.diarize(wav_path, log=log)
        log(f"      {len({t.speaker for t in turns})} мовців")

        log(f"[5/9] Merge")
        segments: list[Segment] = merge(asr_segments, turns)

        if llm_required:
            log(f"[~] Підключення до LM Studio")
            llm = LLMClient(**llm_kwargs)
            llm.health_check()
        else:
            llm = None  # type: ignore[assignment]

        try:
            if options.run_postprocess and llm is not None:
                log(f"[6/9] ASR-постобробка")
                segments = postprocess_module.fix_asr_errors(
                    segments, llm=llm, language=options.language, log=log,
                )
            else:
                log(f"[6/9] ASR-постобробка пропущена")

            log(f"[7/9] Ідентифікація мовців")
            name_map = identify_module.identify_speakers(
                segments,
                language=options.language,
                llm=llm,
                unknown_policy=options.unknown_speaker,  # type: ignore[arg-type]
                names_override=options.names_override,
                log=log,
            )
            for seg in segments:
                if seg.speaker in name_map:
                    seg.name = name_map[seg.speaker]
            log(f"      {len(name_map)} мовців іменовано: {name_map}")

            if options.run_structure and llm is not None:
                log(f"[8/9] Структурування на секції")
                dialog = structure_module.structure_dialog(
                    segments, llm=llm, language=options.language, log=log,
                )
            else:
                log(f"[8/9] Структурування пропущене")
                dialog = structure_module.structure_dialog(
                    segments, llm=None, language=options.language, log=log,
                )

            if options.run_tldr and llm is not None:
                log(f"[9/9] TL;DR")
                tldr_text = tldr_module.generate_tldr(
                    segments, llm=llm, language=options.language, log=log,
                )
            else:
                tldr_text = ""
                log(f"[9/9] TL;DR пропущено")
        finally:
            if llm is not None:
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
