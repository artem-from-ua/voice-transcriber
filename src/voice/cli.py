"""`voice transcribe <audio>` entry-point."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime

from .pipeline import PipelineOptions, run


def _parse_datetime(s: str) -> datetime:
    try:
        return datetime.fromisoformat(s)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid ISO datetime: {s!r}") from exc


def _parse_names(s: str) -> list[str]:
    return [part.strip() for part in s.split(",") if part.strip()]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="voice",
        description="Local audio → diarized Markdown transcript pipeline.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("transcribe", help="Transcribe an audio file end-to-end.")
    t.add_argument("audio", help="Path to the input audio file.")
    t.add_argument("--language", default="uk", help="Conversation language (default: uk).")
    t.add_argument(
        "--asr-bits", type=int, choices=(4, 5, 6, 8), default=6,
        help="VibeVoice-ASR quantisation (default: 6).",
    )
    t.add_argument(
        "--asr-chunk-duration", type=float, default=None,
        metavar="SECONDS",
        help=(
            "ASR chunk size in seconds (default: 45). VibeVoice supports up to "
            "60-minute single-pass inputs and benefits from long context; "
            "shorter chunks tend to drop linguistic continuity."
        ),
    )
    t.add_argument(
        "--asr-temperature", type=float, default=None,
        metavar="T",
        help=(
            "Sampling temperature for ASR (default: 0.0 = greedy, deterministic). "
            "Increase if you hit repetition loops the penalty alone can't break."
        ),
    )
    t.add_argument(
        "--unknown-speaker", choices=("ask", "keep"), default="ask",
        help="When self-intro is missing: ask interactively or keep SPEAKER_XX.",
    )
    t.add_argument(
        "--names", type=_parse_names, default=None,
        metavar='"NAME1,NAME2"',
        help="Override automatic naming: comma-separated names in time order.",
    )
    t.add_argument(
        "--datetime", dest="datetime_override", type=_parse_datetime, default=None,
        help="Override recording start time (ISO 8601, e.g. 2026-05-10T15:44:02).",
    )
    t.add_argument(
        "--llm-model", default=None,
        help=(
            "Path to a local MLX model directory (default: "
            "~/.cache/lm-studio/models/mlx-community/gemma-3-12b-it-qat-4bit)."
        ),
    )
    t.add_argument(
        "--output", "-o", default=None,
        help="Output Markdown path (default: <audio_basename>.md alongside the audio).",
    )
    t.add_argument("--no-proofread", action="store_true", help="Skip per-segment ASR proof-reading.")
    t.add_argument("--no-tldr", action="store_true", help="Skip TL;DR generation.")
    t.add_argument("--no-structure", action="store_true", help="Skip section structuring.")

    cs = t.add_argument_group("clearspeech (audio cleanup for ASR)")
    cs.add_argument(
        "--clearspeech-chain", type=str, default="autogain", metavar="EFFECTS",
        help=(
            "Comma-separated effect names in execution order. Known effects: "
            "autogain, bandpass, presence, denoise, dereverb. Default: 'autogain'. "
            "Use '' to disable preprocessing entirely. WARNING: enabling "
            "'denoise' degrades ASR on Ukrainian content even at its "
            "empirically best position (after autogain); see ADR 0014 for "
            "Metric A and the OpenAI Whisper-rule check. WARNING: using "
            "'presence' without 'bandpass' first catastrophically breaks ASR "
            "(see ADR 0013); enable bandpass whenever you enable presence. "
            "'dereverb' is per-turn (uses pyannote turns) — see ADR 0015 for "
            "Metric A."
        ),
    )
    cs.add_argument(
        "--clearspeech-autogain-target-dbfs", type=float, default=-20.0, metavar="DBFS",
        help="Target RMS for per-turn autogain (default: -20.0).",
    )
    cs.add_argument(
        "--clearspeech-autogain-max-gain-db", type=float, default=16.0, metavar="DB",
        help="Max boost per turn; quieter turns clipped at this gain (default: 16.0).",
    )
    cs.add_argument(
        "--clearspeech-bandpass-low-hz", type=float, default=150.0, metavar="HZ",
        help=(
            "Bandpass low cutoff in Hz (default: 150 — tuned by listening "
            "test, see ADR 0011)."
        ),
    )
    cs.add_argument(
        "--clearspeech-bandpass-high-hz", type=float, default=5_500.0, metavar="HZ",
        help=(
            "Bandpass high cutoff in Hz (default: 5500 — tuned by listening "
            "test, see ADR 0011)."
        ),
    )
    cs.add_argument(
        "--clearspeech-presence-center-hz", type=float, default=3_000.0, metavar="HZ",
        help=(
            "Presence boost center frequency in Hz (default: 3000, starting "
            "anchor — see ADR 0013 for listening-loop results)."
        ),
    )
    cs.add_argument(
        "--clearspeech-presence-boost-db", type=float, default=6.0, metavar="DB",
        help=(
            "Presence boost gain in dB (default: +6.0 — tuned by listening "
            "test, see ADR 0013)."
        ),
    )
    cs.add_argument(
        "--clearspeech-presence-q", type=float, default=1.0, metavar="Q",
        help="Presence boost Q / sharpness; higher = narrower peak (default: 1.0).",
    )
    cs.add_argument(
        "--clearspeech-denoise-noise-floor-db", type=float, default=-25.0, metavar="DB",
        help=(
            "ffmpeg afftdn noise-floor estimate in dB (default: -25, tuned by "
            "listening test, see ADR 0014)."
        ),
    )
    cs.add_argument(
        "--clearspeech-denoise-reduction-db", type=float, default=12.0, metavar="DB",
        help=(
            "ffmpeg afftdn subtraction amount in dB (default: 12)."
        ),
    )
    cs.add_argument(
        "--clearspeech-dereverb-rt60-floor-ms", type=float, default=300.0, metavar="MS",
        help=(
            "Per-turn RT60 estimate is clamped above this floor in ms (default: "
            "300). Lower lets the estimator pick shorter RT60s on already-dry "
            "segments; higher forces stronger dereverb everywhere."
        ),
    )
    cs.add_argument(
        "--clearspeech-dereverb-subtract-factor", type=float, default=1.0, metavar="ALPHA",
        help=(
            "How aggressively to subtract predicted late-reverb power "
            "(0.0 = no-op, 1.0 = full Lebart-Polack). Default: 1.0."
        ),
    )
    cs.add_argument(
        "--clearspeech-dereverb-crossfade-ms", type=float, default=50.0, metavar="MS",
        help="Crossfade duration at per-turn boundaries (default: 50).",
    )

    t.add_argument(
        "--dump-stages", dest="dump_stages_dir", default=None, metavar="DIR",
        help=(
            "Write each stage's output to DIR for troubleshooting "
            "(01-meta.json, 02-diarize.json, 02b-clearspeech-config.json, "
            "02b-clearspeech-N-<effect>.wav per applied effect — autogain, "
            "bandpass, presence — 03-asr.json, 04-merge.json, "
            "05-proofread.json, 06-identify.json, 07-segments-named.json, "
            "08-structure.json, 09-tldr.txt). Disabled when omitted."
        ),
    )
    t.add_argument("-v", "--verbose", action="store_true", help="Verbose progress logs to stderr.")
    return p


def _opts_from_args(args: argparse.Namespace) -> PipelineOptions:
    return PipelineOptions(
        audio_path=args.audio,
        output_path=args.output,
        language=args.language,
        asr_bits=args.asr_bits,
        asr_chunk_duration=args.asr_chunk_duration,
        asr_temperature=args.asr_temperature,
        unknown_speaker=args.unknown_speaker,
        names_override=args.names,
        datetime_override=args.datetime_override,
        llm_model=args.llm_model,
        run_proofread=not args.no_proofread,
        run_tldr=not args.no_tldr,
        run_structure=not args.no_structure,
        clearspeech_chain=args.clearspeech_chain,
        clearspeech_autogain_target_dbfs=args.clearspeech_autogain_target_dbfs,
        clearspeech_autogain_max_gain_db=args.clearspeech_autogain_max_gain_db,
        clearspeech_bandpass_low_hz=args.clearspeech_bandpass_low_hz,
        clearspeech_bandpass_high_hz=args.clearspeech_bandpass_high_hz,
        clearspeech_presence_center_hz=args.clearspeech_presence_center_hz,
        clearspeech_presence_boost_db=args.clearspeech_presence_boost_db,
        clearspeech_presence_q=args.clearspeech_presence_q,
        clearspeech_denoise_noise_floor_db=args.clearspeech_denoise_noise_floor_db,
        clearspeech_denoise_reduction_db=args.clearspeech_denoise_reduction_db,
        clearspeech_dereverb_rt60_floor_ms=args.clearspeech_dereverb_rt60_floor_ms,
        clearspeech_dereverb_subtract_factor=args.clearspeech_dereverb_subtract_factor,
        clearspeech_dereverb_crossfade_ms=args.clearspeech_dereverb_crossfade_ms,
        dump_stages_dir=args.dump_stages_dir,
        verbose=args.verbose,
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.cmd == "transcribe":
        try:
            out = run(_opts_from_args(args))
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(out)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
