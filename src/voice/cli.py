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

    loud = t.add_argument_group("loudness normalization")
    loud.add_argument(
        "--no-loudness-normalize", action="store_true",
        help="Skip per-segment AGC; ASR runs on the raw WAV (pre-0.12 behaviour).",
    )
    loud.add_argument(
        "--loudness-target-dbfs", type=float, default=-20.0, metavar="DBFS",
        help="Target RMS for per-turn AGC (default: -20.0).",
    )
    loud.add_argument(
        "--loudness-max-gain-db", type=float, default=16.0, metavar="DB",
        help="Max boost per turn; quieter turns clipped at this gain (default: 16.0).",
    )

    t.add_argument(
        "--dump-stages", dest="dump_stages_dir", default=None, metavar="DIR",
        help=(
            "Write each stage's output to DIR for troubleshooting "
            "(01-meta.json, 02-diarize.json, 02b-normalized.wav, 03-asr.json, "
            "04-merge.json, 05-proofread.json, 06-identify.json, "
            "07-segments-named.json, 08-structure.json, 09-tldr.txt). "
            "Disabled when omitted."
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
        run_loudness_normalize=not args.no_loudness_normalize,
        loudness_target_dbfs=args.loudness_target_dbfs,
        loudness_max_gain_db=args.loudness_max_gain_db,
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
