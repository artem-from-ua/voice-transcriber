#!/usr/bin/env python3
"""Run ASR-only on a pre-existing clear_speech WAV and write 03-asr.json.

Companion to scripts/asr-chunk-boundary-quality.py. The full
`voice transcribe` pipeline includes diarize + lang_detect + clear_speech
+ four LLM stages, which add ~5 min on the 48-min reference recording
on top of the ~4 min ASR pass. When iterating on `whisper_asr` itself
(e.g. dedup variants for #172, snap variants for #159), the dump dir
from a prior strict-time run already contains the upstream artefacts
that are byte-identical regardless of dedup choice. This script reuses
that dump's `02b-clear_speech-*-autogain.wav` and runs only the ASR
stage, producing a fresh `03-asr.json` that
`scripts/asr-chunk-boundary-quality.py extract` can immediately consume.

Usage:

    uv run scripts/asr-only-bench.py \\
        --dump-dir /tmp/asr-text-dedup \\
        --language en

  -> writes /tmp/asr-text-dedup/03-asr.json

The script does not modify production code or run any LLM. It loads
Whisper-large-v3-MLX once, calls `whisper_asr.transcribe`, and dumps
the segments. Wall-clock on the 48-min reference: ~4 min.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path


def _log(msg: str) -> None:
    print(f"[bench] {msg}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dump-dir",
        required=True,
        help=(
            "Directory containing 02b-clear_speech-*-autogain.wav (output "
            "of the clear_speech stage from a prior `voice transcribe` run)."
        ),
    )
    parser.add_argument(
        "--language",
        default=None,
        help=(
            "ISO language code (e.g. 'en', 'uk'). Defaults to None, which "
            "lets mlx-whisper auto-detect on the first 30 s window. Set "
            "this to match the original `voice transcribe --language` or "
            "the value the lang_detect stage produced in the dump's logs."
        ),
    )
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Output path for the segments JSON (default: "
            "<dump-dir>/03-asr.json, matching pipeline.py's convention)."
        ),
    )
    args = parser.parse_args(argv)

    dump_dir = Path(args.dump_dir).expanduser().resolve()
    if not dump_dir.is_dir():
        print(f"[error] not a directory: {dump_dir}", file=sys.stderr)
        return 1

    wav_candidates = sorted(dump_dir.glob("02b-clear_speech-*-autogain.wav"))
    if not wav_candidates:
        print(
            f"[error] no 02b-clear_speech-*-autogain.wav under {dump_dir}. "
            f"Run `voice transcribe --dump-stages {dump_dir}` first.",
            file=sys.stderr,
        )
        return 1
    wav_path = wav_candidates[-1]
    _log(f"input WAV: {wav_path}")

    out_path = (
        Path(args.out).expanduser().resolve()
        if args.out
        else dump_dir / "03-asr.json"
    )

    from voice import whisper_asr

    model, load_elapsed = whisper_asr.load_model(log=_log)
    _log(f"model loaded in {load_elapsed:.1f}s")

    t0 = time.perf_counter()
    segments = whisper_asr.transcribe(
        wav_path,
        language=args.language,
        model=model,
        log=_log,
    )
    elapsed = time.perf_counter() - t0
    _log(f"got {len(segments)} segments in {elapsed:.1f}s")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps([asdict(s) for s in segments], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _log(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
