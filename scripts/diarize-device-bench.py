#!/usr/bin/env python3
"""Measure pyannote diarization wall-clock on CPU vs MPS, and detect
silent MPS-to-CPU op fallback. Answers issue #163.

Three configurations, run sequentially:

  A  cpu       — VOICE_DIARIZE_DEVICE=cpu
  B  mps       — default, no env (production behaviour)
  C  mps-trace — PYTORCH_ENABLE_MPS_FALLBACK=1, fallback warnings captured

Per run we record load wall-clock, diarize wall-clock, peak RSS,
peak `torch.mps.current_allocated_memory()`, and fallback-warning
counts grouped by op-name. A JSON result is written to
docs/measurements/163/<config>-<audio-stem>.json.

The B configuration also runs a gate check: peak MPS memory must be
> 0 (otherwise we're calling something "MPS" that ran on CPU).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import resource
import sys
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Any


CONFIGS = ("cpu", "mps", "mps-trace")
FALLBACK_RE = re.compile(r"The operator '([^']+)'.*MPS.*falling back", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wav", required=True, type=Path, help="Path to input .wav (16 kHz mono).")
    p.add_argument("--config", required=True, choices=CONFIGS, help="Device configuration to measure.")
    p.add_argument("--label", required=True,
                   help="Fictional short name used in result filename and JSON instead of the "
                        "actual wav path (real names/companies must not enter git).")
    p.add_argument("--out-dir", type=Path, default=Path("docs/measurements/163"),
                   help="Where to write the JSON result. Default: docs/measurements/163")
    p.add_argument("--num-speakers", type=int, default=None,
                   help="Hint pyannote with a known speaker count.")
    p.add_argument("--dry-run", action="store_true",
                   help="Validate arguments and exit without importing voice/torch.")
    return p.parse_args()


def apply_env(config: str) -> None:
    """Set environment before importing torch/voice."""
    if config == "cpu":
        os.environ["VOICE_DIARIZE_DEVICE"] = "cpu"
        os.environ.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
    elif config == "mps":
        os.environ["VOICE_DIARIZE_DEVICE"] = "mps"
        os.environ.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
    elif config == "mps-trace":
        os.environ["VOICE_DIARIZE_DEVICE"] = "mps"
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


def _probe_duration(wav: Path) -> float:
    """Read duration via ffprobe; rounded to 0.1s."""
    import subprocess  # noqa: PLC0415
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(wav)],
        text=True,
    ).strip()
    return round(float(out), 1)


def peak_rss_bytes() -> int:
    """macOS reports ru_maxrss in bytes (Linux: kilobytes). We're macOS-only here."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def mps_allocated_bytes(torch_mod: Any) -> int:
    try:
        return int(torch_mod.mps.current_allocated_memory())
    except (AttributeError, RuntimeError):
        return 0


def mps_driver_bytes(torch_mod: Any) -> int:
    """Total Metal driver pool — survives torch.mps.empty_cache(), so it stays
    non-zero even after diarize_speakers.diarize() drains the allocator."""
    try:
        return int(torch_mod.mps.driver_allocated_memory())
    except (AttributeError, RuntimeError):
        return 0


def install_mps_peak_probe(torch_mod: Any, peak: dict[str, int]) -> None:
    """Hook torch.mps.empty_cache so we capture allocated memory right
    before diarize() drains it (free_torch_mps is called from inside
    diarize()). Without this hook our post-call read always sees 0."""
    if not hasattr(torch_mod, "mps") or not hasattr(torch_mod.mps, "empty_cache"):
        return
    original = torch_mod.mps.empty_cache

    def _wrapped() -> None:
        try:
            now = int(torch_mod.mps.current_allocated_memory())
            if now > peak.get("current", 0):
                peak["current"] = now
        except Exception:  # noqa: BLE001
            pass
        original()

    torch_mod.mps.empty_cache = _wrapped


def run_measurement(args: argparse.Namespace) -> dict[str, Any]:
    import torch  # noqa: PLC0415 — must follow apply_env()
    from voice.diarize_speakers import diarize  # noqa: PLC0415

    mps_available = bool(torch.backends.mps.is_available())
    pre_mps_bytes = mps_allocated_bytes(torch)
    pre_driver_bytes = mps_driver_bytes(torch)
    peak_probe: dict[str, int] = {"current": pre_mps_bytes}
    install_mps_peak_probe(torch, peak_probe)

    fallback_log: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t0 = time.perf_counter()
        turns, load_elapsed = diarize(
            args.wav,
            num_speakers=args.num_speakers,
            log=lambda msg: print(f"  [diarize] {msg}", flush=True),
        )
        total_elapsed = time.perf_counter() - t0
        for w in caught:
            text = str(w.message)
            if "MPS" in text or "fallback" in text.lower():
                fallback_log.append(text)

    # Two complementary signals: peak_probe captures current_allocated_memory
    # at the moment diarize() drained the cache (the real working-set peak),
    # and driver_allocated_memory shows the Metal pool size, which is not
    # released by empty_cache.
    peak_mps_bytes = peak_probe["current"]
    peak_driver_bytes = mps_driver_bytes(torch)
    diarize_elapsed = total_elapsed - load_elapsed

    op_counts: Counter[str] = Counter()
    for msg in fallback_log:
        m = FALLBACK_RE.search(msg)
        op_counts[m.group(1) if m else "<unparsed>"] += 1

    try:
        audio_duration_s = _probe_duration(args.wav)
    except Exception:  # noqa: BLE001
        audio_duration_s = None

    return {
        "config": args.config,
        "label": args.label,
        "audio_duration_s": audio_duration_s,
        "num_speakers": args.num_speakers,
        "turns_count": len(turns),
        "load_elapsed_s": round(load_elapsed, 3),
        "diarize_elapsed_s": round(diarize_elapsed, 3),
        "total_elapsed_s": round(total_elapsed, 3),
        "peak_rss_bytes": peak_rss_bytes(),
        "pre_mps_allocated_bytes": pre_mps_bytes,
        "peak_mps_allocated_bytes": peak_mps_bytes,
        "pre_mps_driver_bytes": pre_driver_bytes,
        "peak_mps_driver_bytes": peak_driver_bytes,
        "mps_available": mps_available,
        "fallback_warning_count": len(fallback_log),
        "fallback_ops": dict(op_counts),
        "torch_version": torch.__version__,
    }


def gate_check(result: dict[str, Any]) -> list[str]:
    """Return list of gate failures; empty list means MPS was real."""
    failures: list[str] = []
    config = result["config"]
    if config in ("mps", "mps-trace"):
        if not result["mps_available"]:
            failures.append("torch.backends.mps.is_available() == False")
        driver_grew = result["peak_mps_driver_bytes"] > result["pre_mps_driver_bytes"]
        if result["peak_mps_allocated_bytes"] == 0 and not driver_grew:
            failures.append(
                "neither current_allocated_memory (probed at empty_cache time) nor "
                "driver_allocated_memory grew during diarize — pipeline likely ran on CPU"
            )
    if config == "mps" and result["fallback_warning_count"] > 0:
        failures.append(
            f"{result['fallback_warning_count']} MPS-fallback warnings in default config — "
            "ops silently ran on CPU; rerun as mps-trace"
        )
    return failures


def main() -> int:
    args = parse_args()
    if not args.wav.is_file():
        print(f"ERROR: wav not found: {args.wav}", file=sys.stderr)
        return 2
    if args.dry_run:
        print(f"DRY RUN ok: config={args.config} wav={args.wav}")
        return 0

    apply_env(args.config)
    print(f"== diarize-device-bench: config={args.config} label={args.label} ==", flush=True)
    print(f"   env VOICE_DIARIZE_DEVICE={os.environ.get('VOICE_DIARIZE_DEVICE', '<unset>')}", flush=True)
    print(f"   env PYTORCH_ENABLE_MPS_FALLBACK={os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '<unset>')}", flush=True)

    result = run_measurement(args)
    failures = gate_check(result)
    result["gate_failures"] = failures
    result["actually_on_gpu"] = (args.config != "cpu" and not failures)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{args.config}-{args.label}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"\n-- result written: {out_path}")
    print(f"   diarize {result['diarize_elapsed_s']}s, load {result['load_elapsed_s']}s, "
          f"peak MPS {result['peak_mps_allocated_bytes']/1e9:.2f} GB, "
          f"fallback warnings: {result['fallback_warning_count']}")
    if failures:
        print(f"\n!! GATE FAILED for config={args.config}:")
        for f in failures:
            print(f"   - {f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
