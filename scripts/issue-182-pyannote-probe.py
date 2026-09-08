"""Probe pyannote 3.1 tunable hyperparameters on the 12-min reference (issue #182).

pyannote-3.1 does NOT expose onset/offset (PowerSet segmentation). Tunable
runtime params per `pipe.parameters()`:
  - segmentation.min_duration_off  (default 0.0)
  - clustering.threshold           (default 0.7045)
  - clustering.method              (default 'centroid')
  - clustering.min_cluster_size    (default 12)

Sweeps a sensible 1D grid for each candidate, applies to the same audio
(set VOICE_PROBE_WAV to the 12-min reference used by the #177 probe),
prints turn boundaries around our canonical failure window (180-195 s),
and reports how each setting attributes two ground-truth words:
  know  @[185.04-185.16]  truth=SPEAKER_01
  yes   @[186.30-187.20]  truth=SPEAKER_00

Read-only probe — does not write into the repo. Not part of the pipeline.
"""
from __future__ import annotations

import os
import time

WAV = os.environ.get("VOICE_PROBE_WAV", "/tmp/reference-12min.wav")

CANONICAL = [
    {"start": 185.04, "end": 185.16, "content": "know", "truth": "SPEAKER_01"},
    {"start": 186.30, "end": 187.20, "content": "yes",  "truth": "SPEAKER_00"},
]


def _overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def argmax_owner(w_start, w_end, turns):
    best, best_ov = None, 0.0
    for t in turns:
        ov = _overlap(w_start, w_end, t["start"], t["end"])
        if ov > best_ov:
            best_ov = ov
            best = t["speaker"]
    return best


def summarize(turns, label):
    print(f"\n=== {label} ===")
    print("  turns in 178-198 s:")
    for t in turns:
        if 178 <= t["start"] < 198:
            print(f"    [{t['start']:7.2f}-{t['end']:7.2f}] {t['speaker']}")
    print("  canonical words:")
    for c in CANONICAL:
        owner = argmax_owner(c["start"], c["end"], turns)
        flag = "OK " if owner == c["truth"] else "BAD"
        print(f"    {flag} '{c['content']}' [{c['start']:.2f}-{c['end']:.2f}] "
              f"-> {owner} (truth: {c['truth']})")


def run_with(label, params, num_speakers=2):
    import torch
    from pyannote.audio import Pipeline

    token = os.environ.get("HF_TOKEN")
    pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=token)

    device = "cpu"
    if torch.backends.mps.is_available():
        try:
            pipe.to(torch.device("mps"))
            device = "mps"
        except Exception as e:
            print(f"  (MPS not usable: {e}; CPU fallback)")
            pipe.to(torch.device("cpu"))

    if params:
        pipe.instantiate(params)

    t0 = time.perf_counter()
    result = pipe(WAV, num_speakers=num_speakers)
    elapsed = time.perf_counter() - t0
    payload = result.serialize()
    turns = sorted(
        [{"start": t["start"], "end": t["end"], "speaker": t["speaker"]}
         for t in payload["exclusive_diarization"]],
        key=lambda x: x["start"],
    )
    print(f"\n[{label}] wall_clock={elapsed:.1f}s, device={device}, total turns={len(turns)}")
    summarize(turns, label)


def main():
    DEFAULT = {
        "segmentation": {"min_duration_off": 0.0},
        "clustering": {
            "method": "centroid",
            "min_cluster_size": 12,
            "threshold": 0.7045654963945799,
        },
    }
    run_with("BASELINE (defaults)", DEFAULT)

    for mdo in (0.1, 0.25, 0.5, 1.0):
        run_with(
            f"min_duration_off={mdo}",
            {"segmentation": {"min_duration_off": mdo},
             "clustering": DEFAULT["clustering"]},
        )

    for thr in (0.50, 0.60, 0.80, 0.90):
        run_with(
            f"clustering.threshold={thr}",
            {"segmentation": DEFAULT["segmentation"],
             "clustering": {
                 "method": "centroid",
                 "min_cluster_size": 12,
                 "threshold": thr,
             }},
        )


if __name__ == "__main__":
    main()
