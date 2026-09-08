"""Compare lang_detect on the longest pyannote turn vs a greedy-fill window.

Probe for the question: does padding the Whisper detect_language window with
real speech (additional same-speaker turns) instead of silence improve the
uk-vs-competing-language margin on the project's reference recording?

Loads (override the directory with VOICE_PROBE_DIR):
- <reference-recording>.wav (16 kHz mono)
- <reference-recording>.diarize.json (cached pyannote turns)

Runs Whisper-large-v3-MLX detect_language() on two mel inputs, sequentially:
1. baseline: longest single turn (current production behaviour, ADR 0022)
2. greedy-fill: longest turn + next-longest same-speaker turns concatenated
   until total reaches >= 30 s, then trimmed to 30 s.

Reports top-3 probabilities and the uk / runner-up margin for each.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

_DIR = Path(os.environ.get("VOICE_PROBE_DIR", Path.home() / "Downloads"))
_STEM = os.environ.get("VOICE_PROBE_STEM", "reference-recording")
WAV = _DIR / f"{_STEM}.wav"
DIAR = _DIR / f"{_STEM}.diarize.json"
TARGET_S = 30.0


def load_turns() -> list[dict]:
    return json.loads(DIAR.read_text())["diarization"]


def pick_baseline(turns: list[dict]) -> tuple[str, list[dict]]:
    longest = max(turns, key=lambda t: t["end"] - t["start"])
    return longest["speaker"], [longest]


def pick_greedy_fill(
    turns: list[dict], target: float = TARGET_S
) -> tuple[str, list[dict]]:
    """Take longest turn's speaker, then add their next-longest turns until >= target."""
    longest = max(turns, key=lambda t: t["end"] - t["start"])
    sp = longest["speaker"]
    same_sp_sorted = sorted(
        (t for t in turns if t["speaker"] == sp),
        key=lambda t: t["end"] - t["start"],
        reverse=True,
    )
    picked: list[dict] = []
    acc = 0.0
    for t in same_sp_sorted:
        picked.append(t)
        acc += t["end"] - t["start"]
        if acc >= target:
            break
    return sp, picked


def slice_and_concat(audio: np.ndarray, turns: list[dict], sample_rate: int) -> np.ndarray:
    """Concatenate audio slices for the given turns, in chronological order."""
    in_chrono = sorted(turns, key=lambda t: t["start"])
    parts = []
    for t in in_chrono:
        s0 = int(t["start"] * sample_rate)
        s1 = int(t["end"] * sample_rate)
        parts.append(audio[s0:s1])
    return np.concatenate(parts) if len(parts) > 1 else parts[0]


def fmt_top(probs: dict[str, float], k: int = 3) -> str:
    top = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return "  ".join(f"{lang}={p:.3f}" for lang, p in top)


def main() -> None:
    from mlx_whisper.audio import (
        N_FRAMES,
        SAMPLE_RATE,
        load_audio,
        log_mel_spectrogram,
        pad_or_trim,
    )
    from mlx_whisper.load_models import load_model

    from voice.whisper_asr import WHISPER_REPO_ID

    turns = load_turns()
    audio = load_audio(str(WAV))
    print(f"Loaded {len(turns)} turns, audio {len(audio) / SAMPLE_RATE:.1f}s\n")

    sp_b, picked_b = pick_baseline(turns)
    audio_b = slice_and_concat(audio, picked_b, SAMPLE_RATE)
    dur_b = len(audio_b) / SAMPLE_RATE
    print(f"[baseline]   speaker={sp_b}  n_turns={len(picked_b)}  speech={dur_b:.2f}s")

    sp_g, picked_g = pick_greedy_fill(turns, TARGET_S)
    audio_g = slice_and_concat(audio, picked_g, SAMPLE_RATE)
    dur_g = len(audio_g) / SAMPLE_RATE
    print(f"[greedy-fill] speaker={sp_g}  n_turns={len(picked_g)}  speech={dur_g:.2f}s")
    print(
        "  picked turns: "
        + ", ".join(f"[{t['start']:.1f}-{t['end']:.1f}]" for t in picked_g)
    )

    print(f"\nLoading Whisper {WHISPER_REPO_ID} (this takes a few seconds)...")
    t0 = time.time()
    model = load_model(WHISPER_REPO_ID)
    print(f"  loaded in {time.time() - t0:.1f}s\n")

    long_turns = sorted(
        (t for t in turns if (t["end"] - t["start"]) > 10.0),
        key=lambda t: t["end"] - t["start"],
        reverse=True,
    )
    print(f"\nFound {len(long_turns)} turns longer than 10s; will probe each individually.\n")
    per_turn_inputs = []
    for t in long_turns:
        a = slice_and_concat(audio, [t], SAMPLE_RATE)
        d = len(a) / SAMPLE_RATE
        label = (
            f"turn {t['speaker']} [{t['start']:.1f}-{t['end']:.1f}] ({d:.2f}s)"
        )
        per_turn_inputs.append((label, a, d))

    for label, audio_in, dur in [
        ("baseline (longest turn, silence-padded)", audio_b, dur_b),
        ("greedy-fill (same speaker, real speech)", audio_g, dur_g),
        *per_turn_inputs,
    ]:
        mel = log_mel_spectrogram(audio_in, n_mels=model.dims.n_mels)
        mel = pad_or_trim(mel, N_FRAMES, axis=-2)
        t0 = time.time()
        _, probs = model.detect_language(mel)
        elapsed = time.time() - t0
        uk = float(probs.get("uk", 0.0))
        ru = float(probs.get("ru", 0.0))
        margin = uk / ru if ru > 0 else float("inf")
        print(f"[{label}]")
        print(f"  speech in window: {min(dur, 30.0):.2f}s / 30.00s")
        print(f"  detect_language: {elapsed:.2f}s")
        print(f"  top-3: {fmt_top(probs, 3)}")
        print(f"  uk={uk:.4f}  ru={ru:.4f}  margin uk/ru = {margin:.2f}x\n")


if __name__ == "__main__":
    main()
