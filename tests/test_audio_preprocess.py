"""Unit tests for the per-segment AGC stage.

All tests synthesise their own WAVs in `tmp_path` — no checked-in fixtures.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voice.audio_preprocess import (
    AudioPreprocessError,
    SR,
    loudness_normalize,
)
from voice.types import DiarTurn


def _sine(freq: float, dur_s: float, dbfs: float) -> np.ndarray:
    """Sine wave at `freq` Hz, `dur_s` seconds long, RMS at `dbfs`.

    For a pure sine, RMS = peak / sqrt(2). We set peak = 10^(dbfs/20) * sqrt(2)
    so the resulting RMS in dBFS matches the request.
    """
    t = np.arange(int(dur_s * SR)) / SR
    peak = (10 ** (dbfs / 20)) * np.sqrt(2)
    return (peak * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _write(path: Path, arr: np.ndarray) -> None:
    sf.write(path, arr, SR, subtype="PCM_16")


def _rms_dbfs(arr: np.ndarray) -> float:
    return 20 * np.log10(np.sqrt(np.mean(arr.astype(np.float64) ** 2)) + 1e-12)


def test_two_segments_brought_within_2dB_of_target(tmp_path: Path) -> None:
    audio = np.concatenate([_sine(440, 1.0, -10), _sine(660, 1.0, -30)])
    src = tmp_path / "in.wav"
    _write(src, audio)
    turns = [DiarTurn(0.0, 1.0, "A"), DiarTurn(1.0, 2.0, "B")]

    dst = loudness_normalize(src, turns, target_dbfs=-20.0, max_gain_db=12.0)

    out, _ = sf.read(dst, dtype="float32")
    skip = int(0.10 * SR)  # avoid the 100 ms crossfade region on each side
    rms_a = _rms_dbfs(out[skip:SR - skip])
    rms_b = _rms_dbfs(out[SR + skip:2 * SR - skip])
    assert abs(rms_a - (-20.0)) < 2.0, f"segment A RMS {rms_a:.2f} dBFS"
    assert abs(rms_b - (-20.0)) < 2.0, f"segment B RMS {rms_b:.2f} dBFS"


def test_empty_turns_is_byte_identical_copy(tmp_path: Path) -> None:
    src = tmp_path / "in.wav"
    _write(src, _sine(440, 0.5, -20))

    dst = loudness_normalize(src, diarize_segments=[])

    assert dst != src
    assert (
        hashlib.sha256(dst.read_bytes()).hexdigest()
        == hashlib.sha256(src.read_bytes()).hexdigest()
    )


def test_gain_clipped_at_max_gain_db(tmp_path: Path) -> None:
    audio = _sine(440, 1.0, -60)
    src = tmp_path / "in.wav"
    _write(src, audio)

    dst = loudness_normalize(
        src, [DiarTurn(0.0, 1.0, "A")], target_dbfs=-20.0, max_gain_db=12.0
    )

    out, _ = sf.read(dst, dtype="float32")
    skip = int(0.10 * SR)
    rms_out = _rms_dbfs(out[skip:SR - skip])
    # -60 dBFS + 12 dB clamp = -48 dBFS. PCM_16 quantisation noise floor is
    # ~-96 dBFS, but at -48 the signal is far above it; allow 1 dB slack for
    # rounding and the 16-bit re-encode.
    assert abs(rms_out - (-48.0)) < 1.0, f"output RMS {rms_out:.2f} dBFS"


def test_gap_between_turns_keeps_silence_at_unity(tmp_path: Path) -> None:
    a = _sine(440, 0.5, -20)
    silence = np.zeros(int(0.4 * SR), dtype=np.float32)
    b = _sine(440, 0.5, -20)
    src = tmp_path / "in.wav"
    _write(src, np.concatenate([a, silence, b]))
    turns = [DiarTurn(0.0, 0.5, "A"), DiarTurn(0.9, 1.4, "B")]

    dst = loudness_normalize(src, turns)

    out, _ = sf.read(dst, dtype="float32")
    # Interior of the 400 ms gap, away from the 100 ms fades on each side.
    mid = out[int(0.65 * SR):int(0.75 * SR)]
    assert float(np.max(np.abs(mid))) < 1e-6, "silence got amplified"


def test_crossfade_boundary_is_continuous(tmp_path: Path) -> None:
    # 100 Hz keeps the wave smooth at 16 kHz (160 samples / cycle); the
    # sample-to-sample step on the bare signal is tiny, so any added step
    # from a click would dominate.
    audio = np.concatenate([_sine(100, 0.5, -20), _sine(100, 0.5, -32)])
    src = tmp_path / "in.wav"
    _write(src, audio)
    turns = [DiarTurn(0.0, 0.5, "A"), DiarTurn(0.5, 1.0, "B")]

    dst = loudness_normalize(src, turns, crossfade_ms=100)

    out, _ = sf.read(dst, dtype="float32")
    boundary = int(0.5 * SR)
    half = int(0.05 * SR)
    fade_region = out[boundary - half:boundary + half]
    step_max = float(np.max(np.abs(np.diff(fade_region))))
    assert step_max < 0.05, f"max sample-to-sample step {step_max:.4f}"


def test_input_must_be_16k_mono(tmp_path: Path) -> None:
    src = tmp_path / "in.wav"
    sf.write(src, np.zeros(48_000, dtype=np.float32), 48_000)

    with pytest.raises(AudioPreprocessError):
        loudness_normalize(src, [DiarTurn(0.0, 1.0, "A")])
