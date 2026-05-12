"""Unit tests for the clearspeech chain (issue #48, PR-1: autogain + bandpass).

The autogain-only tests mirror the v0.13.0 `test_audio_preprocess.py` suite (same
synthesised inputs, same RMS / continuity / gap-handling invariants) so a
regression in the autogain transplant fails loudly here. The bandpass tests cover
the new E2a effect; the chain tests cover the dispatcher itself.

All inputs are synthesised in `tmp_path`; no fixtures are checked in.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voice.clearspeech import (
    SR,
    ClearspeechError,
    clearspeech,
)
from voice.types import DiarTurn


def _sine(freq: float, dur_s: float, dbfs: float) -> np.ndarray:
    t = np.arange(int(dur_s * SR)) / SR
    peak = (10 ** (dbfs / 20)) * np.sqrt(2)
    return (peak * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _write(path: Path, arr: np.ndarray, sr: int = SR) -> None:
    sf.write(path, arr, sr, subtype="PCM_16")


def _rms_dbfs(arr: np.ndarray) -> float:
    return 20 * np.log10(np.sqrt(np.mean(arr.astype(np.float64) ** 2)) + 1e-12)


# ----------------------------- autogain regression -------------------------


def test_autogain_two_segments_brought_within_2dB_of_target(tmp_path: Path) -> None:
    audio = np.concatenate([_sine(440, 1.0, -10), _sine(660, 1.0, -30)])
    src = tmp_path / "in.wav"
    _write(src, audio)
    turns = [DiarTurn(0.0, 1.0, "A"), DiarTurn(1.0, 2.0, "B")]

    dst, config = clearspeech(
        src, chain=("autogain",), autogain_turns=turns,
        autogain_target_dbfs=-20.0, autogain_max_gain_db=12.0,
    )

    assert config["chain"] == ["autogain"]
    assert config["steps"][0]["applied"] is True
    assert config["steps"][0]["stats"]["turn_count"] == 2

    out, _ = sf.read(dst, dtype="float32")
    skip = int(0.10 * SR)
    rms_a = _rms_dbfs(out[skip:SR - skip])
    rms_b = _rms_dbfs(out[SR + skip:2 * SR - skip])
    assert abs(rms_a - (-20.0)) < 2.0, f"segment A RMS {rms_a:.2f} dBFS"
    assert abs(rms_b - (-20.0)) < 2.0, f"segment B RMS {rms_b:.2f} dBFS"


def test_autogain_empty_turns_is_byte_identical_copy(tmp_path: Path) -> None:
    src = tmp_path / "in.wav"
    _write(src, _sine(440, 0.5, -20))

    dst, _ = clearspeech(src, chain=("autogain",), autogain_turns=[])

    assert dst != src
    assert (
        hashlib.sha256(dst.read_bytes()).hexdigest()
        == hashlib.sha256(src.read_bytes()).hexdigest()
    )


def test_autogain_gain_clipped_at_max_gain_db(tmp_path: Path) -> None:
    audio = _sine(440, 1.0, -60)
    src = tmp_path / "in.wav"
    _write(src, audio)

    dst, _ = clearspeech(
        src, chain=("autogain",),
        autogain_turns=[DiarTurn(0.0, 1.0, "A")],
        autogain_target_dbfs=-20.0, autogain_max_gain_db=12.0,
    )

    out, _ = sf.read(dst, dtype="float32")
    skip = int(0.10 * SR)
    rms_out = _rms_dbfs(out[skip:SR - skip])
    assert abs(rms_out - (-48.0)) < 1.0, f"output RMS {rms_out:.2f} dBFS"


def test_autogain_gap_between_turns_keeps_silence_at_unity(tmp_path: Path) -> None:
    a = _sine(440, 0.5, -20)
    silence = np.zeros(int(0.4 * SR), dtype=np.float32)
    b = _sine(440, 0.5, -20)
    src = tmp_path / "in.wav"
    _write(src, np.concatenate([a, silence, b]))
    turns = [DiarTurn(0.0, 0.5, "A"), DiarTurn(0.9, 1.4, "B")]

    dst, _ = clearspeech(src, chain=("autogain",), autogain_turns=turns)

    out, _ = sf.read(dst, dtype="float32")
    mid = out[int(0.65 * SR):int(0.75 * SR)]
    assert float(np.max(np.abs(mid))) < 1e-6, "silence got amplified"


def test_autogain_crossfade_boundary_is_continuous(tmp_path: Path) -> None:
    audio = np.concatenate([_sine(100, 0.5, -20), _sine(100, 0.5, -32)])
    src = tmp_path / "in.wav"
    _write(src, audio)
    turns = [DiarTurn(0.0, 0.5, "A"), DiarTurn(0.5, 1.0, "B")]

    dst, _ = clearspeech(
        src, chain=("autogain",), autogain_turns=turns, autogain_crossfade_ms=100,
    )

    out, _ = sf.read(dst, dtype="float32")
    boundary = int(0.5 * SR)
    half = int(0.05 * SR)
    fade_region = out[boundary - half:boundary + half]
    step_max = float(np.max(np.abs(np.diff(fade_region))))
    assert step_max < 0.05, f"max sample-to-sample step {step_max:.4f}"


def test_autogain_input_must_be_16k_mono(tmp_path: Path) -> None:
    src = tmp_path / "in.wav"
    sf.write(src, np.zeros(48_000, dtype=np.float32), 48_000)

    with pytest.raises(ClearspeechError):
        clearspeech(
            src, chain=("autogain",),
            autogain_turns=[DiarTurn(0.0, 1.0, "A")],
        )


def test_autogain_writes_sibling_autogain_wav(tmp_path: Path) -> None:
    src = tmp_path / "audio.wav"
    _write(src, _sine(440, 0.5, -20))
    dst, _ = clearspeech(
        src, chain=("autogain",), autogain_turns=[DiarTurn(0.0, 0.5, "A")]
    )
    assert dst.name == "audio.autogain.wav"
    assert dst.parent == src.parent


# --------------------------- Bandpass effect --------------------------------


def test_bandpass_passes_in_band_sine_through(tmp_path: Path) -> None:
    """4 kHz sits comfortably inside 80–7900 Hz: RMS preserved within 0.5 dB."""
    src = tmp_path / "in.wav"
    audio = _sine(4_000.0, 1.0, -20.0)
    _write(src, audio)

    dst, config = clearspeech(
        src, chain=("bandpass",),
        bandpass_low_hz=80.0, bandpass_high_hz=7_900.0,
    )

    assert dst.name == "in.bandpass.wav"
    assert config["chain"] == ["bandpass"]
    bp_step = config["steps"][0]
    assert bp_step["applied"] is True
    assert bp_step["params"] == {"low_hz": 80.0, "high_hz": 7_900.0, "order": 4}

    out, _ = sf.read(dst, dtype="float32")
    skip = int(0.05 * SR)
    assert abs(_rms_dbfs(out[skip:-skip]) - _rms_dbfs(audio[skip:-skip])) < 0.5


def test_bandpass_attenuates_out_of_band_tones(tmp_path: Path) -> None:
    """30 Hz (below low cut) and 7990 Hz (above high cut at 6 kHz) get killed."""
    src_lo = tmp_path / "low.wav"
    src_hi = tmp_path / "high.wav"
    _write(src_lo, _sine(30.0, 1.0, -20.0))
    _write(src_hi, _sine(7_990.0, 1.0, -20.0))

    dst_lo, _ = clearspeech(
        src_lo, chain=("bandpass",),
        bandpass_low_hz=80.0, bandpass_high_hz=6_000.0,
    )
    dst_hi, _ = clearspeech(
        src_hi, chain=("bandpass",),
        bandpass_low_hz=80.0, bandpass_high_hz=6_000.0,
    )
    out_lo, _ = sf.read(dst_lo, dtype="float32")
    out_hi, _ = sf.read(dst_hi, dtype="float32")
    skip = int(0.05 * SR)
    assert _rms_dbfs(out_lo[skip:-skip]) < -45.0
    assert _rms_dbfs(out_hi[skip:-skip]) < -45.0


def test_bandpass_invalid_cutoffs_raise(tmp_path: Path) -> None:
    src = tmp_path / "in.wav"
    _write(src, _sine(440, 0.25, -20.0))

    with pytest.raises(ClearspeechError):
        clearspeech(
            src, chain=("bandpass",),
            bandpass_low_hz=500.0, bandpass_high_hz=500.0,
        )
    with pytest.raises(ClearspeechError):
        # high >= Nyquist (=8000 at SR=16000)
        clearspeech(
            src, chain=("bandpass",),
            bandpass_low_hz=80.0, bandpass_high_hz=9_000.0,
        )
    with pytest.raises(ClearspeechError):
        clearspeech(
            src, chain=("bandpass",),
            bandpass_low_hz=0.0, bandpass_high_hz=4_000.0,
        )


def test_bandpass_wrong_sample_rate_raises(tmp_path: Path) -> None:
    src = tmp_path / "in.wav"
    sf.write(src, np.zeros(48_000, dtype=np.float32), 48_000)
    with pytest.raises(ClearspeechError):
        clearspeech(src, chain=("bandpass",))


# ------------------------------ Chain orchestration -------------------------


def test_empty_chain_returns_input_path_unchanged(tmp_path: Path) -> None:
    src = tmp_path / "in.wav"
    _write(src, _sine(440, 0.25, -20))

    dst, config = clearspeech(src, chain=())

    assert dst == src
    assert config["any_enabled"] is False
    assert config["chain"] == []
    assert config["steps"] == []


def test_full_chain_runs_autogain_then_bandpass(tmp_path: Path) -> None:
    """Final sibling carries both suffixes; autogain then bandpass writes intermediates."""
    src = tmp_path / "audio.wav"
    _write(src, _sine(4_000.0, 1.0, -30.0))
    turns = [DiarTurn(0.0, 1.0, "A")]

    seen: list[tuple[int, str, str]] = []

    def dump(idx, name, path):
        seen.append((idx, name, path.name))

    dst, config = clearspeech(
        src, chain=("autogain", "bandpass"),
        autogain_turns=turns,
        bandpass_low_hz=80.0, bandpass_high_hz=7_900.0,
        dump=dump,
    )

    assert dst.name == "audio.autogain.bandpass.wav"
    assert config["chain"] == ["autogain", "bandpass"]
    assert [s["name"] for s in config["steps"]] == ["autogain", "bandpass"]
    assert seen == [(1, "autogain", "audio.autogain.wav"), (2, "bandpass", "audio.autogain.bandpass.wav")]
    # Both intermediates physically exist (the dumper would otherwise dump
    # nothing useful in real --dump-stages runs).
    assert (tmp_path / "audio.autogain.wav").exists()
    assert (tmp_path / "audio.autogain.bandpass.wav").exists()


def test_chain_missing_autogain_turns_raises(tmp_path: Path) -> None:
    src = tmp_path / "in.wav"
    _write(src, _sine(440, 0.25, -20))
    with pytest.raises(ClearspeechError, match="autogain_turns"):
        clearspeech(src, chain=("autogain",), autogain_turns=None)


def test_chain_rejects_unknown_effect(tmp_path: Path) -> None:
    src = tmp_path / "in.wav"
    _write(src, _sine(440, 0.25, -20))
    with pytest.raises(ValueError, match="unknown effect"):
        clearspeech(src, chain=("nonexistent",))


def test_chain_rejects_duplicate_effect(tmp_path: Path) -> None:
    src = tmp_path / "in.wav"
    _write(src, _sine(440, 0.25, -20))
    with pytest.raises(ValueError, match="duplicate effect"):
        clearspeech(src, chain=("autogain", "autogain"), autogain_turns=[])


@pytest.mark.parametrize("chain", [
    ("bandpass", "autogain"),
    ("presence", "autogain"),
    ("autogain", "presence", "bandpass"),
    ("presence", "bandpass", "autogain"),
])
def test_chain_allows_any_known_order(tmp_path: Path, chain: tuple[str, ...]) -> None:
    """Free-order is enabled in PR-2 — any permutation of known effects works."""
    src = tmp_path / "audio.wav"
    _write(src, _sine(1_000.0, 0.5, -20))
    turns = [DiarTurn(0.0, 0.5, "A")]

    dst, config = clearspeech(src, chain=chain, autogain_turns=turns)

    assert config["chain"] == list(chain)
    assert [s["name"] for s in config["steps"]] == list(chain)
    assert dst.exists()


# --------------------------- Presence effect --------------------------------


def test_presence_amplifies_at_center_freq(tmp_path: Path) -> None:
    """Sine at center frequency rises by ~boost_db (±0.5 dB) after filtfilt.

    `filtfilt` applies the biquad twice (forward + back), so the effective
    boost is 2× the per-pass design value. We pass boost_db=3 expecting +6 dB
    of post-filtfilt gain at center.
    """
    src = tmp_path / "in.wav"
    audio = _sine(3_000.0, 1.0, -20.0)
    _write(src, audio)

    dst, config = clearspeech(
        src, chain=("presence",),
        presence_center_hz=3_000.0, presence_boost_db=3.0, presence_q=1.0,
    )

    assert dst.name == "in.presence.wav"
    bp_step = config["steps"][0]
    assert bp_step["applied"] is True
    assert bp_step["params"] == {"center_hz": 3000.0, "boost_db": 3.0, "q": 1.0}

    out, _ = sf.read(dst, dtype="float32")
    skip = int(0.05 * SR)
    rise_db = _rms_dbfs(out[skip:-skip]) - _rms_dbfs(audio[skip:-skip])
    # filtfilt doubles the peak gain (each pass adds boost_db; effective 2×).
    assert 5.5 < rise_db < 6.5, f"rise {rise_db:.2f} dB out of [5.5, 6.5]"


def test_presence_passes_far_band_unchanged(tmp_path: Path) -> None:
    """100 Hz sine sits below the 3 kHz peak — RMS preserved within 0.5 dB."""
    src = tmp_path / "in.wav"
    audio = _sine(100.0, 1.0, -20.0)
    _write(src, audio)

    dst, _ = clearspeech(
        src, chain=("presence",),
        presence_center_hz=3_000.0, presence_boost_db=6.0, presence_q=1.0,
    )

    out, _ = sf.read(dst, dtype="float32")
    skip = int(0.05 * SR)
    assert abs(_rms_dbfs(out[skip:-skip]) - _rms_dbfs(audio[skip:-skip])) < 0.5


def test_presence_invalid_params_raise(tmp_path: Path) -> None:
    src = tmp_path / "in.wav"
    _write(src, _sine(440, 0.25, -20.0))

    # center >= Nyquist
    with pytest.raises(ClearspeechError, match="center_hz"):
        clearspeech(src, chain=("presence",), presence_center_hz=9_000.0)
    # center == 0
    with pytest.raises(ClearspeechError, match="center_hz"):
        clearspeech(src, chain=("presence",), presence_center_hz=0.0)
    # boost out of range
    with pytest.raises(ClearspeechError, match="boost_db"):
        clearspeech(src, chain=("presence",), presence_boost_db=30.0)
    # q out of range
    with pytest.raises(ClearspeechError, match="q"):
        clearspeech(src, chain=("presence",), presence_q=0.05)


def test_presence_writes_sibling_presence_wav(tmp_path: Path) -> None:
    src = tmp_path / "voice.autogain.bandpass.wav"
    _write(src, _sine(2_000.0, 0.5, -20))

    dst, _ = clearspeech(src, chain=("presence",))

    assert dst.name == "voice.autogain.bandpass.presence.wav"
    assert dst.parent == src.parent


# ----------------------------- Denoise effect ------------------------------


def test_denoise_ffmpeg_writes_valid_wav(tmp_path: Path) -> None:
    """ffmpeg afftdn produces a valid 16 kHz mono WAV with recorded params."""
    src = tmp_path / "in.wav"
    rng = np.random.default_rng(seed=7)
    noise = rng.standard_normal(2 * SR).astype(np.float32) * 0.05
    _write(src, noise + _sine(1_000.0, 2.0, -20.0))

    dst, config = clearspeech(
        src, chain=("denoise",),
        denoise_noise_floor_db=-25.0, denoise_reduction_db=12.0,
    )

    assert dst.exists() and dst.name == "in.denoise.wav"
    step = config["steps"][0]
    assert step["params"] == {"noise_floor_db": -25.0, "reduction_db": 12.0}
    out, sr = sf.read(dst, dtype="float32")
    assert sr == SR
    assert out.ndim == 1


def test_denoise_invalid_params_raise(tmp_path: Path) -> None:
    src = tmp_path / "in.wav"
    _write(src, _sine(440, 0.25, -20))
    with pytest.raises(ClearspeechError, match="noise_floor_db"):
        clearspeech(src, chain=("denoise",), denoise_noise_floor_db=10.0)
    with pytest.raises(ClearspeechError, match="reduction_db"):
        clearspeech(src, chain=("denoise",), denoise_reduction_db=200.0)


def test_denoise_writes_sibling_denoise_wav(tmp_path: Path) -> None:
    src = tmp_path / "voice.raw.wav"
    _write(src, _sine(1_000.0, 0.5, -20))

    dst, _ = clearspeech(src, chain=("denoise",))

    assert dst.name == "voice.raw.denoise.wav"
    assert dst.parent == src.parent


# ----------------------------- Dereverb effect -----------------------------


def _exponential_decay_tail(seg_len: int, rt60_s: float, sr: int = SR) -> np.ndarray:
    """Synthesise an impulse followed by an exponentially-decaying tail.

    Useful for verifying that RT60 estimation recovers the planted decay
    and that subtraction reduces the late portion without erasing the impulse.
    """
    out = np.zeros(seg_len, dtype=np.float32)
    out[0] = 1.0
    decay_per_sample = 10 ** (-60.0 / (rt60_s * sr * 10.0))
    rng = np.random.default_rng(seed=11)
    noise = rng.standard_normal(seg_len).astype(np.float32) * 0.05
    env = decay_per_sample ** np.arange(seg_len)
    out[1:] = (noise * env)[1:]
    return out


def test_dereverb_estimator_monotonic_in_decay_rate(tmp_path: Path) -> None:
    """Longer planted RT60 → longer estimated RT60.

    Absolute calibration is not tight (Schroeder backward-integration on a
    500-2000 Hz bandpass tends to underestimate by ~2× on broadband noise
    envelopes), but the relative ordering across two clearly different
    decay rates should be stable.
    """
    from voice.clearspeech import _estimate_rt60

    short = _estimate_rt60(_exponential_decay_tail(int(2.0 * SR), rt60_s=0.2))
    long_ = _estimate_rt60(_exponential_decay_tail(int(2.0 * SR), rt60_s=0.8))
    assert short is not None and long_ is not None
    assert long_ > short, f"expected long > short, got short={short:.2f} long={long_:.2f}"


def test_dereverb_no_turns_is_passthrough(tmp_path: Path) -> None:
    """Empty turns list ⇒ output is a byte-identical copy of the input WAV."""
    import hashlib

    src = tmp_path / "in.wav"
    _write(src, _sine(1_000.0, 0.5, -20))

    dst, config = clearspeech(src, chain=("dereverb",), dereverb_turns=[])

    assert dst != src
    assert (
        hashlib.sha256(dst.read_bytes()).hexdigest()
        == hashlib.sha256(src.read_bytes()).hexdigest()
    )
    assert config["steps"][0]["stats"]["turn_count"] == 0


def test_dereverb_subtracts_reverb_tail(tmp_path: Path) -> None:
    """A reverb-like input should have its late-energy reduced.

    We compare late-tail RMS (after the first 50 ms) before vs after the
    Lebart-Polack subtraction. The subtraction should knock at least 2 dB
    off the tail while leaving the early portion (where the direct sound
    lives) largely intact.
    """
    src = tmp_path / "in.wav"
    seg = _exponential_decay_tail(int(2.0 * SR), rt60_s=0.5)
    _write(src, seg)
    turns = [DiarTurn(0.0, 2.0, "A")]

    dst, _ = clearspeech(
        src, chain=("dereverb",), dereverb_turns=turns,
        dereverb_rt60_floor_ms=300.0, dereverb_subtract_factor=1.0,
        dereverb_crossfade_ms=0.0,
    )

    out, _ = sf.read(dst, dtype="float32")
    # Tail = samples after 50 ms (skip the direct-sound region entirely).
    tail_start = int(0.05 * SR)
    in_tail_rms = _rms_dbfs(seg[tail_start:])
    out_tail_rms = _rms_dbfs(out[tail_start:])
    assert out_tail_rms < in_tail_rms - 2.0, (
        f"expected ≥2 dB reduction; got {in_tail_rms:.2f} → {out_tail_rms:.2f}"
    )


def test_dereverb_invalid_params_raise(tmp_path: Path) -> None:
    src = tmp_path / "in.wav"
    _write(src, _sine(440, 0.25, -20))
    turns = [DiarTurn(0.0, 0.25, "A")]
    with pytest.raises(ClearspeechError, match="rt60_floor_ms"):
        clearspeech(
            src, chain=("dereverb",), dereverb_turns=turns,
            dereverb_rt60_floor_ms=10.0,  # below floor
        )
    with pytest.raises(ClearspeechError, match="subtract_factor"):
        clearspeech(
            src, chain=("dereverb",), dereverb_turns=turns,
            dereverb_subtract_factor=1.5,
        )
    with pytest.raises(ClearspeechError, match="crossfade_ms"):
        clearspeech(
            src, chain=("dereverb",), dereverb_turns=turns,
            dereverb_crossfade_ms=-10.0,
        )


def test_dereverb_writes_sibling_dereverb_wav(tmp_path: Path) -> None:
    src = tmp_path / "voice.raw.wav"
    _write(src, _sine(1_000.0, 0.5, -20))
    dst, _ = clearspeech(
        src, chain=("dereverb",),
        dereverb_turns=[DiarTurn(0.0, 0.5, "A")],
    )
    assert dst.name == "voice.raw.dereverb.wav"
    assert dst.parent == src.parent


def test_dereverb_in_chain_without_turns_raises(tmp_path: Path) -> None:
    """`dereverb` is per-turn — refuse to run without diarisation info."""
    src = tmp_path / "in.wav"
    _write(src, _sine(1_000.0, 0.5, -20))
    with pytest.raises(ClearspeechError, match="dereverb"):
        clearspeech(src, chain=("dereverb",))  # no turns at all


def test_full_chain_autogain_denoise_bandpass_presence(tmp_path: Path) -> None:
    """Four effects, denoise placed after autogain (the Metric-A-best order)."""
    src = tmp_path / "audio.wav"
    rng = np.random.default_rng(seed=1)
    noise = rng.standard_normal(SR).astype(np.float32) * 0.03
    _write(src, _sine(2_000.0, 1.0, -25.0) + noise)
    turns = [DiarTurn(0.0, 1.0, "A")]

    seen: list[tuple[int, str, str]] = []

    def dump(idx, name, path):
        seen.append((idx, name, path.name))

    dst, config = clearspeech(
        src, chain=("autogain", "denoise", "bandpass", "presence"),
        autogain_turns=turns, dump=dump,
    )

    assert dst.name == "audio.autogain.denoise.bandpass.presence.wav"
    assert config["chain"] == ["autogain", "denoise", "bandpass", "presence"]
    assert [s["name"] for s in config["steps"]] == [
        "autogain", "denoise", "bandpass", "presence",
    ]
    assert [s[1] for s in seen] == ["autogain", "denoise", "bandpass", "presence"]


def test_full_chain_autogain_bandpass_presence(tmp_path: Path) -> None:
    """Three effects in canonical order; intermediate WAVs all written."""
    src = tmp_path / "audio.wav"
    _write(src, _sine(3_000.0, 1.0, -25.0))
    turns = [DiarTurn(0.0, 1.0, "A")]

    seen: list[tuple[int, str, str]] = []

    def dump(idx, name, path):
        seen.append((idx, name, path.name))

    dst, config = clearspeech(
        src, chain=("autogain", "bandpass", "presence"),
        autogain_turns=turns, dump=dump,
    )

    assert dst.name == "audio.autogain.bandpass.presence.wav"
    assert config["chain"] == ["autogain", "bandpass", "presence"]
    assert seen == [
        (1, "autogain", "audio.autogain.wav"),
        (2, "bandpass", "audio.autogain.bandpass.wav"),
        (3, "presence", "audio.autogain.bandpass.presence.wav"),
    ]
    for inter in seen:
        assert (tmp_path / inter[2]).exists()
