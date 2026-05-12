"""Audio cleanup chain between diarize and ASR.

The chain-of-effects design lets future PRs add new DSP effects as ordered
links without growing the pipeline-stage count (the slot stays `[5/11]`)
or rewriting dump layout per release.

Chain effects (issue #48 + dereverb follow-up):
  - "autogain" — per-pyannote-turn RMS normalization (was loudness_normalize)
  - "bandpass" — Butterworth IIR bandpass via scipy.signal.sosfiltfilt (E2a)
  - "presence" — RBJ-cookbook peaking EQ via scipy.signal.filtfilt (E2b)
  - "denoise"  — FFT-domain spectral subtraction via `ffmpeg afftdn` (E2d).
                 Empirically best **after** autogain, even though physics-of-noise
                 intuition argues for pre-autogain — see ADR 0014. Default-off.
  - "dereverb" — per-pyannote-turn Lebart-Polack spectral subtraction (E2e).
                 Per-segment RT60 estimated via Schroeder backward integration,
                 then late reverberation power subtracted from the STFT. See
                 ADR 0015. Default-off.

Effect order is the user's choice — any permutation of known effects is
allowed; duplicates and unknown names raise `ValueError`. See ADR 0012 for
the CLI shape (single `--clearspeech-chain` string).

Hard input invariant: 16 kHz mono PCM_16 WAV — the format that the upstream
ffmpeg stage produces. The 16 kHz Nyquist of 8 kHz caps `bandpass_high_hz`
strictly below 8000; defaults were picked by listening tests on the
project's reference recording — see ADR 0011 (bandpass) and ADR 0013
(presence) for the empirical reasoning.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf
from scipy.signal import butter, filtfilt, hilbert, istft, sosfilt, sosfiltfilt, stft

from .types import DiarTurn


SR = 16_000
BANDPASS_ORDER = 4  # Butterworth IIR; 4 ≈ 24 dB/octave, doubled by sosfiltfilt
SILENT_RMS_THRESHOLD = 1e-6

KNOWN_EFFECTS: frozenset[str] = frozenset({
    "autogain", "bandpass", "presence", "denoise", "dereverb",
})

ClearspeechDumpHook = Callable[[int, str, Path], None]


class ClearspeechError(RuntimeError):
    """Raised when the input WAV does not match the pipeline invariants
    or when an effect's parameters are out of range."""


def _noop_log(_msg: str) -> None:
    return None


def _validate_chain(chain: tuple[str, ...]) -> None:
    """Reject unknown effect names and duplicates. Order is the caller's call."""
    seen: set[str] = set()
    for effect in chain:
        if effect not in KNOWN_EFFECTS:
            raise ValueError(
                f"unknown effect in chain: {effect!r}. "
                f"Known effects: {sorted(KNOWN_EFFECTS)}"
            )
        if effect in seen:
            raise ValueError(
                f"duplicate effect in chain: {effect!r}. "
                f"Each effect must appear at most once."
            )
        seen.add(effect)


def clearspeech(
    wav_path: str | Path,
    chain: tuple[str, ...],
    *,
    autogain_turns: list[DiarTurn] | None = None,
    autogain_target_dbfs: float = -20.0,
    autogain_max_gain_db: float = 16.0,
    autogain_crossfade_ms: float = 100.0,
    bandpass_low_hz: float = 150.0,
    bandpass_high_hz: float = 5_500.0,
    presence_center_hz: float = 3_000.0,
    presence_boost_db: float = 6.0,
    presence_q: float = 1.0,
    denoise_noise_floor_db: float = -25.0,
    denoise_reduction_db: float = 12.0,
    dereverb_turns: list[DiarTurn] | None = None,
    dereverb_rt60_floor_ms: float = 300.0,
    dereverb_subtract_factor: float = 1.0,
    dereverb_crossfade_ms: float = 50.0,
    log: Callable[[str], None] = _noop_log,
    dump: ClearspeechDumpHook | None = None,
) -> tuple[Path, dict]:
    """Run the requested chain of DSP effects against `wav_path`.

    Each enabled effect writes a sibling WAV (`<stem>.autogain.wav`,
    `<stem>.autogain.bandpass.wav`, …) and the next effect reads it. `dump`, if
    provided, is invoked with `(step_index, effect_name, intermediate_path)`
    after each effect so the pipeline can mirror intermediates into the
    `--dump-stages` directory.

    Returns `(final_path, config)`:
      - `final_path` — output of the last applied effect, or `wav_path` itself
        when `chain` is empty (no decode/encode round-trip).
      - `config` — a JSON-serialisable dict capturing chain, params and per-
        step stats, suitable for `02b-clearspeech-config.json`.

    Raises ClearspeechError on bad inputs (wrong SR, non-mono, missing
    `autogain_turns` when "autogain" is requested, invalid bandpass / presence params).
    Raises ValueError when `chain` contains unknown effects or duplicates.
    """
    chain = tuple(chain)
    _validate_chain(chain)

    src = Path(wav_path)
    config: dict = {
        "any_enabled": bool(chain),
        "chain": list(chain),
        "steps": [],
    }

    if not chain:
        log("clearspeech: no-op (empty chain)")
        return src, config

    current = src
    for idx, effect in enumerate(chain, start=1):
        if effect == "autogain":
            if autogain_turns is None:
                raise ClearspeechError(
                    "chain contains 'autogain' but autogain_turns is None — pass the "
                    "pyannote turns the autogain needs as per-segment guard-rails"
                )
            current, step_info = _apply_autogain(
                current,
                turns=autogain_turns,
                target_dbfs=autogain_target_dbfs,
                max_gain_db=autogain_max_gain_db,
                crossfade_ms=autogain_crossfade_ms,
                log=log,
            )
        elif effect == "bandpass":
            current, step_info = _apply_bandpass(
                current,
                low_hz=bandpass_low_hz,
                high_hz=bandpass_high_hz,
                log=log,
            )
        elif effect == "presence":
            current, step_info = _apply_presence(
                current,
                center_hz=presence_center_hz,
                boost_db=presence_boost_db,
                q=presence_q,
                log=log,
            )
        elif effect == "denoise":
            current, step_info = _apply_denoise(
                current,
                noise_floor_db=denoise_noise_floor_db,
                reduction_db=denoise_reduction_db,
                log=log,
            )
        elif effect == "dereverb":
            # `dereverb` shares pyannote turns with `autogain` — same physical
            # boundary information. We fall back to `autogain_turns` if the caller
            # didn't pass an explicit `dereverb_turns`.
            turns_for_dereverb = (
                dereverb_turns if dereverb_turns is not None else autogain_turns
            )
            if turns_for_dereverb is None:
                raise ClearspeechError(
                    "chain contains 'dereverb' but no turns provided — "
                    "pass `dereverb_turns` (or reuse `autogain_turns`) so each "
                    "speaker turn gets its own RT60 estimate"
                )
            current, step_info = _apply_dereverb(
                current,
                turns=turns_for_dereverb,
                rt60_floor_ms=dereverb_rt60_floor_ms,
                subtract_factor=dereverb_subtract_factor,
                crossfade_ms=dereverb_crossfade_ms,
                log=log,
            )
        else:
            # Unreachable while _validate_chain stays consistent with the
            # dispatch arms above. Explicit so a future loosening of the
            # validator fails loudly here rather than silently dropping work.
            raise ValueError(f"unknown effect in chain: {effect!r}")

        config["steps"].append({"name": effect, "applied": True, **step_info})
        if dump is not None:
            dump(idx, effect, current)

    return current, config


# ---------------------------------------------------------------------------
# Effect: autogain (per-turn RMS normalization). Logic transplanted byte-for-byte
# from the v0.13.0 `audio_preprocess.loudness_normalize`; the only changes are
# the output suffix (`.autogain.wav` instead of `.normalized.wav`) and returning a
# stats dict alongside the path so the chain's config can record per-step
# metrics.
# ---------------------------------------------------------------------------


def _apply_autogain(
    wav_path: Path,
    *,
    turns: list[DiarTurn],
    target_dbfs: float,
    max_gain_db: float,
    crossfade_ms: float,
    log: Callable[[str], None],
) -> tuple[Path, dict]:
    src = Path(wav_path)
    dst = src.with_suffix(".autogain.wav")
    params = {
        "target_dbfs": float(target_dbfs),
        "max_gain_db": float(max_gain_db),
        "crossfade_ms": float(crossfade_ms),
    }

    if not turns:
        shutil.copyfile(src, dst)
        return dst, {"params": params, "stats": {"turn_count": 0}}

    samples, sr = sf.read(src, dtype="float32", always_2d=False)
    if sr != SR:
        raise ClearspeechError(
            f"expected {SR} Hz sample rate, got {sr} Hz: {src}"
        )
    if samples.ndim != 1:
        raise ClearspeechError(
            f"expected mono input, got {samples.ndim}-D array: {src}"
        )

    n = int(samples.shape[0])
    target_amp = 10 ** (target_dbfs / 20)
    max_gain = 10 ** (max_gain_db / 20)
    fade_samples = max(1, int(round(SR * crossfade_ms / 1000)))

    raw_regions: list[tuple[int, int, float]] = []
    pre_rms_db: list[float] = []
    for turn in sorted(turns, key=lambda t: t.start):
        s = max(0, int(round(turn.start * SR)))
        e = min(n, int(round(turn.end * SR)))
        if e <= s:
            continue
        seg = samples[s:e]
        rms = float(np.sqrt(np.mean(seg * seg))) if seg.size else 0.0
        if rms < SILENT_RMS_THRESHOLD:
            gain = 1.0
        else:
            gain = min(target_amp / rms, max_gain)
        raw_regions.append((s, e, gain))
        pre_rms_db.append(20 * np.log10(rms + 1e-12))

    regions = _fill_gaps(raw_regions, n)
    if not regions:
        shutil.copyfile(src, dst)
        return dst, {"params": params, "stats": {"turn_count": 0}}

    env = _build_envelope(regions, n, fade_samples)
    out = samples * env
    np.clip(out, -1.0, 1.0 - 1e-7, out=out)

    post_rms_db: list[float] = []
    ceiling_hits = 0
    for (s, e, _g), pre in zip(raw_regions, pre_rms_db):
        seg = out[s:e]
        rms = float(np.sqrt(np.mean(seg * seg))) if seg.size else 0.0
        post = 20 * np.log10(rms + 1e-12)
        post_rms_db.append(post)
        if post - pre > max_gain_db - 0.5:
            ceiling_hits += 1
    pre_std = float(np.std(pre_rms_db))
    post_std = float(np.std(post_rms_db))
    spread_drop = (pre_std - post_std) / pre_std * 100 if pre_std > 0 else 0.0

    sf.write(dst, out, SR, subtype="PCM_16")
    log(
        f"clearspeech.autogain: {len(raw_regions)} turn(s) | "
        f"per-turn RMS std-dev {pre_std:.2f} → {post_std:.2f} dB "
        f"({spread_drop:+.1f}%) | "
        f"{ceiling_hits}/{len(raw_regions)} hit +{max_gain_db:.0f} dB ceiling | "
        f"target={target_dbfs:+.1f} dBFS, crossfade={crossfade_ms:.0f} ms "
        f"→ {dst.name}"
    )
    stats = {
        "turn_count": len(raw_regions),
        "pre_rms_std_db": round(pre_std, 3),
        "post_rms_std_db": round(post_std, 3),
        "spread_drop_pct": round(spread_drop, 1),
        "ceiling_hits": ceiling_hits,
    }
    return dst, {"params": params, "stats": stats}


def _fill_gaps(
    raw_regions: list[tuple[int, int, float]], total: int
) -> list[tuple[int, int, float]]:
    """Tile `[0, total)` with regions; insert unity-gain regions for gaps."""
    if total <= 0:
        return []
    filled: list[tuple[int, int, float]] = []
    cursor = 0
    for s, e, g in raw_regions:
        if s < cursor:
            s = cursor
        if e <= s:
            continue
        if s > cursor:
            filled.append((cursor, s, 1.0))
        filled.append((s, e, g))
        cursor = e
    if cursor < total:
        filled.append((cursor, total, 1.0))
    return filled


def _build_envelope(
    regions: list[tuple[int, int, float]], total: int, fade_samples: int
) -> np.ndarray:
    """Per-sample gain envelope with linear crossfades on every boundary.

    Fade width is clamped per-boundary so it never spills past the midpoint
    of either neighbouring region — adjacent fades from a short region can
    therefore meet but never overlap.
    """
    env = np.ones(total, dtype=np.float32)
    for s, e, g in regions:
        env[s:e] = np.float32(g)

    half_fade = fade_samples // 2
    if half_fade < 1:
        return env

    for (s1, e1, g1), (s2, e2, g2) in zip(regions[:-1], regions[1:]):
        if e1 != s2:
            continue
        if g1 == g2:
            continue
        room = min(half_fade, (e1 - s1) // 2, (e2 - s2) // 2)
        if room < 1:
            continue
        start = e1 - room
        stop = e1 + room
        env[start:stop] = np.linspace(
            g1, g2, 2 * room, endpoint=False, dtype=np.float32
        )
    return env


# ---------------------------------------------------------------------------
# Effect: bandpass (zero-phase Butterworth via sosfiltfilt). Cuts sub-vocal
# rumble and super-vocal noise so ASR sees a cleaner spectral envelope.
# ---------------------------------------------------------------------------


def _apply_bandpass(
    wav_path: Path,
    *,
    low_hz: float,
    high_hz: float,
    log: Callable[[str], None],
) -> tuple[Path, dict]:
    src = Path(wav_path)
    dst = src.with_suffix(".bandpass.wav")
    params = {
        "low_hz": float(low_hz),
        "high_hz": float(high_hz),
        "order": BANDPASS_ORDER,
    }

    _validate_bandpass_cutoffs(low_hz, high_hz)

    samples, sr = sf.read(src, dtype="float32", always_2d=False)
    if sr != SR:
        raise ClearspeechError(
            f"expected {SR} Hz sample rate, got {sr} Hz: {src}"
        )
    if samples.ndim != 1:
        raise ClearspeechError(
            f"expected mono input, got {samples.ndim}-D array: {src}"
        )

    # sosfiltfilt runs the filter forward and back, which doubles the effective
    # order (≈ 48 dB/octave at the design point) but cancels phase distortion.
    # That matters here: ASR is sensitive to envelope timing, and a min-phase
    # IIR would smear consonant onsets.
    nyquist = SR / 2
    sos = butter(
        BANDPASS_ORDER,
        [low_hz / nyquist, high_hz / nyquist],
        btype="band",
        output="sos",
    )
    out = sosfiltfilt(sos, samples).astype(np.float32)
    np.clip(out, -1.0, 1.0 - 1e-7, out=out)

    sf.write(dst, out, SR, subtype="PCM_16")
    log(
        f"clearspeech.bandpass: {low_hz:.0f}–{high_hz:.0f} Hz "
        f"(order {BANDPASS_ORDER}) → {dst.name}"
    )
    return dst, {"params": params, "stats": {}}


def _validate_bandpass_cutoffs(low_hz: float, high_hz: float) -> None:
    nyquist = SR / 2
    if not (0.0 < low_hz < high_hz < nyquist):
        raise ClearspeechError(
            f"invalid bandpass cutoffs: 0 < low ({low_hz}) "
            f"< high ({high_hz}) < Nyquist ({nyquist}) is required"
        )


# ---------------------------------------------------------------------------
# Effect: presence (RBJ-cookbook peaking EQ, zero-phase via filtfilt). Boosts
# (or cuts) a configurable band around `center_hz` by `boost_db`; widens or
# narrows the boost via `q`. Used to restore consonant intelligibility for
# distant speakers whose 2-5 kHz energy was attenuated by room transit.
# ---------------------------------------------------------------------------


_PRESENCE_BOOST_BOUNDS_DB = (-24.0, 24.0)
_PRESENCE_Q_BOUNDS = (0.1, 10.0)


def _apply_presence(
    wav_path: Path,
    *,
    center_hz: float,
    boost_db: float,
    q: float,
    log: Callable[[str], None],
) -> tuple[Path, dict]:
    src = Path(wav_path)
    dst = src.with_suffix(".presence.wav")
    params = {
        "center_hz": float(center_hz),
        "boost_db": float(boost_db),
        "q": float(q),
    }

    _validate_presence_params(center_hz, boost_db, q)

    samples, sr = sf.read(src, dtype="float32", always_2d=False)
    if sr != SR:
        raise ClearspeechError(
            f"expected {SR} Hz sample rate, got {sr} Hz: {src}"
        )
    if samples.ndim != 1:
        raise ClearspeechError(
            f"expected mono input, got {samples.ndim}-D array: {src}"
        )

    b, a = _design_peaking_eq(center_hz, boost_db, q)
    # `filtfilt` runs the biquad forward and back, doubling effective slope
    # but cancelling phase distortion — same rationale as the bandpass effect
    # (ASR is sensitive to consonant onset timing; min-phase IIR smears it).
    out = filtfilt(b, a, samples).astype(np.float32)
    np.clip(out, -1.0, 1.0 - 1e-7, out=out)

    sf.write(dst, out, SR, subtype="PCM_16")
    log(
        f"clearspeech.presence: {center_hz:.0f} Hz, "
        f"{boost_db:+.1f} dB, Q={q:.2f} → {dst.name}"
    )
    return dst, {"params": params, "stats": {}}


def _validate_presence_params(center_hz: float, boost_db: float, q: float) -> None:
    nyquist = SR / 2
    if not (0.0 < center_hz < nyquist):
        raise ClearspeechError(
            f"invalid presence center_hz ({center_hz}): "
            f"must satisfy 0 < center_hz < Nyquist ({nyquist})"
        )
    lo, hi = _PRESENCE_BOOST_BOUNDS_DB
    if not (lo <= boost_db <= hi):
        raise ClearspeechError(
            f"invalid presence boost_db ({boost_db}): "
            f"must be within [{lo}, {hi}] dB"
        )
    qlo, qhi = _PRESENCE_Q_BOUNDS
    if not (qlo <= q <= qhi):
        raise ClearspeechError(
            f"invalid presence q ({q}): must be within [{qlo}, {qhi}]"
        )


def _design_peaking_eq(
    center_hz: float, boost_db: float, q: float
) -> tuple[np.ndarray, np.ndarray]:
    """Robert Bristow-Johnson Audio EQ Cookbook peaking EQ biquad.

    Reference: https://www.w3.org/TR/audio-eq-cookbook/#peaking-eq

    Returns (b, a) coefficient arrays normalised so a[0] == 1, suitable for
    `scipy.signal.filtfilt`. The peak gain at `center_hz` equals `boost_db`
    (positive = boost, negative = dip); `q` controls bandwidth (higher = narrower).
    """
    A = 10 ** (boost_db / 40)
    w0 = 2 * np.pi * center_hz / SR
    cos_w0 = np.cos(w0)
    alpha = np.sin(w0) / (2 * q)

    b0 = 1 + alpha * A
    b1 = -2 * cos_w0
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * cos_w0
    a2 = 1 - alpha / A

    b = np.array([b0, b1, b2], dtype=np.float64) / a0
    a = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)
    return b, a


# ---------------------------------------------------------------------------
# Effect: denoise (FFT-domain spectral subtraction via ffmpeg's afftdn). The
# expected place in a chain is **after** autogain, despite the physics intuition
# that says raw signal is the right input for noise estimation — see ADR 0014
# Metric A grid: pre-autogain denoise gave 27.5 % RU-glyph drift (5× the default
# baseline), while post-autogain denoise gave 8.2 %. Autogain's per-turn normalisation
# fixes the SNR before afftdn estimates noise, and the residual subtraction
# artefacts ride at a constant level instead of being amplified per-turn.
# ---------------------------------------------------------------------------


_DENOISE_NF_BOUNDS_DB = (-80.0, 0.0)
_DENOISE_NR_BOUNDS_DB = (0.0, 97.0)  # ffmpeg afftdn caps at 97 dB


def _apply_denoise(
    wav_path: Path,
    *,
    noise_floor_db: float,
    reduction_db: float,
    log: Callable[[str], None],
) -> tuple[Path, dict]:
    src = Path(wav_path)
    dst = src.with_suffix(".denoise.wav")

    _validate_denoise_params(noise_floor_db, reduction_db)

    # We re-decode through ffmpeg so it can compute the FFT-domain subtraction
    # itself. The 16 kHz mono PCM_16 invariant is enforced on the output side.
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-i", str(src),
        "-af", f"afftdn=nf={noise_floor_db}:nr={reduction_db}",
        "-ac", "1", "-ar", str(SR),
        "-c:a", "pcm_s16le",
        str(dst),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise ClearspeechError("ffmpeg not found in PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise ClearspeechError(
            f"ffmpeg afftdn failed: {exc.stderr.strip()}"
        ) from exc

    log(
        f"clearspeech.denoise: "
        f"nf={noise_floor_db:.1f} dB, nr={reduction_db:.1f} dB → {dst.name}"
    )
    params = {
        "noise_floor_db": float(noise_floor_db),
        "reduction_db": float(reduction_db),
    }
    return dst, {"params": params, "stats": {}}


def _validate_denoise_params(
    noise_floor_db: float,
    reduction_db: float,
) -> None:
    nflo, nfhi = _DENOISE_NF_BOUNDS_DB
    if not (nflo <= noise_floor_db <= nfhi):
        raise ClearspeechError(
            f"invalid denoise noise_floor_db ({noise_floor_db}): "
            f"must be within [{nflo}, {nfhi}] dB"
        )
    nrlo, nrhi = _DENOISE_NR_BOUNDS_DB
    if not (nrlo <= reduction_db <= nrhi):
        raise ClearspeechError(
            f"invalid denoise reduction_db ({reduction_db}): "
            f"must be within [{nrlo}, {nrhi}] dB"
        )


# ---------------------------------------------------------------------------
# Effect: dereverb (per-pyannote-turn Lebart-Polack spectral subtraction).
# Estimates RT60 from each turn's energy decay (Schroeder backward
# integration), then subtracts the predicted late-reverberation power from
# the STFT magnitude before iSTFT-ing back to time domain.
#
# Reference: K. Lebart, J.M. Boucher, P.N. Denbigh, "A new method based on
# spectral subtraction for speech dereverberation," Acta Acustica, 2001.
# ---------------------------------------------------------------------------


_DEREVERB_RT60_FLOOR_MS_BOUNDS = (50.0, 2000.0)
_DEREVERB_SUBTRACT_BOUNDS = (0.0, 1.0)
_DEREVERB_CROSSFADE_MS_BOUNDS = (0.0, 500.0)
_DEREVERB_MIN_TURN_S = 0.2  # turns shorter than 200 ms are left untouched
_DEREVERB_FALLBACK_RT60_S = 0.4  # used when the estimator can't fit a slope
_DEREVERB_STFT_NPERSEG = 512
_DEREVERB_STFT_NOVERLAP = 256
_DEREVERB_POWER_FLOOR = 0.1  # keep at least 10% of the original spectrum power


def _apply_dereverb(
    wav_path: Path,
    *,
    turns: list[DiarTurn],
    rt60_floor_ms: float,
    subtract_factor: float,
    crossfade_ms: float,
    log: Callable[[str], None],
) -> tuple[Path, dict]:
    src = Path(wav_path)
    dst = src.with_suffix(".dereverb.wav")
    params = {
        "rt60_floor_ms": float(rt60_floor_ms),
        "subtract_factor": float(subtract_factor),
        "crossfade_ms": float(crossfade_ms),
    }

    _validate_dereverb_params(rt60_floor_ms, subtract_factor, crossfade_ms)

    if not turns:
        shutil.copyfile(src, dst)
        return dst, {"params": params, "stats": {"turn_count": 0}}

    samples, sr = sf.read(src, dtype="float32", always_2d=False)
    if sr != SR:
        raise ClearspeechError(
            f"expected {SR} Hz sample rate, got {sr} Hz: {src}"
        )
    if samples.ndim != 1:
        raise ClearspeechError(
            f"expected mono input, got {samples.ndim}-D array: {src}"
        )

    n = int(samples.shape[0])
    out = samples.copy()
    rt60_estimates_s: list[float] = []
    fade_samples = max(1, int(round(SR * crossfade_ms / 1000)))
    min_seg_samples = int(_DEREVERB_MIN_TURN_S * SR)
    rt60_floor_s = rt60_floor_ms / 1000.0

    for turn in sorted(turns, key=lambda t: t.start):
        s = max(0, int(round(turn.start * SR)))
        e = min(n, int(round(turn.end * SR)))
        if e - s < min_seg_samples:
            continue

        seg = samples[s:e]
        estimated = _estimate_rt60(seg)
        rt60_s = max(estimated if estimated is not None else _DEREVERB_FALLBACK_RT60_S,
                      rt60_floor_s)
        rt60_estimates_s.append(rt60_s)

        processed = _lebart_polack_subtract(seg, rt60_s, subtract_factor)
        _blend_into(out, s, e, processed, fade_samples)

    np.clip(out, -1.0, 1.0 - 1e-7, out=out)
    sf.write(dst, out, SR, subtype="PCM_16")

    rt60_avg = float(np.mean(rt60_estimates_s)) if rt60_estimates_s else 0.0
    log(
        f"clearspeech.dereverb: {len(rt60_estimates_s)} turn(s) | "
        f"avg RT60 {rt60_avg*1000:.0f} ms | "
        f"floor {rt60_floor_ms:.0f} ms, subtract={subtract_factor:.2f} → {dst.name}"
    )
    stats = {
        "turn_count": len(rt60_estimates_s),
        "rt60_avg_ms": round(rt60_avg * 1000, 1),
        "rt60_estimates_ms": [round(x * 1000, 1) for x in rt60_estimates_s],
    }
    return dst, {"params": params, "stats": stats}


def _estimate_rt60(seg: np.ndarray) -> float | None:
    """Schroeder backward-integration RT60 estimator on a single segment.

    Returns None if the segment is too short, the decay slope is unreliable,
    or the resulting RT60 falls outside a physically plausible range.
    """
    if seg.size < int(_DEREVERB_MIN_TURN_S * SR):
        return None

    # Bandpass 500-2000 Hz — the voice-band range standard for room-decay
    # measurements (lower frequencies are dominated by speech fundamental,
    # higher by fricative noise, neither tracks the room reverb cleanly).
    nyquist = SR / 2
    sos = butter(4, [500.0 / nyquist, 2000.0 / nyquist], btype="band", output="sos")
    band = sosfilt(sos, seg)
    env = np.abs(hilbert(band))

    # Schroeder backward integration → energy decay curve in dB, normalised to 0.
    edc = np.cumsum(env[::-1] ** 2)[::-1]
    if edc[0] <= 0:
        return None
    edc_db = 10.0 * np.log10(edc / edc[0] + 1e-12)

    # Fit a line on the -5..-25 dB region — the canonical RT60 working range.
    mask = (edc_db < -5.0) & (edc_db > -25.0)
    if mask.sum() < 100:
        return None
    t = np.arange(edc_db.size, dtype=np.float64) / SR
    slope, _ = np.polyfit(t[mask], edc_db[mask], 1)
    if slope >= 0.0:
        return None  # decay went the wrong way — segment is noise-dominated

    rt60_s = -60.0 / slope
    if not (0.05 <= rt60_s <= 2.0):
        return None  # outside plausible bounds for a phone-recorded room
    return float(rt60_s)


def _lebart_polack_subtract(
    seg: np.ndarray, rt60_s: float, subtract_factor: float
) -> np.ndarray:
    """Subtract predicted late-reverberation power from the STFT magnitude.

    `reverb_power[t, f] = decay * (reverb_power[t-1, f] + |X[t-1, f]|^2)`
    is the per-bin IIR running estimate of late reverberation energy at
    frame `t`; `decay` corresponds to the per-frame fraction of the RT60
    energy that survives into the next frame.
    """
    nperseg = _DEREVERB_STFT_NPERSEG
    noverlap = _DEREVERB_STFT_NOVERLAP
    hop = nperseg - noverlap
    frame_period_s = hop / SR
    # 60 dB drop over rt60_s → per-frame power factor.
    decay = 10.0 ** (-60.0 * frame_period_s / (rt60_s * 10.0))

    _, _, spec = stft(seg, fs=SR, nperseg=nperseg, noverlap=noverlap, window="hann")
    power = np.abs(spec) ** 2
    reverb = np.zeros_like(power)
    for k in range(1, power.shape[1]):
        reverb[:, k] = decay * (reverb[:, k - 1] + power[:, k - 1])

    clean_power = np.maximum(
        power - subtract_factor * reverb,
        _DEREVERB_POWER_FLOOR * power,
    )
    clean_mag = np.sqrt(clean_power)
    spec_clean = clean_mag * np.exp(1j * np.angle(spec))
    _, out = istft(spec_clean, fs=SR, nperseg=nperseg, noverlap=noverlap, window="hann")
    out = np.asarray(out, dtype=np.float32)
    # iSTFT may yield slightly longer output than the input — trim to match.
    if out.size > seg.size:
        out = out[: seg.size]
    elif out.size < seg.size:
        out = np.concatenate([out, np.zeros(seg.size - out.size, dtype=np.float32)])
    return out


def _blend_into(
    out: np.ndarray, s: int, e: int, processed: np.ndarray, fade_samples: int
) -> None:
    """Write `processed` into `out[s:e]` with linear crossfades on both edges.

    The crossfade is clamped per-edge so it never spills past the midpoint
    of the surrounding region; ensures adjacent turns can meet but never
    overlap their fades.
    """
    seg_len = e - s
    if seg_len <= 0 or processed.size == 0:
        return
    half_fade = min(fade_samples // 2, seg_len // 2)
    if half_fade < 1:
        out[s:e] = processed[:seg_len]
        return
    fade_in = np.linspace(0.0, 1.0, half_fade, dtype=np.float32)
    fade_out = np.linspace(1.0, 0.0, half_fade, dtype=np.float32)
    out[s:s + half_fade] = (
        out[s:s + half_fade] * fade_out + processed[:half_fade] * fade_in
    )
    out[s + half_fade:e - half_fade] = processed[half_fade:seg_len - half_fade]
    out[e - half_fade:e] = (
        out[e - half_fade:e] * fade_in + processed[seg_len - half_fade:seg_len] * fade_out
    )


def _validate_dereverb_params(
    rt60_floor_ms: float, subtract_factor: float, crossfade_ms: float
) -> None:
    flo, fhi = _DEREVERB_RT60_FLOOR_MS_BOUNDS
    if not (flo <= rt60_floor_ms <= fhi):
        raise ClearspeechError(
            f"invalid dereverb rt60_floor_ms ({rt60_floor_ms}): "
            f"must be within [{flo}, {fhi}] ms"
        )
    slo, shi = _DEREVERB_SUBTRACT_BOUNDS
    if not (slo <= subtract_factor <= shi):
        raise ClearspeechError(
            f"invalid dereverb subtract_factor ({subtract_factor}): "
            f"must be within [{slo}, {shi}]"
        )
    clo, chi = _DEREVERB_CROSSFADE_MS_BOUNDS
    if not (clo <= crossfade_ms <= chi):
        raise ClearspeechError(
            f"invalid dereverb crossfade_ms ({crossfade_ms}): "
            f"must be within [{clo}, {chi}] ms"
        )
