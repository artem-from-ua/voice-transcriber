"""Audio cleanup chain between diarize and ASR.

This module replaces the older `voice.audio_preprocess` (which only did
per-turn AGC). The chain-of-effects design lets future PRs add denoise /
presence boost / de-esser as ordered links without growing the pipeline-stage
count ([N/10] stays constant) or rewriting dump layout per release.

PR-1 chain effects (issue #48):
  - "agc"      — per-pyannote-turn RMS normalization (was loudness_normalize)
  - "bandpass" — Butterworth IIR bandpass via scipy.signal.sosfiltfilt (E2a)

PR-1 enforces the canonical `agc → bandpass` order; free-order arrives in PR-2
along with the presence-boost effect, when reordering becomes meaningful.

Hard input invariant: 16 kHz mono PCM_16 WAV — the format that the upstream
ffmpeg stage produces. The 16 kHz Nyquist of 8 kHz caps `bandpass_high_hz`
just below 8000 (default 7900); issue #48's nominal 10 kHz upper bound is
unreachable at this sample rate and would require resampling first.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

from .types import DiarTurn


SR = 16_000
BANDPASS_ORDER = 4  # Butterworth IIR; 4 ≈ 24 dB/octave, doubled by sosfiltfilt
SILENT_RMS_THRESHOLD = 1e-6

# PR-1 only allows this canonical order. PR-2 will replace this gate with a
# free-order parser once "presence" lands and reordering becomes meaningful.
_PR1_ALLOWED_CHAINS: frozenset[tuple[str, ...]] = frozenset({
    (),
    ("agc",),
    ("bandpass",),
    ("agc", "bandpass"),
})

ClearspeechDumpHook = Callable[[int, str, Path], None]


class ClearspeechError(RuntimeError):
    """Raised when the input WAV does not match the pipeline invariants
    or when an effect's parameters are out of range."""


def _noop_log(_msg: str) -> None:
    return None


def clearspeech(
    wav_path: str | Path,
    chain: tuple[str, ...],
    *,
    agc_turns: list[DiarTurn] | None = None,
    agc_target_dbfs: float = -20.0,
    agc_max_gain_db: float = 16.0,
    agc_crossfade_ms: float = 100.0,
    bandpass_low_hz: float = 80.0,
    bandpass_high_hz: float = 7_900.0,
    log: Callable[[str], None] = _noop_log,
    dump: ClearspeechDumpHook | None = None,
) -> tuple[Path, dict]:
    """Run the requested chain of DSP effects against `wav_path`.

    Each enabled effect writes a sibling WAV (`<stem>.agc.wav`,
    `<stem>.agc.bandpass.wav`, …) and the next effect reads it. `dump`, if
    provided, is invoked with `(step_index, effect_name, intermediate_path)`
    after each effect so the pipeline can mirror intermediates into the
    `--dump-stages` directory.

    Returns `(final_path, config)`:
      - `final_path` — output of the last applied effect, or `wav_path` itself
        when `chain` is empty (no decode/encode round-trip).
      - `config` — a JSON-serialisable dict capturing chain, params and per-
        step stats, suitable for `02b-clearspeech-config.json`.

    Raises ClearspeechError on bad inputs (wrong SR, non-mono, missing
    `agc_turns` when "agc" is requested, invalid bandpass cutoffs).
    Raises ValueError when `chain` is outside the PR-1 allowed set.
    """
    chain = tuple(chain)
    if chain not in _PR1_ALLOWED_CHAINS:
        allowed = sorted(_PR1_ALLOWED_CHAINS, key=lambda c: (len(c), c))
        raise ValueError(
            f"PR-1 supports only canonical agc→bandpass order; got {chain!r}. "
            f"Allowed chains: {allowed}"
        )

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
        if effect == "agc":
            if agc_turns is None:
                raise ClearspeechError(
                    "chain contains 'agc' but agc_turns is None — pass the "
                    "pyannote turns the AGC needs as per-segment guard-rails"
                )
            current, step_info = _apply_agc(
                current,
                turns=agc_turns,
                target_dbfs=agc_target_dbfs,
                max_gain_db=agc_max_gain_db,
                crossfade_ms=agc_crossfade_ms,
                log=log,
            )
        elif effect == "bandpass":
            current, step_info = _apply_bandpass(
                current,
                low_hz=bandpass_low_hz,
                high_hz=bandpass_high_hz,
                log=log,
            )
        else:
            # Defensive — the chain validator above should make this
            # unreachable, but the dispatch is explicit so the failure mode is
            # clear if the validator is ever loosened.
            raise ValueError(f"unknown effect in chain: {effect!r}")

        config["steps"].append({"name": effect, "applied": True, **step_info})
        if dump is not None:
            dump(idx, effect, current)

    return current, config


# ---------------------------------------------------------------------------
# Effect: AGC (per-turn RMS normalization). Logic transplanted byte-for-byte
# from the v0.13.0 `audio_preprocess.loudness_normalize`; the only changes are
# the output suffix (`.agc.wav` instead of `.normalized.wav`) and returning a
# stats dict alongside the path so the chain's config can record per-step
# metrics.
# ---------------------------------------------------------------------------


def _apply_agc(
    wav_path: Path,
    *,
    turns: list[DiarTurn],
    target_dbfs: float,
    max_gain_db: float,
    crossfade_ms: float,
    log: Callable[[str], None],
) -> tuple[Path, dict]:
    src = Path(wav_path)
    dst = src.with_suffix(".agc.wav")
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
        f"clearspeech.agc: {len(raw_regions)} turn(s) | "
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
