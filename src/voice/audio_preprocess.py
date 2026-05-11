"""Per-segment loudness normalization (AGC) bounded by pyannote turns.

The pipeline runs diarisation before this stage so the pyannote turns can
serve as guard-rails for the gain envelope: gain is computed per turn (one
RMS reading → one factor), so amplification never leaks across speaker
boundaries. Gaps between turns stay at unity gain — silence is never
amplified into hiss.

Boundaries are smoothed with linear gain ramps (linear is artefact-free
when both ramp ends multiply the *same* signal — there is no decorrelation
that would call for half-cosine or equal-power shapes).

Hard input invariant: 16 kHz mono PCM_16 WAV (the format the pipeline's
ffmpeg stage produces).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

from .types import DiarTurn


SR = 16_000  # pipeline invariant: ffmpeg always produces 16 kHz mono PCM_16
SILENT_RMS_THRESHOLD = 1e-6


class AudioPreprocessError(RuntimeError):
    """Raised when the input WAV does not match the pipeline invariants."""


def _noop_log(_msg: str) -> None:
    return None


def loudness_normalize(
    wav_path: str | Path,
    diarize_segments: list[DiarTurn],
    *,
    target_dbfs: float = -20.0,
    max_gain_db: float = 16.0,
    crossfade_ms: float = 100.0,
    log: Callable[[str], None] = _noop_log,
) -> Path:
    """Per-segment AGC bounded by pyannote turns.

    Reads `wav_path` (16 kHz mono PCM_16), computes one gain per turn
    (target_dbfs - rms_dbfs, clamped above by max_gain_db; no floor —
    attenuation is allowed unbounded), applies it sample-wise with linear
    crossfades of `crossfade_ms` at every boundary. Gaps between turns are
    left at unity gain so silence is never amplified. Writes a sibling
    `<stem>.normalized.wav` and returns the path.

    Empty `diarize_segments` → byte-for-byte copy (shutil.copyfile), no
    decode/encode round-trip.

    Raises AudioPreprocessError if the input is not 16 kHz mono.
    """
    src = Path(wav_path)
    dst = src.with_suffix(".normalized.wav")

    if not diarize_segments:
        shutil.copyfile(src, dst)
        return dst

    samples, sr = sf.read(src, dtype="float32", always_2d=False)
    if sr != SR:
        raise AudioPreprocessError(
            f"expected {SR} Hz sample rate, got {sr} Hz: {src}"
        )
    if samples.ndim != 1:
        raise AudioPreprocessError(
            f"expected mono input, got {samples.ndim}-D array: {src}"
        )

    n = int(samples.shape[0])
    target_amp = 10 ** (target_dbfs / 20)
    max_gain = 10 ** (max_gain_db / 20)
    fade_samples = max(1, int(round(SR * crossfade_ms / 1000)))

    raw_regions: list[tuple[int, int, float]] = []
    pre_rms_db: list[float] = []  # for the post-run quality stats
    for turn in sorted(diarize_segments, key=lambda t: t.start):
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
        return dst

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
        f"loudness_normalize: {len(raw_regions)} turn(s) | "
        f"per-turn RMS std-dev {pre_std:.2f} → {post_std:.2f} dB "
        f"({spread_drop:+.1f}%) | "
        f"{ceiling_hits}/{len(raw_regions)} hit +{max_gain_db:.0f} dB ceiling | "
        f"target={target_dbfs:+.1f} dBFS, crossfade={crossfade_ms:.0f} ms "
        f"→ {dst.name}"
    )
    return dst


def _fill_gaps(
    raw_regions: list[tuple[int, int, float]], total: int
) -> list[tuple[int, int, float]]:
    """Tile `[0, total)` with regions; insert unity-gain regions for gaps.

    `raw_regions` must already be sorted by start. Overlapping regions
    (which pyannote's exclusive timeline shouldn't produce, but defensive
    coding is cheap) are clipped to the previous region's end.
    """
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
