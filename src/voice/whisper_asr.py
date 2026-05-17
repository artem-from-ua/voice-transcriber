"""Whisper-large-v3-MLX ASR backend.

`transcribe()` runs `mlx-whisper` in-process against
`mlx-community/whisper-large-v3-mlx`. This is the only ASR backend in the
project since v0.23.0 — see ADR 0017 (how it was chosen) and ADR 0021 (why
the legacy VibeVoice backend was removed).

The model is read out of the HuggingFace cache that `huggingface_hub`
populates, because `mlx-whisper` expects the standard HF layout
(blobs/refs/snapshots) and resolves repo ids through it directly. The
model is never auto-downloaded inside `transcribe`; the user runs
`voice download-whisper` once at onboarding and we hard-fail with an
actionable error if the cache is empty.

For long recordings the wrapper slices the audio in Python and calls
`mlx_whisper.transcribe` once per slice with a fresh MLX cache between
calls — see ADR 0031 / issue #147. Whisper's own 30-second windows live
*inside* a transcribe() call; the slicing here is a layer above that,
bounding the per-call peak so an hour-long recording does not trip the
wired-memory ceiling on a 16 GB Mac.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from ._memory import free_mlx
from .types import AsrSegment


WHISPER_REPO_ID = "mlx-community/whisper-large-v3-mlx"

# ASR chunking knobs — see ADR 0031.
#
# On the M1/16 GB reference machine a 6-min input produces a 7.17 GB MLX
# peak with the model resident (~3 GB) + per-call working set scaling with
# audio duration. Empirical baseline:
#     6 min   -> peak 7.17 GB
#     ~10 min -> headroom still safe alongside other resident artefacts
#     48 min  -> SIGKILL during the call (predates peak instrumentation)
#
# 10 min chosen as a conservative single-pass ceiling. 8 min per chunk
# keeps each call comfortably inside the working envelope; 5 s overlap
# prevents word-level truncation across chunk boundaries (Whisper's own
# 30 s windows live *inside* one transcribe() call, so chunks are not
# aligned to them — we rely on overlap-then-dedup to absorb any drift).
ASR_CHUNK_THRESHOLD_S = 600.0
ASR_CHUNK_SIZE_S = 480.0
ASR_CHUNK_OVERLAP_S = 5.0


class AsrError(RuntimeError):
    pass


def _noop_log(_msg: str) -> None:
    return None


def _verify_model_cached(repo_id: str = WHISPER_REPO_ID) -> None:
    """Raise AsrError if the Whisper model is not present in the HF cache.

    Uses `try_to_load_from_cache` which is the cheap, no-network lookup —
    `snapshot_download` would silently start a 3 GB fetch.
    """
    from huggingface_hub import try_to_load_from_cache
    from huggingface_hub.errors import CacheNotFound

    try:
        hit = try_to_load_from_cache(repo_id=repo_id, filename="config.json")
    except CacheNotFound:
        hit = None
    if hit is None:
        raise AsrError(
            f"Whisper model {repo_id!r} is not in the HuggingFace cache. "
            f"Run `voice download-whisper` once to fetch it (~3 GB)."
        )


def _parse_segments(result: dict) -> list[AsrSegment]:
    """Convert mlx-whisper's `{text, segments, language}` payload to AsrSegment[].

    Whisper has no speaker-prediction head; speaker labels are assigned
    downstream by `merge.py` via max-overlap with pyannote turns.
    """
    segments: list[AsrSegment] = []
    for seg in result.get("segments", []):
        content = (seg.get("text") or "").strip()
        if not content:
            continue
        segments.append(AsrSegment(
            start=float(seg.get("start", 0.0)),
            end=float(seg.get("end", 0.0)),
            content=content,
        ))
    return segments


def load_model(
    log: Callable[[str], None] = _noop_log,
) -> tuple[Any, float]:
    """Load Whisper-large-v3-MLX weights and return (model, load_elapsed_s).

    Separated from transcribe() so the pipeline can measure model load time
    independently from inference time and report both in the Markdown header.
    """
    _verify_model_cached()
    log(f"Whisper-large-v3-MLX from HF cache ({WHISPER_REPO_ID})")

    from mlx_whisper.load_models import load_model as _mlx_load_model

    t0 = time.perf_counter()
    model = _mlx_load_model(WHISPER_REPO_ID)
    load_elapsed = time.perf_counter() - t0
    log(f"Whisper loaded in {load_elapsed:.1f}s.")
    return model, load_elapsed


def _call_mlx_whisper(
    audio_input: Any,
    *,
    language: str | None,
    label: str,
    log: Callable[[str], None],
) -> tuple[dict, float, float]:
    """One mlx_whisper.transcribe call wrapped in MLX peak instrumentation.
    Returns (result_dict, mlx_peak_gb, elapsed_s)."""
    import mlx.core as mx
    import mlx_whisper

    mx.clear_cache()
    mx.reset_peak_memory()
    before_gb = mx.get_active_memory() / 1e9
    log(f"whisper_asr: {label} pre-transcribe mlx_active={before_gb:.2f}GB")

    t0 = time.perf_counter()
    result = mlx_whisper.transcribe(
        audio_input,
        path_or_hf_repo=WHISPER_REPO_ID,
        language=language,
        condition_on_previous_text=False,
    )
    elapsed = time.perf_counter() - t0
    peak_gb = mx.get_peak_memory() / 1e9
    log(
        f"whisper_asr: {label} post-transcribe "
        f"mlx_peak={peak_gb:.2f}GB elapsed={elapsed:.1f}s"
    )
    return result, peak_gb, elapsed


def _shift_segments(segments: list[AsrSegment], offset_s: float) -> list[AsrSegment]:
    if offset_s == 0.0:
        return segments
    return [
        AsrSegment(start=s.start + offset_s, end=s.end + offset_s, content=s.content)
        for s in segments
    ]


def _dedup_overlap(
    accumulated: list[AsrSegment],
    incoming: list[AsrSegment],
    *,
    chunk_start_s: float,
    overlap_s: float,
) -> list[AsrSegment]:
    """Drop segments from `incoming` whose start lies inside the overlap
    window with the previous chunk. Whisper's per-window output can repeat
    a phrase across the boundary; trimming the leading slice of the second
    chunk by `overlap_s` removes those duplicates cleanly without trying
    to match text — segments earlier than the overlap zone are kept as-is."""
    if not accumulated or overlap_s <= 0.0:
        return incoming
    cutoff = chunk_start_s + overlap_s
    return [s for s in incoming if s.start >= cutoff]


def transcribe(
    wav_path: str | Path,
    *,
    language: str | None = None,
    model: Any = None,
    log: Callable[[str], None] = _noop_log,
) -> list[AsrSegment]:
    """Run Whisper-large-v3-MLX on a WAV. Returns a list of AsrSegment.

    `language=None` lets mlx-whisper auto-detect on the first 30 s window.
    The pipeline normally precedes this with [4] lang_detect (ADR 0022),
    which provides a stronger detection signal from the longest pyannote
    turn, but direct callers that skip the diarize step can still get a
    best-effort transcript by passing None.

    When `model` is provided (pre-loaded via `load_model()`), weights are
    reused — mlx_whisper.transcribe accepts a model object directly.
    When `model` is None, mlx_whisper loads weights internally (backwards-
    compatible path for callers that skip the separate load step).

    For recordings longer than ASR_CHUNK_THRESHOLD_S the audio is sliced
    in Python and Whisper is called once per slice with a fresh MLX cache
    between calls. See ADR 0031 / issue #147.
    """
    # Lazy import: mlx_whisper pulls in numba/tiktoken/llvmlite, ~1 s warmup
    # we do not want to pay on every import of this module (e.g. in tests
    # that don't exercise the Whisper branch).
    import mlx_whisper

    if model is None:
        _verify_model_cached()
        log(f"Whisper-large-v3-MLX from HF cache ({WHISPER_REPO_ID})")

    # Probe audio duration with the same loader Whisper uses, so we make
    # the chunk/single-pass decision off the exact sample count Whisper
    # will see (no ffprobe round-trip, no extra file I/O).
    from mlx_whisper.audio import SAMPLE_RATE, load_audio

    audio = load_audio(str(wav_path))
    duration_s = len(audio) / SAMPLE_RATE

    if duration_s <= ASR_CHUNK_THRESHOLD_S:
        # Short audio: pass the path through verbatim (preserves existing
        # behaviour and lets mlx_whisper handle file I/O itself).
        del audio
        result, peak_gb, elapsed = _call_mlx_whisper(
            str(wav_path),
            language=language, label="single-pass",
            log=log,
        )
        log(f"ASR finished in {elapsed:.1f}s.")
        segments = _parse_segments(result)
        del result
        free_mlx(log)
        return segments

    # Long audio: slice in Python, call Whisper per slice, stitch.
    chunk_samples = int(ASR_CHUNK_SIZE_S * SAMPLE_RATE)
    overlap_samples = int(ASR_CHUNK_OVERLAP_S * SAMPLE_RATE)
    step = chunk_samples - overlap_samples
    n_chunks = (len(audio) + step - 1) // step
    log(
        f"whisper_asr: long audio ({duration_s:.0f}s) -> "
        f"{n_chunks} chunks of {ASR_CHUNK_SIZE_S:.0f}s "
        f"with {ASR_CHUNK_OVERLAP_S:.0f}s overlap"
    )

    segments: list[AsrSegment] = []
    overall_t0 = time.perf_counter()
    for chunk_idx in range(n_chunks):
        start_sample = chunk_idx * step
        end_sample = min(start_sample + chunk_samples, len(audio))
        if start_sample >= len(audio):
            break
        chunk_audio = audio[start_sample:end_sample]
        chunk_start_s = start_sample / SAMPLE_RATE
        label = f"chunk {chunk_idx + 1}/{n_chunks} @{chunk_start_s:.0f}s"

        result, _peak_gb, _elapsed = _call_mlx_whisper(
            chunk_audio,
            language=language, label=label,
            log=log,
        )
        chunk_segments = _shift_segments(_parse_segments(result), chunk_start_s)
        chunk_segments = _dedup_overlap(
            segments, chunk_segments,
            chunk_start_s=chunk_start_s, overlap_s=ASR_CHUNK_OVERLAP_S,
        )
        segments.extend(chunk_segments)
        del result

    del audio
    log(f"ASR finished in {time.perf_counter() - overall_t0:.1f}s.")
    # Drop the Whisper weights and KV cache before downstream LLM stages —
    # on 16 GB Macs they otherwise sit alongside the LLM and trigger a
    # Metal OOM at structure_dialog.
    free_mlx(log)
    return segments
