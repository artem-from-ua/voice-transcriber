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

Why no chunking/temperature/context kwargs in the public signature: mlx-whisper
chunks internally (30-second windows are baked into the Whisper architecture)
and the cookbook decode-loop already retries with a temperature schedule when
a window fails its compression / logprob thresholds. The settings copied here
mirror the stone-scriber project, which has been used in production on
Ukrainian recordings — see ADR 0017.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from ._memory import free_mlx
from .types import AsrSegment


WHISPER_REPO_ID = "mlx-community/whisper-large-v3-mlx"


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
    """
    # Lazy import: mlx_whisper pulls in numba/tiktoken/llvmlite, ~1 s warmup
    # we do not want to pay on every import of this module (e.g. in tests
    # that don't exercise the Whisper branch).
    import mlx_whisper

    if model is None:
        _verify_model_cached()
        log(f"Whisper-large-v3-MLX from HF cache ({WHISPER_REPO_ID})")
        path_or_model: Any = WHISPER_REPO_ID
    else:
        path_or_model = model

    t0 = time.perf_counter()
    result = mlx_whisper.transcribe(
        str(wav_path),
        path_or_hf_repo=path_or_model,
        language=language,
        condition_on_previous_text=False,
    )
    log(f"ASR finished in {time.perf_counter() - t0:.1f}s.")

    segments = _parse_segments(result)
    # Drop the Whisper weights and KV cache before downstream LLM stages —
    # on 16 GB Macs they otherwise sit alongside the LLM and trigger a
    # Metal OOM at structure_dialog.
    del result
    free_mlx(log)
    return segments
