"""Whisper-large-v3-MLX ASR backend.

A second `transcribe()` callable with the same signature shape as
`speech2text.transcribe()` so `pipeline.run()` can route between them based
on `PipelineOptions.asr_engine`.

The model is read out of the HuggingFace cache that `huggingface_hub`
populates — not the LM Studio cache used for VibeVoice — because mlx-whisper
expects the standard HF layout (blobs/refs/snapshots) and resolves repo ids
through it directly. The model is never auto-downloaded inside `transcribe`;
the user runs `voice download-whisper` once at onboarding and we hard-fail
with an actionable error if the cache is empty.

Why no chunking/temperature/context kwargs in the public signature: mlx-whisper
chunks internally (30-second windows are baked into the Whisper architecture)
and the cookbook decode-loop already retries with a temperature schedule when
a window fails its compression / logprob thresholds. The settings copied here
mirror the stone-scriber project, which has been used in production on
Ukrainian recordings — see ADR 0017.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from ._memory import free_mlx
from .speech2text import AsrError
from .types import AsrSegment


WHISPER_REPO_ID = "mlx-community/whisper-large-v3-mlx"


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

    `speaker_asr` is left as None — Whisper has no speaker-prediction head,
    and `merge.py` does not need it (max-overlap with pyannote turns is the
    sole source of speaker labels downstream).
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
            speaker_asr=None,
        ))
    return segments


def transcribe(
    wav_path: str | Path,
    *,
    language: str = "uk",
    log: Callable[[str], None] = _noop_log,
) -> list[AsrSegment]:
    """Run Whisper-large-v3-MLX on a WAV. Returns a list of AsrSegment."""
    _verify_model_cached()
    log(f"Whisper-large-v3-MLX from HF cache ({WHISPER_REPO_ID})")

    # Lazy import: mlx_whisper pulls in numba/tiktoken/llvmlite, ~1 s warmup
    # we do not want to pay on every import of this module (e.g. in tests
    # that don't exercise the Whisper branch).
    import mlx_whisper

    import time
    t0 = time.time()
    result = mlx_whisper.transcribe(
        str(wav_path),
        path_or_hf_repo=WHISPER_REPO_ID,
        language=language,
        condition_on_previous_text=False,
    )
    log(f"ASR finished in {time.time() - t0:.1f}s.")

    segments = _parse_segments(result)
    # Drop the Whisper weights and KV cache before downstream LLM stages —
    # on 16 GB Macs they otherwise sit alongside gemma and trigger a Metal
    # OOM at structure_dialog (mirrors speech2text.py:149–152).
    del result
    free_mlx(log)
    return segments
