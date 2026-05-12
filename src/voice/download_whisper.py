"""Onboarding helper that fetches the Whisper-MLX model into the HF cache.

Run via `voice download-whisper`. Idempotent — re-running with the same
repo_id just re-verifies the existing cache entry and exits quickly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from .whisper_asr import WHISPER_REPO_ID


def _noop_log(_msg: str) -> None:
    return None


def download(
    repo_id: str = WHISPER_REPO_ID,
    *,
    log: Callable[[str], None] = _noop_log,
) -> Path:
    """Download or verify a HuggingFace repo into the local HF cache."""
    from huggingface_hub import snapshot_download

    log(f"Downloading {repo_id} into the HuggingFace cache (~3 GB, 5–10 min on first run)...")
    path = snapshot_download(repo_id=repo_id)
    log(f"Done: {path}")
    return Path(path)
