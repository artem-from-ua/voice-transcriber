"""pyannote 3.1 speaker diarization wrapper.

Returns `exclusive_diarization` (no overlapping turns) — that's what
`merge.py` consumes downstream.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import torch
from pyannote.audio import Pipeline

from .types import DiarTurn


class DiarizationError(RuntimeError):
    pass


def _read_hf_token() -> str:
    token_path = Path("~/.cache/huggingface/token").expanduser()
    if not token_path.is_file():
        raise DiarizationError(
            f"HF token not found at {token_path}. "
            "Create it: echo 'hf_xxx' > ~/.cache/huggingface/token && chmod 600 ~/.cache/huggingface/token"
        )
    return token_path.read_text().strip()


def diarize(
    wav_path: str | Path,
    *,
    num_speakers: int | None = None,
    log: Callable[[str], None] = print,
) -> list[DiarTurn]:
    """Run pyannote diarization. Returns the exclusive (no-overlap) timeline."""
    token = _read_hf_token()

    log("Loading pyannote pipeline...")
    t0 = time.time()
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=token)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    try:
        pipeline.to(device)
    except Exception as exc:  # noqa: BLE001 — defensive fallback
        log(f"Could not move pipeline to {device}: {exc}; falling back to CPU")
        pipeline.to(torch.device("cpu"))
    log(f"Pipeline loaded in {time.time() - t0:.1f}s on {device}.")

    t1 = time.time()
    kwargs = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers
    result = pipeline(str(wav_path), **kwargs)
    log(f"Diarization done in {time.time() - t1:.1f}s.")

    payload = result.serialize()
    return [
        DiarTurn(start=t["start"], end=t["end"], speaker=t["speaker"])
        for t in payload["exclusive_diarization"]
    ]
