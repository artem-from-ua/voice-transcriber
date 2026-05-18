"""pyannote 3.1 speaker diarization wrapper.

Returns `exclusive_diarization` (no overlapping turns) — that's what
`merge.py` consumes downstream.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable

import torch
from pyannote.audio import Pipeline

from ._memory import free_torch_mps
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


def _build_progress_hook(
    set_state: Callable[..., None],
) -> Callable[..., None]:
    """Adapter for pyannote's hook signature `(step_name, step_artifact,
    file=None, total=None, completed=None)`. Forwards each step update
    to the caller's structured setter as `(step, completed, total)`.

    Do NOT use pyannote.audio.pipelines.utils.hook.ProgressHook directly
    — it creates its own rich.Progress, which conflicts with the live
    display our ProgressReporter already owns.
    """
    def hook(
        step_name: str,
        step_artifact,
        file=None,
        total: int | None = None,
        completed: int | None = None,
    ) -> None:
        set_state(step_name, completed, total)
    return hook


def diarize(
    wav_path: str | Path,
    *,
    num_speakers: int | None = None,
    log: Callable[[str], None] = print,
    progress_state: Callable[..., None] | None = None,
) -> tuple[list[DiarTurn], float]:
    """Run pyannote diarization. Returns (turns, model_load_elapsed_s).

    If `progress_state` is supplied, it is called as
    `progress_state(step_name, completed, total)` on every pyannote step
    update (segmentation → embeddings → clustering). Otherwise behaviour
    is byte-identical to callers that omit the kwarg.
    """
    token = _read_hf_token()

    log("Loading pyannote pipeline...")
    t0 = time.perf_counter()
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=token)
    # VOICE_DIARIZE_DEVICE forces "cpu" or "mps" for benchmarking; unset = auto.
    override = os.environ.get("VOICE_DIARIZE_DEVICE", "").strip().lower()
    if override in ("cpu", "mps"):
        device = torch.device(override)
    else:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    try:
        pipeline.to(device)
    except Exception as exc:  # noqa: BLE001 — defensive fallback
        log(f"Could not move pipeline to {device}: {exc}; falling back to CPU")
        pipeline.to(torch.device("cpu"))
    load_elapsed = time.perf_counter() - t0
    log(f"Pipeline loaded in {load_elapsed:.1f}s on {device}.")

    t1 = time.perf_counter()
    kwargs = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers
    if progress_state is not None:
        kwargs["hook"] = _build_progress_hook(progress_state)
    result = pipeline(str(wav_path), **kwargs)
    log(f"Diarization done in {time.perf_counter() - t1:.1f}s.")

    payload = result.serialize()
    turns = [
        DiarTurn(start=t["start"], end=t["end"], speaker=t["speaker"])
        for t in payload["exclusive_diarization"]
    ]

    # Drop pyannote's weights and KV before any LLM stage runs — the
    # alternative is LM Studio crashing on long prompts on a 16 GB Mac.
    del pipeline, result, payload
    free_torch_mps(log)
    return turns, load_elapsed
