"""Memory-pressure helpers for releasing big models between pipeline stages.

The pipeline keeps three large weights resident: VibeVoice-ASR (~7 GB),
pyannote (~1.5 GB on MPS), and the LLM held inside LM Studio (~8 GB). On a
16 GB unified-memory Mac that sum exceeds RAM, and LM Studio crashes with
"The model has crashed without additional information" the first time the
LLM is asked for a long-context prompt (structure_dialog).

These helpers force Python GC, drop the MLX cache, and ask Metal/MPS to
release its allocator pools after each of the in-process model stages.
They are best-effort — wrapped so a missing optional dependency just no-ops.
"""

from __future__ import annotations

import gc
from typing import Callable


def free_mlx(log: Callable[[str], None] = lambda _s: None) -> None:
    """Release MLX-side caches (for VibeVoice-ASR via mlx-audio)."""
    gc.collect()
    try:
        import mlx.core as mx  # type: ignore[import-not-found]
    except ImportError:
        return
    for fn_name in ("clear_cache",):
        fn = getattr(mx, fn_name, None)
        if callable(fn):
            try:
                fn()
            except Exception as exc:  # noqa: BLE001
                log(f"memory: mx.{fn_name}() raised {exc!r}; continuing")
    metal = getattr(mx, "metal", None)
    if metal is not None and callable(getattr(metal, "clear_cache", None)):
        try:
            metal.clear_cache()
        except Exception as exc:  # noqa: BLE001
            log(f"memory: mx.metal.clear_cache() raised {exc!r}; continuing")


def free_torch_mps(log: Callable[[str], None] = lambda _s: None) -> None:
    """Release MPS-side caches (for pyannote via torch)."""
    gc.collect()
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError:
        return
    mps = getattr(torch, "mps", None)
    if mps is None:
        return
    for fn_name in ("empty_cache", "synchronize"):
        fn = getattr(mps, fn_name, None)
        if callable(fn):
            try:
                fn()
            except Exception as exc:  # noqa: BLE001
                log(f"memory: torch.mps.{fn_name}() raised {exc!r}; continuing")
