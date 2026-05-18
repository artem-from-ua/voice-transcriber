"""Memory snapshot for the progress footer.

Best-effort: any field that can't be read becomes None. The progress
reporter rendering must tolerate that and skip the missing column.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


_GB = 1024 ** 3
# Allocator pools idle below this threshold are noise — skip them in the
# footer so non-MLX stages don't print "MLX 0.0 GB" and pre-allocated MPS
# scratch space doesn't masquerade as live diarize memory.
_DISPLAY_THRESHOLD_GB = 0.01


@dataclass(frozen=True)
class MemoryStats:
    """RAM, MLX, and MPS memory in gigabytes. Any field may be None."""
    ram_used_gb: float | None
    ram_total_gb: float | None
    mlx_active_gb: float | None
    mlx_peak_gb: float | None
    mps_active_gb: float | None
    mps_driver_gb: float | None

    @property
    def ram_pct(self) -> int | None:
        if self.ram_used_gb is None or not self.ram_total_gb:
            return None
        return int(round(100 * self.ram_used_gb / self.ram_total_gb))

    @property
    def mps_display_gb(self) -> float | None:
        """Larger of active vs driver pool. `driver` survives empty_cache()
        so it keeps a non-zero signal after diarize() drains the allocator."""
        candidates = [v for v in (self.mps_active_gb, self.mps_driver_gb) if v is not None]
        return max(candidates) if candidates else None

    def format(self) -> str:
        """Compact one-line summary; columns silently omitted if unknown
        or below the display threshold (~10 MB allocator noise).

        Shows only values that change during a run: RAM used (with percentage
        relative to total), MLX active, and the larger MPS pool reading.
        Totals and ratchet-only peaks add noise without information.
        """
        parts: list[str] = []
        if self.ram_used_gb is not None:
            if self.ram_pct is not None:
                parts.append(f"RAM {self.ram_used_gb:.1f} GB ({self.ram_pct}%)")
            else:
                parts.append(f"RAM {self.ram_used_gb:.1f} GB")
        if self.mlx_active_gb is not None and self.mlx_active_gb >= _DISPLAY_THRESHOLD_GB:
            parts.append(f"MLX {self.mlx_active_gb:.1f} GB")
        mps = self.mps_display_gb
        if mps is not None and mps >= _DISPLAY_THRESHOLD_GB:
            parts.append(f"MPS {mps:.1f} GB")
        return " · ".join(parts)


def read_memory_snapshot() -> MemoryStats:
    """Read RAM, MLX, and MPS memory once. Never raises."""
    ram_used, ram_total = _read_ram()
    mlx_active, mlx_peak = _read_mlx()
    mps_active, mps_driver = _read_mps()
    return MemoryStats(
        ram_used, ram_total, mlx_active, mlx_peak, mps_active, mps_driver,
    )


def _read_ram() -> tuple[float | None, float | None]:
    """Read macOS Activity Monitor's "Memory Used" via psutil.

    Activity Monitor shows `App Memory + Wired + Compressed`. psutil's
    `virtual_memory().used` follows that convention on Darwin.
    """
    try:
        import psutil
    except ImportError:
        return None, None
    try:
        vm = psutil.virtual_memory()
    except Exception:  # noqa: BLE001 — telemetry must not crash the pipeline
        return None, None
    return vm.used / _GB, vm.total / _GB


def _read_mlx() -> tuple[float | None, float | None]:
    """Active + peak MLX memory in GB. None on CPU-only environments.

    Prefers the top-level `mx.get_active_memory()` / `get_peak_memory()`
    (introduced in mlx ≥ 0.21). Falls back to the older `mx.metal.*`
    namespace for compatibility; both spellings are equivalent on Metal.
    """
    try:
        import mlx.core as mx
    except ImportError:
        return None, None

    def _try(*candidates: str) -> Callable[[], int] | None:
        for path in candidates:
            obj = mx
            for part in path.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            if callable(obj):
                return obj
        return None

    active = _try("get_active_memory", "metal.get_active_memory")
    peak = _try("get_peak_memory", "metal.get_peak_memory")
    try:
        active_val = active() / _GB if active is not None else None
        peak_val = peak() / _GB if peak is not None else None
    except Exception:  # noqa: BLE001
        return None, None
    return active_val, peak_val


def _read_mps() -> tuple[float | None, float | None]:
    """Active + driver-pool MPS memory in GB. None on non-Apple-Silicon.

    `current_allocated_memory()` mirrors what torch holds right now and
    drops to 0 after `torch.mps.empty_cache()` — useful as a live signal.
    `driver_allocated_memory()` reports the full Metal pool size, which
    survives `empty_cache()` and stays non-zero until torch fully releases.
    Issue #163 documented this distinction: a post-`diarize()` reading of
    `current_allocated_memory` alone falsely suggested the pipeline ran on
    CPU; the driver pool grew, proving GPU was exercised.
    """
    try:
        import torch
    except ImportError:
        return None, None
    mps = getattr(torch, "mps", None)
    if mps is None:
        return None, None
    current = getattr(mps, "current_allocated_memory", None)
    driver = getattr(mps, "driver_allocated_memory", None)
    try:
        current_val = current() / _GB if callable(current) else None
        driver_val = driver() / _GB if callable(driver) else None
    except Exception:  # noqa: BLE001 — telemetry must not crash the pipeline
        return None, None
    return current_val, driver_val
