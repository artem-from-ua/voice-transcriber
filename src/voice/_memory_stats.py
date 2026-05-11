"""Memory snapshot for the progress footer.

Best-effort: any field that can't be read becomes None. The progress
reporter rendering must tolerate that and skip the missing column.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


_GB = 1024 ** 3


@dataclass(frozen=True)
class MemoryStats:
    """RAM and MLX-side memory in gigabytes. Any field may be None."""
    ram_used_gb: float | None
    ram_total_gb: float | None
    mlx_active_gb: float | None
    mlx_peak_gb: float | None

    @property
    def ram_pct(self) -> int | None:
        if self.ram_used_gb is None or not self.ram_total_gb:
            return None
        return int(round(100 * self.ram_used_gb / self.ram_total_gb))

    def format(self) -> str:
        """Compact one-line summary; columns silently omitted if unknown.

        Shows only the values that change during a run: RAM used (with the
        percentage relative to total) and MLX active. RAM total never moves
        and MLX peak only ratchets up — both add noise without information.
        """
        parts: list[str] = []
        if self.ram_used_gb is not None:
            if self.ram_pct is not None:
                parts.append(f"RAM {self.ram_used_gb:.1f} GB ({self.ram_pct}%)")
            else:
                parts.append(f"RAM {self.ram_used_gb:.1f} GB")
        if self.mlx_active_gb is not None:
            parts.append(f"MLX {self.mlx_active_gb:.1f} GB")
        return " · ".join(parts)


def read_memory_snapshot() -> MemoryStats:
    """Read RAM + MLX memory once. Never raises."""
    ram_used, ram_total = _read_ram()
    mlx_active, mlx_peak = _read_mlx()
    return MemoryStats(ram_used, ram_total, mlx_active, mlx_peak)


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
