"""Tests for the memory snapshot helper."""

from __future__ import annotations

from voice._memory_stats import MemoryStats, read_memory_snapshot


def test_memory_stats_format_includes_ram_percent_only():
    """Total stays out of the line (it never changes); peak too (only ratchets up)."""
    s = MemoryStats(ram_used_gb=9.4, ram_total_gb=16.0, mlx_active_gb=8.0, mlx_peak_gb=9.6)
    out = s.format()
    assert "RAM 9.4 GB (59%)" in out
    assert "/16.0" not in out
    assert "MLX 8.0 GB" in out
    assert "peak" not in out


def test_memory_stats_format_skips_missing_fields():
    s = MemoryStats(ram_used_gb=4.0, ram_total_gb=8.0, mlx_active_gb=None, mlx_peak_gb=None)
    out = s.format()
    assert "RAM 4.0 GB" in out
    assert "MLX" not in out
    assert "peak" not in out


def test_memory_stats_format_without_total():
    """Even if total is unknown, RAM value alone is still printed."""
    s = MemoryStats(ram_used_gb=4.0, ram_total_gb=None, mlx_active_gb=None, mlx_peak_gb=None)
    out = s.format()
    assert "RAM 4.0 GB" in out
    assert "(" not in out  # no percent without a total


def test_memory_stats_format_empty_when_nothing_known():
    s = MemoryStats(None, None, None, None)
    assert s.format() == ""


def test_ram_pct_handles_zero_total():
    s = MemoryStats(ram_used_gb=4.0, ram_total_gb=0.0, mlx_active_gb=None, mlx_peak_gb=None)
    assert s.ram_pct is None


def test_read_snapshot_returns_real_ram_on_this_machine():
    """Smoke check: psutil is installed and reports plausible numbers."""
    s = read_memory_snapshot()
    assert s.ram_used_gb is not None
    assert s.ram_total_gb is not None
    assert 0 < s.ram_used_gb < s.ram_total_gb
    assert s.ram_total_gb >= 4  # any modern Mac
