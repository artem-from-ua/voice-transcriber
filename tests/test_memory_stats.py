"""Tests for the memory snapshot helper."""

from __future__ import annotations

from voice._memory_stats import MemoryStats, read_memory_snapshot


def _stats(**overrides) -> MemoryStats:
    """Helper: build MemoryStats with all fields None unless overridden."""
    defaults = dict(
        ram_used_gb=None, ram_total_gb=None,
        mlx_active_gb=None, mlx_peak_gb=None,
        mps_active_gb=None, mps_driver_gb=None,
    )
    defaults.update(overrides)
    return MemoryStats(**defaults)


def test_memory_stats_format_includes_ram_percent_only():
    """Total stays out of the line (it never changes); peak too (only ratchets up)."""
    s = _stats(ram_used_gb=9.4, ram_total_gb=16.0, mlx_active_gb=8.0, mlx_peak_gb=9.6)
    out = s.format()
    assert "RAM 9.4 GB (59%)" in out
    assert "/16.0" not in out
    assert "MLX 8.0 GB" in out
    assert "peak" not in out


def test_memory_stats_format_skips_missing_fields():
    s = _stats(ram_used_gb=4.0, ram_total_gb=8.0)
    out = s.format()
    assert "RAM 4.0 GB" in out
    assert "MLX" not in out
    assert "MPS" not in out
    assert "peak" not in out


def test_memory_stats_format_without_total():
    """Even if total is unknown, RAM value alone is still printed."""
    s = _stats(ram_used_gb=4.0)
    out = s.format()
    assert "RAM 4.0 GB" in out
    assert "(" not in out  # no percent without a total


def test_memory_stats_format_empty_when_nothing_known():
    assert _stats().format() == ""


def test_ram_pct_handles_zero_total():
    s = _stats(ram_used_gb=4.0, ram_total_gb=0.0)
    assert s.ram_pct is None


def test_mps_shown_when_active_pool_nonzero():
    """During diarize, current_allocated_memory holds real data."""
    s = _stats(ram_used_gb=2.0, ram_total_gb=16.0, mps_active_gb=1.5, mps_driver_gb=0.5)
    out = s.format()
    assert "MPS 1.5 GB" in out  # max(active, driver)


def test_mps_falls_back_to_driver_after_empty_cache():
    """current_allocated_memory drops to 0 after empty_cache; driver pool stays."""
    s = _stats(ram_used_gb=2.0, ram_total_gb=16.0, mps_active_gb=0.0, mps_driver_gb=4.16)
    out = s.format()
    assert "MPS 4.2 GB" in out


def test_mps_hidden_below_threshold():
    """Tiny allocator residue (a few MB) is noise — don't display it."""
    s = _stats(ram_used_gb=2.0, ram_total_gb=16.0, mps_active_gb=0.001, mps_driver_gb=0.005)
    out = s.format()
    assert "MPS" not in out


def test_mlx_hidden_below_threshold():
    """Same threshold applies to MLX so non-MLX stages don't print 'MLX 0.0 GB'."""
    s = _stats(ram_used_gb=2.0, ram_total_gb=16.0, mlx_active_gb=0.0)
    out = s.format()
    assert "MLX" not in out


def test_read_snapshot_returns_real_ram_on_this_machine():
    """Smoke check: psutil is installed and reports plausible numbers."""
    s = read_memory_snapshot()
    assert s.ram_used_gb is not None
    assert s.ram_total_gb is not None
    assert 0 < s.ram_used_gb < s.ram_total_gb
    assert s.ram_total_gb >= 4  # any modern Mac
