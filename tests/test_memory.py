"""Offline tests for the memory-release helpers."""

from __future__ import annotations

import sys
import types

from voice import _memory


def test_free_mlx_prefers_top_level_clear_cache(monkeypatch):
    """mlx ≥ 0.21 exposes clear_cache on the top level; the legacy
    `metal.clear_cache` triggers a deprecation warning and must be
    skipped when the new one is available."""
    calls: list[str] = []
    fake_mx = types.SimpleNamespace(
        clear_cache=lambda: calls.append("mx.clear_cache"),
        metal=types.SimpleNamespace(
            clear_cache=lambda: calls.append("mx.metal.clear_cache"),
        ),
    )
    monkeypatch.setitem(sys.modules, "mlx", types.SimpleNamespace(core=fake_mx))
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)
    _memory.free_mlx(log=lambda _s: None)
    assert calls == ["mx.clear_cache"]


def test_free_mlx_falls_back_to_metal_clear_cache(monkeypatch):
    """Older mlx (no top-level clear_cache) still works via metal.*"""
    calls: list[str] = []
    fake_mx = types.SimpleNamespace(
        # No top-level clear_cache here.
        metal=types.SimpleNamespace(
            clear_cache=lambda: calls.append("mx.metal.clear_cache"),
        ),
    )
    monkeypatch.setitem(sys.modules, "mlx", types.SimpleNamespace(core=fake_mx))
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)
    _memory.free_mlx(log=lambda _s: None)
    assert calls == ["mx.metal.clear_cache"]


def test_free_mlx_swallows_missing_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "mlx.core", None)
    # Should not raise even though "mlx.core" can't be imported.
    _memory.free_mlx(log=lambda _s: None)


def test_free_torch_mps_calls_empty_cache_when_present(monkeypatch):
    calls: list[str] = []
    fake_mps = types.SimpleNamespace(
        empty_cache=lambda: calls.append("empty_cache"),
        synchronize=lambda: calls.append("synchronize"),
    )
    fake_torch = types.SimpleNamespace(mps=fake_mps)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    _memory.free_torch_mps(log=lambda _s: None)
    assert calls == ["empty_cache", "synchronize"]


def test_free_torch_mps_handles_missing_mps(monkeypatch):
    fake_torch = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    _memory.free_torch_mps(log=lambda _s: None)
