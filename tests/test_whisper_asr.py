"""Unit tests for the Whisper ASR backend.

Patches `mlx_whisper.transcribe` and `huggingface_hub.try_to_load_from_cache`
so the tests never touch the real model or network. The 4 assertions verify:
  - mlx-whisper segments are parsed into `AsrSegment`
  - missing-cache raises `AsrError` with an actionable message
  - the `language` kwarg flows through to mlx-whisper
  - `condition_on_previous_text=False` is locked in (matches stone-scriber)
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from voice import whisper_asr
from voice.whisper_asr import AsrError
from voice.types import AsrSegment


def _install_fake_mlx_whisper(monkeypatch, return_payload: dict, capture: dict) -> None:
    """Inject a stub `mlx_whisper` module so `import mlx_whisper` in the SUT
    picks it up without hitting the real package (which lazy-loads MLX)."""

    fake = types.ModuleType("mlx_whisper")

    def fake_transcribe(audio, **kwargs):
        capture["audio"] = audio
        capture["kwargs"] = kwargs
        return return_payload

    fake.transcribe = fake_transcribe
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake)


def _patch_cache_hit(monkeypatch) -> None:
    monkeypatch.setattr(
        whisper_asr,
        "_verify_model_cached",
        lambda *a, **kw: None,
    )


def test_transcribe_parses_mlx_whisper_segments(monkeypatch):
    capture: dict[str, Any] = {}
    _install_fake_mlx_whisper(
        monkeypatch,
        return_payload={
            "text": "  privet  ",
            "segments": [
                {"start": 0.0, "end": 1.5, "text": "  привіт  "},
                {"start": 1.5, "end": 3.0, "text": "як справи"},
                {"start": 3.0, "end": 3.1, "text": "   "},  # blank → dropped
            ],
            "language": "uk",
        },
        capture=capture,
    )
    _patch_cache_hit(monkeypatch)

    out = whisper_asr.transcribe("/tmp/a.wav")

    assert out == [
        AsrSegment(start=0.0, end=1.5, content="привіт"),
        AsrSegment(start=1.5, end=3.0, content="як справи"),
    ]


def test_transcribe_raises_when_model_not_cached(monkeypatch):
    # Force try_to_load_from_cache to return None so _verify_model_cached fails.
    fake_hf = types.ModuleType("huggingface_hub")
    fake_hf.try_to_load_from_cache = lambda **kw: None
    fake_errors = types.ModuleType("huggingface_hub.errors")
    fake_errors.CacheNotFound = type("CacheNotFound", (Exception,), {})
    fake_hf.errors = fake_errors
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", fake_errors)

    with pytest.raises(AsrError, match="voice download-whisper"):
        whisper_asr._verify_model_cached()


def test_transcribe_passes_language_to_mlx_whisper(monkeypatch):
    capture: dict[str, Any] = {}
    _install_fake_mlx_whisper(
        monkeypatch,
        return_payload={"segments": []},
        capture=capture,
    )
    _patch_cache_hit(monkeypatch)

    whisper_asr.transcribe("/tmp/a.wav", language="en")

    assert capture["kwargs"]["language"] == "en"


def test_transcribe_disables_condition_on_previous_text(monkeypatch):
    """Locks in the stone-scriber policy: each 30-second window decodes
    fresh, no prompt-conditioning from the previous window. Trades a bit of
    cross-window consistency for far fewer repetition loops on Ukrainian.
    """
    capture: dict[str, Any] = {}
    _install_fake_mlx_whisper(
        monkeypatch,
        return_payload={"segments": []},
        capture=capture,
    )
    _patch_cache_hit(monkeypatch)

    whisper_asr.transcribe("/tmp/a.wav")

    assert capture["kwargs"]["condition_on_previous_text"] is False
    assert capture["kwargs"]["path_or_hf_repo"] == whisper_asr.WHISPER_REPO_ID
