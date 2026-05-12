"""Unit tests for the pre-ASR lang_detect sub-step.

Patches `mlx_whisper.load_models.load_model` and `mlx_whisper.audio.*` so
the tests never touch the real model or any audio file. Each test asserts
one specific guarantee of `detect_language_on_longest_turn`.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from voice import lang_detect
from voice.types import DiarTurn


# Importable Whisper repo id alias to keep test asserts readable.
WHISPER = lang_detect.WHISPER_REPO_ID


def _install_fake_mlx_whisper(monkeypatch, capture: dict) -> None:
    """Inject stubs for `mlx_whisper.audio` and `mlx_whisper.load_models`
    so the SUT can `from mlx_whisper.{audio,load_models} import ...` without
    pulling in the real package (which lazy-loads MLX and reads audio
    files).

    The stub records the audio slice fed to `log_mel_spectrogram` and the
    language probabilities returned by `model.detect_language`. Tests can
    overwrite `capture["fake_probs"]` to drive the SUT into different
    branches.
    """
    sr = 16000

    fake_audio_module = types.ModuleType("mlx_whisper.audio")
    fake_audio_module.SAMPLE_RATE = sr
    fake_audio_module.N_FRAMES = 3000

    # `load_audio` returns one sample per second so `audio[s0:s1]` yields a
    # slice whose length equals (end - start) seconds — trivial to assert
    # against in the audio-slice test.
    def fake_load_audio(path):
        capture["load_audio_path"] = path
        return list(range(sr * 600))  # 600 seconds of "audio"

    def fake_log_mel_spectrogram(turn_audio, *, n_mels):
        capture["mel_input_length"] = len(turn_audio)
        capture["mel_n_mels"] = n_mels
        # Return a dummy "mel" — the SUT only forwards it to `detect_language`.
        return f"mel-of-{len(turn_audio)}-samples"

    def fake_pad_or_trim(mel, length, *, axis):
        capture["pad_or_trim_length"] = length
        return mel  # pass-through is fine; the SUT never inspects the value

    fake_audio_module.load_audio = fake_load_audio
    fake_audio_module.log_mel_spectrogram = fake_log_mel_spectrogram
    fake_audio_module.pad_or_trim = fake_pad_or_trim

    fake_load_models = types.ModuleType("mlx_whisper.load_models")

    class _FakeDims:
        n_mels = 128

    class _FakeModel:
        dims = _FakeDims()

        def detect_language(self, mel):
            capture["detect_language_mel"] = mel
            # `model.detect_language` returns `(lang_tokens, probs)` where
            # `probs` is a flat `{lang: probability}` dict — matches
            # `mlx_whisper/transcribe.py:173-174` usage.
            return None, capture.get("fake_probs", {"uk": 0.9, "ru": 0.05})

    def fake_load_model(repo_id):
        capture["load_model_repo_id"] = repo_id
        return _FakeModel()

    fake_load_models.load_model = fake_load_model

    monkeypatch.setitem(sys.modules, "mlx_whisper.audio", fake_audio_module)
    monkeypatch.setitem(sys.modules, "mlx_whisper.load_models", fake_load_models)


def _patch_cache_hit(monkeypatch) -> None:
    monkeypatch.setattr(lang_detect, "_verify_model_cached", lambda *a, **k: None)


def test_picks_longest_turn_and_returns_top1_language(monkeypatch):
    """Among several turns, the longest one is fed to `detect_language`, and
    its top-1 probability becomes the return value."""
    capture: dict[str, Any] = {"fake_probs": {"uk": 0.84, "ru": 0.15, "en": 0.01}}
    _install_fake_mlx_whisper(monkeypatch, capture)
    _patch_cache_hit(monkeypatch)

    turns = [
        DiarTurn(start=0.0, end=2.5, speaker="SPEAKER_00"),
        DiarTurn(start=10.0, end=33.4, speaker="SPEAKER_01"),  # longest, 23.4 s
        DiarTurn(start=40.0, end=45.0, speaker="SPEAKER_00"),
    ]
    out = lang_detect.detect_language_on_longest_turn("/tmp/a.wav", turns)

    assert out == "uk"
    # Whisper repo id passed through unchanged.
    assert capture["load_model_repo_id"] == WHISPER
    # Audio slice corresponds to the longest turn (10.0 to 33.4 s).
    sr = 16000
    assert capture["mel_input_length"] == int(33.4 * sr) - int(10.0 * sr)


def test_picks_top1_even_when_uk_is_not_first(monkeypatch):
    """Detect should return whichever language has highest probability,
    not bias toward `uk`."""
    capture: dict[str, Any] = {"fake_probs": {"uk": 0.10, "en": 0.85, "fr": 0.04}}
    _install_fake_mlx_whisper(monkeypatch, capture)
    _patch_cache_hit(monkeypatch)

    turns = [DiarTurn(start=0.0, end=20.0, speaker="SPEAKER_00")]
    out = lang_detect.detect_language_on_longest_turn("/tmp/a.wav", turns)

    assert out == "en"


def test_empty_turns_returns_fallback_uk(monkeypatch):
    """No diarization turns at all → fallback (`"uk"`), no model load."""
    capture: dict[str, Any] = {}
    _install_fake_mlx_whisper(monkeypatch, capture)
    _patch_cache_hit(monkeypatch)

    out = lang_detect.detect_language_on_longest_turn("/tmp/a.wav", [])

    assert out == lang_detect.FALLBACK_LANGUAGE
    # Model should NOT have been loaded for a no-turn input.
    assert "load_model_repo_id" not in capture


def test_too_short_turn_returns_fallback_uk(monkeypatch):
    """When the longest turn is shorter than MIN_TURN_DURATION_S, skip
    detection and return the fallback."""
    capture: dict[str, Any] = {}
    _install_fake_mlx_whisper(monkeypatch, capture)
    _patch_cache_hit(monkeypatch)

    turns = [
        DiarTurn(start=0.0, end=1.0, speaker="SPEAKER_00"),
        DiarTurn(start=1.5, end=2.4, speaker="SPEAKER_01"),  # longest, < 2 s
    ]
    out = lang_detect.detect_language_on_longest_turn("/tmp/a.wav", turns)

    assert out == lang_detect.FALLBACK_LANGUAGE
    # Model should NOT have been loaded for a too-short input.
    assert "load_model_repo_id" not in capture


def test_empty_probs_falls_back(monkeypatch):
    """If model.detect_language returns no probabilities (defensive), the
    fallback applies."""
    capture: dict[str, Any] = {}
    _install_fake_mlx_whisper(monkeypatch, capture)
    _patch_cache_hit(monkeypatch)

    # Make detect_language return an empty probability dict.
    fake_load_models = sys.modules["mlx_whisper.load_models"]

    class _EmptyModel:
        class dims:
            n_mels = 128

        def detect_language(self, mel):
            return None, {}

    monkeypatch.setattr(fake_load_models, "load_model", lambda repo: _EmptyModel())

    turns = [DiarTurn(start=0.0, end=20.0, speaker="SPEAKER_00")]
    out = lang_detect.detect_language_on_longest_turn("/tmp/a.wav", turns)

    assert out == lang_detect.FALLBACK_LANGUAGE


def test_verifies_model_cached_before_load(monkeypatch):
    """Hard-fail with AsrError-style message if Whisper weights are not in
    the HF cache — same contract as whisper_asr.transcribe()."""
    capture: dict[str, Any] = {}
    _install_fake_mlx_whisper(monkeypatch, capture)
    # _verify_model_cached itself raises AsrError when nothing is cached.
    from voice.whisper_asr import AsrError

    def fail(*a, **k):
        raise AsrError("Whisper model is not in the HuggingFace cache.")

    monkeypatch.setattr(lang_detect, "_verify_model_cached", fail)

    turns = [DiarTurn(start=0.0, end=20.0, speaker="SPEAKER_00")]
    with pytest.raises(AsrError, match="HuggingFace cache"):
        lang_detect.detect_language_on_longest_turn("/tmp/a.wav", turns)
