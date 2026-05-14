"""Unit tests for the pre-ASR lang_detect sub-step.

Patches `mlx_whisper.load_models.load_model` and `mlx_whisper.audio.*` so
the tests never touch the real model or any audio file. Each test asserts
one specific guarantee of `detect_language`.
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
    overwrite `capture["fake_probs"]` to drive the SUT into a single fixed
    branch, or set `capture["fake_probs_queue"]` to a list to feed
    successive `detect_language` calls with different probs (used by the
    two-attempt voting tests).
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
        capture.setdefault("mel_input_lengths", []).append(len(turn_audio))
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
            capture.setdefault("detect_language_mels", []).append(mel)
            capture["detect_language_mel"] = mel
            queue = capture.get("fake_probs_queue")
            if queue:
                # Pop from the head so the queue order matches attempt order.
                return None, queue.pop(0)
            # `model.detect_language` returns `(lang_tokens, probs)` where
            # `probs` is a flat `{lang: probability}` dict — matches
            # `mlx_whisper/transcribe.py:173-174` usage.
            return None, capture.get("fake_probs", {"uk": 0.9, "ru": 0.05})

    def fake_load_model(repo_id):
        capture["load_model_repo_id"] = repo_id
        capture["load_model_call_count"] = capture.get("load_model_call_count", 0) + 1
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
    out = lang_detect.detect_language("/tmp/a.wav", turns)

    assert out.language == "uk"
    assert out.probabilities == {"uk": 0.84, "ru": 0.15, "en": 0.01}
    # Whisper repo id passed through unchanged.
    assert capture["load_model_repo_id"] == WHISPER
    # First mel input is the longest turn (10.0 to 33.4 s).
    sr = 16000
    assert capture["mel_input_lengths"][0] == int(33.4 * sr) - int(10.0 * sr)


def test_picks_top1_even_when_uk_is_not_first(monkeypatch):
    """Detect should return whichever language has highest probability,
    not bias toward `uk`."""
    capture: dict[str, Any] = {"fake_probs": {"uk": 0.10, "en": 0.85, "fr": 0.04}}
    _install_fake_mlx_whisper(monkeypatch, capture)
    _patch_cache_hit(monkeypatch)

    turns = [DiarTurn(start=0.0, end=20.0, speaker="SPEAKER_00")]
    out = lang_detect.detect_language("/tmp/a.wav", turns)

    assert out.language == "en"


def test_empty_turns_returns_fallback_uk(monkeypatch):
    """No diarization turns at all → fallback (`"uk"`), no model load."""
    capture: dict[str, Any] = {}
    _install_fake_mlx_whisper(monkeypatch, capture)
    _patch_cache_hit(monkeypatch)

    out = lang_detect.detect_language("/tmp/a.wav", [])

    assert out.language == lang_detect.FALLBACK_LANGUAGE
    assert out.probabilities is None
    assert out.agreement == "single"
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
    out = lang_detect.detect_language("/tmp/a.wav", turns)

    assert out.language == lang_detect.FALLBACK_LANGUAGE
    assert out.probabilities is None
    assert out.agreement == "single"
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
    out = lang_detect.detect_language("/tmp/a.wav", turns)

    assert out.language == lang_detect.FALLBACK_LANGUAGE
    assert out.probabilities is None


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
        lang_detect.detect_language("/tmp/a.wav", turns)


# ---------------------------------------------- two-attempt voting (new)


def test_two_attempts_agree_picks_higher_top1(monkeypatch):
    """Two speakers, both vote `uk`; the more confident attempt wins."""
    capture: dict[str, Any] = {
        "fake_probs_queue": [
            {"uk": 0.60, "ru": 0.30, "en": 0.10},  # attempt 1, longest
            {"uk": 0.85, "ru": 0.10, "en": 0.05},  # attempt 2, other speaker
        ]
    }
    _install_fake_mlx_whisper(monkeypatch, capture)
    _patch_cache_hit(monkeypatch)

    turns = [
        DiarTurn(start=0.0, end=23.0, speaker="SPEAKER_01"),  # longest overall
        DiarTurn(start=30.0, end=42.0, speaker="SPEAKER_00"),  # longest of S00
    ]
    out = lang_detect.detect_language("/tmp/a.wav", turns)

    assert out.language == "uk"
    # Winner's full probs surface in `probabilities`.
    assert out.probabilities == {"uk": 0.85, "ru": 0.10, "en": 0.05}
    assert out.agreement == "agree"
    assert len(out.attempts) == 2
    # Two detect_language calls were made against the same loaded model.
    assert capture["load_model_call_count"] == 1
    assert len(capture["detect_language_mels"]) == 2


def test_two_attempts_disagree_picks_higher_top1(monkeypatch):
    """Top-1 differs across attempts; higher top-1 prob wins, agreement="disagree"."""
    capture: dict[str, Any] = {
        "fake_probs_queue": [
            {"uk": 0.55, "ru": 0.40, "en": 0.05},  # attempt 1, longest
            {"ru": 0.80, "uk": 0.15, "en": 0.05},  # attempt 2, other speaker
        ]
    }
    _install_fake_mlx_whisper(monkeypatch, capture)
    _patch_cache_hit(monkeypatch)

    turns = [
        DiarTurn(start=0.0, end=20.0, speaker="SPEAKER_01"),
        DiarTurn(start=30.0, end=42.0, speaker="SPEAKER_00"),
    ]
    out = lang_detect.detect_language("/tmp/a.wav", turns)

    assert out.language == "ru"
    assert out.probabilities == {"ru": 0.80, "uk": 0.15, "en": 0.05}
    assert out.agreement == "disagree"
    assert len(out.attempts) == 2


def test_single_speaker_uses_second_longest_same_speaker(monkeypatch):
    """One speaker, multiple eligible turns → attempt 2 = second-longest of same speaker."""
    capture: dict[str, Any] = {
        "fake_probs_queue": [
            {"uk": 0.70, "ru": 0.25, "en": 0.05},  # attempt 1 (longest)
            {"uk": 0.50, "ru": 0.45, "en": 0.05},  # attempt 2 (second-longest)
        ]
    }
    _install_fake_mlx_whisper(monkeypatch, capture)
    _patch_cache_hit(monkeypatch)

    turns = [
        DiarTurn(start=0.0, end=20.0, speaker="SPEAKER_00"),  # longest, 20 s
        DiarTurn(start=30.0, end=44.0, speaker="SPEAKER_00"),  # 14 s
        DiarTurn(start=50.0, end=58.0, speaker="SPEAKER_00"),  # 8 s
    ]
    out = lang_detect.detect_language("/tmp/a.wav", turns)

    assert out.language == "uk"
    assert out.agreement == "agree"
    assert len(out.attempts) == 2
    # Both AttemptInfo entries are SPEAKER_00; first is the 20 s turn,
    # second is the 14 s turn (sorted by duration desc within same speaker).
    speakers = [a.speaker for a in out.attempts]
    durations = [a.end - a.start for a in out.attempts]
    assert speakers == ["SPEAKER_00", "SPEAKER_00"]
    assert durations == pytest.approx([20.0, 14.0])


def test_two_speakers_second_speaker_too_short_falls_through(monkeypatch):
    """Second speaker's only turn < MIN_TURN_DURATION_S → fall through to
    second-longest turn of attempt-1's speaker."""
    capture: dict[str, Any] = {
        "fake_probs_queue": [
            {"uk": 0.70, "ru": 0.25, "en": 0.05},
            {"uk": 0.60, "ru": 0.35, "en": 0.05},
        ]
    }
    _install_fake_mlx_whisper(monkeypatch, capture)
    _patch_cache_hit(monkeypatch)

    turns = [
        DiarTurn(start=0.0, end=20.0, speaker="SPEAKER_01"),    # longest
        DiarTurn(start=21.0, end=21.5, speaker="SPEAKER_00"),   # too short, ignored
        DiarTurn(start=30.0, end=40.0, speaker="SPEAKER_01"),   # second-longest of S01
    ]
    out = lang_detect.detect_language("/tmp/a.wav", turns)

    assert out.agreement == "agree"
    assert len(out.attempts) == 2
    assert [a.speaker for a in out.attempts] == ["SPEAKER_01", "SPEAKER_01"]


def test_only_one_eligible_turn_runs_single_attempt(monkeypatch):
    """When only one turn meets MIN_TURN_DURATION_S, run a single attempt."""
    capture: dict[str, Any] = {
        "fake_probs_queue": [{"uk": 0.95, "ru": 0.03, "en": 0.02}]
    }
    _install_fake_mlx_whisper(monkeypatch, capture)
    _patch_cache_hit(monkeypatch)

    turns = [
        DiarTurn(start=0.0, end=20.0, speaker="SPEAKER_00"),    # eligible
        DiarTurn(start=21.0, end=21.5, speaker="SPEAKER_01"),   # too short
        DiarTurn(start=22.0, end=23.5, speaker="SPEAKER_00"),   # too short
    ]
    out = lang_detect.detect_language("/tmp/a.wav", turns)

    assert out.language == "uk"
    assert out.agreement == "single"
    assert len(out.attempts) == 1
    assert len(capture["detect_language_mels"]) == 1


def test_attempt_two_empty_probs_returns_attempt_one(monkeypatch):
    """If attempt 2 fails (empty probs), keep attempt 1 with agreement="single"."""
    capture: dict[str, Any] = {
        "fake_probs_queue": [
            {"uk": 0.80, "ru": 0.15, "en": 0.05},  # attempt 1 succeeds
            {},                                       # attempt 2 fails
        ]
    }
    _install_fake_mlx_whisper(monkeypatch, capture)
    _patch_cache_hit(monkeypatch)

    turns = [
        DiarTurn(start=0.0, end=20.0, speaker="SPEAKER_01"),
        DiarTurn(start=30.0, end=42.0, speaker="SPEAKER_00"),
    ]
    out = lang_detect.detect_language("/tmp/a.wav", turns)

    assert out.language == "uk"
    assert out.probabilities == {"uk": 0.80, "ru": 0.15, "en": 0.05}
    assert out.agreement == "single"
    assert len(out.attempts) == 1
