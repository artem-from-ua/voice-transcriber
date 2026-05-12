"""Pre-ASR spoken language detection on the longest pyannote turn.

A sub-step between `[3] diarize` and `[4] clearspeech` that runs only when
the user did not pass `--language` explicitly. It picks the longest turn
from pyannote's diarization output, feeds its mel-spectrogram into
Whisper-large-v3-MLX's `model.detect_language()`, and returns the
top-1 ISO language code that the pipeline then uses for the ASR stage and
every LLM/render stage downstream.

Why this exists (see [ADR 0022](../docs/adr/0022-asr-language-autodetect.md)):
Whisper's internal `transcribe(language=None)` auto-detect runs on the
first 30 s window of the audio, which often contains silence, an
acknowledgement ("ага"), or just one short speaker turn — not enough
lexical content to disambiguate Slavic languages (uk vs ru in particular).
The longest pyannote turn is a much richer signal — typically tens of
seconds of one speaker with complete sentences. Empirically on the
project's Ukrainian reference recording, Whisper detect on the longest
turn gives uk=0.84 vs ru=0.15 (5.6x margin), while detect on the first
30 s gives ru=0.85 (catastrophically wrong).

Not to be confused with the PyPI package `langdetect`, which performs
text-based language identification — this module is audio-based and runs
the same Whisper-large-v3 backend that the ASR stage will later use.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from ._memory import free_mlx
from .types import DiarTurn
from .whisper_asr import WHISPER_REPO_ID, _verify_model_cached


# Pyannote turns shorter than this are too brief for Whisper to classify
# the language reliably — Whisper's classifier head was trained on roughly
# utterance-length windows. We treat such tiny inputs as a "no signal"
# case and fall back to a hard-coded language, telling the caller via log.
MIN_TURN_DURATION_S = 2.0

# Defensive fallback when (a) pyannote returned no turns at all, (b) the
# longest turn is shorter than MIN_TURN_DURATION_S, or (c) Whisper returns
# an empty `lang_probs`. "uk" is the project's primary language; users who
# work with non-Ukrainian audio should pass `--language` explicitly anyway.
FALLBACK_LANGUAGE = "uk"


class LangDetectError(RuntimeError):
    pass


def _noop_log(_msg: str) -> None:
    return None


def detect_language_on_longest_turn(
    wav_path: str | Path,
    turns: list[DiarTurn],
    *,
    log: Callable[[str], None] = _noop_log,
) -> str:
    """Return the ISO language code Whisper-large-v3 detects on the longest
    pyannote turn from `turns`.

    Falls back to FALLBACK_LANGUAGE (`"uk"`) when there are no turns or the
    longest turn is shorter than MIN_TURN_DURATION_S — both are pathological
    states for a successful pipeline run, but the function never raises on
    them so the caller can keep going with a best-effort default.
    """
    if not turns:
        log(
            f"lang_detect: pyannote returned no turns — fallback "
            f"{FALLBACK_LANGUAGE!r}. Pass --language explicitly to silence."
        )
        return FALLBACK_LANGUAGE

    longest = max(turns, key=lambda t: t.end - t.start)
    duration = longest.end - longest.start
    if duration < MIN_TURN_DURATION_S:
        log(
            f"lang_detect: longest turn is only {duration:.2f}s "
            f"(< {MIN_TURN_DURATION_S}s) — fallback {FALLBACK_LANGUAGE!r}. "
            f"Pass --language explicitly to silence."
        )
        return FALLBACK_LANGUAGE

    _verify_model_cached()

    # Lazy import: mlx_whisper pulls in numba/tiktoken/llvmlite, ~1 s warmup
    # we do not want to pay on every import of this module (e.g. in tests
    # that don't exercise the language-detect branch).
    from mlx_whisper.audio import (
        N_FRAMES,
        SAMPLE_RATE,
        load_audio,
        log_mel_spectrogram,
        pad_or_trim,
    )
    from mlx_whisper.load_models import load_model

    log(
        f"lang_detect: longest turn {duration:.2f}s "
        f"({longest.speaker}, {longest.start:.2f}-{longest.end:.2f}s)"
    )

    audio = load_audio(str(wav_path))
    s0 = int(longest.start * SAMPLE_RATE)
    s1 = int(longest.end * SAMPLE_RATE)
    turn_audio = audio[s0:s1]

    t0 = time.time()
    model = load_model(WHISPER_REPO_ID)
    mel = log_mel_spectrogram(turn_audio, n_mels=model.dims.n_mels)
    mel = pad_or_trim(mel, N_FRAMES, axis=-2)
    # `model.detect_language` accepts a (n_mels, n_frames) mel and adds the
    # batch dimension internally — same call shape that
    # `mlx_whisper/transcribe.py:172` uses, no manual reshape needed.

    # `model.detect_language(mel)` returns (lang_tokens, probs), where
    # `probs` is a dict of {lang_code: probability} — matches the call
    # shape used in `mlx_whisper/transcribe.py:173`. The module-level
    # `decoding.detect_language(model, mel, tokenizer)` returns a *list*
    # of such dicts (one per batch row); the method version skips the
    # batching wrapper.
    _, probs = model.detect_language(mel)
    log(f"lang_detect: model.detect_language done in {time.time() - t0:.1f}s.")

    if not probs:
        log(
            f"lang_detect: model.detect_language returned no probabilities — "
            f"fallback {FALLBACK_LANGUAGE!r}."
        )
        del model
        free_mlx(log)
        return FALLBACK_LANGUAGE

    top1_lang = max(probs, key=probs.get)
    top1_prob = float(probs[top1_lang])
    ru_prob = float(probs.get("ru", 0.0))
    log(
        f"lang_detect: top1 {top1_lang} (p={top1_prob:.2f}, ru p={ru_prob:.2f})"
    )

    # Drop the Whisper weights and encoder KV cache. The ASR stage will load
    # the same model again with its own buffers; sharing across stages would
    # complicate the per-stage `free_mlx` contract for very little win.
    del model
    free_mlx(log)
    return top1_lang
