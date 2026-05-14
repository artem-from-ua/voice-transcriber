"""Pre-ASR spoken language detection with two-attempt voting.

A sub-step between `[3] diarize` and `[5] clearspeech` that runs only when
the user did not pass `--language` explicitly. It picks two pyannote turns
and runs Whisper-large-v3-MLX `model.detect_language()` on each, then uses
top-1 agreement across the two attempts as a confidence signal:

- Attempt 1: longest pyannote turn overall.
- Attempt 2: longest turn of a different speaker if at least two speakers
  exist; otherwise the second-longest turn of the same speaker.

When both attempts agree on top-1, the more-confident one wins. When they
disagree, the more-confident one still wins (the user's design choice in
the AGREE/DISAGREE-gate decision documented in ADR 0022's 2026-05-14
postscript). When only one eligible turn exists (very short recording or
single-turn audio), the stage degrades to a single attempt.

Why this exists (see [ADR 0022](../docs/adr/0022-asr-language-autodetect.md)):
Whisper's internal `transcribe(language=None)` auto-detect runs on the
first 30 s window of the audio, which often contains silence, an
acknowledgement ("ага"), or just one short speaker turn — not enough
lexical content to disambiguate Slavic languages (uk vs ru in particular).
The longest pyannote turn is a much richer signal — typically tens of
seconds of one speaker with complete sentences. Empirical probe on the
project's reference recording (`scripts/lang-detect-fill-probe.py`) showed
single-turn margin ranges from 1.01x to 9.25x with no correlation to turn
duration; running a second independent attempt on a different turn turns
that variance into a usable agreement signal.

Not to be confused with the PyPI package `langdetect`, which performs
text-based language identification — this module is audio-based and runs
the same Whisper-large-v3 backend that the ASR stage will later use.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ._memory import free_mlx
from .types import DiarTurn
from .whisper_asr import WHISPER_REPO_ID, _verify_model_cached


@dataclass(frozen=True)
class AttemptInfo:
    """One detect_language() call's input turn + ranked top-2 result."""

    speaker: str
    start: float
    end: float
    top_lang: str
    top_prob: float
    second_lang: str | None
    second_prob: float | None


@dataclass(frozen=True)
class LangDetectResult:
    language: str
    probabilities: dict[str, float] | None
    # "single" — exactly one attempt was run (degenerate inputs or fallback);
    # "agree"  — two attempts ran and their top-1 languages matched;
    # "disagree" — two attempts ran and disagreed (winner = higher top1-prob).
    agreement: str = "single"
    attempts: tuple[AttemptInfo, ...] = field(default_factory=tuple)


# Pyannote turns shorter than this are too brief for Whisper to classify
# the language reliably — Whisper's classifier head was trained on roughly
# utterance-length windows. We treat such tiny inputs as a "no signal"
# case and fall back to a hard-coded language, telling the caller via log.
MIN_TURN_DURATION_S = 2.0

# Defensive fallback when (a) pyannote returned no turns at all, (b) the
# longest turn is shorter than MIN_TURN_DURATION_S, or (c) Whisper returns
# an empty `lang_probs` on the first attempt. "uk" is the project's primary
# language; users who work with non-Ukrainian audio should pass `--language`
# explicitly anyway.
FALLBACK_LANGUAGE = "uk"


class LangDetectError(RuntimeError):
    pass


def _noop_log(_msg: str) -> None:
    return None


def _pick_attempts(turns: list[DiarTurn]) -> list[DiarTurn]:
    """Choose 1 or 2 turns to feed Whisper, by the speaker-aware rule.

    Pre-condition: caller has already filtered out the "no turns" and
    "longest < MIN_TURN_DURATION_S" cases — those don't reach this helper.

    - Attempt 1: longest eligible turn overall.
    - Attempt 2: longest eligible turn of a *different* speaker if such a
      turn exists; otherwise the second-longest eligible turn of the same
      speaker; otherwise no second attempt.
    """
    eligible = sorted(
        (t for t in turns if (t.end - t.start) >= MIN_TURN_DURATION_S),
        key=lambda t: t.end - t.start,
        reverse=True,
    )
    if not eligible:
        return []
    first = eligible[0]
    other_speaker = next(
        (t for t in eligible[1:] if t.speaker != first.speaker), None
    )
    if other_speaker is not None:
        return [first, other_speaker]
    same_speaker_second = next(
        (t for t in eligible[1:] if t.speaker == first.speaker), None
    )
    if same_speaker_second is not None:
        return [first, same_speaker_second]
    return [first]


def _run_one_attempt(
    model: Any,
    audio: Any,
    turn: DiarTurn,
    *,
    sample_rate: int,
    n_frames: int,
    log_mel_spectrogram: Callable[..., Any],
    pad_or_trim: Callable[..., Any],
    log: Callable[[str], None],
) -> dict[str, float] | None:
    """Slice the turn out of `audio`, build the mel, call detect_language.

    Returns the probability dict, or None when Whisper returned an empty
    result (defensive — same condition the legacy code treated as fallback).
    """
    s0 = int(turn.start * sample_rate)
    s1 = int(turn.end * sample_rate)
    turn_audio = audio[s0:s1]
    mel = log_mel_spectrogram(turn_audio, n_mels=model.dims.n_mels)
    mel = pad_or_trim(mel, n_frames, axis=-2)
    t0 = time.time()
    _, probs = model.detect_language(mel)
    log(
        f"lang_detect: detect_language done in {time.time() - t0:.1f}s "
        f"({turn.speaker} {turn.start:.2f}-{turn.end:.2f}s)"
    )
    return dict(probs) if probs else None


def _attempt_info(turn: DiarTurn, probs: dict[str, float]) -> AttemptInfo:
    top_lang = max(probs, key=probs.get)
    top_prob = float(probs[top_lang])
    others = [(k, float(v)) for k, v in probs.items() if k != top_lang]
    if others:
        s_lang, s_prob = max(others, key=lambda kv: kv[1])
    else:
        s_lang, s_prob = None, None
    return AttemptInfo(
        speaker=turn.speaker,
        start=turn.start,
        end=turn.end,
        top_lang=top_lang,
        top_prob=top_prob,
        second_lang=s_lang,
        second_prob=s_prob,
    )


def _aggregate(
    results: list[tuple[DiarTurn, dict[str, float]]],
    *,
    log: Callable[[str], None],
) -> LangDetectResult:
    """Build the final LangDetectResult from 1 or 2 attempt results.

    Single attempt → `agreement="single"`, return its top-1.
    Two attempts → winner is the one with higher top-1 probability.
    `agreement="agree"` if both top-1 languages match, else `"disagree"`.
    """
    infos = tuple(_attempt_info(t, p) for t, p in results)
    if len(infos) == 1:
        info = infos[0]
        winning_probs = results[0][1]
        log(f"lang_detect: single attempt -> {info.top_lang} (p={info.top_prob:.2f})")
        return LangDetectResult(
            language=info.top_lang,
            probabilities=winning_probs,
            agreement="single",
            attempts=infos,
        )

    a, b = infos
    winner_idx = 0 if a.top_prob >= b.top_prob else 1
    winner = infos[winner_idx]
    winning_probs = results[winner_idx][1]
    agreement = "agree" if a.top_lang == b.top_lang else "disagree"
    log(
        f"lang_detect: 2 attempts {agreement} "
        f"(a:{a.top_lang}={a.top_prob:.2f}, b:{b.top_lang}={b.top_prob:.2f}) "
        f"-> winner {winner.top_lang} (p={winner.top_prob:.2f})"
    )
    return LangDetectResult(
        language=winner.top_lang,
        probabilities=winning_probs,
        agreement=agreement,
        attempts=infos,
    )


def detect_language(
    wav_path: str | Path,
    turns: list[DiarTurn],
    *,
    log: Callable[[str], None] = _noop_log,
) -> LangDetectResult:
    """Return a LangDetectResult with the ISO language code and probability dict.

    Falls back to FALLBACK_LANGUAGE (`"uk"`) when there are no turns, the
    longest turn is shorter than MIN_TURN_DURATION_S, or Whisper returns an
    empty probability dict on the first attempt — all are pathological
    states for a successful pipeline run, but the function never raises on
    them so the caller can keep going with a best-effort default.
    The `probabilities` field is None for all fallback paths.
    """
    if not turns:
        log(
            f"lang_detect: pyannote returned no turns — fallback "
            f"{FALLBACK_LANGUAGE!r}. Pass --language explicitly to silence."
        )
        return LangDetectResult(FALLBACK_LANGUAGE, None)

    longest = max(turns, key=lambda t: t.end - t.start)
    longest_duration = longest.end - longest.start
    if longest_duration < MIN_TURN_DURATION_S:
        log(
            f"lang_detect: longest turn is only {longest_duration:.2f}s "
            f"(< {MIN_TURN_DURATION_S}s) — fallback {FALLBACK_LANGUAGE!r}. "
            f"Pass --language explicitly to silence."
        )
        return LangDetectResult(FALLBACK_LANGUAGE, None)

    picked = _pick_attempts(turns)
    if len(picked) == 1:
        log("lang_detect: only 1 eligible turn — running a single attempt.")

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

    audio = load_audio(str(wav_path))
    t0 = time.time()
    model = load_model(WHISPER_REPO_ID)
    log(f"lang_detect: model loaded in {time.time() - t0:.1f}s.")

    # Run all picked attempts against the same loaded model. Reusing the
    # model across attempts is safe within this stage (the per-stage
    # `free_mlx` contract from ADR 0022 talks about sharing across stages,
    # not within one) and saves ~2 s vs reloading the weights.
    results: list[tuple[DiarTurn, dict[str, float]]] = []
    for idx, turn in enumerate(picked, start=1):
        log(
            f"lang_detect: attempt {idx}/{len(picked)} "
            f"on {turn.speaker} [{turn.start:.2f}-{turn.end:.2f}s, "
            f"{turn.end - turn.start:.2f}s]"
        )
        probs = _run_one_attempt(
            model,
            audio,
            turn,
            sample_rate=SAMPLE_RATE,
            n_frames=N_FRAMES,
            log_mel_spectrogram=log_mel_spectrogram,
            pad_or_trim=pad_or_trim,
            log=log,
        )
        if probs is None:
            if idx == 1:
                log(
                    f"lang_detect: attempt 1 returned no probabilities — "
                    f"fallback {FALLBACK_LANGUAGE!r}."
                )
                del model
                free_mlx(log)
                return LangDetectResult(FALLBACK_LANGUAGE, None)
            log("lang_detect: attempt 2 returned no probabilities — using attempt 1.")
            break
        results.append((turn, probs))

    # Drop the Whisper weights and encoder KV cache. The ASR stage will load
    # the same model again with its own buffers; sharing across stages would
    # complicate the per-stage `free_mlx` contract for very little win.
    del model
    free_mlx(log)

    return _aggregate(results, log=log)
