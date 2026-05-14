---
status: accepted
date: 2026-05-12
---

> **Note (v0.25.0):** modules formerly called `diarize` and `identify` are now `diarize_speakers` and `identify_speakers`. Body of this ADR preserves the historical names.

# ADR 0022 — `--language` defaults to detection on the longest pyannote turn

## Context

Up to v0.23.0 the CLI flag `--language` defaulted to `"uk"`. That value flowed into eight pipeline points: as a hint to `mlx_whisper.transcribe()`, into every LLM system prompt via the `<<language>>` placeholder, as the selector for the TL;DR prompt file in `tldr.py:_pick_prompt_name`, as the driver of `_fallback_section`'s title in `structure.py` (`"Розмова"` vs `"Conversation"`), and verbatim in the rendered Markdown header (`🌐 Мова: uk`).

The hard-coded default made sense when the project was Ukrainian-only. After Whisper became the sole ASR backend (ADR 0021), it became a liability: Whisper-large-v3 is a multilingual model that auto-detects language from audio, and shipping every recording with a forced `"uk"` hint loses accuracy on English / mixed-language audio. We wanted to make `--language` optional.

A first attempt set `--language` default to `None` and read the detected language out of `mlx_whisper.transcribe()`'s `result["language"]` field — which is populated by Whisper's internal classifier running on the first 30-second window of the audio. **End-to-end testing on the project's reference Ukrainian recording showed this fails catastrophically:**

- The first 30 s contained "Підбирається" + a 13 s silence + "Подойшов?" — three short utterances with little lexical content.
- Whisper classified this as Russian with confidence 0.85.
- The pipeline then ran ASR with `language="ru"`, which rewrote Ukrainian phonetics as Russian transliteration ("Подойшов?" → "Подошёл?", "Ліворуч" → "Левую ручку", "Відбирається" → "Вернется").
- Downstream LLM stages received Russian prompts: proofread fixed only 24 of 107 segments (vs 34 of 90 with `language="uk"`); structure produced invalid JSON on 2 of 4 chunks; TL;DR was generated in Russian; the rendered header showed `🌐 Мова: ru`.

The transcript was unusable. The diagnosis was clean: Whisper's classifier is fine, but the **first 30 s window is the wrong sample** — it is structurally biased toward silences and acknowledgement utterances rather than substantive speech.

## Decision

Add a new stage **`[4] lang_detect`** between `[3] diarize` and `[5] clearspeech` (which is renumbered from `[4]` to `[5]`; all subsequent stages shift by one). The new stage runs only when `--language` is omitted. The step:

1. Picks the longest pyannote turn from `[3] diarize`'s output (`turns: list[DiarTurn]`).
2. Slices the corresponding range from the raw 16 kHz mono WAV.
3. Computes the log-mel spectrogram and pads/trims to Whisper's 30-second window (`N_FRAMES = 3000`).
4. Calls `model.detect_language(mel)` on Whisper-large-v3-MLX (the same model the ASR stage will later use for transcription).
5. Returns the top-1 ISO language code.

The result is bound to a local `effective_language` variable in `pipeline.run()`. Every downstream stage that previously consumed `options.language` now reads `effective_language` instead: ASR, proofread, identify, structure (with-LLM and no-LLM paths), TL;DR, render. The CLI flag `--language uk|en|...` remains as an explicit override — when set, it skips the lang_detect sub-step entirely (no extra model load) and flows everywhere as before.

**Verification on the project's reference Ukrainian recording:** the longest pyannote turn is SPEAKER_01, 219.54–242.90 s (23.4 s of continuous speech with full Ukrainian sentences). On this fragment, `model.detect_language()` returns `uk` at probability 0.84 vs `ru` at 0.15 — a 5.6× margin. The same Whisper model that gave `ru` on the first 30 s gives `uk` with high confidence on the longest turn, because the input signal is fundamentally better.

**Why this module is named `lang_detect`, not `langdetect`.** The PyPI package [`langdetect`](https://pypi.org/project/langdetect/) is a popular text-based language identifier. We are audio-based and live inside the `voice` namespace, so technically there's no collision; nevertheless, the underscored name reads more clearly in code (`from voice import lang_detect`) and avoids any chance that a reader confuses our module with the PyPI one.

## Consequences

- **New optional stage adds ~3-5 s to the pipeline runtime.** On a 16 GB Mac Whisper-large-v3 loads in ~2 s; `detect_language` itself is sub-second. We load the model, run `detect_language`, then drop weights with `free_mlx(log)` so the ASR stage can load fresh state — that means Whisper loads twice across a single transcription run, which is acceptable given the alternative (catastrophic mis-detection).
- **Failure mode on very short recordings.** If pyannote returns no turns at all, or the longest turn is shorter than `MIN_TURN_DURATION_S = 2.0`, `lang_detect` skips Whisper entirely and returns the fallback `"uk"`, logging a hint that the user should pass `--language` explicitly. The 2-second threshold is below Whisper's recommended utterance length but covers degenerate inputs gracefully.
- **Behaviour with explicit `--language`.** Identical to v0.23.0: no lang_detect call, no extra model load, no log line about detection. This is the recommended way for non-Ukrainian recordings where the user already knows the language, since it saves the 3-5 s detect step.
- **`whisper_asr.transcribe()` signature.** The `language` parameter is now `str | None` (default `None`) instead of `str` (default `"uk"`). `pipeline.run()` always passes a concrete string (`effective_language`), so the change is invisible to it; direct callers of `whisper_asr.transcribe()` that omitted the kwarg now get auto-detect inside `mlx-whisper` rather than a forced `"uk"` hint.
- **No new dependencies.** Same `mlx-whisper` package, same model, same HF cache path — only a new call site (`model.detect_language()` instead of `model.transcribe()`).
- **Render header.** When auto-detect runs, the Markdown header reflects the detected language (`🌐 Мова: en` for an English recording). This is correct behaviour, even though it changes what users who left `--language` at its default will see on non-Ukrainian audio.
- **Architecture diagram and pipeline numbering.** The new stage takes the slot `[4]`; clearspeech / speech2text / merge / proofread / identify / structure / tldr / render all shift by one (clearspeech is now `[5]`, render is `[12]`). The progress label denominator therefore changes from `[N/10]` to `[N/11]` (render is not counted in the progress denominator — `render` rolls into the total wall-clock instead of getting its own `_timed` block).

## Alternatives considered

- **Internal Whisper detect on the first 30 s** (the failed first attempt). Empirically catastrophic on the project's reference recording; rejected.
- **Internal Whisper detect averaged over rolling windows.** Adds complexity and still works on potentially-quiet windows. The longest pyannote turn is a strictly better signal than any blind windowing strategy.
- **A separate language ID model: SpeechBrain VoxLingua107 ECAPA (~86 MB).** Adds a PyTorch dependency and a second model on a 16 GB Mac. Since Whisper-large-v3 already gives correct answers on the longest turn (5.6× margin in the verification experiment), introducing a second model is over-engineering for our actual failure mode.
- **A separate language ID model: MMS LID 126/256/512/1024/4017 (~4 GB).** Same trade-off, worse: 4 GB for a binary uk-vs-ru decision is prohibitive on 16 GB unified memory when we also run Whisper-large-v3 and Qwen2.5-7B in the same session.
- **Two-stage detect: Whisper-large-v3 first, fall back to SpeechBrain on low confidence.** A reasonable future escalation if real recordings show edge cases where `detect_language` on the longest turn is still wrong. Not warranted now — the reference recording resolves cleanly without a second tier.
- **Keep `--language` default `"uk"` and add an opt-in `--language auto`.** Rejected because auto-detect produces correct behaviour for the common Ukrainian case (confidence 0.84+) and adds correctness for non-Ukrainian users by default. Making auto-detect opt-in would keep the toxic forced-uk default for any English / mixed-language user who didn't read the docs.
- **Detect on the raw WAV before diarize, in a separate pre-stage.** Possible, but pyannote's turn boundaries are exactly the signal we want (a "rich" stretch of speech without inter-speaker boundaries), so reusing diarize's output is both correct and cheap. Running detect before diarize would mean blind windowing again.

## Postscript — 2026-05-14: multi-attempt probe on the reference recording

**Question raised.** The reference recording's longest pyannote turn is 23.4 s, i.e. < Whisper's 30-second classifier window. Whisper still classifies it correctly (uk top-1) but the input is silence-padded for the last ~7 s. Could we improve robustness by using more of the available speech — either by concatenating multiple turns to fill the 30 s window, or by running multiple independent attempts and aggregating?

**Probed two hypotheses on the same reference fixture** (`scripts/lang-detect-fill-probe.py`, identical Whisper-large-v3-MLX, sequential CPU/GPU runs on M1/16 GB):

1. **Greedy-fill** — concatenate the longest turn with the next-longest same-speaker turns until the input reaches 30 s of real speech. On the reference: 23.4 s + 10.2 s = 33.6 s, trimmed to 30 s, both from SPEAKER_01.
2. **Per-long-turn detection** — run `detect_language()` independently on every pyannote turn longer than 10 s (4 turns on the reference: 23.4 s, 10.8 s, 10.2 s, 10.1 s), tabulate top-3 probabilities per turn.

**Results.** Margin is `uk_prob / ru_prob`:

| Input | Speech in window | uk | ru | margin |
|---|---|---|---|---|
| Baseline (longest turn, silence-padded) | 23.4 s | 0.735 | 0.238 | **3.09×** |
| Greedy-fill (2 same-speaker turns, no silence) | 30.0 s | 0.549 | 0.422 | 1.30× |
| Single turn SPEAKER_01 [219.5–242.9] | 23.4 s | 0.735 | 0.238 | 3.09× |
| Single turn SPEAKER_00 [173.3–184.2] | 10.8 s | 0.488 | 0.483 | 1.01× |
| Single turn SPEAKER_01 [254.2–264.5] | 10.2 s | 0.551 | 0.415 | 1.33× |
| Single turn SPEAKER_01 [295.2–305.3] | 10.1 s | 0.896 | 0.097 | 9.25× |

**Findings.**

- **Greedy-fill is empirically worse than the silence-padded baseline** (1.30× vs 3.09×). Filling the window with a second turn brings new lexical content that diluted the classifier's confidence rather than reinforcing it; silence is apparently a more neutral filler than additional speech with different lexical context.
- **Single-turn margin does not correlate with turn duration.** The 10.1 s turn produced a 9.25× margin; the 10.8 s turn produced 1.01× (essentially a coin flip). Picking "the longest turn" leaves the strongest individual signals on the table.
- **Top-1 agreement across attempts is the useful signal**, not probability averaging. All four turns voted `uk` top-1 — independent confirmation that the recording really is Ukrainian. Averaging the four probability vectors gives uk=0.611 / ru=0.361 (margin 1.69×), worse than the baseline because weak attempts pull the average down.

**Decisions for the next iteration** (to be implemented as a follow-up — this postscript records the experimental result, the next ADR will record the design choice):

1. **Reject greedy-fill.** Concatenating turns into one input is a worse use of compute than running multiple independent detections.
2. **Keep "single Whisper detection" as the inference unit**, but run it more than once and use the votes for confidence.
3. **Two attempts by default**, with the second attempt picked structurally (longest turn of a different speaker if ≥2 speakers; second-longest turn of the same speaker if 1 speaker).
4. **AGREE/DISAGREE gate, not probability averaging.** When attempts agree on top-1, accept the most-confident attempt's probabilities as the result. When they disagree, either escalate (third attempt) or fall back to the hard-coded `FALLBACK_LANGUAGE` with a log hint asking the user to pass `--language` explicitly. The CLI flag `--language` remains a hard pre-stage override that bypasses lang_detect entirely (unchanged from this ADR's main body).

**Reproducer.** `scripts/lang-detect-fill-probe.py` — runs against the cached pyannote output `~/Downloads/two-speakers-diar-test-ukr.diarize.json` and the cached 16 kHz WAV. Re-running it requires only the Whisper model already used by the ASR stage; no new dependencies.

**Status of this postscript.** Records empirical results that constrain the design space for a future change. The lang_detect code itself is unchanged at the time of writing — the main body of this ADR still describes current behaviour.
