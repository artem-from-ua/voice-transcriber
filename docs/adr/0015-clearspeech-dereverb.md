---
status: accepted
date: 2026-05-12
---

# ADR 0015 — Defaults for the clearspeech dereverb effect (per-turn Lebart-Polack, opt-in)

## Context

Issue #48 was closed with three of four sub-experiments shipped (E2a / E2b / E2d as opt-in; E2c declined empirically). E2e (dereverb) was always tracked as a separate issue. PR-4 adds it as the fifth `clearspeech` effect.

The user's constraints going in were strict:
- **No AI.** No DeepFilterNet, no Demucs, no learned denoisers.
- **Per-segment, per-speaker.** Each `pyannote` turn should get its own RT60 estimate so the effect adapts to a speaker moving in the room.
- **ffmpeg if possible.** No new heavy deps.

Discovery: **ffmpeg 8.1.1 has no native dereverb filter.** `aecho` adds echo, `deconvolve` is video-only, `afftdn` denoises stationary noise rather than reverb tails. The implementation therefore had to be hand-rolled in scipy/numpy.

The chosen algorithm is the **Lebart-Polack** spectral-subtraction dereverb (Acta Acustica, 2001). Mono-signal, no ML, well-understood failure modes. For each STFT frame it estimates how much energy is reverberation from past frames (running per-bin IIR keyed to the segment's RT60), then subtracts it from the magnitude before iSTFT. RT60 itself is estimated per turn via Schroeder backward integration on a 500-2000 Hz bandpassed envelope.

## Decision

- **Algorithm:** Lebart-Polack spectral subtraction (custom scipy implementation).
- **RT60 estimator:** Schroeder backward integration, fitted on the -5 dB to -25 dB section of the decay curve. Returns `None` on unreliable fits; the caller falls back to `_DEREVERB_FALLBACK_RT60_S = 0.4 s`.
- **Per-segment:** every `pyannote` turn ≥ 200 ms gets its own RT60 estimate; shorter turns are left untouched.
- **Defaults:** `rt60_floor_ms=300`, `subtract_factor=1.0`, `crossfade_ms=50`. The floor is an empirical "minimum RT60 to assume" — clamps the estimator's output up when it returns suspiciously low values.
- **Chain position:** opt-in only via `--clearspeech-chain`. Recommended position is documented as **after AGC** for users who do enable it, but see *Metric A* below — the position-vs-quality picture is non-trivial.
- **Default chain stays `"agc"`.** Even though the isolated `dereverb` Metric A run came out at 0 % RU-glyph drift (down from the 5.1 % `agc` baseline), the side-by-side text comparison showed that the 0 % is a **Simpson's-paradox artefact**, not a genuine ASR improvement. See *Findings worth recording* below.

## How the listening test ran

Three rounds on the 30-second slice (`/tmp/voice-bp-loop/slice-raw.wav`), all with the candidate chain `("dereverb","agc")` so the post-dereverb signal still gets AGC for a fair perceptual A/B against the no-dereverb baseline.

| Round | Anchor | Candidate | Result |
| ----- | ------ | --------- | ------ |
| 1 | baseline (`agc`, no dereverb) | `subtract_factor=1.0` (full) | **better** — late tails audibly shorter, direct sound intact |
| 1 | baseline | `subtract_factor=0.5` (gentle) | **better** as well — slightly less of the same effect |
| 1.5 | full vs gentle | direct A/B | barely audible difference; user could not call it |
| 2 | full (rt60_floor=300) | rt60_floor=200 | identical-sounding (estimator already produced ~585 ms RT60, floor doesn't bite) |
| 2 | full (rt60_floor=300) | rt60_floor=500 | "hard to say — needs measurement" — user could not pick a winner perceptually |

The estimator's behaviour was internally consistent: 14 of 15 turns produced an RT60 estimate ≥ 500 ms, averaging 585 ms across the slice. This is a phone-recorded living-room signal, and the estimator landing in the 500-600 ms range is plausible.

The listening loop is the **first one in the four-PR DSP series** where the candidates audibly beat the baseline at round 1 — `bandpass` won round 1 too but only after the comparison shifted to the 80/7900-Hz cuts, and `presence` / `denoise` had ambivalent round-1 outcomes. Dereverb is the cleanest perceptual win in the series.

## Metric A — four-run grid

Four full ASR runs on the reference recording, all in the new "fast mode" (`--no-proofread --no-structure --no-tldr`) since Metric A reads `03-asr.json` and LLM stages don't change it. Two existing rows from PR-2 (A1 from ADR 0013 B3, A4 from PR-2 B2 baseline) reused as anchors.

| Run | Chain | Segments | RU-glyph | Share |
| --- | ------ | --------:| --------:| ----- |
| **A1** | `agc` (current default, cached) | 39 | 2 | **5.1 %** ⭐ |
| A2 | `dereverb,agc` (pre-AGC) | 42 | 3 | 7.1 % |
| A3 | `agc,dereverb` (post-AGC) | 38 | 4 | 10.5 % |
| **A4** | **`dereverb`** (isolated, no AGC) | 41 | **0** | **0.0 %** ⚠️ Simpson's paradox |

A4's 0 % looks like a runaway win — until you read the actual transcripts side-by-side against A1.

## Side-by-side transcript audit

A1 (`agc`) and A4 (`dereverb` isolated) produce dramatically different texts on the same recording. Character-level similarity (`difflib`) between the two transcripts is **0.146** — well below the 0.4-0.7 typical for two variants of the same ASR pass. Word-set Jaccard is 0.345; A1 has 367 unique words, A4 has 327, and 189 of A1's words don't appear at all in A4.

Looking at the actual segments:

- A4 produces **9 % less transcribed text** (3 747 chars vs 4 119) than A1.
- A4 routes ~30 seconds of mid-recording speech into `[Human Sounds]` / `[Breath]` / `[Noise]` markers instead of transcribing it.
- One mid-recording segment that A1 captures as `"У мене там є account, я не дуже шарю. Там якось можна закинути бабло і використовувати подібні моделі. І в принципі ми можемо протестувати."` becomes A4's `"Хоча і те, що ми переговарювали знову."` — different content at a different timestamp.
- The last segments of A4 contain phrases like `"Потому что… по йому голову"` and `"шойго ми штукавий ресурс короче який треба розглядати"` — markedly more gibberish than A1's `"Тобто це, мабуть, просто рекламний вайданчик? …"`

A2 (`dereverb,agc`) and A1 also diverge heavily (char-similarity 0.214) — A2 *concatenates* shorter turns into long Russian-leaning blocks (`"Доброе утро, все. Вот зашел?"`, `"Ну я вот запустил, короче... Так и куплять шторку ведь открывателя?..."`), which is what drives its 7.1 % RU rate up.

The pattern that the four runs show, when read together:

| Run | What dereverb did | What VibeVoice did with it |
| --- | --- | --- |
| A1 (no dereverb) | — | balanced transcription, mild Russian drift |
| A2 (pre-AGC) | tightened tails, AGC then amplified residual artefacts and noise floor | VAD glued short turns into long blocks → more Russian drift |
| A3 (post-AGC) | AGC first lifted quiet-turn reverb tails to speech level, dereverb couldn't tell them apart | over-cleaning of speech, even worse drift |
| A4 (isolated) | dereverb on raw signal, distant speaker stays quiet, no AGC compensation | VAD reads quiet Ukrainian as `[Human Sounds]` / `[Breath]` and drops it → **0 RU-glyph segments because the segments were never transcribed in the first place** |

The 0 % in A4 is **rate-not-volume**: less Ukrainian gets transcribed, so the absolute count of Russian-drift segments drops to zero, but so does the volume of correctly-transcribed Ukrainian.

This is the same shape as Simpson's paradox in statistics: a metric improves on one slice while the underlying quantity it's a rate of has shrunk past the point of usefulness. The PR-2 segmentation-shift finding (bandpass +59 % segments, presence +0 % segments, ADR 0011/0013) and the PR-3 catastrophe (denoise pre-AGC at 27.5 %) were both warning signs that this metric, on its own, doesn't fully describe what's happening to the transcript. PR-4's A4 makes the limitation impossible to ignore.

## OpenAI Whisper rule — does it apply here?

The OpenAI cautionary rule was about *spectral subtraction* (`afftdn`-class denoising) before Whisper. Lebart-Polack is also spectral subtraction, just keyed to RT60 rather than noise floor. The rule generalises directly, and ADR 0014 already showed that pre-AGC denoising on VibeVoice produces Russian drift in the same shape. PR-4 confirms it again with a different signal-conditioning target (reverb tails instead of stationary noise): pre-AGC spectral subtraction produces the strongest drift (A2), post-AGC is "merely worse" (A3), isolated-without-AGC suppresses transcript volume (A4).

The cross-effect lesson, repeated four times now: **VibeVoice trusts AGC-normalised inputs and reacts to spectral-subtraction artefacts as language-shift cues**. Re-checking this rule is the right move when the ASR engine changes (#25, #50).

## Consequences

- **Default chain stays `"agc"`.** PR-4 ships dereverb as a fifth opt-in effect; no breaking changes to default behaviour.
- **`KNOWN_EFFECTS` grows to 5.** The chain validator continues to accept any permutation; the recommended chain for users who enable dereverb is documented in `--clearspeech-chain` help.
- **`dereverb` is the first effect that genuinely needs `agc_turns` (or its `dereverb_turns` alias).** Earlier opt-in effects (bandpass / presence / denoise) could run on the raw WAV without diarisation; dereverb cannot, because each turn needs its own RT60 estimate. Without turns, the dispatcher raises `ClearspeechError`.
- **No new dependencies.** scipy already had `stft` / `istft` / `hilbert` / `sosfilt`.
- **A follow-up issue is required** to validate dereverb (and the other opt-in effects) against a different ASR engine and a reference transcript. The Simpson's-paradox finding here is the strongest single argument the project has for *not* trusting RU-glyph drift in isolation — we need an absolute-accuracy comparator before any of these effects can move to default-on.

## Alternatives considered

- **pyroomacoustics WPE.** Multi-channel-optimised; mono performance documented as suboptimal. ~50 MB dep just to underperform a 150-line scipy implementation on our material. Rejected.
- **Per-speaker average RT60 instead of per-turn.** Stable estimate from more frames, but loses sensitivity to a speaker moving in the room. The user explicitly preferred per-turn; the per-segment RT60s vary across turns in the loop (300-800 ms range), so the freedom is real.
- **EMA-smoothed per-speaker RT60.** Considered as a refinement. Deferred: the per-turn estimator already produces stable values within a speaker (most turns from each speaker land in a ±150 ms band), so EMA buys little over raw per-turn. Worth revisiting if a future recording shows wider per-speaker variance.
- **Default-on dereverb on the A4 result.** The empirical case looks airtight (0 % vs 5.1 %, well past the 20 % gate). Rejected because the side-by-side audit shows A4 is suppressing transcript volume, not improving accuracy. Default-on would silently lose ~9 % of the user's transcribed Ukrainian content.
- **Default-on `dereverb,agc` chain.** A2's 7.1 % RU rate is worse than A1's 5.1 %, and the segment-merging pattern (long Russian-leaning blocks at the start) is qualitatively worse, not better. No case for flipping.
- **Reject the effect outright on the same Simpson's-paradox grounds.** Considered. Kept the effect in the codebase as opt-in because (a) it does perceptual work the listening loop confirmed, (b) the segmentation behaviour is well-characterised in this ADR so users opting in know what they get, and (c) the cross-engine validation issue may show that some future ASR engine *doesn't* exhibit the Russian-drift pattern, at which point dereverb's opt-in availability matters.

## Findings worth recording

1. **Simpson's paradox in RU-drift metric.** The clearest sign yet that "% of segments containing ы/ъ/э" is a *rate*, not a *quantity*. A4 looks like it eliminated drift entirely — and it did, by eliminating ~9 % of the actual Ukrainian transcript in the same move. Future metric-driven decisions in this codebase should pair the rate with a *total transcribed character count* sanity check. Adding that automatically to the dump artefact is plausible follow-up work.

2. **Dereverb is the first DSP effect in the series that genuinely needed per-turn information from diarisation.** AGC needs turns to set the gain envelope; bandpass / presence / denoise can all run on the raw WAV. Dereverb's RT60 estimator only works on speech-active windows long enough to fit a Schroeder decay — random spans of speech-plus-silence pollute the estimate. This validates the architectural choice from ADR 0010 to pass `agc_turns` (now also `dereverb_turns`) through the chain rather than computing turns on the fly.

3. **The "physics intuition" vs Metric A pattern continues, but reverses direction.** ADR 0014 (denoise) showed pre-AGC catastrophic, post-AGC merely-degrading — i.e. physics-correct order was empirically wrong. PR-4 shows pre-AGC dereverb *less bad* (7.1 %) than post-AGC dereverb (10.5 %), with isolated dereverb taking a third path entirely. The interaction with AGC is effect-specific; there is no universally-right chain position.

4. **Listening tests and Metric A agree only in extremes.** The perceptual loop for dereverb was the cleanest win in the series, and Metric A on the isolated-dereverb run (A4) came out at 0 %. They appeared to agree — and the side-by-side audit showed they were both right *about different things*. Metric A's "0 %" measured rate; the listener's "audibly better" measured tail-suppression; the audit measured transcript volume. Three different things all called "quality" by different observers. Worth keeping this lesson visible: a Metric A win is necessary but not sufficient for shipping.

5. **The case for ASR-engine cross-validation just got concrete.** PRs 1-3 left the question "is the wall here in the DSP chain or in VibeVoice's spectrum priors?" hanging on theoretical grounds. PR-4 makes the question urgent: A4's behaviour is explainable entirely by "VibeVoice's VAD reads quiet Ukrainian speech as non-speech", which is a model-specific failure mode. A reference transcript plus a second ASR (Whisper, MS Azure) would let us measure absolute accuracy and decide whether *any* of these opt-in effects actually help when the wall is no longer the model.
