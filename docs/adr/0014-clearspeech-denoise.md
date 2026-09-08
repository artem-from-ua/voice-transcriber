---
status: accepted
date: 2026-05-12
---

> **Note (v0.25.0):** the module formerly called `clearspeech` is now `clear_speech`. Body of this ADR preserves the historical name.

# ADR 0014 — Defaults for the clearspeech denoise effect (post-AGC, opt-in)

## Context

Issue #48 sub-experiment E2d adds the fourth clearspeech effect — FFT-domain spectral subtraction of stationary background noise. Unlike the three earlier effects (`agc` from issue #47, `bandpass` from E2a, `presence` from E2b), denoise targets a different failure mode: not signal envelope or per-turn balance, but the **noise floor** the recording itself carries.

Two implementations were evaluated during PR-3:

- **`noisereduce`** — pure Python (numpy/scipy spectral gating), ~3 MB direct dep, stationary + non-stationary modes.
- **`ffmpeg afftdn`** — subprocess shell-out, no new Python deps (ffmpeg already required for stage `[1]`), knobs `nf` (noise floor dB) and `nr` (subtraction amount dB).

Two architectural questions were live going in:

1. **Backend choice.** The plan was to ship both behind a temporary `--clearspeech-denoise-backend` flag, measure on Metric A + perceptual A/B, and drop the loser before merge.
2. **Chain position.** Physics intuition argues for pre-AGC — noise estimation should run on the raw signal, otherwise AGC amplifies the noise floor in quiet turns before denoise sees it. ADR 0014 records what we actually measured.

A third standing question was the **OpenAI Whisper cautionary rule** — the OpenAI team has publicly warned against denoising before Whisper, on the grounds that spectral-subtraction artefacts are read as phonemes. We do not run Whisper; ADR 0014 records whether the rule extends to VibeVoice on Ukrainian content.

## Decision

- **Backend:** `ffmpeg afftdn`. `noisereduce` lost both the listening test and (preliminarily) the Metric A direction. Removed before merge; `noisereduce` direct dep dropped from `pyproject.toml`.
- **Default values:** `nf=-25 dB`, `nr=12 dB` (the agressive variant from the listening loop's round 1).
- **Chain position:** **after AGC** — `agc,denoise,…`. The physics intuition was wrong on this material; Metric A is dramatically worse pre-AGC (see grid below).
- **Default enabled?** **No.** Default `--clearspeech-chain` stays `"agc"`. Denoise is opt-in; using it costs ASR accuracy in every chain we measured.

## How the listening test ran

Both backends ran on the 30-second slice (`/tmp/voice-bp-loop/slice-raw.wav`) used in ADR 0011 / 0013. Each candidate was produced with `chain = ("denoise","agc")` (denoise pre-AGC at this point — pre-AGC was still the assumed canonical order). AGC normalised the post-denoise level so A/B comparisons against the AGC-only baseline were fair.

### Backend A — `noisereduce`

| Round | Anchor | Candidate | Result |
| ----- | ------ | --------- | ------ |
| 1 | baseline (`agc`, no denoise) | `stationary=True` (A1) | **worse** — musical-noise rings on fricatives |
| 1 | baseline | `stationary=False` (A2) | **worse** — same artefacts, plus phase drift on changing noise |
| 2 (sanity) | baseline | `stationary=True, prop_decrease=0.5` (A3) | "maybe slightly better" — at the edge of perception |

The verdict for `noisereduce` on this material: at full strength both modes are perceptibly worse than no denoising; at half-strength the listener cannot reliably tell it apart from baseline. That is the cleanest "no win" signal a perceptual A/B can produce. We did **not** spend Metric A budget on `noisereduce` after this — the listening-loop ceiling already disqualifies it.

### Backend B — `ffmpeg afftdn`

| Round | Anchor | Candidate | Result |
| ----- | ------ | --------- | ------ |
| 1 | baseline (`agc`) | `nf=-25, nr=12` (B1, aggressive) | **better** — background noise audibly reduced, no rings |
| 1 | baseline | `nf=-20, nr=6` (B2, mild) | better than baseline, but **worse than B1** |
| 2 | B1 (`nf=-25`) | `nf=-30` (lower floor) | worse — speech itself starts thinning |
| 2 | B1 | `nf=-20` (higher floor) | worse than B1 (= B2 already showed this) |

Backend B clearly wins the listening test at `(nf=-25, nr=12)`. This is what we took into the Metric A grid.

## Metric A — five-run grid

Each row is a full ASR run on the reference recording with `--unknown-speaker keep`. Cells from prior PRs are reused where the same chain was already measured.

| Run | Chain | Denoise position | Segments | RU-glyph | Share |
| --- | ------ | ---------------- | -------- | -------- | ----- |
| **A1** | `agc` (current default) | — | 39 | 2 | **5.1 %** ⭐ best |
| **A2** | `denoise,agc` | **pre-AGC** | 40 | 11 | **27.5 %** 💥 catastrophe |
| **A3** | `denoise,agc,bandpass,presence` | pre-AGC | 47 | 4 | 8.5 % |
| **A4** | `agc,bandpass,presence` | — | 39 | 4 | 10.3 % (from ADR 0013 B2) |
| **A5** | `agc,denoise` | **post-AGC** | 49 | 4 | 8.2 % |

Three observations are load-bearing:

1. **A1 stays the best.** No chain involving denoise improves on the bare `agc` baseline. The default is left unchanged.
2. **Pre-AGC denoise (A2) is catastrophic.** Adding denoise before AGC to the *same* chain that performs at 5.1 % takes it to 27.5 % — a 5.4× regression. Inspection of the 11 flagged segments confirms real language drift, not stray letters — whole utterances come back in a different language.
3. **Post-AGC denoise (A5) is the salvageable position.** It is still worse than `agc` (8.2 % vs 5.1 %), but 3.4× better than pre-AGC. With `bandpass` and `presence` added (A3 = 8.5 %), the result is essentially the same as A5 — bandpass+presence appear to mask the residual spectral-subtraction artefacts that AGC alone could not fix.

The physics intuition for pre-AGC ("estimate noise on the raw signal") was therefore wrong on this material. AGC's per-turn RMS normalisation is what afftdn needs to read a stable noise floor; without it the per-turn amplitude swings throw off the FFT-frame noise estimate enough that the subtraction artefacts spike in proportion to per-turn gain, and VibeVoice reads the spikes as Russian phonemes.

## Backend comparison summary

| Axis | `noisereduce` | `ffmpeg afftdn` |
| ---- | ------------- | --------------- |
| Listening A/B vs baseline | worse (full) or "maybe slightly better" (half) | clear win at `(nf=-25, nr=12)` |
| Direct deps added | +1 (~3 MB) | 0 |
| Wall-clock on 374 s recording | ~1× audio length | ~0.5× audio length (subprocess + I/O included) |
| Knob ergonomics | `stationary` boolean + `prop_decrease` 0..1 | `nf`/`nr` in dB, intuitive for audio engineers |

`ffmpeg afftdn` wins on every axis. `noisereduce` is dropped from the codebase and `pyproject.toml`.

## OpenAI Whisper rule — does it hold for VibeVoice?

The OpenAI team's caution against denoising before Whisper rests on the observation that spectral subtraction introduces musical-noise artefacts which Whisper interprets as phonemes — typically as random words inserted into otherwise-correct transcripts. Our measurement on VibeVoice confirms a similar but not identical failure mode:

- **Pre-AGC denoise (A2, 27.5 %)** behaves consistently with the OpenAI warning — the model interprets the artefact-laden envelope as a different language altogether (drifts heavily into Russian).
- **Post-AGC denoise (A5, 8.2 %)** still degrades vs baseline but in a much milder way — drift increases by 3.1 pp, not 22.4 pp.

So the rule extends to VibeVoice with two refinements:

1. The **failure direction** in our case is language drift, not random word insertion. That is consistent with VibeVoice's training data (mixed Ukrainian/Russian, where one is a strong neighbour of the other in feature space) versus Whisper's (predominantly English).
2. The **position** in the chain matters more than the OpenAI guidance implies. Pre-AGC denoise is the catastrophic configuration; post-AGC is "merely worse than nothing." If a future ASR engine swap (`#25`, `#50`) lands a model less prone to Russian drift, post-AGC denoise may become viable.

## Consequences

- **Default chain stays `"agc"`.** No user-facing breaking change in v0.16.0. Callers who never touched `--clearspeech-*` get identical behaviour to v0.14.0 / v0.15.0.
- **`KNOWN_EFFECTS` grows by one** to `{agc, bandpass, presence, denoise}`. The chain validator accepts any permutation; in particular, both `denoise,agc` and `agc,denoise` parse fine, but the CLI `--help` for `--clearspeech-chain` carries an explicit warning that **denoise degrades ASR on Ukrainian content** even at its best position.
- **`noisereduce` direct dep removed.** Nothing in the runtime depends on it any more.
- **`--clearspeech-denoise-backend` is gone from the final PR.** It was a tournament knob and the tournament is over.
- **CLI surface for denoise:** two knobs (`--clearspeech-denoise-noise-floor-db`, `--clearspeech-denoise-reduction-db`) that pass straight to `afftdn`.
- **The chain abstraction continues to pay off.** Adding a fourth effect required no pipeline-stage renumbering and no breaking CLI rearrangement; the only API surface added is two knobs. PR-4 (`de-esser`) and the eventual dereverb effect can follow the same shape.

## Alternatives considered

- **Ship `noisereduce` instead.** Rejected at listening test — it never produced a clear A/B win over baseline at any setting we tested. The full-strength modes brought in audible rings; half-strength was at the edge of perception. No reason to carry an extra direct dep for that.
- **Default denoise on at the post-AGC position (A5 = 8.2 %).** Rejected at Metric A — even at its best position denoise is 3.1 pp worse than no denoise. The default-on threshold from the plan was a 20 % relative *reduction* in RU-glyph rate; we got a regression instead.
- **Block `denoise,agc` (the catastrophic order) at the validator level.** Rejected on the same grounds as ADR 0013's `agc,presence` foot-gun: free-order chains are an explicit feature (ADR 0010 / 0012), and rejecting specific permutations re-introduces the allowlist gate PR-2 lifted. Documentation (this ADR + CLI `--help`) is the right tool.
- **RNNoise / Demucs / DeepFilterNet (learned denoisers).** Skipped. The DSP-class denoise here is the right granularity for a clearspeech effect; learned denoisers are bigger than the entire VibeVoice ASR run and represent a different architectural decision. Worth a separate ADR if a recording surfaces where DSP denoising fails for a reason a learned model would fix.
- **Drop denoise entirely.** Tempting given the negative Metric A across all chains. We kept it on three grounds: (a) it is genuinely useful for *listening to the dump artefacts*, where a clean post-denoise WAV is easier on the ear than the raw recording; (b) future recordings with different noise floor characteristics may flip the Metric A direction; (c) ADR 0014 already documents the cautionary findings, so leaving the effect in the codebase as opt-in is informational, not load-bearing.

## Findings worth recording

1. **Physics intuition can lose to per-turn dynamics.** "Estimate noise on the raw signal" is correct in an abstract sense — but our pipeline AGC normalises per-pyannote-turn, which makes the noise floor *more* stable, not less. afftdn benefits from the stabilised envelope. The general lesson: when a stage downstream of yours equalises per-segment dynamics, your noise-floor estimator works better on its output than on the raw input.

2. **Bandpass + presence partially mask denoise artefacts.** A3 (full chain with pre-AGC denoise) lands at 8.5 %, very close to A5 (post-AGC denoise alone, 8.2 %). Bandpass and presence appear to clean up the residual spectral-subtraction noise that AGC alone leaves behind. This is a useful interaction to know for future tuning of recordings where pre-AGC denoise might be the only option (e.g. for a future ASR engine that does its own per-turn normalisation internally).

3. **OpenAI Whisper rule generalises but with a chain-position caveat.** Whisper's "do not denoise" guidance is real for VibeVoice too — but the severity is *highly* sensitive to where in the chain denoise sits. This is worth keeping in mind whenever a new ASR engine lands (#25, #50): re-run the A2-vs-A5 comparison on the new engine before declaring denoise universally unhelpful.

4. **Listening tests catch backend losers cheaply.** Spending GPU on Metric A only for the listening-test winner (ffmpeg afftdn) saved ~25 minutes of ASR time vs running the full grid for both backends. The pattern from ADRs 0011 / 0013 was that perceptual A/B and Metric A frequently disagree — but they agree at least on which backend is *audibly broken*, and that's enough to prune the search space.
