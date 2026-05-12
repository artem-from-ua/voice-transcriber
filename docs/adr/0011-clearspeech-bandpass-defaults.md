---
status: accepted
date: 2026-05-11
---

> **Note (v0.25.0):** the module formerly called `clearspeech` is now `clear_speech`. Body of this ADR preserves the historical name.

# ADR 0011 — Defaults for the clearspeech bandpass effect (150 Hz – 5500 Hz)

## Context

Issue #48 sub-experiment E2a added a Butterworth bandpass to the clearspeech chain (see [ADR 0010](0010-clearspeech-chain.md) for the chain design). The issue text proposed nominal cutoffs of 80 Hz – 10 kHz. Two facts narrowed those down before any listening:

- The pipeline operates at **16 kHz mono PCM_16** — the format the ffmpeg stage produces — so the high cut is capped at the 8 kHz Nyquist frequency. The issue's 10 kHz upper bound is unreachable at this sample rate.
- Issue #48's defaults were a priori choices, not measurements. Acceptance criteria for the wider issue (Metric A: language drift; sanity-listen) demanded an empirical pass before the numbers shipped as defaults.

The bandpass is **default off** in v0.14.0 (`--clearspeech-bandpass` opts in), so these are the values a user gets when they enable the flag without further tuning.

## Decision

Defaults:
- `bandpass_low_hz = 150.0` (raised from issue-proposed 80)
- `bandpass_high_hz = 5500.0` (lowered from issue-proposed 10000, then from the Nyquist-capped 7900)
- `bandpass` enabled? **No** (default off — experimental opt-in, pending Metric A on a full ASR run)

These cutoffs were picked by a five-round binary-search listening test on a 30-second slice of `~/Downloads/two-speakers-diar-test-ukr.m4a` (the same reference recording used for ADR 0007). The slice was taken at the 30 s offset so it contains both speakers at their typical microphone distances (Ostap close to the phone, Artem across the room — see ADR 0007 for the room-physics context).

Each round produced candidate WAVs at `/tmp/voice-bp-loop/out-NN-bp-LOW-HIGH-NOTE.wav`, all sharing the same AGC pre-step (transplanted v0.13.0 logic, 49.4% per-turn RMS spread reduction) so only the bandpass change varied between A/B comparisons. Listening was binary: better / worse / same vs the previous round's winner.

| Round | Anchor | Candidate | Result | Decision |
| ----- | ------ | --------- | ------ | -------- |
| 1 | none (AGC-only) | `80 / 7900` (issue-default) | **better** | bandpass helps in principle |
| 2 | `80 / 7900` | `80 / 5500` | **better** | high=5500 cleaner than 7900; the 5.5–8 kHz band in this recording was mostly noise |
| 3 | `80 / 5500` | `100 / 5500` | **better** | low=100 trims rumble without thinning the voice |
| 3.5 | `100 / 5500` | `120 / 5500` | **better** | 120 cuts further without losing body |
| 4 | `120 / 5500` | `135 / 5500` | slightly better | continuing |
| 4.5 | `135 / 5500` | `150 / 5500` | slightly better | very close to the wall |
| 5 | `150 / 5500` | `165 / 5500` | **no difference** | wall hit — stop |

The round-2 win on the **high** side surfaced a perceptual artefact worth recording (see *Findings*, below).

## Findings worth recording

### 1. The 5.5–8 kHz band was carrying noise, not articulation

Round 1 already preferred `80 / 7900` over no bandpass. Round 2 then preferred `80 / 5500` over `80 / 7900` — trimming a full 2.4 kHz off the top. The listener's report on the round-2 pair was:

> *"`07` is clearer, but the background noise that was up high in `05` seems to have moved down lower."*

This is a psychoacoustic effect, not a physics one. The low-end rumble (~80–200 Hz) was present in both `05` and `07` — `07` just removed the high-frequency hash that was masking it. The unmasked rumble *felt* louder relative to the now-quiet upper band, but its absolute level was unchanged. The diagnostic value of this report is that it correctly predicted rounds 3+: with the high band cleaned, the low cut became the next live tuning surface.

### 2. The 80 Hz low cut from issue #48 was way too generous for this recording

Issue #48 picked 80 Hz as a standard "everything above sub-bass rumble" cut. On a clean headset recording that would be fine. On this material — phone microphone, room with HVAC, mild handling noise — there is rumble all the way up to ~120 Hz. Rounds 3–5 walked the low cut from 80 → 100 → 120 → 135 → 150 with each step audibly cleaner; 165 Hz was the first point where no further improvement was perceived (and where some chest resonance for low male voices starts to drop).

The 150 Hz default is therefore tuned to this kind of input — phone-recorded, mid-quality, two speakers at different distances. A user recording on a directional desk mic in a treated room would likely prefer a lower cut. This is why bandpass is opt-in, not default-on, and why the flag exposes both cutoffs.

### 3. Listening fatigue is real; stop when the listener stops being sure

At round 4 the report shifted from confident "better" to hedging "mildly better" — a sign that the ear was saturating on the 30-second clip. Round 5 confirmed the wall ("no difference") and stopped. The PR could have pushed for 165 or 180 with another fresh-ear session, but the marginal gain past 150 was clearly under the noise floor of human A/B judgement on this material, and the asymmetric risk (overcutting low → tinny voice on lower-fundamental speakers; undercutting → mild residual rumble) favours stopping shy of the wall.

## Metric A — language-drift measurement

After the listening loop set the cutoffs, two full ASR runs were executed on the reference recording — one with bandpass off (AGC only, the v0.13.0-equivalent baseline) and one with bandpass on at `150 / 5500`. Russian-only glyphs `ы`/`ъ`/`э` are an unambiguous fingerprint of VibeVoice sliding into Russian (the same metric ADR 0007 used for AGC).

| Run | ASR segments | RU-glyph segments | Share |
| --- | ------------ | ----------------- | ----- |
| Baseline (`--no-clearspeech-bandpass`, AGC only) | 39 | 2 | 5.1 % |
| Bandpass on (`--clearspeech-bandpass`, 150 / 5500) | **62** | 4 | 6.5 % |

The expected result, by analogy with AGC's −69 % drop in ADR 0007, would be a clear reduction in the RU-glyph share. **Bandpass did not deliver that.** Absolute RU-glyph count went up (+2 segments); the share went up slightly (+1.4 pp); the per-segment drift rate is statistically indistinguishable.

What did change, dramatically, was the **segment count**: 39 → 62 (+59 %). VibeVoice's internal VAD slices the cleaned audio into many more chunks. Inspecting the new chunks, this is segmentation, not regression — short English code-switches that baseline ran into one Ukrainian segment ("Hugging Face", "couple of directions", "White spectrum недоступний") now land as their own short segments. Several of the new short segments do carry RU-glyph drift, which is what drives the absolute count up.

Conclusion: bandpass changes how the ASR slices the timeline, not how often it confuses Ukrainian for Russian on a per-second-of-speech basis. The perceptual listening win was real (each individual round in the table above stands), but it does **not** translate into a Metric A win on this recording.

## Consequences

- Default-on for the AGC effect (kept from v0.13.0); default-**off** for bandpass. The listening-loop produced a perceptual win, but Metric A is flat — without an objective improvement that matches AGC's −69 % bar, opt-in is the responsible ship.
- `--clearspeech-bandpass` with no further arguments applies `150 / 5500` to the test-recording-class of inputs out of the box. Other recording conditions will likely need explicit `--clearspeech-bandpass-low-hz` / `--clearspeech-bandpass-high-hz`.
- The 5500 Hz high cut sits noticeably below the 16 kHz Nyquist (8 kHz). The pipeline does not enforce this gap — `_validate_bandpass_cutoffs` accepts any `0 < low < high < 8000`. Users who experimentally want more high-frequency information can raise the cut, but the default reflects what was perceptually best on the reference recording.
- The Metric A result here applies to a single recording; future E2-sub-experiments (E2b presence, E2c de-ess, E2d denoise, E2e dereverb) may shift the picture by addressing different artefacts. The default-on question is re-opened when the clearspeech chain has more than one optional effect to compose.

## Alternatives considered

- **Ship issue-proposed defaults (`80 / 7900` after Nyquist clamp).** Rejected: round 1 already showed `80 / 7900` is the *threshold* of usefulness, not the optimum. Walking to `150 / 5500` raised perceived quality on every round-by-round comparison.
- **Auto-tune per recording (estimate the input's noise floor and set cuts).** Tempting but the same trap as auto-tuning AGC (ADR 0007, *Alternatives considered*): a recording that happens to be clean would get a useless tight cut, hiding the issue the next time levels drift. Static defaults are more predictable; users with unusual material set the flags.
- **Default-on bandpass with `150 / 5500`.** Rejected. The post-listening-loop Metric A run (table above) showed flat language drift (5.1 % → 6.5 %) with a large segmentation shift (39 → 62 segments). Opt-in stays. Re-opens if a future E2-sub-experiment moves the needle.
- **Symmetric 12 dB/octave (Butterworth order 2) instead of order 4.** Order 2 is gentler, less ringing on transients. Not tested — the user did not report ringing or transient smearing on any of the order-4 candidates, and `sosfiltfilt` cancels phase distortion either way. Order 4 stays; revisit if a future recording shows audible artefacts.
- **Replace the bandpass with high-pass alone (no high cut at all).** The round-2 result rules this out — `80 / 5500` was preferred over `80 / 7900`, so the high cut is contributing perceptual value, not just a fixed safety margin against Nyquist artefacts.
