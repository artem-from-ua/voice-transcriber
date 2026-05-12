---
status: accepted
date: 2026-05-12
---

> **Note (v0.25.0):** the module formerly called `clearspeech` is now `clear_speech`. Body of this ADR preserves the historical name.

# ADR 0013 — Defaults for the clearspeech presence effect (3000 Hz / +6 dB / Q=1)

## Context

Issue #48 sub-experiment E2b added a peaking-EQ "presence boost" effect to the clearspeech chain. Issue text proposed nominal cutoffs around 3 kHz / +3..+6 dB. The exact defaults that ship with v0.15.0 are picked by a listening test plus a four-point Metric A grid on the project's reference recording — same recording (`~/Downloads/two-speakers-diar-test-ukr.m4a`) used for ADR 0007 / ADR 0011.

The presence effect is **default off** in v0.15.0: `--clearspeech-chain` defaults to `"agc"`, and users opt into presence by adding it to the chain (`--clearspeech-chain agc,bandpass,presence`). The values in this ADR are what an opt-in user gets without further tuning.

## Decision

Defaults:
- `center_hz = 3000.0` (broadcast presence center; psychoacoustic ceiling reached on this material)
- `boost_db = 6.0` (raised from issue-proposed +3, picked by round-1 listening test)
- `q = 1.0` (typical sharpness; not measured separately — see *How the listening test ran*)
- `presence` enabled? **No** — opt-in only, and **only meaningful inside a chain that also contains `bandpass`**.

## How the listening test ran

Three rounds on a 30-second slice (the same one used in ADR 0011, `/tmp/voice-bp-loop/slice-raw.wav`, 30 s offset). Every candidate was generated with `chain = "agc,bandpass,presence"` so the presence effect was heard in its real production context. Round-by-round binary A/B judgements:

| Round | Anchor | Candidate | Result | Decision |
| ----- | ------ | --------- | ------ | -------- |
| 1 | baseline `agc,bandpass` (no presence) | presence +3 dB | better | continue |
| 1 | presence +3 dB | presence +6 dB | **better** — fricatives sharper, Artem (far speaker) audibly louder | adopt +6 |
| 2 | center 3000 Hz | center 3500 Hz | no difference | continue |
| 2 | center 3000 Hz | center 4000 Hz | no difference | wall hit |
| 3 (Q) | — | — | skipped: round-2 wall reached at Q=1.0; varying Q within ±0.4 octave at this gain is below the perceptual threshold on this material | keep Q=1 |

Round-1 winner: **+6 dB at 3 kHz, Q=1**. Round 2 confirmed that at Q=1.0 the boost band is wide enough (~1 octave at -3 dB) that 3000 Hz and 3500 Hz are perceptually indistinguishable on this slice; the same reasoning made round 3 (Q tuning) cheap-to-skip. Listening fatigue is real — saturating on the same 30-second clip after 4 rounds — and "no perceived difference" is itself a load-bearing signal that we are at the perceptual ceiling for the chosen knobs.

The listener also recorded a perceptual observation worth keeping: Artem (the far-microphone speaker) became audibly louder relative to Ostap (the close-microphone speaker) after presence +6 dB. This is the expected physics — distance attenuates HF faster than LF, so presence boost re-balances the two speakers' fricative energy. The listener correctly flagged "I'm not sure this is good for ASR" — which set up the Metric A test below.

## Metric A — language-drift measurement

Four full ASR runs on the reference recording, all otherwise identical (`--unknown-speaker keep`, no Metric A is measured against post-LLM text; only the raw VibeVoice output in `03-asr.json` is counted). RU-glyph rate is the share of segments containing `ы`/`ъ`/`э` — the same fingerprint of Russian drift ADR 0007 used for AGC and ADR 0011 used for bandpass.

| Run | Chain | Segments | RU-glyph | RU % | ASCII % | avg / median chars |
| --- | ----- | --------:| --------:| ----:| -------:| ------------------:|
| **B3** | `agc` (default) | 39 | 2 | **5.1 %** | 46.2 % | 105 / 72 |
| B1 | `agc,bandpass` | 62 | 4 | 6.5 % | 40.3 % | 66 / 40 |
| **B4** | **`agc,presence`** (no bandpass) | 46 | **29** | **63.0 %** | 41.3 % | 93 / 63 |
| B2 | `agc,bandpass,presence` | 39 | 4 | 10.3 % | 30.8 % | 96 / 86 |

The single most important finding sits in row B4. Adding presence boost **without** bandpass first **catastrophically breaks ASR**: the Russian-glyph rate jumps from 5.1 % (default `agc`) to 63 %. Inspection of the 29 RU-glyph segments confirms this is real language drift, not stray letters — VibeVoice is transcribing the Ukrainian dialogue as full Russian sentences ("Что я тебе сказал? Это шоу?", "Если ты зараз видишь Voice Isolation", etc.).

The physical explanation is consistent with how presence boost interacts with raw recordings. A +6 dB peaking EQ around 3 kHz lifts everything in that band, including the room noise and HF artefacts that bandpass would have removed first. The resulting spectral envelope shifts towards a higher HF-to-LF ratio than the recording naturally has, and VibeVoice — trained on a corpus where Russian is more HF-prominent than Ukrainian (more frequent sibilants and consonant clusters) — interprets that envelope as Russian. Bandpass before presence removes the band that gets disproportionately boosted (5.5–8 kHz noise above the voice cap), so B2 stays close to B1 in drift rate even though it shares B4's gain.

The other comparison — B2 vs B1 (presence's *isolated* contribution on top of bandpass) — shows a smaller, less dramatic shift: RU rate goes from 6.5 % to 10.3 % (+1 segment, but on a smaller denominator). Presence does not help drift rate here either, but at least it does not catastrophically break the transcript when bandpass cleans up the band first.

**Recap on segment count.** Bandpass roughly doubles segmentation (62 vs 39 in the no-bandpass runs) — same finding as ADR 0011: cleaner HF gives VibeVoice's VAD more confidence to slice mid-sentence. Presence reverses that effect: B2's chain (`agc,bandpass,presence`) brings the count back to 39 segments. The mechanism is plausible — presence boost re-introduces gentle continuity in the consonant band that the VAD reads as one speaker stream — but it is not a regression vs B3, just a partial undo of the bandpass effect.

## Consequences

- **Presence stays default-off.** No measurable Metric A win on this recording in any chain; B2 (`agc,bandpass,presence`) is +3.8 pp worse than B3 default. We do not flip default-on.
- **Recommended chain remains `agc,bandpass,presence` when presence is used.** Anyone who enables presence should also enable bandpass — using `agc,presence` is a foot-gun that produces the B4 catastrophe. The `--clearspeech-chain` help string already names `agc,bandpass,presence` as the recommended full chain; we strengthen that wording slightly to "use bandpass whenever you use presence."
- **Default values for the knobs are set as in *Decision*** (`center_hz=3000, boost_db=6.0, q=1.0`). These are what a user gets if they type `--clearspeech-chain agc,bandpass,presence` without per-knob tuning. Override per recording if the speakers, mic, or room differ from this reference.
- **No code guard against `agc,presence` (without bandpass).** We considered adding one and rejected it. Free-order chains were the explicit promise of ADR 0010; rejecting specific orderings re-introduces exactly the kind of allowlist gate we lifted in PR-2. The documentation route is sufficient — the CLI's `--help` and this ADR both flag the foot-gun. Power-users who genuinely want `agc,presence` for an unusual recording can still do it; they just have to mean it.
- **The listening-vs-Metric-A gap stays load-bearing.** Same lesson as ADR 0011: bandpass also won the listening loop but did not win Metric A. With presence we now have two consecutive opt-in effects whose perceptual benefit outruns their ASR-accuracy benefit on this material. The pattern is what motivates leaving both default-off — a perceptual win is real but not transferable.

## Findings worth recording

### 1. Presence is a chain-context-dependent effect, not a standalone one

The B4 catastrophe is the first time in the chain rollout that an effect's **prerequisites** were emp-observable. AGC works in isolation (ADR 0007); bandpass works in isolation (ADR 0011, even if it doesn't help). Presence does not — it amplifies anything in the 2–5 kHz band, which is fine after bandpass cleanup but pathological on the raw spectrum. This is the first concrete argument for keeping the chain-of-effects abstraction (ADR 0010 / 0012) over independent stages: future effects of this kind (de-ess, presumably the same logic) need to compose, and the chain makes composition observable.

### 2. ASR drift can move in surprising directions

We expected presence to *reduce* drift by giving VibeVoice clearer consonant cues for Ukrainian fricatives. Instead, the **language-spectrum priors in the model** turned out to dominate the **acoustic-feature priors** at this gain. Lower gain (round-1 +3 dB) was not tested via Metric A because round 1 picked +6 dB as the listening winner, but it is plausible that a milder +3 dB on the same chain would have been less drift-positive. We mark this as future work — if someone wants to revisit, ADR 0013 records what was measured and what was not.

### 3. The 30-second slice is a poor stand-in for full-recording drift

The slice the listening loop ran on had no audible Russian drift at any gain — all four loop candidates sounded "more or less Ukrainian." Metric A on the full 374-second recording told a completely different story (the slice happens to land on a content-rich passage where both speakers stay in Ukrainian; the drift accumulates over the longer turns elsewhere). The methodology going forward should run Metric A on the full recording before committing to a default-on flip, not just after; cheap perceptual A/B is fine for parameter sweeps, but the acceptance gate is the full-recording number.

## Alternatives considered

- **Default-on presence within `agc,bandpass,presence` chain.** Rejected. Metric A is *worse*, not better, than the default `agc` chain on this recording.
- **Block `agc,presence` (no bandpass) at the validator level.** Rejected. Free-order is a deliberate feature of PR-2 (ADR 0012); turning around and rejecting specific permutations would re-introduce the exact allowlist gate we lifted. The documentation route (this ADR + `--help` warning) is sufficient.
- **Re-run the listening loop on a longer slice that surfaces drift.** Worth doing for future effects; out of scope for v0.15.0 ship. The numbers here are already enough to set defaults.
- **Try lower presence gain (+3 dB) for Metric A.** Acknowledged as future work. Round 1 of the listening loop picked +6 dB as the perceptual winner, and the project's testing pattern is to push the *listener's* favourite through Metric A; if the +6 dB choice catastrophically broke ASR in B4 and merely shifted it in B2, +3 dB might land somewhere between. We did not measure.
- **Drop presence entirely from the chain effects.** Rejected. The listening win is real, and a user with a different recording (e.g. clean studio mic at one distance) may well see Metric A improvements that this reference recording does not. The mechanism (boost fricative band to help ASR consonant discrimination) is sound; we just need the right input for it to pay off.
