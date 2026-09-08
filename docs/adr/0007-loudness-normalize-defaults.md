---
status: accepted
date: 2026-05-11
---

# ADR 0007 — Defaults for the loudness-normalize stage (target -20 dBFS, max gain 16 dB)

## Context

Issue #47 introduced a per-segment AGC stage between diarisation and ASR. Issue text proposed initial defaults of `target_dbfs=-20.0` and `max_gain_db=12.0`. Those are reasonable a-priori choices (target = broadcast-ish loudness; ceiling = standard "do not amplify breath/silence" rule), but the issue acceptance criteria are stringent enough that the defaults were never going to land unmeasured.

The concrete numbers came out of a listening loop on the project's test recording (`~/Downloads/<reference-recording>.m4a`, 6 minutes, 2 speakers at very different microphone distances), combined with two objective metrics:

- **Metric A — language drift.** Count ASR segments whose output contains Cyrillic glyphs that exist in Russian but not in Ukrainian (`ы`, `ъ`, `э`). Lower is better — those glyphs are an unambiguous fingerprint of VibeVoice sliding into Russian on hard inputs. Issue target: ≥ 20% relative reduction with normalisation on.
- **Metric B — per-pyannote-turn RMS spread.** Std-dev of per-turn RMS in dBFS, comparing raw WAV vs the normalised WAV. Lower is better — this is the direct sanity check that AGC did its job. Issue target: ≥ 50% relative reduction.

## Decision

Defaults:
- `target_dbfs = -20.0` (kept as-issued)
- `max_gain_db = 16.0` (raised from 12.0)
- `crossfade_ms = 100.0` (kept as-issued)
- `SILENT_RMS_THRESHOLD = 1e-6` (skip turns below this RMS — leave at unity gain to avoid amplifying mic noise inside a pyannote-extended turn)

`16.0` was chosen over `12.0` and `18.0` by binary listening test plus per-turn measurement on the full recording. The numbers:

| `max_gain_db` | Per-turn RMS std-dev | Drop vs raw | Turns hitting ceiling |
| ------------- | -------------------- | ----------- | --------------------- |
| (raw)         | 6.40 dB              | —           | 0/111                 |
| **12.0**      | 4.36 dB              | -32%        | 54/111 (49%)          |
| 15.0          | 3.51 dB              | -45%        | 33/111 (30%)          |
| **16.0** ✅   | 3.24 dB              | -49%        | 26/111 (23%)          |
| 18.0          | 2.73 dB              | -57%        | 14/111 (13%)          |

And the language-drift number, computed by re-running the full ASR pipeline:

| Pipeline                                | RU-glyph segments | Share | Drop vs OFF |
| --------------------------------------- | ----------------- | ----- | ----------- |
| `--no-loudness-normalize` (baseline)    | 10 / 53           | 18.9% | —           |
| `--loudness-max-gain-db 12.0`           | 3 / 45            | 6.7%  | -65%        |
| **`--loudness-max-gain-db 16.0`** ✅    | **2 / 34**        | **5.9%** | **-69%** |

Metric A passes by a wide margin at both ceilings (issue target is ≥20%). Metric B is the gating constraint, and `16.0` is the smallest ceiling that gets within the rounding error of the 50% target (49.4%). Anything higher trades real perceptual quality for a fractional metric win. The PR declines to chase the literal 50% threshold and documents the result honestly.

## How the listening test ran

Three 22–30-second slices from different parts of the recording were normalised with `max_gain ∈ {12, 15, 16, 18}` and judged by ear:

1. **A — typical (30–60 s).** Conversational back-and-forth. Reasonable level on both speakers; mostly tests "does the stage do harm on easy material."
2. **B — very quiet (205–227 s).** Contains a turn at -39.8 dBFS — the user must boost ~20 dB to reach target. This is where the ceiling shows.
3. **C — late quiet (340–362 s).** Sanity check on a second quiet stretch, less extreme than B.

User feedback on the slices:

- **A:** `g12` evens levels well; `g18` is too much (the loud speaker no longer feels louder than they "should").
- **B:** `g18` is clearly better than `g12` — the quiet speaker becomes audible.
- **C:** `g12` is already sufficient.

The pattern said the right value was *between* 12 and 18 — closer to 18 on hard turns but never as aggressive as 18 on easy turns. `g16` was the user's intuition and it held up: it sits at -49% spread reduction (vs -32% for g12 and -57% for g18), and on slice A it does not feel "too much."

## Two non-obvious findings worth recording

Both of these came out of the tuning process and would be expensive to re-discover.

### 1. pyannote classifies long non-speech as a "turn"

The first tuning slice (15–45 s) had a turn at `[5.36-11.03] SPEAKER_01` at -36 dBFS that AGC happily boosted by +12 dB. User listening revealed it was **one speaker inhaling from a vape pen**, not speech. pyannote's exclusive timeline doesn't distinguish "speech" from "loud breath" — it just splits the audio between the active diariser labels.

`SILENT_RMS_THRESHOLD = 1e-6` does not save us here: human sounds at -36 dBFS are far above silence. Only the raw `[Human Sounds]` markers (where pyannote *did* mark non-speech, with `speaker=None`) get skipped because our gain envelope leaves the gap regions at unity. Anything pyannote tagged as a speaker turn is fair game for AGC.

Mitigation: none in #47 itself. The follow-up is to apply spectral preprocessing (#48) and/or stricter VAD upstream. Out of scope for the amplitude-only fix.

### 2. The "loud speaker" in the file is the distant one in the room

It is easy to assume that the louder voice in a recording is the closer speaker. On this material the opposite is true: the far-microphone speaker sits across the room from the phone but registers at -19 to -25 dBFS, while the near-microphone speaker sits right next to the phone but registers at -32 to -36 dBFS. The likely culprit is the iPhone's voice-isolation feature, which is mentioned by the speakers themselves on the very recording being analysed.

Practical consequence: when the user reported "I cannot hear a difference between baseline and normalised" on the first listening pass, the temptation was to crank `max_gain` higher. It turned out the per-turn boost was actually working as expected — but the perceived "improvement" was on the *distant* speaker becoming audible relative to the *near* speaker, which read on ear as "the distant speaker is louder now," not "the near speaker became normalised." Without the room-physics context the data would have been misread.

This is a property of the test recording, not the algorithm. Future tuning on a different recording should expect different patterns.

## Consequences

- AGC catches reasonable distance variation (~16 dB) at default. Speakers who are out by more than that will still be muffled — by design, to avoid amplifying breath, room reverb, or pyannote-misclassified non-speech.
- Headroom: the loudest turn after normalisation reaches 0 dBFS peak on this recording (per the per-turn measurements above). The implementation does explicit `np.clip(out, -1.0, 1.0 - 1e-7)` so PCM_16 quantisation does not roll over, but users with hotter source material may want a lower `target_dbfs`.
- For users whose recording has uniform levels (everyone on a headset, broadcast setup), the stage is mostly a no-op anyway and the defaults cost nothing. For users with a "phone on the desk" setup like our test recording, they get a measurable ASR-quality win without touching the knobs.
- Future ASR backends (#25, #50) will be compared *against the normalised baseline*. The 16.0 default is what they will see.

## Alternatives considered

- **Keep `max_gain_db=12.0`.** Cheaper from a clipping/headroom standpoint, but fails Metric B by a wide margin (-32% vs the 50% target) and leaves half of all turns hitting the ceiling. Acceptable on its own merits if the user weighs "do no harm" higher than "fix the quiet speaker," but does not match what the listening test said.
- **Raise to `max_gain_db=18.0`.** Best on Metric B (-57%), but user found it audibly excessive on the typical-conversation slice. The marginal Metric B win is not worth that perceptual loss.
- **LUFS instead of RMS.** Broadcast standard, more accurate perceptually. Adds the `pyloudnorm` dep and ~10× the per-turn compute. RMS proved sufficient on this material; promote if a later recording shows RMS-vs-LUFS divergence.
- **Auto-tune per recording.** Compute the recording's existing RMS distribution and pick `max_gain` to fit. Tempting, but the failure mode is bad: a recording that happens to be uniform would get an artificially low ceiling, hiding the problem the next time levels drift. Static default is more predictable.
- **Separate ceiling for boost vs cut.** AGC currently allows unbounded attenuation but clamps boost. We considered also clamping cut, but the asymmetry is correct: a loud turn dropped by 20 dB just sits at -40 dBFS, which is fine; a quiet turn boosted by 20 dB amplifies breath. The clamp belongs only on the boost side.

## Note (v0.14.0)

The standalone `loudness_normalize` stage was absorbed into the new `clearspeech` chain (see [ADR 0010](0010-clearspeech-chain.md)). The numbers in this ADR — `target_dbfs = -20.0`, `max_gain_db = 16.0`, `crossfade_ms = 100.0` — are unchanged; they live on as the `agc` effect's defaults inside that chain (`--clearspeech-agc-target-dbfs`, `--clearspeech-agc-max-gain-db`). The CLI rename is the only behaviour-visible change: `--no-loudness-normalize` → `--no-clearspeech-agc`, `--loudness-*` → `--clearspeech-agc-*`. The empirical reasoning above stays load-bearing for future tuning.
