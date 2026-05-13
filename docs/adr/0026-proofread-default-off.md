---
status: accepted
date: 2026-05-13
see_also: [0005, 0017, 0021]
---

# 0026 — Proofread stage: default off, opt-in via `--proofread`

## Context

The proofread stage (`src/voice/proofread.py:fix_asr_errors`) was introduced
when the project shipped on the **VibeVoice** ASR backend, which produced
systematic Russian-phonetic drift on Ukrainian conversational audio
([ADR 0005](0005-segment-level-asr-proofread.md)). On VibeVoice output it
rewrote ~26 of 54 segments (~48 % hit-rate) on the reference recording —
"earning its keep" by patching obvious mishearings.

VibeVoice was removed entirely in v0.23.0 ([ADR 0021](0021-remove-vibevoice-backend.md));
Whisper-large-v3-MLX became the sole ASR backend ([ADR 0017](0017-whisper-asr-backend.md)).
Whisper on Ukrainian is qualitatively much better — it preserves
code-switched English, doesn't drift to Russian, and produces fewer
phonetic garblings. The question (tracked in #57) was whether proofread
still earns its place in the default pipeline now that the input is
already clean.

## Measurement

A controlled run on the project's 6:14 reference recording
(`two-speakers-diar-test-ukr.m4a`, two-speaker Ukrainian conversation with
code-switched English and IT terms) — see
[`docs/benchmarks/proofread-hit-rate.md`](../postprocess-hit-rate.md) for the full
method and `docs/measurements/57/` for the raw artefacts.

| Metric | Value |
|---|---|
| Total segments | 90 |
| Wall-clock for proofread stage | 140.05 s (37 % of audio duration) |
| LLM calls | 86 |
| `unchanged` (proofread did nothing) | 71 (78.9 %) |
| `cosmetic` (punctuation / case only) | 3 (3.3 %) |
| `proper_noun_fix` (Latin / CamelCase tokens) | 7 (7.8 %) |
| `substantive_rewrite` (Ukrainian → different Ukrainian) | 9 (10.0 %) |
| **Hit-rate** (any change) | **21.1 %** |

The hit-rate alone is ambiguous — it tells us *how much* proofread does,
not *whether the work is correct*. We therefore manually evaluated the
9 `substantive_rewrite` segments with a two-channel design:

- **Human evaluator** with audio access.
- **Claude Code text-only judge** as a calibration channel (the audio is
  ground truth; the text-only judgment is what an automated future
  measurement could deliver).

Human verdicts on the 9 contested segments: **3 better, 4 worse,
2 neutral**. The proofread stage *degraded* the transcript on more
segments than it improved.

Two failure modes are particularly hard to ignore:

1. **Systematic regression.** `продуктового` → `продуктівого` and
   `продуктовим` → `продуктівим` (segments 77 and 81) — the model
   "corrects" a standard Ukrainian IT-slang adjective (`продуктовий`,
   from English "product") toward an unrelated, less-common stem
   (`продуктивний`, "productive / manufacturing-related"). Twice in a row
   on the same speaker, same root — this is a model/prompt weakness, not
   noise. Even the text-only Claude judge agreed on both.
2. **Synonym substitution.** Segment 70: `чуваки` → `хлопці`. Replacing a
   correctly transcribed conversational word with a near-synonym is
   strictly outside proofread's stated job ("fix obvious mishearings").
   This kind of edit is invisible to ASR-vs-proofread hit-rate metrics
   but degrades fidelity to actual speech.

The `proper_noun_fix` category (7 segments) was a net positive on this
recording — multiple `Hugging Space` / `HugginsFace` → `Hugging Face`
fixes, one `код-код` → `Claude Code` — but one regression
(`Gradle` → `Graddle`) shows even this category is not bullet-proof.

## Constraints (from CLAUDE.md "Project goal")

- **Audience:** lively, friendly Ukrainian conversations with English
  technical terms — the tool must preserve speech as actually spoken, not
  enforce literary norms.
- **Speed floor:** transcription must be faster than real-time. Proofread
  costing 140 s on a 374-s recording consumes 37 % of the entire
  real-time budget by itself — the most expensive single LLM stage.
- **Hardware floor:** M1 / 16 GB. Proofread's ~7 GB MLX-resident memory
  for an LLM that produces a net-negative outcome is hard to justify.

## Decision

**Proofread is off by default**, opt-in via `--proofread`.

### CLI change

- Remove the off-switch `--no-proofread`.
- Add the on-switch `--proofread` (`action="store_true"`).
- `PipelineOptions.run_proofread` default flips from `True` to `False`.

This is a **breaking CLI change**. Users who relied on
`--no-proofread` (it was the documented "make it faster" flag) can simply
drop the flag — the new default already does what `--no-proofread`
explicitly did. Users who wanted proofread on must now pass
`--proofread`.

### Why not "keep on by default, document the savings"

The alternative — keep on, document `--no-proofread` as the recommended
optimisation — was considered. Rejected because:

- The measurement says proofread is **net-negative on real speech**, not
  merely "marginal". A default that ships a known regression to most
  users contradicts CLAUDE.md's project goal ("preserve real
  conversation"). "It can be turned off" is not a fix for a bad default.
- The proper-noun fixes (the strongest argument to keep it on) are
  *also* unreliable — one `Gradle` → `Graddle` regression in the same
  measurement. Reliability of the win matters as much as average effect.
- Real-time budget is precious. The current default sets a clear
  precedent that *all* LLM stages must justify their wall-clock cost
  against measurement.

## Consequences

- **User experience.** Default-path transcription becomes ~37 % faster
  on the reference recording. Output retains raw ASR text without
  LLM-generated regressions on Ukrainian conversational content.
- **`docs/benchmarks/proofread-hit-rate.md`** is the canonical record of the
  measurement; ADR 0005's original rationale (patch VibeVoice
  russisms) is now historical only.
- **Proofread is not removed.** The stage code, prompts, and `--proofread`
  flag remain. Users with audio where the trade-off goes the other way
  (heavily English-tech recordings, no slang, no IT terms with
  ambiguous Ukrainian roots) can still opt in.
- **A separate critical issue tracks the prompt/model rework** required
  before proofread can become default-on again. The current prompt has
  identifiable failure modes (synonym substitution, dialectal "fixes"
  to standard IT slang) that suggest the rework is feasible but
  non-trivial. Without that rework, opt-in is the honest default.
- **The benchmark framework** introduced for this measurement
  ([`docs/benchmarks/README.md`](../benchmarks/README.md)) is reusable
  for any future stage that needs an evidence-based default decision.

## Refs

- #57 — issue tracking this measurement.
- [`docs/benchmarks/proofread-hit-rate.md`](../benchmarks/proofread-hit-rate.md) — full
  measurement record.
- [`docs/benchmarks/README.md`](../benchmarks/README.md) — the per-stage
  benchmark methodology this measurement instantiates.
- ADR 0005 — original rationale for the stage (now superseded by this
  measurement; the rationale assumed VibeVoice, which is gone).
- ADR 0017 — Whisper as default ASR (the clean-input precondition).
- ADR 0021 — VibeVoice removal (the event that made this re-evaluation
  necessary).
