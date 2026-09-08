---
status: accepted
date: 2026-05-14
see_also: [0005, 0017, 0021, 0026, 0027]
---

# 0028 — Proofread stage: default on again after the iteration-2.1 rework

## Context

[ADR 0026](0026-proofread-default-off.md) (v0.29.0, 2026-05-13) flipped
proofread to default-off after iteration 1 of
[`docs/benchmarks/proofread-hit-rate.md`](../benchmarks/proofread-hit-rate.md)
measured the stage degrading the transcript on Whisper output of lively
Ukrainian conversation: 3 better vs 4 worse human verdicts across 9
contested segments, plus systematic IT-slang regressions and out-of-scope
synonym substitutions. The same ADR called for prompt/model rework
before proofread could earn its way back to default-on. Issue
[#120](https://github.com/artem-from-ua/voice-transcriber/issues/120) tracked
the rework as a critical follow-up: without proofread, the user-facing
transcript ships raw ASR with non-words (`корище цей`, `контролуси`),
mis-heard tech terms (`HugginsFace`, `Gradle` for `Gradio`), and other
artefacts that defeat the project's "readable structured Markdown" goal
([CLAUDE.md "Project goal"](../../CLAUDE.md)).

[ADR 0027](0027-prompt-cache-llm-stages.md) (also v0.29.0) added KV-cache
reuse for the proofread loop, making longer system prompts (which this
rework introduces) significantly cheaper than they would otherwise be.

This ADR records the iteration 2 / 2.1 rework and the decision to flip
the default back to on in v0.31.0.

## What changed in this iteration

### Prompt — fully rewritten

`src/voice/prompts/proofread_system.md` grew from ~14 lines to ~150
lines. Major additions:

- **Genre frame.** Explicit description of what the input is — raw ASR
  output from a lively two-or-more-speaker Ukrainian conversation with
  code-switched English IT terms, mid-sentence interruptions, slang,
  and swearing. Speakers interrupting each other and producing
  incomplete utterances are characterised as normal speech, not
  errors.
- **Closed glossary** of canonical forms for tech terms (Hugging Face,
  Gradio, Claude Code, GitHub, pyannote, Whisper, MLX, ChatGPT, …)
  with their known ASR mishearings. Applied unconditionally.
- **"Words to LEAVE" section** explicitly listing colloquial answers
  (`Нє`, `Ага`, `Угу`, `шо`, …), particles/conjunctions (`от`, `і`,
  `а`, `ну`, …), near-equivalent demonstratives (`оцей`/`цей`, …),
  Ukrainian IT loanwords (`продуктовий`, `шерити`, `мітинг`,
  `деплоїти`, …) — all to pass through unchanged.
- **Obscene-vocabulary rule.** Project policy is explicit: never
  censor, soften, or euphemise swear words. The transcript captures
  what was spoken.
- **Interrupted-sentence rule.** If a segment ends in `...` or trails
  off mid-word, that is a real speaker interruption — do not complete
  the missing tail.
- **Short-segment bias.** For segments of ~3 words or fewer, prefer
  leave-alone over "fix" when the ASR rendering is itself a valid
  word; context does not justify changing case/number/gender of an
  isolated short utterance.

### Sampling parameters tightened (frontmatter)

- `temperature: 0.0` (greedy) — faster and deterministic, so future
  benchmark runs are byte-identical given the same input.
- `repetition_penalty: null` — cheap and unnecessary for short outputs.
- `max_tokens: 256` — observed outputs are 5-100 tokens; tighter cap.

### Context-aware per-segment calls

Each `chat()` call sees up to N neighbouring segments on each side of
the current one, framed in the user prompt as:

- `[CONTEXT BEFORE]` — built from already-proofread outputs (final
  form). Self-bootstraps a running glossary: once `[i-1]` resolves
  `хагінг фейсі` → `Hugging Face`, `[i]` sees the canonical form.
- `[CONTEXT AFTER]` — built from the raw upcoming segments, with an
  explicit "not yet proofread" framing so the model does not match
  against still-garbled terms.
- `[CURRENT SEGMENT — fix this one only]` — the edit target.

`n_context == 0` collapses the user prompt to bare segment text — the
v0.29.0 wire format, kept for the sweep baseline.

### New CLI surface

- `--no-proofread` — opt out (default is on; this is the off-switch).
- `--proofread` — legacy no-op since v0.31.0, kept for backward
  compatibility with v0.29.0 invocations.
- `--proofread-context N` — tune the context window for benchmark
  runs.
- `PipelineOptions.proofread_n_context: int = 3`.

### New tooling

- `scripts/proofread-classify.py sweep` — drives a full pipeline run
  for each value in a context-size grid, classifies each dump,
  aggregates into `sweep-summary.{json,md}`. Reusable for any
  future per-stage parameter sweep.
- `scripts/proofread-classify.py compare-baseline` — diffs a new
  categories.json against a baseline; surfaces per-bucket movement.
- `docs/benchmarks/README.md` gains a "Parameter sweep / knee
  detection" section formalising the pattern, with this proofread
  sweep as the worked example.

## Measurement summary

Audio: `~/Downloads/<reference-recording>.m4a` (6:14, two
speakers, Ukrainian conversation about software with code-switched
English IT terms, casual register, occasional swearing). Same recording
as iteration 1.

### Iteration 2 — context-size sweep (six grid points)

| n_context | unchanged | proper_noun_fix | substantive | hit-rate | wall-clock |
|---|---|---|---|---|---|
| 0 | 59 | 8 | 18 | 34.4 % | 97.6 s |
| 1 | 57 | 11 | 21 | 36.7 % | 125.8 s |
| 2 | 62 | 9 | 16 | 31.1 % | 144.9 s |
| **3** | **63** | **10** | **14** | **30.0 %** | **170.1 s** |
| 5 | 65 | 10 | 15 | 27.8 % | 208.2 s |
| 8 | 66 | 10 | 14 | 26.7 % | 258.3 s |

Knee at **n_context=3**: substantive_rewrite plateaus from 3 onward
(14, 15, 14), but the cost grows linearly (170 → 208 → 258 s). n=5 and
n=8 break the project's real-time floor (374 s audio → must finish
under 374 s total pipeline; proofread alone at 208/258 s leaves
insufficient budget for the remaining stages). n=8 also regresses on
the iteration-1 `продуктовим → продуктівим` case — long context
overload appears to drown out the glossary.

### Iteration 2.1 — sharpened prompt on the winning n_context=3

After the iteration-2 human spot-check (3 better / 6 worse / 5 neutral
on 14 substantive segments) identified six specific failure patterns,
the prompt was extended with explicit rules for each (colloquial
answers, particles, demonstratives, swearing, interrupted sentences,
short-segment bias).

| Metric | Iter 2 (n=3) | Iter 2.1 (n=3) | Δ |
|---|---|---|---|
| unchanged | 63 | 74 | +11 |
| substantive_rewrite | 14 | 8 | -6 |
| hit_rate | 30.0 % | 17.8 % | -12.2 pp |
| wall_clock | 170.05 s | 173.32 s | +3 s |

Failure-mode coverage after iter 2.1:

- ✅ `Нє` no longer normalised to `Ні` (seg 15).
- ✅ Interrupted-sentence `сигнал...` no longer completed (seg 24).
- ✅ Obscenity no longer censored to a milder synonym.
- ✅ `оця` no longer flattened to `ця` (seg 75).
- ❌ Particle change `от → а` still happens (seg 11) — rule did not
  fully transfer.
- ❌ Synonym substitution `чуваки → хлопці` partially regressed
  (seg 70) compared to iter 2 on the same n_context.
- ❌ Short-segment fabrications appeared (seg 4 `контролуси →
  контроліри`, seg 83 `абстрактної → абстрактнії`) — the short-segment
  bias rule landed but the model still occasionally invents non-words
  on phonetically-unrecognised English loanwords.

## The decision criterion that actually applies

Iteration 1's human-spot-check methodology asked "did proofread move
the text closer to or further from what the speaker said?". On that
metric, iteration 2.1 still scores poorly on the contested-segments
slice — many of those segments are exactly the cases where the speaker
used a phonetic English IT term (`контролси` = Controls, `скілами` =
skills, `допилювати`, `короче`) that the model fails to recognise.

But the project goal in CLAUDE.md is **a readable structured Markdown
document**, not a maximally-faithful phonetic transcript. Comparing
proofread output to **raw ASR** (what the user actually gets with
proofread off) on the same segments, proofread is a net win:

- Raw ASR ships `корище цей`, `дотавку`, `контролуси`, `твими` — all
  unreadable non-words for anyone without the audio.
- Proofread iter 2.1 turns most of those into either correct
  Ukrainian (`доставку`, `твоїми`) or at-least-readable approximations
  (`користувач`).
- The categorised regressions (synonym substitutions, particle changes)
  remain readable Ukrainian. They lose some authentic-speech flavour
  but do not break readability.
- Systematic iter-1 regressions (`продуктовий → продуктівого`) are
  gone on n_context=3.

The single largest remaining failure category — phonetic English
loanwords that look like non-words to the model — is **not a prompt
problem**. It is a model-knowledge problem. No amount of prompt rules
can teach a 7B model that `контролси` means `Controls`. That requires
either a different model, a retrieval-augmented glossary, or a
fine-tune — work tracked separately as a new critical issue
(see Refs).

## Decision

**Default-on, with `--no-proofread` as opt-out.** Ships in v0.31.0.

- `PipelineOptions.run_proofread: bool = True` (flipped from `False`).
- `PipelineOptions.proofread_n_context: int = 3` (the sweep knee).
- CLI: `--no-proofread` reintroduced as the off-switch. The legacy
  `--proofread` from v0.29.0 stays as a no-op for backward
  compatibility — removing it would be a second CLI churn in two
  releases.

This is a **behaviour-breaking change** from v0.29.0: users who
relied on the v0.29.0 default-off behaviour must now pass
`--no-proofread` explicitly to keep that behaviour. The CHANGELOG
calls this out.

## Consequences

- **The transcript regains readability on the default path.** This is
  the user's primary complaint against v0.29.0 and the explicit reason
  this ADR exists.
- **Real-time floor remains intact** at n_context=3 (170 s on a 374 s
  recording, ~45 % of the pipeline budget).
- **Authentic-speech fidelity** drops slightly compared to raw ASR —
  swear words and IT slang survive (those rules landed), but synonym
  substitutions on conversational vocabulary (`чуваки → хлопці`) still
  happen occasionally. Users who want maximum fidelity over readability
  pass `--no-proofread`.
- **Phonetic-English-loanword recognition** remains the largest gap;
  it is a model-knowledge problem and tracked as a new critical issue.
- The **benchmark framework** (`docs/benchmarks/`) gained the
  "Parameter sweep / knee detection" pattern, with this rework as the
  worked example.

## Refs

- ADR 0026 — iteration-1 default-off, which this ADR succeeds.
- ADR 0027 — KV-cache reuse that makes the longer iter-2 prompt
  affordable.
- [#120](https://github.com/artem-from-ua/voice-transcriber/issues/120) — critical follow-up
  that this rework closes.
- [#133](https://github.com/artem-from-ua/voice-transcriber/issues/133) — option-D side-by-side
  context format, deferred from this iteration.
- New critical follow-up issue (filed with this PR): "proofread cannot
  recognise Ukrainian-phonetic English IT loanwords (`контролси`,
  `скілами`, `допилювати`)" — the model-knowledge gap that no prompt
  fix can close.
- [`docs/benchmarks/proofread-hit-rate.md`](../benchmarks/proofread-hit-rate.md) — both iterations
  of the measurement.
- `docs/measurements/120/` — raw artefacts:
  the iter-2 sweep, the iter-2.1 single-point run, the spot-check
  files, the comparison tables.
