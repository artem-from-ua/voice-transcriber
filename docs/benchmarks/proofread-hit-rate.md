# Benchmark: proofread hit-rate

Stage-level benchmark for the **proofread** stage
(`src/voice/proofread.py:fix_asr_errors`). Answers the question:
*on Whisper output, does proofread improve or degrade Ukrainian
conversational transcripts?*

This document follows the six-section contract from
[`README.md`](README.md).

## 1. Input

- **Stage under test:** `proofread` (in-pipeline LLM stage that rewrites
  each `Segment.content` via a single LLM call).
- **Code state:** branch `feature/postprocess-hitrate-review`, project
  version 0.29.0.
- **Pipeline configuration:** default flags except `--dump-stages /tmp/postprocess-57/`
  and `--verbose`. ASR backend: Whisper-large-v3-MLX (the only backend
  since v0.23.0). Proofread model: `mlx-community/Qwen2.5-7B-Instruct-4bit`
  (the v0.22 default LLM).
- **Hardware:** M1 Mac, 16 GB unified memory.
- **Input artefacts to the measurement:**
  - `04-merge.json` — `Segment[]` *before* proofread (ASR output merged
    with pyannote turns).
  - `05-proofread.json` — `Segment[]` *after* proofread (same shape,
    same count, same timestamps; `content` may differ).
  - `01-meta.json` — `stages.proofread` field for wall-clock + LLM call
    count (added in this PR; see #57 and follow-up #118).

## 2. Expected output

For a *useful* proofread stage:

- High `unchanged` rate on already-clean Whisper output (Whisper handles
  Ukrainian well — most segments should not need any fix).
- Where proofread *does* change a segment, the change should fix a real
  ASR error: a non-word (`корище`, `контролуси`), a clearly misheard
  technical term (`HugginsFace` → `Hugging Face`), or a grammatical
  impossibility.
- The stage **must not**:
  - swap correctly-transcribed words for synonyms;
  - rewrite colloquial forms into literary forms (or vice versa) when
    the ASR-rendered form already matched the speech;
  - introduce non-standard or dialectal variants when the ASR form was
    the conventional one for the context (especially IT slang).

**Regression signals:** any of the above forbidden behaviours appearing
in the `substantive_rewrite` category. A single occurrence is noteworthy;
the *same word stem* being mis-corrected twice (the `продуктовий` /
`продуктівий` case in this measurement) is systematic and disqualifying
for default-on.

## 3. Evaluation

The metric is **a two-stage cascade**, because hit-rate alone is
ambiguous (it measures *amount of work*, not *quality of work*).

### 3.1 Hit-rate breakdown (text-only, automated)

Each segment-pair `(asr_text, proofread_text)` is classified into one of
four buckets by `scripts/proofread-classify.py classify`:

- **`unchanged`** — texts identical after `strip()` + collapse-whitespace.
- **`cosmetic`** — texts differ only in punctuation, casing, or
  whitespace (identical after aggressive normalisation: lowercase +
  remove `[.,!?;:'"()—–\-«»…]` + collapse whitespace).
- **`proper_noun_fix`** — differing tokens contain Latin letters or
  CamelCase, i.e. proofread touched what looks like a technical term.
- **`substantive_rewrite`** — any other change (Ukrainian text replaced
  by different Ukrainian text). This is the category that requires
  human eyes — the script cannot tell better from worse here.

**Threshold for decision based on hit-rate alone:**
- `unchanged ≥ 90 %` AND `substantive_rewrite ≤ 1` → default-off (proofread
  is doing nothing).
- otherwise → proceed to the manual spot-check on
  `substantive_rewrite` segments.

### 3.2 Manual spot-check + LLM judge (calibration)

Only the `substantive_rewrite` segments go through spot-check. Two
channels in parallel:

- **Human evaluator** with audio access — marks each segment
  `better` / `worse` / `neutral` after listening at the segment's
  timestamp. **This is the authoritative channel.**
- **Claude Code judge**, text-only — marks the same segments using
  text + context. This is a **calibration** channel that tells us
  whether future automated measurements (without listening) can be
  trusted. It is *not* a substitute for the human channel.

**Marks definitions** (identical for both channels, in the spot-check
file the script emits):
- **better** — proofread fixed a real ASR error.
- **worse** — proofread changed text that did not need fixing or moved
  it further from speech.
- **neutral** — immaterial difference, or plausible-but-unverifiable
  guess on a garbled segment.

**Decision rule:**

| Human-mark distribution | Recommended default |
|---|---|
| `better` clearly dominates | keep on |
| `worse` matches or beats `better` | turn off |
| roughly balanced + a systematic error pattern is visible | turn off (the systematic pattern is the deciding factor) |

**LLM-judge agreement metric:**
- `agreement_pct = agreed / shared` × 100.
- `agreement < 70 %` → text-only judgment is not trustworthy for this
  measurement; future iterations must include the human-with-audio pass
  rather than relying on LLM-judge alone.

### 3.3 Cost signal (always reported, never the deciding factor alone)

- Proofread wall-clock as a fraction of audio duration.
- LLM-call count.
- Peak MLX memory during the stage (from log).

A stage that crosses the **real-time floor** (transcription > audio
duration) is a hard regression per CLAUDE.md project goal. Proofread
costing 37 % of audio duration is alarming but does not by itself
justify a default flip — quality is the primary lever.

## 4. Synthetic input / reference output

For unit-level testing of the classifier, hand-crafted pairs live in
`tests/test_proofread_classify.py`. The pairs are parametrised, one per
category, covering at minimum:

- An `unchanged` pair where strings are byte-equal.
- An `unchanged` pair that differs only in trailing/inner whitespace.
- A `cosmetic` pair differing only in punctuation.
- A `cosmetic` pair differing only in case.
- A `proper_noun_fix` pair with a phonetic Cyrillic transliteration →
  English term (`хагінг фейсі` → `Hugging Face`).
- A `proper_noun_fix` pair with a CamelCase token.
- A `substantive_rewrite` pair that changes meaning
  (`думав це працює` → `думав це не працює`).
- A `substantive_rewrite` pair that swaps a noun for another noun
  (`в парк` → `в магазин`).

**Generator rule:** each new pair must come with a one-line comment
explaining *which failure mode it catches*. When the classifier
mis-categorises a real segment from a future run, add the pair here
before fixing the classifier — so the fix has a test that pins it.

## 5. Real test input

- **Recording:** `~/Downloads/two-speakers-diar-test-ukr.m4a`. 374 s
  (6:14). Two speakers, Ukrainian conversation about software / Hugging
  Face / Gradio / Claude Code, code-switched English, casual register,
  occasional swearing. This is the project's canonical test recording
  since ADR 0017.
- **Reference transcript:** none yet. Acknowledged limitation — the
  benchmark currently relies on the human listening rather than diffing
  against a manually-transcribed reference. A full reference transcript
  is a separate work item (see #73 for the cross-engine validation use
  case; the same reference would serve here once it exists).
- **Dump artefacts from the measurement:**
  - `/tmp/postprocess-57/01-meta.json` (with `stages.proofread`
    telemetry — first run that has it).
  - `/tmp/postprocess-57/04-merge.json`, `05-proofread.json` —
    the input pair for the classifier.
- **Measurement results (checked in):**
  [`docs/measurements/57/`](../measurements/57/) — `categories.json`,
  `spot-check.md` (filled), `judge-input.md`, `judge-marks.json`,
  `final-table.md`.

## 6. Result and decision (2026-05-13)

### Hit-rate breakdown

| Category | Count | % |
|---|---|---|
| unchanged | 71 | 78.9 % |
| cosmetic | 3 | 3.3 % |
| proper_noun_fix | 7 | 7.8 % |
| substantive_rewrite | 9 | 10.0 % |

Hit-rate: **21.1 %** (19 of 90 segments changed).
Wall-clock: **140.05 s** on 374 s of audio (37 %). LLM calls: 86.

### Spot-check on `substantive_rewrite` (9 segments)

| Channel | better | worse | neutral |
|---|---|---|---|
| Human (audio-grounded, authoritative) | 3 | **4** | 2 |
| Claude Code judge (text-only) | 2 | 4 | 3 |

`worse` beats `better` in the authoritative human channel. Two
particularly damning failures:

- Segments 77 and 81 — both `worse` in both channels — the model
  rewrote `продуктовий` (standard Ukrainian IT slang from English
  "product") to `продуктівий` (a dialectal/incorrect adjective that
  shifts meaning toward `продуктивний` = "productive"). Same speaker,
  same root, twice in a row → systematic, not noise.
- Segment 70 — proofread substituted `чуваки` → `хлопці` (synonym
  replacement of a correctly transcribed slang word, out of scope).

The `proper_noun_fix` category was a net win on this recording
(`Hugging Space` / `HugginsFace` → `Hugging Face`, `код-код` →
`Claude Code` — five solid fixes) but **not perfect**: one
`Gradle` → `Graddle` regression. Reliability of the win matters as
much as average effect.

### LLM-judge agreement

**44.4 % (4 of 9 segments).** Below the 70 % threshold by a wide
margin. Conclusion: **text-only LLM judgment is not reliable for this
measurement**. Future hit-rate re-runs must include a human-with-audio
pass. The judge channel is retained for calibration on future iterations
once the proofread prompt is reworked.

### Cost signal

37 % of audio duration consumed by one stage; ~7 GB MLX-resident
memory; 86 LLM calls. The stage spends a large fraction of the
real-time budget on net-negative work.

### Decision

**Default off; opt-in via `--proofread`.** See
[ADR 0026](../adr/0026-proofread-default-off.md). This is a breaking
CLI change (`--no-proofread` removed; `--proofread` introduced) and
ships in v0.29.0.

### Follow-ups created

- Critical issue: prompt/model rework needed before proofread can be
  default-on again. The systematic `продуктовий`/`продуктівий` failure
  and the synonym-substitution scope creep both look fixable in the
  prompt; without that work the opt-in default is the honest answer.
- Existing #107 (proofread speedup) and #108 (proofread glossary) are
  related but address different angles. The new issue is specifically
  about *correctness*.

---

## Iteration 2 (2026-05-14) — prompt rework + context-size sweep

This iteration closes #120 (the critical follow-up filed at the end of
iteration 1). Two hypotheses about why proofread degraded transcripts in
iteration 1 are tested together: insufficient per-call context, and a
system prompt that did not describe the conversational genre. Both are
addressed in this iteration.

### Changes vs iteration 1

- **System prompt fully rewritten** (`src/voice/prompts/proofread_system.md`):
  - Genre frame describing raw ASR + lively Ukrainian conversation +
    code-switched English IT terms.
  - Closed glossary block of known-correct canonical forms (Hugging Face,
    Gradio, Claude Code, GitHub, pyannote, …) with their common ASR
    mishearings.
  - Negative rules naming the specific failure modes from iteration 1:
    no synonym substitutions, no literary/colloquial normalisation, no
    loanword "correction" toward unrelated Ukrainian roots, no padding
    or completion of mid-word interruptions.
- **Sampling parameters tightened (`proofread_system.md` frontmatter):**
  `temperature: 0.0` (greedy), `repetition_penalty: null`, `max_tokens: 256`.
  Deterministic and faster.
- **Context-aware per-segment calls.** The LLM now sees up to N
  neighbouring segments around the current one:
  - `[CONTEXT BEFORE]` from already-proofread outputs (final form).
  - `[CONTEXT AFTER]` from the raw upcoming segments (with an explicit
    note that they are not yet proofread).
  - `[CURRENT SEGMENT — fix this one only]` is the edit target.
- **`--proofread-context N` CLI flag and `PipelineOptions.proofread_n_context`**
  expose the new parameter; the default value comes from this iteration's
  sweep (see §6 below). `n_context=0` reproduces the iteration-1 wire
  format for the baseline grid point.
- **KV-cache reuse** ([ADR 0027](../adr/0027-prompt-cache-llm-stages.md))
  is already in place; the longer system prompt of this iteration makes
  the cache more valuable, not less.

### 6. Result and decision

#### Parameter sweep — context size

Audio: `~/Downloads/two-speakers-diar-test-ukr.m4a` (same as iteration 1).
Grid: `n_context ∈ {0, 1, 2, 3, 5, 8}`. Each grid point is one full
`voice transcribe --proofread --proofread-context N --dump-stages …` run;
results aggregated by [`scripts/proofread-classify.py sweep`](../../scripts/proofread-classify.py).
Canonical numbers in
[`docs/measurements/120/sweep/sweep-summary.md`](../measurements/120/sweep/sweep-summary.md).

| n_context | unchanged | proper_noun_fix | substantive | hit-rate | wall-clock | seg 70 | seg 81 |
|---|---|---|---|---|---|---|---|
| 0 | 59 | 8 | 18 | 34.4 % | 97.6 s | ❌ хлопці | ✅ |
| 1 | 57 | 11 | 21 | 36.7 % | 125.8 s | ❌ хлопці | ❌ продуктівим |
| 2 | 62 | 9 | 16 | 31.1 % | 144.9 s | ❌ хлопці | ✅ |
| **3** | **63** | **10** | **14** | **30.0 %** | **170.1 s** | **✅ чуваки** | **✅** |
| 5 | 65 | 10 | 15 | 27.8 % | 208.2 s | ✅ чуваки | ✅ |
| 8 | 66 | 10 | 14 | 26.7 % | 258.3 s | ❌ хлопці | ❌ продуктівим |

Knee at `n_context=3`:

- `substantive_rewrite` plateaus from 3 onward (14, 15, 14) — no
  meaningful drop past this value.
- `unchanged` keeps growing (63 → 65 → 66) — but the marginal trust
  past n=3 is small.
- Wall-clock scales roughly linearly (+38 s per step). n=5 and n=8
  push proofread past the 200 s mark; the full pipeline on a 374 s
  audio would then exceed the project's real-time floor.
- n=8 regresses on both `seg 70` (synonym swap returns) and `seg 81`
  (`продуктівим` returns) — long context appears to drown out the
  glossary block, an instability that disqualifies it for default-on.

`n_context=3` is the cheapest point that hits the quality plateau and
keeps the real-time floor.

#### Human spot-check on `n_context=3` (iteration 2)

[Filled spot-check](../measurements/120/spotcheck-n3/spot-check.md) +
[final-table](../measurements/120/spotcheck-n3/final-table.md):

- Human marks: **3 better, 6 worse, 5 neutral** on the 14
  `substantive_rewrite` segments.
- `worse` exceeded `better`. The iteration-2 prompt fixed iteration-1's
  systematic IT-slang regressions (`продуктовий` no longer touched) but
  introduced six new failure patterns — colloquial-answer normalisation
  (`Нє → Ні`), particle changes (`от → а`), interrupted-sentence
  completion (`сигнал... → сигналізує`), obscenity censorship
  (`Ніхуя → Нічого`), and the iteration-1 `оця → ця` regression
  re-appearing.

This is why iteration 2 by itself did **not** justify default-on. The
sharpened prompt of iteration 2.1 addresses each named failure with
explicit rules and re-runs the same `n_context=3` configuration.

#### Iteration 2.1 — sharpened prompt on `n_context=3`

Prompt extended with: a "Words to LEAVE" section (colloquial answers
`Нє`/`шо`/…, particles `от`/`і`/`а`/…, demonstrative pairs
`оцей`/`цей`/…, IT loanwords); an explicit obscenity rule; an
explicit "never complete an interrupted sentence ending in `...`"
rule; and a short-segment bias toward leaving alone.

| Metric | Iter 2 (n=3) | Iter 2.1 (n=3) | Δ |
|---|---|---|---|
| unchanged | 63 | 74 | +11 |
| cosmetic | 3 | 1 | -2 |
| proper_noun_fix | 10 | 7 | -3 |
| substantive_rewrite | 14 | 8 | -6 |
| hit_rate | 30.0 % | 17.8 % | -12.2 pp |
| wall_clock | 170.05 s | 173.32 s | +3 s |

Failure-mode coverage on the six patterns from iteration 2 spot-check:

- ✅ `Нє` no longer normalised to `Ні` (seg 15).
- ✅ Interrupted-sentence `сигнал...` no longer completed (seg 24).
- ✅ Obscenity `Ніхуя` no longer censored (seg 33).
- ✅ `оця` no longer flattened to `ця` (seg 75).
- ❌ Particle change `от → а` still happens occasionally (seg 11).
- ❌ Synonym substitution `чуваки → хлопці` regressed on n=3 with the
  new prompt (seg 70 — iter 2 had kept it correctly).

The iter-2.1 spot-check ([source](../measurements/120/iter2.1/spotcheck/spot-check.md))
revealed that most of the remaining `substantive_rewrite` segments fall
into one specific category that **no prompt rule can fix**:

- `контролуси → контроліри` (ground truth: `контролси`, from English
  *Controls*)
- `корище цей → користувач цей` (ground truth: `короче, цей` — a Russianism)
- `твими скелами → твоїми скелами` (partial fix; ground truth: `скілами`,
  from English *skills*)
- `абстрактної → абстрактнії` (ground truth: `абстрактної` — should
  have been left alone)
- `допилювати → додати` (ground truth: `допилювати` — legit Ukrainian
  IT slang the model didn't recognise)

This is a **model-knowledge gap**: Qwen2.5-7B does not recognise
Ukrainian-phonetic English IT loanwords. When confronted with such a
word, it either invents a plausible-looking Ukrainian non-word or
substitutes a generic synonym. Prompt rules cannot teach a model
vocabulary it does not have.

A separate critical follow-up tracks this gap; resolving it requires
either a different (larger / fine-tuned) model, a retrieval-augmented
glossary, or both. That work is out of scope for this iteration.

#### Comparison against iteration 1 baseline

```
python scripts/proofread-classify.py compare-baseline \
    --baseline docs/measurements/57/categories.json \
    --current docs/measurements/120/iter2.1/spotcheck/categories.json
```

| Bucket | Iter 1 | Iter 2.1 | Δ |
|---|---|---|---|
| unchanged | 71 | 74 | +3 |
| cosmetic | 3 | 1 | -2 |
| proper_noun_fix | 7 | 7 | 0 |
| substantive_rewrite | 9 | 8 | -1 |
| hit_rate | 21.1 % | 17.8 % | -3.3 pp |

Iteration-1's most damaging failures are gone — the two
`продуктовий → продуктівого/продуктівим` regressions (seg 77, 81) and
the `Gradle → Graddle` regression all return correct values now.

#### Decision

**Default flipped back to on**, `n_context=3`, with `--no-proofread`
as opt-out. Shipped in v0.30.0. See
[ADR 0028](../adr/0028-proofread-default-on-after-rework.md) for the
full rationale.

The decision criterion deserves an explicit note. Iteration-1 used a
"closer to / further from what the speaker said" yardstick on the
contested segments. Under that yardstick, iter 2.1 still has `worse`
outnumbering `better` on the eight remaining substantive_rewrite
segments — because most of those eight are exactly the Ukrainian-phonetic
English-loanword cases the model cannot recognise (`контролси`,
`скілами`, `допилювати`).

But the project goal (CLAUDE.md) is a **readable structured Markdown
document**, not a maximally-faithful phonetic transcript. Compared
against the **default-off baseline** (raw ASR that the user actually
gets), proofread at iter-2.1/n=3 is a net win:

- Raw ASR ships `корище цей`, `дотавку`, `контролуси`, `твими` — all
  unreadable non-words.
- Iter-2.1 proofread turns most of those into either correct
  Ukrainian (`доставку`, `твоїми`) or readable approximations
  (`користувач`).
- The remaining regressions (synonym substitutions, occasional particle
  swaps) stay readable Ukrainian — they lose authenticity but not
  readability.

That trade-off favours default-on for the **primary user-facing
metric** (readability) at an acceptable cost on the secondary metric
(speech fidelity). Users who need maximum fidelity can opt out with
`--no-proofread`.
