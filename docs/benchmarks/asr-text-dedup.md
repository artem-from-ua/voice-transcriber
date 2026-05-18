# Benchmark: text-similarity dedup vs structural dedup at ASR chunk boundaries

Stage-level A/B benchmark for [ADR 0036](../adr/0036-text-similarity-asr-dedup.md).
Answers: *does replacing the structural `_dedup_overlap` (drop by
timestamp) with text-similarity Jaccard reduce the boundary-word loss
rate measured by [`asr-chunk-boundary-quality.md`](asr-chunk-boundary-quality.md)
on the 48-min reference recording?*

The baseline is the strict-time / structural-dedup measurement from
[issue #156](https://github.com/artem-from-ua/voice-transcriber/issues/156)
/ PR #169: `material_rate = 0.50` against the 0.10 threshold →
**material** verdict. This benchmark reuses **the same measurement
script** (`scripts/asr-chunk-boundary-quality.py`) on a fresh ASR
dump produced with the new dedup, so the numbers are directly
comparable.

Filed as [issue #172](https://github.com/artem-from-ua/voice-transcriber/issues/172).

## 1. Input

- **Stage under test:** chunked path of `whisper_asr.transcribe` with
  the new text-similarity `_dedup_overlap`. Geometry constants
  unchanged from ADR 0031: `ASR_CHUNK_SIZE_S = 480`,
  `ASR_CHUNK_OVERLAP_S = 5`, `ASR_CHUNK_THRESHOLD_S = 600`. New dedup
  knobs: `ASR_DEDUP_OVERLAP_WINDOW_S = 30`,
  `ASR_DEDUP_JACCARD_THRESHOLD = 0.5`, `ASR_DEDUP_MIN_TOKEN_COUNT = 3`.
- **Code state:** branch `feature/172-text-similarity-asr-dedup`,
  v0.38.0. Strict-time baseline reproduces the #156 numbers
  byte-for-byte from the unchanged chunked-ASR geometry; the only
  source of difference is the dedup function itself.
- **Pipeline configuration:** default flags. ASR backend
  `mlx-community/whisper-large-v3-mlx` (ADR 0017). The 48-min
  reference contains a single English-language conversation (the same
  recording PR #169 measured for #156).
- **Hardware:** M1 Mac, 16 GB unified memory.
- **Measurement input artefacts (per arm):**
  - `03-asr.json` — `AsrSegment[]` from the chunked path, stitched
    and dedup'd, with the dedup function under test.
  - `02b-clear_speech-*-autogain.wav` — identical between arms (input
    to ASR, not its output).

The measurement does **not** re-run ASR for the strict baseline —
PR #169 already produced those numbers and they're reused verbatim
(see [`docs/measurements/issue-156/summary.md`](../measurements/issue-156/summary.md)).
Only the text-dedup arm is freshly transcribed.

## 2. Expected output

For each chunk boundary, the same four-way verdict as #156:
`clean / missing / duplicated / truncated`. The text-dedup arm should:

- **Reduce `material_rate`** — strict baseline's 3 missing verdicts
  included cases where the structural cutoff over-trimmed legitimate
  continuation (visible post-mortem on #159 V5 where Z dedup dropped
  `"And again, our forecast product does not survive..."` along with
  duplicates). Text similarity should preserve those continuations
  while still dropping true repeats.
- **Not introduce new duplications.** The new logic catches duplicates
  at a wider 30 s window, not just inside the 5 s structural overlap —
  if anything, fewer duplicates should survive.
- **Not introduce new missing.** Short backchannels are protected by
  `min_token_count=3`; long unique-vocabulary segments are protected
  by the Jaccard threshold (`< 0.5` = keep).

**Regression signals:**

- `material_rate_textdedup >= material_rate_strict` → text dedup is not
  improving the measured failure mode.
- Any boundary that was `clean` in strict baseline flipping to
  `missing` or `truncated` — text dedup dropped a legitimate segment.
  Hard to imagine geometrically (Jaccard would have to find a tail
  segment with 50%+ overlapping vocabulary), but worth watching.
- ASR wall-clock change > 2% — the new dedup should be microseconds;
  any visible time increase points at an implementation bug.

## 3. Evaluation

A direct A/B on the metric `asr-chunk-boundary-quality.md` already
defines. Methodology unchanged — only the dump under analysis changes.

### 3.1 Per-arm features (`extract` subcommand, automated)

```bash
uv run python scripts/asr-chunk-boundary-quality.py extract \
    --dump-dir /tmp/asr-text-dedup \
    --out docs/measurements/issue-172
```

Produces `boundaries.json` with `gap_s`, `word_overlap_jaccard`,
`rms_before`, `rms_after`, truncation heuristics — same shape as
PR #169's output. Note: this benchmark uses the *strict-time* cutoffs
(no snap), so the script's default `compute_boundaries()` is the
correct cutoff source — no `--cutoffs` override needed (the snap
extension lives only on the archive branch).

### 3.2 Single-verdict judge (text-only)

```bash
uv run python scripts/asr-chunk-boundary-quality.py judge-prompt \
    --boundaries docs/measurements/issue-172/boundaries.json \
    --out docs/measurements/issue-172/judge-input.md
```

A text-only judge (Claude in-session, same convention as #156) reads
the rendered windows and writes `judge-marks.json`. Same rubric as
`asr-chunk-boundary-quality.md` §3.2.

### 3.3 Aggregation and decision

```bash
uv run python scripts/asr-chunk-boundary-quality.py parse-marks \
    --boundaries docs/measurements/issue-172/boundaries.json \
    --marks docs/measurements/issue-172/judge-marks.json \
    --out docs/measurements/issue-172/summary.md
```

Decision rule (this benchmark's contribution to the family):

- `material_rate_textdedup ≤ 0.33` (≥ 1 boundary fixed) → **partial win
  or better** — ship as default. ADR 0036, version bump, CHANGELOG.
- `material_rate_textdedup > 0.33` and `< material_rate_strict` →
  **marginal**: ship with caveat or rerun with tuned threshold.
- `material_rate_textdedup ≥ material_rate_strict` → **regression** —
  revert, write ADR as negative result, close #172 wontfix.

## 4. Synthetic fixtures

Algorithm-level tests in [`tests/test_whisper_asr.py`](../../tests/test_whisper_asr.py)
lock down the math: rewritten `test_overlap_dedup_drops_repeated_segments_at_boundary`
+ 7 new unit tests cover clear duplicate (drop), clear continuation
(keep), short-token segment (keep regardless), exact-threshold edge,
pairwise-not-aggregated rule, stop-after-first-keep, empty
accumulated, and the trailing-window cutoff for old segments.

No audio fixtures shipped — the failure modes only appear at real
chunk boundaries on long recordings.

## 5. Real test input

Same recording as `asr-chunk-boundary-quality.md` §5: the local
~48-minute English conversational dump
(private phone-interview fixture, kept locally only,
~5 speakers; not redistributed — privacy + size).

For this benchmark only the ASR stage was re-run (via
`scripts/asr-only-bench.py`), reusing the `02b-clear_speech-*-autogain.wav`
from the strict baseline so the upstream pipeline state is provably
identical. Wall-clock: ~247 s on M1/16 GB, indistinguishable from
strict 246 s.

Per-arm artefacts under [`docs/measurements/issue-172/`](../measurements/issue-172/):

- `boundaries.json` — per-cutoff features.
- `judge-input.md` — rendered windows the judge read.
- `judge-marks.json` — judge verdicts + rationale.
- `summary.md` — aggregated table + decision.

The strict-baseline comparison artefacts live under
[`docs/measurements/issue-156/`](../measurements/issue-156/) (PR #169).

## 6. Result and decision

### Numbers

| Variant | clean | missing | duplicated | truncated | material_rate | ASR wall-clock | render-visible dup? |
|---|---|---|---|---|---|---|---|
| **Strict / structural dedup (#156)** | 3 | 3 | 0 | 0 | **0.500** | 246.3 s | no |
| **Text-similarity dedup PR1** (pairwise-only) | 5 | 1 | 0 | 0 | 0.167 | 246.8 s | **yes** (B3/B4 short tail fragments survived alongside the long rewind) |
| **Text-similarity dedup + SUPERSEDE (final)** | 5 | 1 | 0 | 0 | **0.167** | 247.2 s | **no** |

`material_rate` improvement: **0.50 → 0.167 = 3× reduction.** ASR
wall-clock unchanged within noise (+1 s, 0.4%). SUPERSEDE adds no
measurable cost — it triggered at 2 of the 6 boundaries on this
recording (B3, B4) and only when the pairwise check already passed
without dropping.

### Per-boundary breakdown

| Boundary | Cutoff | Strict (#156) | Final SUPERSEDE (#172) | Change |
|---|---|---|---|---|
| B0 | 480 s | clean | clean | informationally richer; SUPERSEDE dropped two short tail fragments superseded by the longer rewind |
| B1 | 955 s | clean | clean | unchanged (no audio overlap → no SUPERSEDE trigger) |
| B2 | 1430 s | missing | missing | unchanged — `#171` class (Whisper segmentation drop, no dedup mechanism can fix) |
| B3 | 1905 s | **missing** | **clean** | `"catch myself on this sometimes"` clause recovered; SUPERSEDE dropped `"and asking questions."` + `"And I need to catch myself."` short fragments that PR1 had left alongside the long rewind |
| B4 | 2380 s | **missing** | **clean** | `"does not survive without our scheduling"` clause recovered; SUPERSEDE dropped `"And again, our Forecast product"` short fragment that PR1 had left alongside the long rewind |
| B5 | 2855 s | clean | clean | unchanged (intra-chunk Whisper duplications survive — out of scope for `_dedup_overlap`, which only touches across-boundary segments) |

### What changed and why

The two boundaries that flipped from `missing` to `clean` share the
same pattern: Whisper, given the additional context at the chunk
boundary, re-emits a longer, richer version of the audio that the
previous chunk had transcribed as several short fragments. The PR1
text-similarity check (pairwise max-Jaccard only) keeps the longer
rewind (low pairwise score against any single short fragment) but
also keeps the short fragments themselves — they pre-existed in
`accumulated`. Both versions then survive into the rendered transcript
as visible duplication.

The SUPERSEDE branch added in this PR's final commit catches the
pattern: when `incoming` has low pairwise but its audio span overlaps
multiple accumulated tail segments AND its Jaccard against the *union*
of those overlapping tail tokens passes a lower aggregate threshold,
drop the superseded tail segments and keep the (richer) incoming.

Walk-through B3:
- tail fragments: `"and asking questions."` (3 tokens) +
  `"And I need to catch myself."` (6 tokens), audio spans
  `[1900.40, 1903.34]` and `[1903.34, 1905.00]`.
- incoming: `"people and asking questions and i need to catch myself
  on this sometimes um and i think like"` (16 tokens), audio span
  `[1900.00, 1908.40]`.
- pairwise max ≈ 0.35 (below 0.5 plain-dup threshold) → not Case 1.
- audio span of incoming covers both tail fragments
  (`1900.00 < 1903.34` and `1908.40 > 1900.40` for the first,
  analogous for the second).
- aggregate Jaccard against union of overlapping tail tokens ≈ 0.5
  (above 0.3 SUPERSEDE threshold) → Case 2 fires, both tail fragments
  dropped, incoming kept.
- final tail: `"...trying to learn / people and asking questions and
  i need to catch myself on this sometimes..."`. Renders cleanly,
  no duplication.

B4 follows the same shape: `"And again, our Forecast product"` short
tail fragment superseded by `"especially with our ai products and
against our forecast product does not survive without our"`.

B2 stays missing for an entirely different reason: Whisper never
transcribed the lost words in *either* chunk's segments. No dedup
mechanism (structural, text-similarity, or SUPERSEDE) can recover
what's not in the input. Tracked by [issue #171](https://github.com/artem-from-ua/voice-transcriber/issues/171).

### Decision

**Text-similarity dedup is shipped as the production default in
v0.38.0.** Structural-cutoff dedup is removed from
`src/voice/whisper_asr.py`. ADR 0036 records the rationale and
defaults.

A subsequent follow-up could narrow the remaining B2-class failures
(see #171: e.g. forcing prompt-conditioning at chunk boundaries, or
re-transcribing suspect joins with a fast model). Those are
orthogonal to dedup choice and out of scope here.

### Links

- [ADR 0031](../adr/0031-chunked-asr.md) — original chunked-ASR
  design (geometry stays, dedup superseded).
- [ADR 0036](../adr/0036-text-similarity-asr-dedup.md) — the decision
  this benchmark validates.
- [`asr-chunk-boundary-quality.md`](asr-chunk-boundary-quality.md) —
  baseline measurement methodology (PR #169) reused verbatim.
- [Issue #156](https://github.com/artem-from-ua/voice-transcriber/issues/156)
  — strict baseline.
- [Issue #159](https://github.com/artem-from-ua/voice-transcriber/issues/159)
  — snap-to-silence attempt, rejected. Post-mortem identified
  structural dedup as the root cause this benchmark addresses.
- [Issue #171](https://github.com/artem-from-ua/voice-transcriber/issues/171)
  — residual B2-class boundaries (open, orthogonal to dedup).
- [Issue #172](https://github.com/artem-from-ua/voice-transcriber/issues/172)
  — this benchmark.
