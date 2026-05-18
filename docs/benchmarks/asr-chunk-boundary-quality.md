# Benchmark: ASR chunk-boundary quality

Stage-level benchmark for the chunked ASR path in
[`src/voice/whisper_asr.py`](../../src/voice/whisper_asr.py). Answers
the question: *when long audio is sliced into 8-min chunks with a 5 s
overlap, how often does the structural `_dedup_overlap` cut lose,
duplicate, or truncate a word across a chunk boundary?*

Filed as [issue #156](https://github.com/artem-from-ua/voice-transcriber/issues/156).
This doc follows the six-section contract in [`README.md`](README.md)
and is the **baseline measurement** that issues
[#159](https://github.com/artem-from-ua/voice-transcriber/issues/159)
(snap-to-silence ASR chunk boundaries) and
[#168](https://github.com/artem-from-ua/voice-transcriber/issues/168)
(compare silence-detection methods) will compare against — both reuse
this measurement script verbatim on a snap-modified dump.

## 1. Input

- **Stage under test:** chunked path of `whisper_asr.transcribe`
  (geometry constants `ASR_CHUNK_SIZE_S = 480`, `ASR_CHUNK_OVERLAP_S = 5`,
  `ASR_CHUNK_THRESHOLD_S = 600`). See
  [ADR 0031](../adr/0031-chunked-asr.md).
- **Code state:** branch `chore/156-asr-chunk-boundary-quality`,
  measurement script `scripts/asr-chunk-boundary-quality.py`. The
  pipeline code that produced the dump being measured is v0.37.0
  (PR #149 chunked-ASR path).
- **Pipeline configuration:** default flags. ASR backend
  `mlx-community/whisper-large-v3-mlx` (ADR 0017), single Ukrainian
  conversation with English IT terms code-switched in.
- **Hardware:** M1 Mac, 16 GB unified memory (reference machine).
- **Measurement input artefacts:**
  - `03-asr.json` — `AsrSegment[]` from the chunked path, already
    stitched and dedup'd.
  - `02b-clear_speech-*-autogain.wav` — the WAV that fed Whisper, used
    only to compute RMS in a ±200 ms window around each cutoff.

The measurement does **not** re-run ASR. It is purely post-hoc analysis
of the dumped artefacts of one chunked run.

## 2. Expected output

For a boundary to count as `clean`:

- (a) `head_text` continues `tail_text` logically with no repeated
  phrase and no obvious dropped word — a future-naive reader of the
  stitched transcript would not be able to tell a chunk cut happened
  here;
- (b) `gap_s = head[0].start - tail[-1].end` is either small (close to
  zero) or matches an obvious natural pause that the RMS-around-cutoff
  feature corroborates (low RMS on both sides);
- (c) `last_token` (end of chunk N) and `first_token` (start of chunk
  N+1) are both complete words — no `Cer` + `nat` style halves.

**Regression signals** (anything *not* clean):

- `missing` — sense breaks across the cutoff; a word or short phrase
  reads as lost. The signature is `gap_s` of a few seconds *combined
  with* high RMS at the cutoff (audio was loud, yet no token landed).
- `duplicated` — the same phrase appears at the end of tail and the
  start of head; `word_overlap_jaccard` is noticeably above the
  baseline for the other boundaries.
- `truncated` — a half-word stranded on either side of the cutoff.

## 3. Evaluation

The measurement is a two-stage cascade — features computed from text
and audio, then a single categorical verdict per boundary.

### 3.1 Per-boundary features (`extract`, automated)

For each cutoff `c = chunk_start_{N+1} + overlap_s` (which is exactly
the timestamp at which `_dedup_overlap` trims chunk N+1):

- `gap_s` = `head[0].start - tail[-1].end` — how much wall-clock the
  stitched transcript silently skips at the join.
- `word_overlap_jaccard` — set-Jaccard on the normalised token bags
  of the last 5 tail segments vs the first 5 head segments. High value
  is a hint at duplication; low value alone is not informative.
- `last_token_truncated_heuristic` / `first_token_truncated_heuristic`
  — regex / very-short-token flags. Imperfect; the judge filters.
- `rms_before` / `rms_after` — root-mean-square energy in a ±200 ms
  window split exactly at the cutoff sample. The single most useful
  audio feature available without listening: low on both sides means
  the cut landed in silence (best case); high on either side means
  Whisper had loud audio right at the boundary.

### 3.2 Single-verdict judge (text-only)

Each boundary gets one of `clean / missing / duplicated / truncated`
from a judge that reads only the rendered text window (±5 segments)
plus the features above. Audio is **not** listened to — the goal of
this measurement is reproducibility, and listen-back inflates the
sample to "judge had a bad-hair day" levels of noise.

The judge in this measurement is Claude (this session) reading the
generated `judge-input.md` and writing `judge-marks.json` directly.
Future runs of this benchmark (e.g. from #159 / #168) reuse the same
script with the same instructions — any consistent text-only judge
will produce comparable numbers.

**Honest limitation:** a text-only judge cannot detect cases where the
ASR acoustically lost a word entirely. If Whisper dropped a word and
the surrounding text still reads grammatical, the judge will mark
`clean`. The RMS feature partially mitigates: a `clean`-looking
boundary with `gap_s > 2 s` *and* high RMS on at least one side is a
heuristic flag the judge takes seriously (this is exactly the signal
that drove the three `missing` verdicts in §6 below).

### 3.3 Decision threshold

- `material_rate = (missing + duplicated + truncated) / n_boundaries`
- `material_rate >= 0.10` → **material** — open a follow-up to fix
  the dedup (overlap bump, text-similarity dedup, or snap-to-silence
  via #159).
- otherwise → **baseline-acceptable** — keep `_dedup_overlap` as-is,
  record the baseline so #159 / #168 have a number to beat.

## 4. Synthetic fixtures

This benchmark does not ship synthetic audio fixtures. An unchunked
control on the M1/16 GB reference machine is unreachable on long
audio (the original symptom that drove PR #149); a short synthetic
clip cannot reproduce the chunk-boundary problem at all.

What ships instead:
[`tests/test_asr_chunk_boundary.py`](../../tests/test_asr_chunk_boundary.py)
— unit tests for the pure helpers in
`scripts/asr-chunk-boundary-quality.py` (tokenisation, Jaccard,
boundary geometry, mark aggregation). These lock down the math so a
future contributor cannot silently break the methodology that
#159 / #168 reuse.

## 5. Real test input

The reference dump used for §6 is from a Ukrainian + English-mixed
conversation, ~48 minutes long, ~5 speakers, conversational style
with IT terminology code-switched in. The recording itself is not
checked in (privacy + size — see CLAUDE.md "Реальні імена/компанії з
тестових записів"); the dump artefacts referenced below are derived
artefacts that contain only the dedup'd ASR output and aggregate
audio features, no raw audio.

- Dump location on the dev machine:
  `~/Downloads/<long-recording-dump>/` (contains `03-asr.json` +
  `02b-clear_speech-*-autogain.wav`).
- Per-run artefacts under
  [`docs/measurements/issue-156/`](../measurements/issue-156/):
  - `boundaries.json` — per-cutoff features.
  - `judge-input.md` — rendered windows the judge read.
  - `judge-marks.json` — judge verdicts + rationale.
  - `summary.md` — aggregated table + decision.

Reproducing the measurement on the same dump:

```bash
uv run python scripts/asr-chunk-boundary-quality.py extract \
    --dump-dir ~/Downloads/<long-recording-dump> \
    --out docs/measurements/issue-156

uv run python scripts/asr-chunk-boundary-quality.py judge-prompt \
    --boundaries docs/measurements/issue-156/boundaries.json \
    --out docs/measurements/issue-156/judge-input.md

# Read judge-input.md, write judge-marks.json with one entry per boundary.

uv run python scripts/asr-chunk-boundary-quality.py parse-marks \
    --boundaries docs/measurements/issue-156/boundaries.json \
    --marks docs/measurements/issue-156/judge-marks.json \
    --out docs/measurements/issue-156/summary.md
```

## 6. Result and decision

### Numbers on the 48-min reference dump

| metric | value |
|---|---|
| boundaries judged | 6 |
| `clean` | 3 |
| `missing` | 3 |
| `duplicated` | 0 |
| `truncated` | 0 |
| material rate | 0.50 |
| threshold | 0.10 |
| decision | **material** |

Full per-boundary table:
[`docs/measurements/issue-156/summary.md`](../measurements/issue-156/summary.md).

### Pattern across the three `missing` boundaries

| boundary | cutoff | gap_s | RMS_before | RMS_after | symptom |
|---|---|---|---|---|---|
| 2 | 23:50 | 3.48 | 0.059 | 0.067 | `rank all of the potential <…> of relevant targets` — noun phrase split, ~3 s of unaccounted speech |
| 3 | 31:45 | 3.40 | 0.092 | 0.089 | `catch myself <…> to somehow like partially related to this` — head starts with bare `to`, no subject |
| 4 | 39:40 | 1.50 | 0.117 | 0.131 | `our Forecast product <…> scheduling product` — fuses into ungrammatical `Forecast product scheduling product`, almost certainly missing `and` |

Every `missing` boundary shares the same signature: **non-trivial
`gap_s` (1.5-3.5 s) AND mid-to-high RMS on both sides** — Whisper had
loud, continuous speech across the cutoff yet the stitched transcript
shows a multi-second gap with no tokens. The three `clean` boundaries
either land in silence (B0: RMS ≈ 0.0004 on both sides) or coincide
with a natural speaker-turn boundary (B5: clean sentence end + new
speaker greeting).

### Decision

The 5 s overlap is **not enough** for the chunked ASR path on the
48-min reference recording. Word-level loss occurs at 50 % of
boundaries when the cut lands in active speech.

This benchmark does **not** ship the fix. Two reasons:

1. The fix surface is exactly what
   [issue #159](https://github.com/artem-from-ua/voice-transcriber/issues/159)
   was opened to design: **snap the chunk boundary to a silence
   region** instead of cutting at a strict time offset. All three
   `missing` boundaries here would be eliminated if the chunk split
   were nudged a few seconds in either direction to land in a quiet
   stretch.
2. #159's PR can now use this measurement script directly on its
   snap-modified dump and report `material_rate` as the headline
   number — same units, same per-boundary breakdown, directly
   comparable to the 50 % baseline above.

A follow-up issue is opened to track the fix delivery via #159 (and
its predecessor #168 for silence-detection method choice).

### Links

- ADR 0031 — original chunked-ASR design.
- Issue #156 — this benchmark.
- Issue #159 — snap-to-silence fix (will land the actual remedy).
- Issue #168 — silence-detection method comparison for #159.
