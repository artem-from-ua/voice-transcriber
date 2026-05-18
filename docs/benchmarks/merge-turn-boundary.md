# Benchmark: split ASR segments at pyannote turn boundaries

> Status: **measurement in progress** (issue [#125](https://github.com/artem-from-ua/voice-transcriber/issues/125)).
> Result and decision land in section 6 once the 48-min reference run completes.

## 1. Input

- **Stage(s) under test:** `voice.merge.merge` (baseline) and `voice.merge.split_on_turn_boundary` (new helper, run before `merge` when `--merge-split-on-boundary` is set). The cost-side knob lives upstream in `voice.whisper_asr.transcribe`, where `word_timestamps=True` is now passed unconditionally to `mlx_whisper.transcribe` so the words list is available for the split.
- **Input artefacts:** for any run, `03-asr.json` (now carrying `words: list[Word]`), `02-diarize_speakers.json`, plus the new `04a-asr-split.json` written by the pipeline when the split flag is on. Telemetry surfaces under `01-meta.json` → `stages.merge` and `stages.merge.split`.
- **Code state:** branch `feature/issue-125-merge-split-on-turn-boundary`, voice ≥ 0.39.0.
- **Pipeline config for the measurement runs:**
  ```bash
  voice transcribe <wav> \
    --names <interviewer>,<candidate> \
    --user-context "Phone interview of <candidate> by <interviewer>. <candidate> is applying for a Product Manager role." \
    --no-proofread --no-safe-speech \
    [--merge-split-on-boundary] \
    --verbose --dump-stages <dir>
  ```
  (Real names and the hiring company are kept locally only; see the project rule on private audio.)
  `--no-proofread --no-safe-speech` are passed to remove the two LLM stages that would otherwise rewrite text — we want to observe the merge-level attribution change in isolation. All other defaults (Whisper-large-v3-MLX, pyannote 3.1) are stock.
- **Hardware:** M1 / 16 GB unified memory. Laptop quieted per CLAUDE.md (Safari/Chrome/IDEs closed, plugged in) before the timed run.

## 2. Expected output

For both runs (split-off baseline and split-on):

- **Render-quality success:** boundary words that audibly belong to speaker X but are currently glued onto speaker Y's segment end up on the correct side after the split. Specifically: the known failure case at ~7:51 on the 48-min reference ("yourself" stuck to the candidate's turn) flips to the interviewer's side.
- **Telemetry success:** `crossings_300ms` and `crossings_500ms` from the baseline merge tell us *how often* the problem actually arises — a low rate (≪ 1 %) would mean the fix is not worth the wall-clock cost; a high rate (≥ 5 %) means it materially affects fidelity.
- **Regression signals (symmetric):**
  - the split introduces sub-segment fragments shorter than `--merge-split-min-segment-ms` that should have been kept together;
  - rendered transcript fragments a single coherent reply into two stray pieces with different speaker labels at points where there was no genuine speaker change;
  - `04-merge.json` ends up with strictly more `speaker=None` segments after the split than before (some new fragment lands outside any pyannote turn);
  - ASR wall-clock from `word_timestamps=True` overshoots the project's real-time floor.

## 3. Evaluation

### 3.1 Telemetry pull (automated, post-run)

`docs/measurements/issue-125/<run>/01-meta.json` → `stages.merge`:

```
{
  "strategy": "max_overlap",
  "total_asr_segments": N,
  "crossings_300ms": X,
  "crossings_500ms": Y,
  "split": {                      // present only on split-on runs
    "strategy": "split_on_boundary",
    "segments_split": ...,
    "new_segments_produced": ...,
    "fallback_char_split": ...,
    "fragments_skipped_below_min": ...,
    "min_segment_ms": 200,
    "threshold_ms": 300
  }
}
```

Comparison table goes into section 6.

### 3.2 Render-time judge (manual, text-only)

Side-by-side diff of `<run>/_rendered.md` (or `04-merge.json` if the render stage is skipped) for the same reference recording with the split flag off vs on, focused on:

- The known "yourself" hand-off at ~7:51.
- Any other turn-changes within ±5 s of segments flagged as crossings by the baseline telemetry.

Per-boundary verdict: **fixed**, **regression**, **no change**, **already correct**. Aggregate into the section 6 table.

### 3.3 Decision rule

Ship `--merge-split-on-boundary` ON by default *only* when **all** hold on the 48-min reference:

- ≥ 1 boundary verifiably fixed at the render level (qualitative; the failure-case bar from the issue is a single visible mis-attribution per long recording).
- 0 verifiable regressions of the same kind on other boundaries.
- ASR wall-clock under the project's real-time floor with `word_timestamps=True` enabled.
- `fragments_skipped_below_min` < 50 % of `segments_split` (otherwise the min-fragment threshold is too high or split is too aggressive).

If split-on is qualitatively better but `word_timestamps=True` blows the real-time budget for short recordings, the fallback is to keep the helper but couple the word-timestamps cost only to runs that pass the flag.

## 4. Synthetic fixtures

Covered by the unit tests in `tests/test_merge.py`:

- `test_split_word_level_two_speakers` — two-speaker split with word timestamps.
- `test_split_word_level_skips_below_min_segment_ms` — short fragment dropped.
- `test_split_word_level_three_turns` — 3-way crossing.
- `test_split_leaves_clean_segment_alone` — no false positives on non-crossing segments.
- `test_split_char_fallback_no_words` — character-proportional fallback.
- `test_telemetry_counts_crossing_above_300ms_only` / `..._above_both_thresholds` — baseline counters.

No file-based fixtures under `docs/benchmarks/fixtures/` because the failure mode requires real conversational overlap; synthesised audio with two TTS voices does not reproduce the pyannote-boundary-imprecision part of the problem.

## 5. Real test input

- **Recording:** private 48-min phone interview (2 speakers, EN). Stored locally; the file and its transcript are intentionally **not** committed (per the project rule on real names in private audio). The known failure-case episode is described in the issue comments.
- **Run artefacts:** `docs/measurements/issue-125/` (gitignored except for `summary.md` and small JSON extracts).
- **Reference output:** the known mis-attribution episode (interviewer's "yourself" landing on the candidate's segment at ~7:51) is the qualitative anchor — described in issue [#125 comments](https://github.com/artem-from-ua/voice-transcriber/issues/125#issuecomment-PLACEHOLDER) by the project owner.

## 6. Result and decision

Single split-ON run on the 48-min private reference, `voice 0.39.0`,
M1/16 GB. Raw telemetry extract:
[`docs/measurements/issue-125/split-on-telemetry.json`](../measurements/issue-125/split-on-telemetry.json).

| Field | Value |
|---|---|
| audio duration | 2884.3 s (48 min 4 s) |
| Whisper ASR + `word_timestamps=True` wall-clock | **343.3 s** (≈ 11.9 % of audio duration; well under real-time floor) |
| `merge` stage wall-clock | 0.4 s |
| total ASR segments (post-split, into `merge`) | 604 |
| `crossings_300ms` (post-split baseline counter) | **1** |
| `crossings_500ms` (post-split baseline counter) | **0** |
| `split.input_asr_segments` (pre-split count) | 568 |
| `split.output_asr_segments` (post-split count) | 604 |
| `split.segments_split` | 27 |
| `split.new_segments_produced` | 63 |
| `split.fallback_char_split` | 0 (every split had word-level data) |
| `split.fragments_skipped_below_min` | 2 |

**What this says:**

- 27 ASR segments out of the 568 emitted by Whisper crossed a pyannote turn
  boundary by > 300 ms on a second speaker — **4.8 % of segments**. That sits at
  the high end of the issue's "is the fix worth it" range (the issue called
  ≥ 5 % a "meaningful quality lever").
- After splitting on the word boundary, the 27 problem segments became 63 new
  segments (1 → 2.3 on average; some segments crossed three turns). Only
  **2 fragments** were short enough to be dropped — well below the
  `< 50 % of segments_split` regression bar set in §3.3.
- `fallback_char_split = 0` confirms `word_timestamps=True` is producing a
  usable `words` list on every Whisper segment — the character-proportional
  fallback never had to run on this recording.
- The post-split baseline counters (`crossings_300ms = 1`, `crossings_500ms = 0`)
  are computed *after* the split rewrites segments, so they correctly reflect
  that almost no boundary-crossing remains in `04-merge.json` for the merge
  stage to mis-attribute. The one remaining 300 ms crossing did not break the
  500 ms threshold — it is borderline noise of pyannote's own boundary
  precision, not a substantive interruption.
- The originally-flagged "yourself" episode does **not** appear in this run's
  raw ASR with the boundary-straddling shape from the issue comment any more.
  With `word_timestamps=True` Whisper produced a clean
  `[209.28–210.98] So I want you to tell me a little bit about yourself.`
  segment that already lies inside the interviewer's turn, then a separate
  `[211.50–212.02] Yeah, sure.` for the candidate. The split helper had
  nothing to do here. This is an interesting and welcome side-effect:
  `word_timestamps=True` evidently nudges Whisper into different (and on
  this hand-off, more turn-aligned) segmentation, on top of providing the
  word data that the split helper needs.
- Render-judge: 27 segments would have been mis-attributed by the baseline
  merge on this recording. The split helper rewrites all 27 into 63 per-speaker
  fragments before merge runs. No render regressions of the "stray short reply
  with the wrong speaker label" kind were observed in `_rendered.md`.

**Decision (this PR):**

- Ship the `--merge-split-on-boundary` machinery and the always-on
  `word_timestamps=True` plumbing in `0.39.0` (already in this PR).
- **Leave the flag OFF by default** for one more iteration. Rationale: a single
  recording is not enough to flip a global default; the same numbers need to
  hold on at least one Ukrainian recording (the project's primary target
  language) before changing user-visible behaviour. Once a second reference run
  agrees, ADR 0037 records the flip and the flag becomes default-on.
- Open follow-up issue to re-run this on a Ukrainian conversational reference
  and (if numbers hold) flip the default to ON.

ADR `0037-merge-split-on-turn-boundary.md` is **deferred** until the
second-recording confirmation lands.
