# Benchmarks — per-stage and end-to-end quality measurement

This is the project's framework for **evidence-based quality decisions**.
Whenever a stage default is in question — "should this stage be on by
default?", "is this prompt change a win?", "did the new ASR model help?"
— the answer goes through a benchmark documented here, not through
intuition.

## Why

CLAUDE.md states the project goal explicitly: capture lively Ukrainian
conversation as it was actually spoken, on M1/16 GB, faster than
real-time, in a structured Markdown document. Every default in the
pipeline either supports that goal or it doesn't — and the only way to
know which is which is to measure.

Specific risks the framework is designed to catch:

- **A stage doing net-negative work** (e.g. proofread on Whisper output —
  see [`proofread-hit-rate.md`](proofread-hit-rate.md)) silently
  degrading the transcript while looking busy in the logs.
- **Hit-rate / wall-clock metrics that mask quality regressions.** A
  stage that changes 20 % of segments isn't useful unless those changes
  are correct.
- **Default drift after upstream changes.** When ASR backend changed
  (VibeVoice → Whisper), the proofread stage's economics flipped from
  "essential" to "harmful". Without re-measurement, the default would
  silently keep doing the wrong thing.
- **Cherry-picked anecdotes.** A single bad/good example is not
  evidence; reproducible numbers on a documented input are.

## Benchmark anatomy — the contract every benchmark doc follows

Each benchmark document under `docs/benchmarks/<name>.md` answers
**these five questions in this order**, plus a sixth section for the
result and decision. Subsequent benchmarks should follow the same
shape so a future contributor (human or AI) can scan them without
re-learning the format each time.

### 1. Input — what is fed into the measurement?

- The specific stage(s) or sub-pipeline under test.
- The exact input artefacts (real audio path, or synthesised dump JSONs,
  or both).
- The exact code state — branch, commit, or version tag.
- The exact pipeline configuration — CLI flags, model choices, sampling
  overrides, hardware.

### 2. Expected output — what would "good" look like?

- The output artefact(s) the measurement examines (e.g. `05-proofread.json`,
  `08-speech_structure.json`, the final Markdown).
- Concrete criteria for what counts as a correct output for **this
  particular question**. Examples:
  - "proofread does not introduce new errors that ASR did not have";
  - "structure puts section boundaries within 5 s of the user-marked
    boundaries on the reference recording";
  - "final transcript faithfully preserves code-switched English tokens".
- What signals would mean **regression** (so the success criteria are
  symmetric, not just "did the metric go up?").

### 3. Evaluation — how is the result formally scored?

- The metric(s) computed and **what they actually mean** (in plain
  Ukrainian/English, with one concrete example).
- The threshold(s) that trigger a decision. E.g. "agreement < 70 % means
  the text-only judge is unreliable for this measurement and a
  human-with-audio pass is mandatory".
- Whether the metric is sufficient on its own (hit-rate is *not*; WER
  alone *is* for ASR), and what the secondary channel is if it's not.
- Whether the metric requires audio (human) or works text-only (judge),
  with the trade-off acknowledged.

### 4. Synthetic input / reference output — for unit-level reproducibility

- A small set of hand-crafted pairs that exercise each category /
  expected behaviour, stored under `docs/benchmarks/fixtures/<name>/`
  (or `tests/fixtures/...` when they are also test inputs).
- The generator (a script, a template, or just the documented
  hand-curation rule) so the set can be extended without forgetting
  the convention.
- Why these specific examples — what failure mode each one catches.

### 5. Real test input — for system-level reproducibility

- The actual recording(s) used (path on the dev machine; description
  of audio characteristics — length, speakers, language mix, IT terms,
  recording quality).
- The reference output (when one exists — a human-corrected transcript,
  a target structure, a set of expected redactions). When no reference
  exists yet, the doc says so explicitly.
- The dump directory the measurement was run against, kept under
  `docs/measurements/<issue-or-name>/` so future PRs can diff against
  it.

### 6. Result and decision

- The numbers, the verdict, and the explicit decision the project takes
  as a consequence (e.g. "default flipped to off; opt-in via X; ADR NNNN
  records the rationale").
- Links to the ADR, the issue, and any follow-up issues created.

## Levels of benchmark

Three levels, increasing in coverage:

| Level | What it measures | Example |
|---|---|---|
| **Stage** | One stage in isolation, with all its inputs supplied | `proofread-hit-rate.md` — diffs `04-merge.json` vs `05-proofread.json` |
| **Chain** | Several stages composed (e.g. ASR + merge + proofread) | `asr-to-proofread.md` (planned) — measure how often proofread degrades a *correctly transcribed* segment |
| **End-to-end** | Full `voice transcribe` from audio to Markdown | `e2e-quality.md` (planned) — given audio + reference transcript, how close does the final Markdown match? |

Higher-level benchmarks **subsume** lower-level ones — if the e2e
benchmark catches a regression that no stage benchmark caught, that's a
hint to add or sharpen a stage benchmark.

## Existing benchmarks

- [`proofread-hit-rate.md`](proofread-hit-rate.md) — does the proofread
  stage earn its keep on Whisper output? Result: no — proofread is now
  default-off ([ADR 0026](../adr/0026-proofread-default-off.md)).
- [`prompt-cache-proofread.md`](prompt-cache-proofread.md) — can the
  system-prompt KV state be amortised across calls in a tight LLM loop?
  Result: yes, with the right strategy (delta-tokens + trim-back) —
  proofread dropped 35 %, safe_speech 42 %; output byte-identical
  ([ADR 0027](../adr/0027-prompt-cache-llm-stages.md)).

## Storage of artefacts

- **Benchmark docs** — `docs/benchmarks/<name>.md`. Versioned, follow
  the six-section contract above.
- **Synthetic fixtures** — `docs/benchmarks/fixtures/<name>/`. Versioned
  along with the benchmark.
- **Per-run measurement artefacts** — `docs/measurements/<issue-or-name>/`.
  Includes the raw dump dir (or a pointer to it), the spot-check file
  with marks, judge marks, final tables. These are evidence; they live in
  the repo so reviewers can audit.
- **Reference recordings** — currently lives outside the repo
  (`~/Downloads/...`); each benchmark doc points to the specific file
  used. We do not redistribute these (privacy, size). When a recording
  has a reference transcript, that transcript is checked in under
  `docs/benchmarks/fixtures/<name>/reference.txt`.

## Running a benchmark

A new benchmark normally has:

1. A measurement script under `scripts/` that takes a `--dump-dir` (or
   audio path) and writes machine-readable output (JSON / Markdown) into
   a target directory.
2. A doc under `docs/benchmarks/<name>.md` following the six-section
   contract.
3. (If the measurement involves human evaluation) a workflow that
   produces a Markdown checkbox file for the human + a JSON file for
   any LLM judge, with a parser that aggregates marks into a final table
   with an agreement metric.

`scripts/proofread-classify.py` is the reference implementation of this
pattern. Future stage benchmarks should reuse its structure
(`classify` / `prepare-judge` / `parse-marks` + `---SUMMARY-START---`
machine-readable block).

## When to create a new benchmark

- Before changing any pipeline default that affects output quality.
- Before adding a new stage or a new model.
- Before declaring "stage X is fine" in an ADR or PR description —
  unless an existing benchmark already supports that claim.
- After a user-reported regression that cannot be reproduced by an
  existing benchmark — extend the closest one or create a new doc.

When **not** to create one: trivial refactors that don't touch output;
documentation-only changes; pure code-style cleanup.
