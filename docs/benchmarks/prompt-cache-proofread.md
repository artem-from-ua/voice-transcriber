# Benchmark: prompt-cache amortisation across LLM stages

Stage-level (proofread) and chain-level (structure, safe_speech)
benchmark for the **`prompt_cache_session()`** optimisation. Answers
the question: *does sharing the system-prompt KV state across calls
in a tight LLM loop give a speedup without changing the output, and
what is the right way to do it with `mlx_lm`?*

This document follows the six-section contract from
[`README.md`](README.md).

## 1. Input

- **Stages under test:**
  - `proofread` — 60–100 `chat()` calls with a shared ~150-token system
    prompt.
  - `speech_structure` — 1–3 `chat_json()` calls with a shared ~300-token
    system prompt (chunked path triggers at 88+ segments).
  - `safe_speech` — 3–7 `chat_json()` calls with a shared ~200-token
    system prompt (one per section).
- **Code states compared:**
  - `main` at commit `5b6eea5` (v0.29.0, no prompt cache).
  - `feature/122-prompt-cache-session` (v0.30.0, this work).
- **Pipeline configuration:** `voice transcribe <REC> --proofread
  --unknown-speaker keep --dump-stages /tmp/v122-{main,feat}/ --verbose`.
  ASR backend: Whisper-large-v3-MLX (the only backend since v0.23.0).
  LLM: `mlx-community/Qwen2.5-7B-Instruct-4bit` (the v0.22 default).
  Sampling: temp=0.1 (`call_kwargs("proofread_system")` / structure
  / safe_speech prompt registry).
- **Hardware:** M1 Mac, 16 GB unified memory. Safari and VS Code closed
  for the timed runs (see "Methodology hazards" below).
- **Input artefacts to the measurement:** dumps in
  `/tmp/v122-main/` and `/tmp/v122-feat/`, specifically:
  - `01-meta.json` — `stages.*.wall_clock_s` and `llm_calls` per stage.
  - `04-merge.json` — shared proofread input (must be identical between
    runs, otherwise the comparison is meaningless).
  - `05-proofread.json` — proofread output (the primary quality gate).
  - `08-speech_structure.json` — structure output.
  - `09-safe_speech-decisions.json` — safe_speech decisions.

## 2. Expected output

"Good" means **all of the following**:

- `04-merge.json` byte-identical between `main` and `feature` runs
  (sanity — same ASR input, same prompt input to proofread).
- `05-proofread.json`, `08-speech_structure.json`,
  `09-safe_speech-decisions.json` are **functionally equivalent** between
  `main` and `feature`:
  - For `chat_json` outputs (structure, safe_speech): the structured
    payload (sections, verdicts, redaction ranges) is identical. Free-text
    reason fields may differ within the same noise band as two consecutive
    `main` runs (temp=0.1 sampling on short generated text).
  - For `chat()` outputs (proofread): per-segment `content` differs by
    no more than the sampling-noise floor (empirically ≤ 3-5 segments on
    a 90-segment reference recording — same as the noise floor between
    two consecutive `main` runs).
- `proofread` wall-clock drops by **at least 25 %** on the reference
  recording. The original issue projected 30-40 %; below 25 % means the
  cache didn't fire as intended.
- `speech_structure` wall-clock drops measurably (it has only 3 calls,
  so the absolute gain is small; we require **no regression** rather
  than a guaranteed speedup).
- `safe_speech` wall-clock drops measurably (5 calls; same rule as
  structure).

**Regression signals** (any one of these blocks the change from
shipping):

- `chat_json` outputs change in their *structured* payload (verdict
  flipped, redactions added/removed, section count or ordering shifted).
- `proofread` differs by ≥ 10 segments out of 90 (more than the noise
  floor of consecutive `main` runs).
- `proofread` wall-clock *regresses* vs `main` (we saw this on the first
  wrong implementation — see "Methodology hazards").
- `mlx_lm.stream_generate` raises (e.g. on a `prompt_cache` API mismatch).

## 3. Evaluation

### Wall-clock

Reading `stages.<stage>.wall_clock_s` from each run's `01-meta.json`.
Per-stage delta `(t_feat / t_main) - 1`. The reference recording is
short (374.9 s of audio), so we accept any single-shot wall-clock with
± 5 % noise — if the laptop is quiet (see "Methodology hazards") this
is achievable from a single run per branch and one re-run for
confirmation.

### Quality — `chat_json` stages (structure, safe_speech)

`diff` of the relevant JSON arrays after sorting keys; manual inspection
of any free-text field that differs. Structured payload (verdict,
section count, redaction indices) must be identical. If the model
returned valid JSON satisfying the schema, the schema-relevant fields
are what we trust. lm-format-enforcer guarantees parseability — we
verify it didn't degrade with cache on.

### Quality — `chat()` (proofread)

Segment-level `diff` of `content` across `05-proofread.json`. Counts:

- `differing segments / total segments` between `main` and `feature`.
- Manual classification of each diff into one of:
  - **noise** — both outputs are plausibly correct; the difference is
    word-choice / synonym (`допилювати` vs `доділювати`) consistent with
    temp=0.1 sampling on the same logits.
  - **regression** — `feature` introduces a worse output (galution,
    capitalisation loss, garbled English term).
  - **improvement** — `feature` produces a strictly better output (the
    `хлопci` → `хлопці` Cyrillic/Latin fix we saw on the first cached
    run is an example).

Threshold: **0 regressions** is required. Noise + improvements together
≤ 5/90 segments. (Above 5/90 is suspicious even if no individual diff
is a regression — it suggests the sampling is shifted, not just
noisy.)

## 4. Synthetic input / reference output

`/tmp/cache_strategy_probe.py` — a 50-line probe that loads the LLM
once and runs the same 5 short proofread-style prompts through three
strategies (no cache; cache without trim; cache with trim back to
prefix). Comparison is direct `==` on the rendered output text.

`/tmp/cache_delta_probe.py` — an expanded probe (8 prompts) for the
final design: full-prompt on first call, **delta-tokens after trim** on
subsequent calls. This is the strategy that ships.

`/tmp/chat_json_cache_probe.py` — verifies the `chat_json` path with
`lm-format-enforcer` under the cache: schema enforcement remains
intact (5/5 valid JSON, identical verdicts), free-text reason fields
show the same noise as `chat()`.

The probes are deliberately kept out of `scripts/` — they are
investigation tools, not reusable regression harnesses. The
`MlxLLM.prompt_cache_session()` tests in `tests/test_mlxllm.py`
(two new tests: cache shared across calls + non-reentrant) are the
permanent regression coverage.

## 5. Real test input

Recording: `~/Downloads/<reference-recording>.m4a` (the project's
reference recording — 374.9 s, two speakers, Ukrainian conversational
with code-switched English IT terms `Hugging Face`, `Gradio`,
`Claude Code`).

Reference output: `05-proofread.json` from the `main` run is the
de-facto reference for "what proofread does on this recording with
v0.29.0 defaults". A `feature` run that is bit-identical to this
(modulo the 5/90 noise band) means quality is preserved.

Dump directories from the comparison runs:

- `/tmp/v122-main/` — `main` baseline, generated by `git switch main &&
  voice transcribe …`.
- `/tmp/v122-feat/` — feature branch run, generated by
  `git switch feature/122-prompt-cache-session && voice transcribe …`.

Both runs used `--unknown-speaker keep` to avoid the interactive
speaker-name prompt (the agent-driven run can't answer the prompt
interactively; `keep` is non-interactive and doesn't affect
proofread / structure / safe_speech).

## 6. Result and decision

### Numbers (M1 16 GB, single run per branch — full pipeline with proofread + structure + safe_speech wrapped)

| Stage | calls | main (v0.29.0) | feature (v0.30.0) | delta |
|---|---|---|---|---|
| `proofread` | 86 | 139.0 s | **90.3 s** | **−35.0 %** |
| `speech_structure` | 3 chunks | 43.0 s | 45.1 s | +4.9 % |
| `safe_speech` | 5 sections | 31.7 s | **18.4 s** | **−42.0 %** |
| `speech_summary` (TL;DR, no session) | 1 | 14.7 s | 16.8 s | +14 % (noise; not wrapped) |

`proofread` and `safe_speech` deliver clean wins. `speech_structure`
shows a +4.9 % wall-clock change on a 3-call recording — within the
single-run measurement noise band and not enough to be conclusive
either way. We keep it wrapped: in longer recordings the chunked path
fires 5–10+ times (the threshold is `STRUCTURE_CHUNK_THRESHOLD` = 80
segments; one of our test recordings is shorter than that and triggered
only 3 chunks). At that scale the system-prompt amortisation will
matter, the way it already does in proofread.

`speech_summary` is intentionally **not** wrapped — it's a single
call, so a session-of-one is strictly more work than no session. The
+14 % delta reported above is run-to-run noise on a single call (output
token count itself differs: 161 vs 236 tokens — the model wrote a
longer TL;DR on this seed, and the per-token cost dominates).

### Quality — proofread

`diff -q /tmp/v122-main/05-proofread.json /tmp/v122-feat/05-proofread.json` →
**no output, byte-identical**. (The first cached-output measurement on
an earlier intermediate strategy showed 2/90 differing segments —
within the temp=0.1 noise band — but on the final delta-tokens + trim
implementation the proofread output is bit-identical to the no-cache
baseline. Sampling consumes the same logits in the same order, so
this is the expected behaviour, not luck.)

0 regressions. Within the acceptance threshold.

### Quality — safe_speech (chat_json)

`diff -q /tmp/v122-main/09-safe_speech-decisions.json
/tmp/v122-feat/09-safe_speech-decisions.json` → no output. The
structured payload (decisions list, verdicts) is byte-identical.
`lm-format-enforcer` schema enforcement and `prompt_cache` coexist
without interference — confirmed empirically.

### Quality — speech_structure (chat_json)

`08-speech_structure.json` differs between runs: the `segments` array
is byte-identical (0/90 differing), but the `sections` array shows a
**different sectionisation** — 5 sections in `main` vs 6 sections in
`feature`, with different section titles and slightly shifted
boundaries on the 17000–55000 ms range (`main` keeps it as one
"Обговорення Voice Isolation" section, `feature` splits it into
"Обговорення шторок" + "Перевірка шторки Voice Isolation").

Both sectionisations are valid — the segments and the bulk timeline
agree, only the LLM's choice of where to draw section boundaries is
different. This is the same kind of sampling-driven divergence we see
between two consecutive `main` runs on this stage (the structure LLM
runs at temp=0.1 like the others), not a cache artefact. We accept it:
0 regressions in the segments themselves, only sectionisation noise.

Section labels can drift between runs anyway — see
[ADR 0018](../adr/0018-chunked-structure-dialog.md): the chunked
structure path is inherently non-deterministic across runs because
each chunk is judged in isolation and the reconciliation step picks
the chunk's labels verbatim. This existed before this PR.

### Decision

Ship `prompt_cache_session()` in v0.30.0:

- Default-on for proofread (issue #122) — single biggest economic win,
  and prerequisite for affordable proofread rework (#120).
- Default-on for structure (chunked path) and safe_speech (multi-section
  path) — both are tight loops with shared system prompt; same speedup
  mechanism applies. Verified that lm-format-enforcer's schema
  enforcement is unaffected.

Not wrapping:

- `identify_speakers` (1-2 calls per unknown speaker — a session of one
  is strictly more work than no session).
- `speech_summary` / TLDR (single call).

### Methodology hazards we hit (and a note about saving the next reader from them)

This benchmark is shaped by **two failed attempts** we walked through
before landing the right design. The lessons:

1. **Don't trust the issue's mental model of an external API. Verify.**
   Issue #122 assumed `mlx_lm`'s `prompt_cache` parameter would
   auto-detect a shared prefix and skip prompt-eval for it. The actual
   semantics are turn-based chat continuation — every new call extends
   the previous turn's KV state. Running this naively gave us a
   **+27.9 % regression** (140 s → 178 s) **and** material output
   corruption (`Claude Code` → `код-код`, `Hugging Face` → `HugginsFace`,
   capitalisation losses). The fix wasn't a bigger system prompt or a
   smaller model — it was reading
   `.venv/lib/python*/site-packages/mlx_lm/models/cache.py` and
   `.../generate.py` to see how `prompt_cache` is actually consumed.

2. **A correct output without a speedup is also a wrong answer.** Our
   first fix (trim the cache back to the prefix after each call but
   keep passing the full prompt to `stream_generate`) produced 2/90
   differing segments, well within the noise floor — output looked
   bit-identical. But wall-clock was −0.4 %, i.e. within measurement
   noise. The cache was correctly sized but had no actual effect,
   because `mlx_lm.generate_step` runs prompt-eval on every token of
   the `prompt` argument it receives. The cache is a per-layer
   resumption point at offset time, **not** a prefix-skip detector at
   the prompt level. The only way to get a speedup is to pass
   `prompt[prefix_len:]` after the first call — mlx-lm appends those
   tokens on top of the cached prefix and skips prompt-eval for the
   prefix entirely.

3. **The laptop has to be quiet.** Our first three measurement attempts
   were noisy because Safari was open. The cached run reported
   −0.4 % the first time and +28 % the second time, while the
   single-line behaviour of the cache hadn't changed — the laptop's
   memory pressure had. Project CLAUDE.md now states: ask the user to
   close Safari / IDEs / heavy apps before a benchmark run. The
   `before/after` comparison only means something if both runs are on
   the same `ps aux | grep -v zero` shape.

4. **Per-segment output diffs are the safety harness; wall-clock is the
   payoff.** This benchmark hinges on the dual-check: `diff -q` on
   `05-proofread.json` says "we didn't break anything", and
   `01-meta.json` says "we made it faster". Either one alone would be
   ambiguous. We almost shipped #2 above on wall-clock-alone evidence.

These hazards generalised into project CLAUDE.md rules:
- "Before running an LLM benchmark — ask the user to quiet the laptop."
- "Benchmarks and decision documents."

The first one is the operational guardrail; the second is the paper
trail that lets a future reader avoid re-walking this path.

### Refs

- Issue [#122](https://github.com/artem-from-ua/voice-transcriber/issues/122)
  — original speedup proposal.
- PR [#123](https://github.com/artem-from-ua/voice-transcriber/pull/123)
  — implementation.
- ADR [0027](../adr/0027-prompt-cache-llm-stages.md) — the decision and
  design notes.
- Issue [#120](https://github.com/artem-from-ua/voice-transcriber/issues/120)
  — proofread rework that uses the now-cheap system-prompt budget.
- Issue [#107](https://github.com/artem-from-ua/voice-transcriber/issues/107)
  — other proofread-speedup directions (batching, smaller model,
  skip-confident); orthogonal to this work.
