---
status: accepted
date: 2026-05-13
see_also: [0006, 0019, 0020, 0026]
---

# 0027 — `prompt_cache_session()`: amortise the system-prompt KV across LLM-stage loops

## Context

Every LLM-driven stage in the pipeline (`proofread`, `speech_structure`,
`safe_speech`, `identify_speakers`, `speech_summary`) follows the same
shape: a fixed **system prompt** that describes the task, plus a series
of **user prompts** that vary per call.

Three stages call the LLM in a tight loop with the system prompt held
constant:

- **proofread** — one `chat()` per ASR segment (60–100 calls on a
  6-minute recording, ~150-token system prompt).
- **speech_structure** — one `chat_json()` per dialogue chunk (1–3 calls
  on a 6-minute recording, ~300-token system prompt).
- **safe_speech** — one `chat_json()` per section (3–7 calls, ~200-token
  system prompt).

Before this change, every call paid the full prompt-eval cost for the
system prompt — `mlx_lm.stream_generate` re-encodes the entire prompt on
each invocation, even when the leading tokens are byte-identical to the
previous call. On the reference recording (Qwen2.5-7B-Instruct-4bit, M1
16 GB) this dominated wall-clock: `proofread` reported `prompt 173 tok,
out 9 tok` per call, i.e. prompt-eval ≈ 95 % of per-call work, and the
stage took ~140 s for 86 calls.

This ADR records the decision to amortise that cost via the mlx-lm
prompt-cache API, the wrong path we walked down first, and the rules
that prevent silent quality regressions.

## Decision

Introduce `MlxLLM.prompt_cache_session(prefix_messages)`, a context
manager that:

1. Renders `prefix_messages` through the tokenizer's chat template **with
   `add_generation_prompt=False`** (so it ends after the last system /
   user-turn marker, leaving the assistant-marker slot for each
   per-call user prompt to fill in).
2. Allocates an `mlx_lm.models.cache.make_prompt_cache(model)`.
3. On each `chat()` / `chat_json()` call inside the block:
   - Builds the full chat-template prompt as usual.
   - Checks whether the cache currently holds exactly the prefix
     (`cache[0].offset == prefix_len` and the first `prefix_len` tokens
     of the rendered prompt match `prefix_ids`).
   - If yes, tokenises the full prompt and passes **only the delta
     tokens** (`full_ids[prefix_len:]`) to `mlx_lm.stream_generate` —
     mlx-lm appends them on top of the cached prefix KV state instead
     of re-encoding from scratch.
   - If no (first call inside the session, or the prompt doesn't match
     the prefix), passes the full rendered prompt; cache fills with
     whatever the call produces.
4. After each call, **trims the cache back to `prefix_len`** via
   `trim_prompt_cache(cache, cache[0].offset - prefix_len)`. This keeps
   the system-prompt KV alive while discarding the per-call user / asst
   tokens — otherwise the next call would resume generation from the
   previous turn instead of treating the new user prompt independently.
5. On context exit, drops the cache and runs `mx.clear_cache()` once,
   matching the long-standing per-call clear behaviour callers outside
   the session see.

Both `chat()` and `chat_json()` participate — including the
`lm-format-enforcer`-constrained JSON path used by structure /
safe_speech. The enforcer is a logits processor: it tracks its own
parser state independently of the KV cache, so the cache does not
interfere with schema enforcement.

Stages without a tight loop (`identify_speakers` — 1-2 calls;
`speech_summary` — 1 call) are **not** wrapped: a session-of-one is
strictly more work than no session, and the surface-area saved is zero.

## Consequences

### Quality

`05-proofread.json` is **byte-identical** between cache-on and
cache-off runs on the reference recording (`diff -q` empty). Sampling
consumes the same logits in the same order under both code paths, so
this is the expected behaviour for the final design, not a coincidence.

`09-safe_speech-decisions.json` is **byte-identical** as well —
`lm-format-enforcer` schema enforcement and `prompt_cache` coexist
without interference.

`08-speech_structure.json` shows a different *sectionisation* between
runs (6 sections vs 5, slightly different boundaries on the same
17-second range) — but the `segments` array is byte-identical (0/90
differing). The sectionisation differs the same way it would between
two consecutive `main` runs: structure runs at temp=0.1 and the chunked
path ([ADR 0018](0018-chunked-structure-dialog.md)) makes each chunk's
labels non-deterministic across runs. We accept this as pre-existing
noise, not a cache regression.

### Speed

Measured on the reference recording (M1 16 GB, Qwen2.5-7B-Instruct-4bit,
single run per branch with Safari + IDEs closed):

| Stage | calls | main (0.29.0) | feature (0.30.0) | delta |
|---|---|---|---|---|
| `proofread` | 86 | 139.0 s | **90.3 s** | **−35.0 %** |
| `speech_structure` | 3 | 43.0 s | 45.1 s | +4.9 % (within single-run noise) |
| `safe_speech` | 5 | 31.7 s | **18.4 s** | **−42.0 %** |

`speech_structure` is left wrapped despite the negligible delta on this
recording — at the chunked threshold (80+ segments) longer dialogues
will trigger 5–10+ chunks, where the system-prompt amortisation matters
the way it already does in proofread. See
[`docs/benchmarks/prompt-cache-proofread.md`](../benchmarks/prompt-cache-proofread.md)
for the per-stage numbers and the dual `diff -q` quality gates.

### System-prompt budget

Because the system prompt is now KV-evaluated **once** per stage instead
of once per call, the cost of growing it drops by orders of magnitude.
A 150-token → 400-token expansion that would have added 250 × 86 ≈ 21500
prompt-eval tokens to a proofread run now adds 250 tokens, paid once on
the first call. This unlocks issue #120 (proofread rework), which plans
to add a glossary section to the system prompt — previously that would
have made proofread unaffordably slow.

### What this does NOT change

- The `chat_json` JSON-schema correctness guarantee from
  `lm-format-enforcer` — verified empirically and by construction (the
  enforcer's state is logits-side, not KV-side).
- The transition between stages — `mx.clear_cache()` still runs on
  session exit, so e.g. `proofread`'s cache does not leak into
  `speech_structure`'s much-larger system prompt.
- Behaviour outside a session — when `prompt_cache_session()` is **not**
  on the stack, `chat()` and `chat_json()` behave exactly as in v0.29.0
  (full prompt → `mlx_lm.stream_generate` → `mx.clear_cache()`).

## Alternatives considered

### A. Cache without trim (initial attempt)

We first wired up the cache to be shared across calls with **no trim**
between calls — letting mlx-lm's prompt-cache logic "figure it out". On
the reference recording this produced a **+27.9 % wall-clock regression**
(140 → 178 s) **and** 12/90 segments where the output text materially
diverged in shameful ways: `Claude Code` → `код-код`,
`Hugging Face` → `HugginsFace`, capitalisation losses on proper nouns.

Root cause: mlx-lm's `prompt_cache` API assumes **chat-mode
continuation**. Each call's user+asst tokens stay in the cache so the
*next* call extends them with another turn. Our loop sent independent
same-prefix queries, but mlx-lm interpreted each new call as continuing
the previous one — sampling logits conditioned on a steadily-growing
hallucinated dialogue, not on the system prompt alone.

This branch never shipped. The investigation became the basis of the
"check the API's actual contract before assuming it does what the issue
says it does" lesson, captured in
[`docs/benchmarks/prompt-cache-proofread.md`](../benchmarks/prompt-cache-proofread.md).

### B. Cache with trim but full prompt re-passed each call

A correct-output but **no-speedup** intermediate. We trimmed the cache
back to `prefix_len` after each call (fixing quality — 2/90 differing
segments, same as the final design) but kept passing the full prompt to
`mlx_lm.stream_generate`. Result: −0.4 % wall-clock — within noise.

Root cause: `mlx_lm.generate_step` does **not** look at the cache to
decide which prompt tokens to skip. It runs prompt-eval on **every**
token of the `prompt` argument, regardless of cache state. The cache is
only consulted on a per-layer basis at offset time. So passing the full
prompt on call N+1 silently overwrote the cache's prefix KV with freshly
recomputed values — correct logits, zero speed gain.

### C. Per-stage shared LLM with a manually pre-filled prefix

Considered: run `model(prefix_ids)` once at stage start, save cache to
disk, reload per call. Rejected: more code, no measurable advantage
over (D) on a 6-minute recording, and complicates the cross-stage
unload-and-reload flow ([ADR 0019](0019-per-stage-llm-models.md)).

### D. Final design (chosen): delta-tokens + trim-back

Pass full prompt only on the first call inside the session (fills
cache). On subsequent calls, pass `prompt[prefix_len:]` — mlx-lm
appends these tokens to the cached prefix and runs prompt-eval only on
them. Trim back to `prefix_len` after each call so the cache always
"starts" at the same offset for the next call's delta append.

Result: 2/90 differing segments (within sampling noise) **and** ~37 %
wall-clock reduction on `proofread`. This is the implementation that
shipped.

## Refs

- Issue [#122](https://github.com/artem-from-ua/voice-transcriber/issues/122)
  — the original speedup proposal (which assumed mlx-lm's prompt cache
  auto-detected shared prefixes; turned out to require explicit
  delta-tokens + trim).
- PR [#123](https://github.com/artem-from-ua/voice-transcriber/pull/123)
  — implementation.
- Benchmark — [`docs/benchmarks/prompt-cache-proofread.md`](../benchmarks/prompt-cache-proofread.md).
- Issue [#120](https://github.com/artem-from-ua/voice-transcriber/issues/120)
  — proofread rework that uses the now-cheap system-prompt budget.
- [ADR 0006](0006-mlx-lm-over-lm-studio.md) — in-process mlx-lm, which
  is what made direct cache access possible (LM Studio's HTTP API
  doesn't expose prompt_cache).
