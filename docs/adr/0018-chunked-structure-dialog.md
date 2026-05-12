---
status: accepted
date: 2026-05-12
---

> **Note (v0.25.0):** the module formerly called `structure` is now `speech_structure`. Body of this ADR preserves the historical name.

# ADR 0018 — Chunk the structure_dialog prompt for hour-long recordings

## Context

[ADR 0017](0017-whisper-asr-backend.md) noted that on a 16 GB Mac the gemma-3-12b LLM runs out of Metal memory when proceeding from `identify` into `structure_dialog`. The workaround documented in v0.20.0 (`--no-structure --no-tldr`) avoids the crash but disables two of the four LLM stages, which produces a flat transcript without thematic sections or a summary.

Issue #78 then raised the bar: the pipeline must **complete end-to-end on a one-hour Ukrainian recording on the same hardware** — no `--no-*` flags, no manual chunking. That target turns the OOM from a known-issue into a blocker for the primary use case.

The structural mismatch behind the failure:

- `structure_dialog` in v0.20.0 sends **every speech segment in one prompt**. A typical 6-minute test recording has ~90 segments and the prompt fits comfortably; a one-hour recording has ~900–1200 segments and the rendered script alone is 40–50 KB before the system prompt and the 2048-token output budget are added.
- gemma-3-12b 4-bit (~8 GB resident) plus its KV cache plus the 1200-input/2048-output activations exceeds the Metal pool that survives the proofread stage's allocator fragmentation (ADR 0006).
- The proofread KV churn (~90 short prompts on the test file, ~1200 on a one-hour recording) is the heaviest contributor to fragmentation; by the time `structure_dialog` runs, the largest free Metal block is already smaller than what a single-pass long-context prompt needs.

We have three independent axes we can pull:

1. Shrink the per-call working set in `structure_dialog` itself (chunking).
2. Spread the load across stages by letting different stages use differently-sized models (covered separately by [ADR 0019](0019-per-stage-llm-models.md)).
3. Free more memory between stages with explicit `gc.collect()` + `mx.clear_cache()` (a small one-liner already in `pipeline.py`).

This ADR is about axis 1. Axes 2 and 3 ship in the same PR but are decided independently.

## Decision

`structure_dialog` now has two execution paths, selected by speech-segment count:

- **Fast path** (≤ `STRUCTURE_CHUNK_THRESHOLD` segments, default 60): one LLM call, validator forces full coverage and 2–7 sections — exactly v0.20.0 behaviour, byte-for-byte. Short recordings pay no overhead.
- **Chunked path** (> threshold): the speech list is split into overlapping chunks (`STRUCTURE_CHUNK_SIZE=35`, `STRUCTURE_CHUNK_OVERLAP=2`), each chunk gets its own `chat_json` call, and the per-chunk results are reconciled into a single contiguous section list.

The chunked path's per-call validator is permissive: 1–4 sections per chunk, no exact-bounds requirement (start_ms / end_ms may drift to the nearest pause marker the LLM picked). Reconciliation does the global tightening:

1. Concatenate all chunk sections, sort by `start_ms`.
2. Drop sections wholly contained in the previous one (overlap-region duplicates).
3. Snap edges: first section starts at `total_start_ms`, each subsequent `start_ms` is forced to the previous `end_ms`. Last section's `end_ms` is forced to `total_end_ms`.
4. Merge adjacent sections with identical case-insensitive titles (chunks frequently emit the same topic name on both sides of a boundary when one thread spans them).
5. If the count still exceeds `MAX_SECTIONS=7`, greedily fuse the two shortest neighbours under the longer side's title until it fits.

A chunk that returns invalid JSON (or whose payload fails the permissive validator) is silently skipped; reconcile papers over the gap by snapping edges. If **every** chunk fails, we fall back to a single section titled "Розмова" / "Conversation" — the same v0.20.0 last-resort behaviour.

The constants are tuned to keep each chunked call's working set well under what the test box can spare after proofread:

- 35 segments × ~50 tokens/segment ≈ 1.7K-token user prompt
- 2048-token output budget (unchanged from v0.20.0)
- Peak MLX during one chunked call stays below ~10 GB on the 6-minute test recording (verified via the new `MlxLLM(log_memory=True)` instrumentation; the empirical pre-/peak/post lines land in `--verbose` output).

The overlap (2 segments) gives the LLM enough context at chunk seams to make a sensible decision about whether the topic is continuing or shifting; tighter values risk a content-free first paragraph in every chunk, looser values inflate the prompt.

## Consequences

**More LLM calls on long inputs.** A one-hour recording produces ~35 chunked calls instead of one. Each call is short (~2K input, ≤2K output) and runs in seconds, so wall-clock for the stage scales roughly linearly with segment count — predictable, no Metal cliff.

**Section duplication at chunk boundaries** is real and partially mitigated. The same-title merge step handles the common case (LLM independently chose the same title for the last section of chunk N and the first of chunk N+1). The case it doesn't handle is **renaming the same thread** ("обговорення задачі" vs "обговорення задач") — the merge step is conservative on purpose, because false fusing of unrelated topics is worse than emitting two near-duplicate titles. This is an accepted limitation; see Alternatives below for harder approaches we rejected.

**Edge-snapping can shorten a section.** If chunk N's last section ends at 12:34 and chunk N+1's first section starts at 11:50, reconcile keeps chunk N+1's later boundary and pushes chunk N's section back to 12:34. The displaced span (50 seconds in this example) is absorbed by whichever section the snap landed on. In practice this happens at pause markers, which is precisely where a section break should land.

**Failure semantics are unchanged for callers.** A `StructuredDialog` with one fallback section is what callers received before, and is what they receive now when chunked reconcile cannot produce a valid layout. No new exception types, no new option flags.

**`structure_user.md` is reused as-is.** The chunked path passes the chunk's own bounds to the same prompt template — the LLM is told "the dialogue spans X–Y", which happens to be a chunk's bounds, and the validator accepts any 1–4 sections within them. We do not introduce a `structure_chunk_user.md` variant; the prompt is generic enough.

**Validator now takes optional bounds-strictness.** `_validate(payload, *, total_start_ms, total_end_ms, min_sections=2, max_sections=7, require_exact_bounds=True)` keeps backward-compat defaults for the single-pass path; the chunked path passes `min_sections=1`, `max_sections=4`, `require_exact_bounds=False`.

**Constants are public for tests.** `STRUCTURE_CHUNK_THRESHOLD`, `STRUCTURE_CHUNK_SIZE`, `STRUCTURE_CHUNK_OVERLAP` are top-level so the test suite can use them to compute "expected number of chunks" deterministically.

## Alternatives considered

- **Smaller LLM for structure only.** Possible (and orthogonally enabled by ADR 0019), but does not on its own solve the 40 KB prompt problem — even a 1B model has to hold the prompt + KV in memory, and the issue is the **prompt size**, not the model weights. Chunking is the structural fix; smaller models are an optimisation.

- **Hierarchical / map-reduce summarisation** (`structure_dialog` calls itself recursively on chunk titles). Rejected for v0.21.0: too many failure modes per call, and the simple flat reconcile already produces clean output on the test recording. We can revisit if the same-title merge proves too coarse on real long recordings.

- **Pre-segment the dialogue by long pauses, then call `structure_dialog` per pause-block.** Effectively manual chunking, but tied to acoustic features instead of segment count. Rejected because pause distribution is bimodal in conversational Ukrainian — half-hour gaps don't exist, and a 4-second gap doesn't usefully partition a discussion. Fixed-size chunking with small overlap is more predictable.

- **Refuse to run `structure_dialog` on long inputs and emit a one-section transcript.** This is the current v0.20.0 workaround in disguise. Rejected: ships a degraded experience as the documented behaviour, which contradicts the issue #78 goal.

- **Stream the structure prompt via mlx-lm's prompt-caching API.** Not viable today: `lm-format-enforcer`'s `TokenEnforcer` keeps state per-call, so prompt caching does not amortise the JSON-constrained path. Worth revisiting when mlx-lm gains a chunked-prompt KV-reuse API that the enforcer can hook into.

- **Larger chunk size (~60) with no overlap.** Tried in scratch experiments: at 60 segments the prompt + 2K output started hitting the same Metal ceiling on some pre-fragmented runs. The plan's hypothesis (`STRUCTURE_CHUNK_SIZE = 35`) survived contact with the test recording; we'd rather pay the cost of two extra calls than gamble on the edge of the Metal pool.
