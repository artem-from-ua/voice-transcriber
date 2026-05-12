---
status: accepted
date: 2026-05-12
---

> **Note (v0.25.0):** modules formerly called `identify`, `structure`, `tldr` are now `identify_speakers`, `speech_structure`, `speech_summary`. Body of this ADR preserves the historical names.

# ADR 0019 — Allow a different LLM model per pipeline stage

## Context

Until v0.20.0 the four LLM-driven stages (proofread, identify, structure, tldr) all shared a single `MlxLLM` session running one model — `mlx-community/gemma-3-12b-it-qat-4bit` by default, overridable globally by `--llm-model`. That made sense when the test bed was a 6-minute recording: a single ~8 GB resident model handles a few dozen short prompts and one long prompt without Metal pressure, and the cost of loading weights only once dominates everything else.

Two observations shifted the design:

1. **Stages have wildly different working sets.** Proofread does 90–1200 short per-segment calls (each prompt is one segment's text, the LLM must judge a tiny edit); identify does at most a handful of short snippet calls; structure does one (or, after [ADR 0018](0018-chunked-structure-dialog.md), ~30) long-context calls with structured output; tldr does one long-context call. A 12B model is overkill for proofread per-segment edits and is the only stage that benefits from its full capacity for structure/tldr reasoning.

2. **Issue #78 stress-tests this exactly.** A one-hour recording sends ~1200 proofread calls through a 12B model. Each call leaves a fragment in the Metal allocator. By the time structure runs, even chunked, the pool is too fragmented for a 12B prompt to fit cleanly. Letting the user run proofread on a smaller model (3-4B class) and only swap to 12B for structure/tldr removes that pressure at its source.

The natural shape, then, is: keep the single-model path as the fast/default behaviour, but expose four per-stage overrides for users who want to dial in their own trade-off. Switching models between stages costs a cold load (~5–15 seconds for a 4 GB MLX weights file on an M-series Mac), which is meaningful on a 6-minute job but invisible on a one-hour job; the user pays the cost only when they explicitly ask for a different model.

## Decision

`PipelineOptions` gains four new optional fields:

```python
llm_proofread_model:  str | None = None
llm_identify_model:   str | None = None
llm_structure_model:  str | None = None
llm_tldr_model:       str | None = None
```

CLI exposes them as `--llm-proofread-model`, `--llm-identify-model`, `--llm-structure-model`, `--llm-tldr-model`. Each falls back to `--llm-model` (the global override), which itself falls back to `MlxLLM.DEFAULT_MODEL` if unset. **No defaults change**: with no flags, the v0.20.0 behaviour is reproduced byte-for-byte — one model for all four stages.

A new internal helper, `_ensure_llm(current, want_path, *, log, log_memory, progress)`, replaces the old "load one LLM in a try/finally" pattern in `pipeline.py`:

- If `current` is None: instantiate `MlxLLM(model_path=want_path)`, load, return.
- If `current.model_path == want_path` (or `want_path` is None, meaning "use whatever current is"): return `current` unchanged.
- Otherwise: `current.close()`, `free_mlx(log)`, then load a fresh `MlxLLM` at `want_path`.

The pipeline calls `_ensure_llm` four times — once before each LLM stage, only when that stage is enabled. Disabled stages (`--no-proofread`, `--no-tldr`, `--no-structure`, plus the `names_override + unknown_speaker=keep` shortcut for identify) skip the call entirely. The `finally` block at the end of the LLM section closes whatever LLM is currently resident, regardless of which stage installed it.

To make the swap auditable, `MlxLLM.log_memory` (introduced in the same PR) is turned on when `options.verbose` is set: every chat / chat_json prints `pre`, `peak`, `post-clear` MLX memory plus prompt/output token counts. After a model swap the new instance's first call's `pre` is the post-load size, which lets us measure cold-load overhead from the verbose log without extra instrumentation.

## Consequences

**Cold reload cost dominates on short recordings, vanishes on long ones.** On a 6-minute test recording with three model swaps the overhead is ~30 seconds wall-clock (3 × ~10 s). On a one-hour recording the same three swaps are ~30 seconds out of 30+ minutes of pipeline work — under 2 % overhead. The single-model fast path (no swaps) remains untouched.

**Default behaviour is unchanged.** Anyone who does not pass per-stage flags sees the same models, the same memory profile, and the same output as v0.20.0. No migration path needed.

**Per-stage defaults are deliberately NOT changed in this ADR.** The plan that motivated this PR (issue #78) called for a benchmark across `gemma-3-12b`, `gemma-3-4b`, `gemma-3-1b`, `Qwen2.5-3B`, `Llama-3.2-3B` to pick empirical defaults per stage. That benchmark requires subjective quality judgements on real recordings, which is out of scope for an autonomous coding session — running it would mean burning ~2 hours of compute on partial data. We ship the mechanism here; the empirical defaults stay an open follow-up, tracked under issue #78. Users who already know they want a smaller model on proofread can do so today via `--llm-proofread-model`.

**Stage label in the verbose log is per-call, not per-stage.** The `MlxLLM.chat()` / `chat_json()` lines are labeled with the operation (`llm.chat` / `llm.chat_json`), not with the pipeline stage, because the LLM does not know which stage called it. Cross-referencing is done by reading the verbose log linearly — stage banners ([7/10] Proofread, [9/10] Структурування) are emitted by `pipeline.py` immediately before the corresponding LLM calls.

**Helper-of-helpers temptation.** `_ensure_llm` is the only orchestration helper in `pipeline.py`; the rest of the file is straight-line per-stage logic. We resist the urge to factor out per-stage progress-bar plumbing or per-stage dump writes, because the per-stage variation is what gives the file its narrative: a reader can see what runs in what order without chasing call graphs.

**No retroactive renaming.** `--llm-model` keeps its name; it is now documented as "the default for all four stages". A literal-minded reader might want `--llm-default-model`, but renaming a long-published flag would break user scripts for no functional gain.

## Alternatives considered

- **Run one LLM with multiple LoRA adapters loaded simultaneously.** Out of scope for `mlx-lm` 0.21 (no adapter-swap API stable enough to rely on) and orthogonal to the OOM problem — adapters live on top of the same base weights, so they don't help with the fragmentation that comes from KV churn.

- **Use prompt engineering instead of swapping models.** Done already for proofread (the prompt is strict, short, output is constrained to "single line, same content"). The bottleneck on a one-hour recording is not prompt quality, it is Metal allocator fragmentation accumulated across 1200 calls. Smaller weights = smaller per-call working set = less fragment per call.

- **Auto-pick the model per stage based on dialogue length.** Tempting but premature. We do not have measurements showing which model is best for each stage on Ukrainian conversations; until the empirical benchmark exists, an "auto" mode would just be an opinion encoded in code. Better to expose the four flags and let users (and a future benchmark PR) pick the defaults explicitly.

- **Single `--llm-models "stage1=path1,stage2=path2,…"` combined flag.** Compact but harder to autocomplete in shell history. The four explicit flags map 1:1 to the per-stage fields in `PipelineOptions`, which keeps the docs and the code aligned.

- **Reload the model between stages even when it is the same path** (force full GC cycle). Tried in scratch experiments: the wall-clock cost (5–15 s × 3) is real and the memory benefit is marginal — `mx.clear_cache()` already releases the KV pool, and the model weights themselves are not the source of fragmentation. The plain `free_mlx(log)` between stages (introduced in the same PR independently) gives most of the win without the cold-load tax.
