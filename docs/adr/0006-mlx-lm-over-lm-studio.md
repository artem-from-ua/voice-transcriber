---
status: accepted
date: 2026-05-11
supersedes: 0001-local-llm-via-lm-studio.md
---

# ADR 0006 — Run the LLM in-process via mlx-lm

## Context

ADR 0001 chose LM Studio as the LLM runtime because it was already installed, managed model downloads, and served an OpenAI-compatible HTTP API. Subsequent end-to-end runs against real 6-minute Ukrainian audio uncovered a hard limit: LM Studio's Metal allocator fragments after dozens of short postprocess prompts, and the next long-context prompt (structure or TL;DR) crashes with `Insufficient Memory (kIOGPUCommandBufferCallbackErrorOutOfMemory)`. The server then enters a state where every subsequent chat request fails with "model has crashed" until the app is restarted manually.

Bandaids (best-effort `unload_model()` between stages, MLX/MPS cache eviction in our own modules) reduced the symptom but did not eliminate it. The underlying cause is that LM Studio's MLX runtime does not expose the controls we need to keep the allocator clean between prompts.

`mlx-lm` lets us load the very same MLX-quantised model file in-process (`mlx_lm.load(path)`), call `mx.clear_cache()` ourselves between prompts, and `del` the model on shutdown. JSON-constrained generation via `lm-format-enforcer` (as an `mlx_lm.generate` logits processor) replaces LM Studio's `response_format: json_schema`, which itself crashed under memory pressure.

## Decision

Replace `LLMClient` (HTTP, LM Studio) with `MlxLLM` (in-process, `mlx-lm`). LM Studio is no longer in the runtime path — we still rely on it to download model files into its cache directory (`~/.cache/lm-studio/models/mlx-community/…`), but no Server tab, no HTTP, no native unload API.

## Consequences

- Pipeline runs in a single Python process. No subprocess crashes to recover from.
- Full control over the KV cache and model weights. `MlxLLM.close()` frees the model immediately; tests can `monkeypatch.setattr(mlx_lm, "load", …)` without spinning up a server.
- JSON output is guaranteed valid on the first try via `lm-format-enforcer`. No retry loop, no parse failures in practice.
- Cold start is paid once per pipeline run (~10 s for `gemma-3-12b-it-qat-4bit`). LM Studio used to amortise this across runs by keeping the model resident — we trade that for predictability.
- Three model sessions are now strictly serialised: VibeVoice loads → frees, pyannote loads → frees, then `MlxLLM` loads → frees. Never co-resident on the 16 GB Mac.
- CLI loses `--llm-base-url`. `--llm-model` keeps its name but its value is now a filesystem path (or any string `mlx_lm.load()` accepts).

## Alternatives considered

- **Keep LM Studio + harder mitigations.** Tried (PRs #14, #16). Crash rate dropped, did not reach zero. Operational pain of "user has to restart LM Studio after each crash" was unacceptable.
- **`ollama` as runtime.** Would require a second copy of the same model in GGUF format; explicit user preference is to reuse the existing MLX weights.
- **Pure `transformers` on MPS.** No support for the MLX-quantised checkpoints LM Studio downloaded. Re-conversion is a non-starter.
