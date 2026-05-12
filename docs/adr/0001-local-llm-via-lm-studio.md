---
status: superseded
superseded_by:
  - 0006-mlx-lm-over-lm-studio.md
  - 0020-default-llm-qwen25-7b.md
date: 2026-05-11
---

# ADR 0001 — Use LM Studio as the local LLM runtime

> Superseded by [ADR 0006](0006-mlx-lm-over-lm-studio.md) (mlx-lm runtime
> replaces the LM Studio HTTP server) and [ADR 0020](0020-default-llm-qwen25-7b.md)
> (default LLM resolves via HuggingFace cache, not the LM Studio path).
> LM Studio is no longer in the runtime path.

## Context

The pipeline needs a local LLM for four stages: identify, proofread, structure, TL;DR. Three reasonable runtimes exist on macOS / Apple Silicon:

1. **LM Studio** — desktop app with a built-in OpenAI-compatible HTTP server on `:1234`. Supports MLX natively, manages model downloads through its UI.
2. **ollama** — Go daemon, native HTTP server on `:11434`. Supports GGUF (and recently MLX), CLI-driven model management.
3. **`mlx-lm` as a library** — load and call the model inside the Python process.

The user already runs LM Studio for the ASR model (VibeVoice via mlx-audio looks up models in `~/.cache/lm-studio/models/`).

## Decision

Use LM Studio as the LLM runtime. The pipeline talks to it over its OpenAI-compatible API (`/v1/chat/completions`, `/v1/models`).

## Consequences

- Zero extra setup: LM Studio is already installed and the model cache is shared with VibeVoice.
- The pipeline does not embed model code; it remains a thin client. Swapping models is a CLI flag (`--llm-model`).
- Same client works against any OpenAI-compatible endpoint (Anthropic-OpenAI proxies, ollama, custom servers), keeping the door open without code changes.
- LM Studio must be running and the server tab started before any LLM stage. The pipeline fails fast with a clear message via `LLMClient.health_check()`.

## Alternatives considered

- **ollama**: separate model cache, would require a second download for the LLM. Most MLX-quantised models are published as `mlx-community/*` and LM Studio's MLX runtime is better optimised than ollama's at the time of writing.
- **`mlx-lm` library**: cold start of a 12B model inside the pipeline process is slow (5–10 s) and the model occupies memory while VibeVoice is also active. LM Studio unloads idle models on demand.
