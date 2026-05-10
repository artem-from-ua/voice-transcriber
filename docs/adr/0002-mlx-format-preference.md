---
status: accepted
date: 2026-05-11
---

# ADR 0002 — Prefer MLX format over GGUF for local models

## Context

Quantised model weights for local Apple Silicon inference are available in two main formats:

- **MLX** — Apple's array framework, used by `mlx-lm`, LM Studio's MLX runtime, and `mlx-audio`. The community publishes most modern models under `mlx-community/*` on Hugging Face.
- **GGUF** — llama.cpp's format. Mature, very portable, runs anywhere.

Both work on Apple Silicon. The pipeline lets the user override the model with `--llm-model`, so we still need a default and a guideline.

## Decision

Choose MLX as the default and prefer `mlx-community/*` repositories everywhere — for the LLM, the ASR model, and any future additions.

## Consequences

- Faster cold start and slightly better throughput on M-series hardware (LM Studio's MLX runtime is faster than its llama.cpp runtime in our environment).
- Some niche fine-tunes (e.g. INSAIT's MamayLM, the Ukrainian Gemma 2 9B variant) are published as GGUF only; using them requires either GGUF (acceptable but not the default) or a manual `mlx_lm.convert -q`.
- Documentation (README, models.md) recommends MLX downloads in LM Studio first.

## Alternatives considered

- **Default to GGUF**: wider model availability, but worse speed and a different model cache layout in LM Studio.
- **Don't pick a default**: forces every user to read docs before the first run; bad UX.
