---
status: accepted
date: 2026-05-17
supersedes: []
---

# ADR 0033 — `--user-context` injected as a prefix to every LLM stage's system prompt

## Context

[Epic #134](https://github.com/artem-from-ua/voice-transcriber/issues/134) proposed a per-run, user-supplied topic hint that propagates into the LLM-driven stages. The user often knows the topic of a recording going in ("phone interview between two software engineers about ML deployments", "internal architecture review about MLX memory pressure"). That signal is currently discarded: every LLM stage runs with a fixed system prompt, with no per-run channel for a one-line description of *this* conversation.

The epic decomposed the work into two research subtasks ([#135](https://github.com/artem-from-ua/voice-transcriber/issues/135) for ASR via Whisper `initial_prompt` and [#136](https://github.com/artem-from-ua/voice-transcriber/issues/136) for proofread system-prompt addendum), each with its own benchmark and ADR obligations. Before either research arm starts, however, both need the same plumbing: a CLI flag, a `PipelineOptions` field, and a way to thread the value into a stage's system prompt without rewriting every prompt file.

This ADR records the shape of that plumbing — explicitly **not** the quality-vs-cost trade-off of any given stage (those live in #135 and #136 with measurements).

Two viable shapes for the system-prompt-side injection were considered:

1. **Prefix outside the prompt file.** A helper that takes a rendered system prompt plus the user's free-text string and returns either the original prompt (no-op) or `"<prefix>\n<text>\n\n<original>"`.
2. **`<<user_context>>` placeholder inside every `*_system.md` file.** Each system prompt declares the placeholder; the renderer substitutes the user-supplied string (or an empty string when the flag is absent).

## Decision

Adopt the **prefix-outside** shape.

- New CLI flag: `--user-context "..."` → `PipelineOptions.user_context: str | None`. Empty / whitespace-only input is normalised to `None` in the CLI layer so a stage cannot accidentally receive `""` instead of `None`.
- New helper in `src/voice/_prompts.py`: `with_user_context(system_text: str, user_context: str | None, *, language: str) -> str`. Returns `system_text` unchanged when `user_context` is `None` or blank; otherwise prepends a short header followed by the trimmed context and a blank line.
- Prefix wording is language-specific to match the surrounding prompt:
  - `language == "uk"` → `"Ця розмова описана користувачем як:"`
  - anything else → `"The user described this conversation as:"` (the project currently ships only Ukrainian and English prompts, so English is the safe default for any non-`uk` ISO code the language detector might surface).
- Every LLM stage's public entry point accepts `user_context: str | None = None` and wraps each `render_prompt("..._system", ...)` call in `with_user_context(...)`. The five wired stages are: proofread (per-segment + cache-prefix), identify_speakers, speech_structure (single-pass + chunked + cache-prefix), safe_speech (composes on top of the existing `<<topics>>` substitution), speech_summary (per-section + aggregate + final).
- ASR (Whisper `initial_prompt`) stays out of scope: it is measured separately in #135 with its own 224-token ceiling and off-topic-hallucination risk profile.

The default is `None`, so the entire pipeline behaves identically to the previous release when the flag is absent.

## Consequences

### Positive

- **One CLI surface for five stages.** Adding a sixth LLM stage tomorrow needs three lines: an extra parameter, one `with_user_context()` wrap, and the call from `pipeline.run`. No prompt-file edits.
- **Zero impact on default behaviour.** The helper short-circuits on `None`/blank, so the system prompt is byte-identical to the pre-change one. The `prompt_cache_session` fingerprint, the KV-cache hit rate, and every existing benchmark stay valid without re-running.
- **Composable with the existing `<<topics>>` placeholder in `safe_speech_system.md`.** The prefix is applied to the already-rendered prompt, so prompt-internal placeholders remain unaffected.
- **Documentable as a single concept** ("the user-context block goes in front of the system prompt") rather than as eight near-duplicate `<<user_context>>` placeholders to keep in sync.

### Negative

- **KV-cache rebuilds whenever `--user-context` changes between runs.** This is unavoidable for any mechanism that changes the system prompt and is the same cost the static glossary work in #124 / #108 will pay. Within a single `voice transcribe` invocation the cache amortises normally — the prefix is part of `prefix_messages` for every `prompt_cache_session`.
- **Convention, not enforcement.** A future LLM stage that forgets to call `with_user_context()` silently drops the user's input on the floor. Mitigation: the pipeline test suite already iterates all stage entry points; a one-line "does this stage thread `user_context` through?" assertion can be added if the convention starts drifting.

### Why the alternative was rejected

The `<<user_context>>` placeholder shape would require:

- editing eight `*_system.md` files (proofread, identify, structure × 2, safe_speech × 2, tldr_system_uk, tldr_system_en) to add the placeholder and a no-content path;
- every `render()` site passing `user_context=""` even when the feature is off, polluting call sites that have nothing to do with this feature;
- rendered output with leading blank lines (or conditional whitespace) when the value is empty, since Markdown prompts cannot conditionally hide a block.

The prefix-outside shape sidesteps all three concerns at the cost of one helper function and an extra parameter per stage. The trade is clearly worth it.

## Relationship to other ADRs

- Complements [ADR 0027](0027-prompt-cache-llm-stages.md): the prefix sits inside the cache-prefix message, so the session-level cache reuse stays correct.
- Plumbing only — the quality-vs-cost measurement that decides whether `--user-context` is worth using for proofread lives in [#136](https://github.com/artem-from-ua/voice-transcriber/issues/136); the ASR companion lives in [#135](https://github.com/artem-from-ua/voice-transcriber/issues/135). Either may eventually flip a default on or off; that decision will be a separate ADR.
