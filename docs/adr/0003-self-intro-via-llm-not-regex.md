---
status: accepted
date: 2026-05-11
---

# ADR 0003 — Detect self-introductions with an LLM, not regexes

## Context

The pipeline maps pyannote speaker clusters (`SPEAKER_00`, `SPEAKER_01`, …) to human names by looking for self-introductions in each cluster's first ~60 s. Two implementations were on the table:

1. **Regex over the language.** Match `я ([А-ЯІЇЄҐ]\w+)`, `мене звати ([А-ЯІЇЄҐ]\w+)`, `i['m]+ ([A-Z]\w+)`, etc. Per-language pattern tables.
2. **LLM prompt.** Send the snippet to the local LLM with a strict "did the speaker name themselves?" instruction; require JSON output `{name, confidence}`.

ASR text is noisy: dropped punctuation, mid-word boundaries, accidental capitalisations, and the speaker may say "Привіт, тут Артем" rather than "Привіт, я Артем" — regex coverage explodes. The LLM, by contrast, generalises across languages and phrasings without per-language code.

## Decision

Use the LLM. The prompt files live under [`src/voice/prompts/identify_*.md`](../../src/voice/prompts/) and their wire-level contract (temperature, max_tokens, response_format) is documented in [`docs/prompts.md`](../prompts.md). CLI flags still allow regex-free escape hatches: `--names "A,B"` skips the LLM entirely, and `--unknown-speaker {ask,keep}` handles missed detections.

## Consequences

- Adds one LLM call per speaker (cheap; small max_tokens).
- Works for any language the LLM understands; switching `--language` only changes the prompt's `{language}` slot.
- Failure mode is benign: unknown clusters fall back to `ask` or `keep`. Wrong matches are filtered by capitalisation, min length, and confidence checks.
- Conflict resolution is straightforward: if two clusters claim the same name, the higher-confidence one wins.

## Alternatives considered

- **Regex per language**: rejected — high maintenance cost, low recall on noisy ASR.
- **Speaker enrollment with reference recordings**: separately useful but solves a different problem (matching against known speakers) and was explicitly deprioritised in the planning conversation.
