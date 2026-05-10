---
status: accepted
date: 2026-05-11
---

# ADR 0005 — Proof-read ASR per segment, not over the whole transcript

## Context

VibeVoice mishears some terms in a recurring way ("Hugging Space" → "Hugging Face", "градіо" → "Gradio", "CloudCop" → "Claude Code"). An LLM is well placed to fix those — but at what scope?

Two options:

1. **Whole-transcript proofread.** Send the entire dialogue and ask the LLM for a corrected version.
2. **Per-segment proofread.** Send each `Segment.content` independently with the same strict instruction.

In testing, option (1) drifts: the LLM normalises slang, removes filler words, merges short utterances, or hallucinates context. Even with strict prompts, the temptation to "edit" is high when the model sees the whole text.

## Decision

Proof-read per segment. The LLM sees one short text at a time with no surrounding context. The pipeline rejects any reply that diverges from the original by:

- Length ratio > 2× in either direction, or
- Levenshtein distance / max(len(orig), len(fixed)) > 0.5.

Markers like `[Human Sounds]` and segments under 10 characters skip the LLM entirely.

## Consequences

- The model cannot summarise, reorder, or smuggle ideas across the dialogue — it physically does not see more than one line.
- Cost: N small LLM calls instead of one big one. Still well within local-LLM budgets (~30–60 s for a 6-minute recording).
- We deliberately give up cross-segment correction (e.g. resolving "Hugging Space" later but "Hugging" earlier). In practice these almost always co-occur in a single segment.
- LLM failures (`LLMError`, length-ratio reject, Levenshtein reject) silently keep the original; the pipeline never aborts here.

## Alternatives considered

- **Whole-transcript proofread**: rejected for the drift reason above.
- **Skip proofreading altogether**: leaves "Hugging Space" / "CloudCop" in user-facing output, which is exactly the kind of artefact this stage exists to fix. Can still be opted out with `--no-postprocess`.
