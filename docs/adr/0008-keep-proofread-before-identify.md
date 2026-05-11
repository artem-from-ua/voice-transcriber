---
status: accepted
date: 2026-05-11
---

# ADR 0008 — Keep proofread before identify; do not reorder

## Context

The pipeline runs `proofread` (stage 7, LLM proof-reads ASR per segment) **before** `identify` (stage 8, LLM detects speaker names from segment content). A reasonable proposal: swap the order so `identify` runs first, then feed each detected speaker name into the `proofread` prompt to stop it "correcting" Ukrainian names into Russian transliterations.

The proposal sounds appealing on first inspection but does not survive a look at the actual prompts and data flow.

## Decision

Keep the current order: **proofread → identify**.

## Why the reorder fails without further changes

1. **`proofread` does not currently receive speaker context.** The system prompt instructs the LLM to "fix obvious mishearings" and the user prompt is literally just `Text: <<text>>`. There is no `{speaker}` or `{name}` placeholder. Putting `identify` first would fill `seg.name` on each segment, but `proofread` ignores it — the reorder gains nothing.

2. **`identify` benefits from cleaned content.** `identify` looks for self-introductions ("Я — Артем") in the first ~60 seconds of each speaker cluster. Garbled ASR output like "Я Яртем" or "Я артим" reduces its detection rate. Today `proofread` runs first and cleans those up before `identify` sees them — a quiet upstream improvement we would lose.

3. **Both effects compound.** Even if `proofread` were rewritten to take speaker context, the gain would be partially offset by `identify` working on raw ASR text instead of proof-read text. The combined direction of effect on this material is unclear without measurement.

## Consequences

- The "stop proofread from correcting Ukrainian names to Russian" idea is **not** abandoned, just deferred. It would require:
  - A new placeholder in `proofread_user.md` (e.g. `Speaker: <<name>>. Text: <<text>>`).
  - Updating `proofread.fix_segment()` to pass the speaker name from `seg.name`.
  - Reordering identify → proofread.
  - Re-measuring proofread's accuracy and identify's detection rate on the test recording.
- Until that lands, the existing order produces measurable wins (per the `proofread` proof-read counts in CHANGELOG `[0.12.0]` and `[0.13.0]`).

## Alternatives considered

- **Reorder only (no prompt change).** Rejected: as analysed above, this is a regression. `identify` would work on dirtier text without any compensating benefit.
- **Reorder + name-aware proofread prompt.** Rejected as out of scope for the current PR work. Worth doing as its own follow-up issue once someone has time to measure the impact end-to-end.
- **Run proofread twice — once before identify, once after with names.** Doubles the LLM call count for stage 7 (already the slowest of the four LLM stages). The win would have to be very large to justify the cost. Not measured; not pursued.
