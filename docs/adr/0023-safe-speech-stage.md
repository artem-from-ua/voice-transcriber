---
status: accepted
date: 2026-05-12
see_also: [0005, 0008, 0018, 0019, 0024]
---

# 0023 — `safe_speech` stage for sensitive-content redaction

## Context

The pipeline produces a faithful, unredacted transcript of everything said in a recording.
When users share notes externally (archives, excerpts for third parties), they must manually strip
sensitive utterances — a tedious and error-prone step. Issue #92 requested an automated redaction stage.

The stage must decide what to redact at *data-transform time*, before the Markdown output is
rendered, so the dump artifacts remain a faithful source of truth for regression triage (per ADR 0005)
while the final output is clean.

## Decision

Add a new `safe_speech` stage placed **between `speech_structure` and `speech_summary`**:

```
[10] speech_structure → [11] safe_speech → [12] speech_summary → [13] render
```

### Why before `speech_summary`

The TL;DR is generated from `dialog.segments`. If redaction runs first, `speech_summary` receives
already-clean segments and cannot accidentally surface sensitive content in bullet points.
This eliminates a separate TL;DR-redaction pass and removes any dependency on the TL;DR bullet format.

Placing redaction *after* `speech_summary` (as originally proposed in issue #92) would require a
second LLM pass or regex scan to clean the generated bullets — brittle and format-coupled.

### Chunking: per section

The stage calls `chat_json` once per section produced by `speech_structure`, not once for the whole
dialog. This bounds the LLM working set to the largest section rather than the total recording length,
making memory requirements statically predictable before a run (analogous to the chunked approach
in ADR 0018, but using natural section boundaries instead of fixed-size chunks).

### Redaction granularity: utterance or contiguous utterance range

The minimum unit the LLM may redact is a whole `Segment` (one utterance). The LLM returns
`(start_idx, end_idx, topic)` ranges in section-local indexing; the stage collapses the range into a
single synthetic placeholder. The LLM never edits text inside a segment — it only flags ranges.

This is conservative by design: flagging an entire utterance is less risky than trying to excise
a phrase mid-sentence and accidentally leaving a semantic trace.

### Synthetic placeholder, not in-place mutation

Unlike the proofread stage (ADR 0005), which edits `Segment.content` in place, `safe_speech` replaces
a redacted range with a single new `Segment` whose `speaker` and `name` are `None`. This:

- Prevents the render stage from attributing the placeholder to any participant.
- Makes the redaction visually unambiguous — `[muted, 3.4s]` with a duration lets the reader
  cross-reference the original audio.
- Avoids fabricating speaker identity for synthetic content.

### Graceful degradation on LLM error

If `chat_json` raises `LLMError` for a section, the stage logs a warning and leaves that section
untouched, then continues with the remaining sections. This follows the pattern established in
ADR 0019 (per-stage model swap) and `speech_summary`.

## Consequences

- `04-merge.json`, `05-proofread.json`, and `08-speech_structure.json` remain unredacted
  (source-of-truth for regression triage). Only `09-safe_speech-decisions.json` onward reflects
  the clean world.
- The TL;DR is guaranteed to be free of sensitive content because `speech_summary` runs after
  redaction — no separate TL;DR scan is needed.
- The render stage is entirely unaware of redaction; it treats `[muted, X.Xs]` as ordinary text.
- Stage numbering shifted: `[10/11]` → `[10/13]`, `[11/11]` → `[12/13]`, render = `[13]`.

## Alternatives considered

**Placement after `speech_summary`** (original issue proposal): rejected because it requires a
second redaction pass over TL;DR bullets whose format may change independently. Placing before
summary eliminates this coupling entirely.

**Whole-dialog single prompt**: rejected for baseline because working set grows with recording
length, making memory requirements unpredictable. Possible future opt-in for short recordings.

**Per-segment (one LLM call per utterance, like proofread)**: rejected — ~900–1200 calls for a
1-hour recording, excessive Metal fragmentation.

**Per-chunk (fixed-size windows, like ADR 0018)**: rejected because per-section already provides
natural boundaries from `speech_structure`, without the reconcile logic needed when chunks
cross topic boundaries.

**Word/phrase granularity**: rejected — LLM may leave a semantic trace if it partially excises a
sentence, and distinguishing "personal disclosure" from "neutral mention" is harder at the phrase
level. Whole-utterance redaction is the safer conservative default.

**Redaction at render time**: rejected — would undermine testability and violate the dump-as-source-
of-truth contract (ADR 0005).

## See also

ADR 0008 (stage ordering rationale), ADR 0018 (chunked prompting precedent),
ADR 0019 (per-stage model), ADR 0024 (defaults — topics, policy, placeholder format).
