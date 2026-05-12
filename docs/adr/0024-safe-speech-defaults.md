---
status: accepted
date: 2026-05-12
see_also: [0011, 0013, 0014, 0015, 0023]
---

# 0024 — Defaults for the `safe_speech` stage

## Context

ADR 0023 established the `safe_speech` stage. This ADR records the product defaults: which topics
are covered out of the box, what the default redaction policy is, and what the placeholder format
looks like. These are likely to change with user feedback, so they are separated from the
architectural decisions in ADR 0023 — following the clearspeech-defaults pattern (ADR 0011, 0013–0015).

## Decision

### Built-in topics

`["health", "drugs", "alcohol"]` — the three categories named in issue #92.

Topics are passed to the LLM as English ISO tags. The system prompt translates them for the model
(the Ukrainian system prompt includes a translate-hint). Users override via `--safe-speech-topics`.

More categories (`legal`, `finance`, `minors`, `sexual`) are documented as examples in `--help` but
not included in the default list — they are too aggressive for a general-purpose transcription tool
and would produce false positives on common professional recordings.

### Default policy: `placeholder`

`placeholder` replaces a redacted range with `[muted, 3.4s]`. This is preferred over `drop` because:

- The reader understands that *something was said* but it has been hidden — no deceptive gap.
- The duration lets them cross-reference the original audio if needed.
- Section continuity is preserved: surrounding utterances retain their context.

`drop` removes segments entirely. It is available via `--safe-speech-policy drop` for use cases where
even the presence of a gap must not be visible.

### Placeholder format: `[muted, {dur:.1f}s]`

- Duration is rounded to one decimal place.
- For a range of utterances, a single placeholder with the sum of durations replaces the whole range.
- If an entire section is dropped under `drop` policy, one `[muted, 0.0s]` marker is inserted to
  prevent the render stage from seeing an empty section.
- The placeholder deliberately does not include the topic name (`[muted: health]`) — doing so would
  leak information the user is trying to hide.

### Opt-out

`--safe-speech-topics ""` (empty string) maps to `[]`, which skips the stage entirely.
`--no-safe-speech` also disables the stage. Both are backwards-compatible with recordings processed
before 0.27.0.

## Consequences

Users with different sensitivity profiles supply their own topic list via `--safe-speech-topics`.
The built-in list covers the most common personal-disclosure categories without over-triggering on
professional content.

## Alternatives considered

**Include `legal` and `finance` by default**: rejected — too aggressive; a typical project-status
recording will mention contracts and budgets without being sensitive.

**Placeholder without duration** (`[muted]`): rejected — duration aids audio cross-reference.

**Placeholder with topic** (`[muted: health, 3.4s]`): rejected — leaks the very category the user
is redacting.

**`drop` as default**: rejected — creates an invisible gap that could mislead readers who assume
the transcript is complete.

## See also

ADR 0023 (stage architecture), ADR 0011, 0013, 0014, 0015 (clearspeech-defaults pattern).
