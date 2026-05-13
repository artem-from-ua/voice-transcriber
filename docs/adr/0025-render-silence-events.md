---
status: accepted
date: 2026-05-13
supersedes: [0024]
see_also: [0017, 0021, 0023, 0024]
---

# 0025 — Silence events: unified pause/muted rendering with timestamp ranges

## Context

The render stage historically emitted two distinct, uncoordinated signals for "quiet time":

1. **`[пауза Nс]` blockquotes** — synthesised by `render.py` for any inter-segment ASR gap
   ≥ `EXPLICIT_PAUSE_S = 3.0 s`.  A typical hour-long recording produces 40–80 of these,
   most of which are 3–9-second breathing/transition pauses.  They create visual noise without
   adding navigational value for the human reader.

2. **`[muted, X.Xs]` inline text** — inserted by the `safe_speech` stage as `Segment.content`
   of a speakerless segment, showing the *summed duration* of the redacted utterances.

The two signals used different formats (seconds-count vs duration-string), were not merged when
they appeared adjacent (e.g. `[muted, 4.2s]` + 5 s gap + `[muted, 3.1s]` rendered as three
separate items), and neither made it easy to locate the event in the source audio (a timestamp
range is more useful than a duration).

Issue #88 originally proposed re-introducing VibeVoice-style in-band pause markers by inserting
synthetic `Segment` objects in `merge.py`.  After investigation this was rejected: the underlying
user need — a cleaner, more navigable transcript — is better served entirely within the render
layer, without touching the pipeline's data model.

## Decision

### New module: `src/voice/silence.py`

Introduces `SilenceEvent(start, end, kind)` and `extract_silence_events(segments, min_silence_s)`.
The extractor walks a segment list once and builds events from:

- inter-segment ASR gaps ≥ `min_silence_s`,
- `[muted, ...]` placeholder segments (any duration, since the redaction itself is the signal),
- chains of adjacent muted/gap/muted with no real speaker content between them (merged into one
  event; `kind = "muted"` if any placeholder was part of the chain).

Leading gaps between the previous speaker segment and a muted chain are absorbed into the event
(the silence started when speech stopped, not when the muted block starts).

Events shorter than `min_silence_s` after merging are discarded.

### Render changes

`_render_section_body` now uses `extract_silence_events` to pre-compute all events for a section,
then emits them as blockquote lines at the appropriate position:

```
> _[пауза 00:01:23–00:01:38]_   ← pure ASR gap ≥ min_silence_s
> _[muted 00:03:45–00:04:12]_   ← chain that contained at least one redacted segment
```

The old `if gap >= EXPLICIT_PAUSE_S: blocks.append(f"> _[пауза {int(round(gap))}с]_")` path
is removed.  `[muted, X.Xs]` segments are no longer emitted directly; they are consumed by
the silence extractor instead.

### Threshold and CLI

`MIN_SILENCE_S = 10.0` replaces `EXPLICIT_PAUSE_S = 3.0` as the render-time threshold.
Configurable via `--render-min-silence-s SECONDS` (passed through `PipelineOptions`).
The 3.0 s threshold used by `speech_structure._build_script` for its LLM script is unchanged
(that stage operates before `safe_speech` and serves a different consumer).

### TL;DR experiment flag

`--tldr-include-silence` (default OFF) injects silence-event lines into the TL;DR prompt.
This is an opt-in experiment to test whether long-pause context helps or hurts the summary LLM.
The flag is not documented as a stable feature; a follow-up issue will evaluate the outcome and
either promote it to default-ON or remove it.

## Consequences

- Markdown output is significantly cleaner: a 1-hour recording that previously showed ~60 pause
  markers now shows only the structurally meaningful long pauses and redacted regions.
- Muted regions are now displayed with absolute timestamps, making it easy to jump to that
  position in the source audio for review.
- Adjacent muted/gap chains appear as a single block, correctly representing the total quiet span.
- `src/voice/silence.py` is a new public module within the `voice` package.  Its API
  (`SilenceEvent`, `extract_silence_events`) may be used by other future consumers.
- The `safe_speech` stage's `Segment.content` format (`[muted, {dur:.1f}s]`) is **unchanged** —
  see ADR 0024.  Only the *render-time display* has changed.
- Issue #88's original proposal (synthetic segments in `merge.py` for LLM stages) is rejected
  for this PR.  If a future recording demonstrates that structure or TL;DR quality degrades
  without pause markers, that work should be revisited as a separate, evidence-gated task.

## Alternatives considered

**Raise `EXPLICIT_PAUSE_S` without unifying muted/gap**: does not fix the inconsistency between
the two silence representations, and does not enable merging of adjacent chains.

**Implement in `merge.py`** (original issue #88 text): `speech_structure` runs before
`safe_speech` in the pipeline (`[10/13]` vs `[11/13]`), so a merged silence stream built in
`merge.py` cannot include muted regions.  The LLM prompt for `speech_structure` already has its
own pause-injection logic (`PAUSE_GAP_S = 3.0` in `_build_script`); duplicating it in `merge.py`
would add complexity without benefit for the primary goal (cleaner human-readable transcript).

**Make `--tldr-include-silence` default ON**: rejected — no empirical evidence of benefit.
The TL;DR summarises content, not rhythm.  Keeping it default-OFF avoids prompt pollution until
side-by-side testing on real recordings provides a clear answer.

## See also

ADR 0024 (safe_speech placeholder format — `Segment.content` form unchanged, render display
superseded by this ADR), ADR 0023 (safe_speech stage architecture), ADR 0021 (removal of
VibeVoice in-band markers; background for issue #88), ADR 0017 (Whisper migration context).
