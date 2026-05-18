---
status: accepted
date: 2026-05-18
supersedes: []
---

# ADR 0037 — Merge-stage optimisation pipeline order

## Context

Issue #125 / PR #175 introduced `split_on_turn_boundary` (cut ASR
segments at pyannote turn boundaries) and a snap-to-low-probability-
word refinement on top of it. Issue #177 / PR #(this) added two more
opt-in word-level refinements — filler-bias and deadband — to recover
single straddling words that snap cannot reach.

Each refinement runs at a specific point in the merge pipeline. The
relative order is **load-bearing**: rearranging the steps either
breaks back-compat with prior calibration, or quietly drops the
correction. Several plausible rewrites turn out to silently regress
attribution. This ADR records the order and the reason, so a future
contributor adding a sixth refinement does not have to re-discover it.

## Decision

The merge stage runs word-level optimisations in this fixed order:

```
1. naive owner       per-word argmax over pyannote turn overlap
2. filler-bias       (opt-in) reassign known-filler words near a boundary
                     to the turn containing word.end
3. deadband          (opt-in) reassign any word whose centre is within
                     ±deadband_ms of a pyannote boundary to the
                     downstream turn
4. snap              (opt-in) shift each naive cut to the nearest
                     low-probability word within ±snap_window_ms
5. speaker_hint      majority-vote owner per emitted fragment, stored
                     on `AsrSegment.speaker_hint`
6. merge             produces final `Segment`s; honours `speaker_hint`
                     when present, otherwise falls back to max-overlap
```

The order is encoded in `voice.merge._split_segment_by_words` (steps
1-5) and `voice.merge.merge` (step 6). All opt-in refinements default
to OFF; turning them on is byte-additive — the unrefined baseline is
preserved when every refinement knob is at its default.

## Consequences

**Wins.**

- Each new optimisation has one obvious slot. "Where does my new
  per-word heuristic go?" has a defined answer: between step 1 and
  step 4. The default slot for any new word-level refinement is
  immediately after deadband and before snap.
- Failure modes are isolated by step. When a render-time misattribution
  appears, the telemetry counters (`filler_bias_applied`,
  `deadband_applied`, `snaps_applied`) say which step did or did not
  fire on the affected segment. Bisecting which refinement caused a
  regression is mechanical, not investigative.
- `speaker_hint` carries the post-refinement decision through `merge`
  unchanged. Without it, max-overlap would silently revert any
  refinement that shifted a fragment's time span outside its owner's
  pyannote turn — exactly the case snap and deadband produce on purpose.

**Costs.**

- `AsrSegment` carries a `speaker_hint: str | None` field used only by
  the merge stage. Downstream stages (`proofread`, `identify_speakers`,
  `speech_structure`, `safe_speech`, `speech_summary`, `render`)
  consume `Segment.speaker` and never need the hint, but the field
  still travels with each ASR segment.
- The single-fragment-with-hint case (`_split_segment_by_words` returns
  one fragment, but with a `speaker_hint` that differs from baseline
  max-overlap) needs special handling in `split_on_turn_boundary` —
  the obvious `if len(new_segs) > 1` early-out would drop the hint.
- Per-stage doc `docs/stages/merge.md` is the contract for new
  optimisations: any change to `src/voice/merge.py` must update the
  doc in the same PR. CLAUDE.md enforces this.

## Alternatives considered

- **Run filler-bias and deadband AFTER snap.** Tried during #177
  prototyping. Snap moves a cut between words; filler-bias and
  deadband operate on per-word owners. Running them after snap means
  re-deriving per-word owners from already-grouped fragments — the
  same code with one extra round-trip, and a regression vector if
  the re-derivation disagrees with the original assignment.
- **Run merge first, then re-attribute on fragments.** `Segment` has
  no `words` field; the per-word evidence needed by every refinement
  is gone by the time merge has run. Re-attributing on fragments
  would require shipping word-level data through to downstream stages
  that do not need it.
- **Single combined per-word "best-owner" function.** Considered for
  #177. Rejected because the refinements have different signals
  (filler whitelist vs centre-of-word distance vs lowest-probability-
  word within a window) and combining them into one decision loses
  the per-step telemetry that catches regressions during validation.
- **Default the refinements ON.** Rejected for #175 and #177. The
  project's primary target language is Ukrainian (`CLAUDE.md` project
  goal). All measurements so far are on a single English phone-
  interview reference. Flipping defaults requires a second reference
  recording in Ukrainian, tracked separately in #180 / #181.

## Links

- Issue [#125](https://github.com/artem-from-ua/voice-transcriber/issues/125) — original research.
- Issue [#177](https://github.com/artem-from-ua/voice-transcriber/issues/177) — single-word straddling, addressed by filler-bias + deadband.
- Issue [#180](https://github.com/artem-from-ua/voice-transcriber/issues/180) / [#181](https://github.com/artem-from-ua/voice-transcriber/issues/181) — default-flip decisions (deferred).
- PR [#175](https://github.com/artem-from-ua/voice-transcriber/pull/175) — split + snap shipped.
- [`docs/stages/merge.md`](../stages/merge.md) — load-bearing per-stage doc; impact-table view of all optimisations.
- [`docs/benchmarks/merge-turn-boundary.md`](../benchmarks/merge-turn-boundary.md) — single-source benchmark numbers.
