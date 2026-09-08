# `[7] merge` — assigning speakers to ASR text

`src/voice/merge.py` glues the two upstream signals together: ASR text
segments (from `[6] speech2text`) and pyannote diarization turns (from
`[3] diarize_speakers`). The default assignment is one speaker per ASR
segment by **maximum temporal overlap**; everything else in this file
is an opt-in correction for known failure modes of that default.

## Pipeline at a glance

<details>
<summary>Diagram source</summary>
<!-- plantuml-generated -->

```plantuml
@startuml
title merge stage — optimisation pipeline

skinparam ActivityBorderColor #95A5A6
skinparam ActivityFontSize 12
skinparam ArrowColor #5B9BD5
skinparam DefaultFontName "Helvetica"

start
:ASR segments + pyannote turns;

:[1] naive owner
__always-on__
per-word argmax over turn overlap; <<#E8F4FD>>

:[2] filler-bias  (opt-in · impact **)
reassign known-filler words near a boundary
to the turn containing word.end
""--merge-split-filler-bias-tie-ms""; <<#E8F5E9>>

:[3] deadband  (opt-in · impact **)
reassign any word whose centre is within
±N ms of a boundary to the downstream turn
""--merge-split-deadband-ms""; <<#E8F5E9>>

:[4] snap  (opt-in · impact ***)
shift each split cut to the nearest
low-probability word within ±window
""--merge-split-snap-window-ms""; <<#E8F5E9>>

:[5] split_on_turn_boundary  (opt-in · impact ***)
cut a multi-speaker ASR segment
into per-speaker fragments
""--merge-split-on-boundary""; <<#E8F5E9>>

:[6] speaker_hint
__always-on__
majority-vote owner per emitted fragment
(carries refinements through merge); <<#E8F4FD>>

:[7] merge
__always-on__
emit final Segments; honour speaker_hint
when present, else max-overlap; <<#E8F4FD>>

:[8] crossings telemetry  (impact *)
__always-on__
counts straddling cases per run
into 01-meta.json; <<#FFF8E1>>

stop

legend right
  <back:#E8F4FD>   </back> always-on (baseline / structural)
  <back:#E8F5E9>   </back> opt-in refinement
  <back:#FFF8E1>   </back> telemetry (no render change)
endlegend
@enduml
```

</details>

![PlantUML Diagram](https://www.plantuml.com/plantuml/svg/XPJFKjim4CRlUegTS0cqXj8IohzXW4Ac9mxbsCbaDlR65iWhZoJ7JKzzYBu8fzvvb3v9LtO20SBquaaahRxVxdwhCn_GXReqAkXW24foEO4iolVlhzzWgw1BxJ5eor3fYencKihVQQxGOGddQT0p7UONrcNalbXZ7MmS3iu7v_jhJWqjXnlzas3tu-gkSxPPX0uk3Yyk1okRbpJ5seGOUOqbGUyhcHa5dM9FE2AzEZg_-GQUyf8uU7W7rHoPRI08jMD_hDJHzzqHCEePWMsOd1gFqJGuzudbyLXLv996-05TNk8Fi3DoRMpxos1r32Sd6rS7mxtXvUbflExZ2ARQ68cRQFG0VQbJeXaU_e0kAqm3R6rjAaVelSuPxbXaaou2ef879dI0CB4rP-ZcAbW8HKSCgUIAcZNdxU5juaprUadIDYVnbT4XMP5FWgQazBtU4dDmTTXXVXf1HfXDaBF_CYBFMpLe2ki9Kgcb8z0U6XqApUhX_XfA3tQwGWqBwapIyt9SsXKJU0MxX5XFkJS2prYj9nH4N-Xf0CAqWFOwIEkmL8vb91-KiKrIEJl1YPOJoqnQSdYuRpGBuYkig9fqU-l91gDESImsYOcD7nD_YpMY8PHYM2qYX7VIy1LtAiq27lsst9mwx7pxYivoifHRWxOVqTehnf9ZUE7e4c-japeaipW6hUUZAb2fGw3iKLNrKtHEamT7KvdiRe92uMoT5zrZiFdA_9z7tSuBpNYtJ0MZWPl5B1v3OTdMxZbgKv2y9T8sEV8Uo8ZRPEgIjsRjO0Ifis9IpWMDZ426jWFBkc--04cbQfA561AppCGfIj5BYh40hkQk1nzsfTW1js-zvLPoE1mUNEr6IHzifPIXNEOEdCwBe01E9fZU7It1G1PsuiefF2f3Vo8oyNs4dIXUfs8PD9lFWcF_Le8N9deg_jFX1S_AuQVS-smbXkN1XRH0bYOf-TSHgpFvo8F-3m00)

## Optimisations at a glance

| Step - Impact | What it does | Before | After |
|---|---|---|---|
| **[1] naive owner**<br>*always-on*<br>— | Each word goes to the pyannote turn it overlaps most. The baseline merge has always done this; everything below is a refinement on top. No CLI knob. | — | — |
| **[2] filler-bias**<br>*opt-in*<br>⭐⭐ | A back-channel `yes` / `okay` / `sure` sometimes lands on the previous speaker because Whisper put its timestamp half-and-half on both turns. When the word is a known filler AND its two top per-turn overlaps differ by less than the tie-window, reassign it to the turn containing `word.end`. CLI: `--merge-split-filler-bias-tie-ms 120 --merge-split-filler-lang en`. | <pre>Alice: ...the field pretty<br>       well yes<br>Bob:   yes yes i am lucky…</pre> | <pre>Alice: ...the field pretty<br>       well<br>Bob:   yes yes yes i am<br>       lucky…</pre> |
| **[3] deadband**<br>*opt-in*<br>⭐⭐ | Sometimes diarization marks the speaker change a beat late, so the new speaker's first word (`know`, `the`, ...) gets glued to the previous one. Any word whose centre lies within ±N ms of a boundary is moved forward to the downstream speaker. CLI: `--merge-split-deadband-ms 100`. | <pre>Alice: ...so yeah so you<br>       know<br>Bob:   the field pretty well</pre> | <pre>Alice: ...so yeah so you<br>Bob:   know the field<br>       pretty well</pre> |
| **[4] snap**<br>*opt-in*<br>⭐⭐⭐ | Diarization boundaries can be a couple of words off. Move each split cut to the word the recogniser is least sure about — that is almost always where the speaker actually changes. CLI: `--merge-split-snap-window-ms 500`. | <pre>Alice: ...so yeah so you<br>       know<br>Bob:   the field pretty well<br>       yes<br>Alice: yes yes i am lucky…</pre> | <pre>Alice: ...so yeah<br>Bob:   so you know the field<br>       pretty well yes<br>Alice: yes yes i am lucky…</pre> |
| **[5] split_on_turn_boundary**<br>*opt-in*<br>⭐⭐⭐ | When two people talk over each other, split the transcript line at the moment the second person joins in. CLI: `--merge-split-on-boundary`. | <pre>Alice: ...about the field<br>Bob:   (silent)<br>Alice: pretty well yes yes i<br>       am lucky…</pre> | <pre>Alice: ...about the field<br>Bob:   pretty well<br>Alice: yes yes i am lucky…</pre> |
| **[6] speaker_hint**<br>*always-on*<br>— | Pick the majority-vote owner across a fragment's words and stamp it on `AsrSegment.speaker_hint`. Lets refinements 2-4 override `merge`'s max-overlap decision when their corrected owner disagrees with the fragment's time span. No CLI knob. | — | — |
| **[7] merge**<br>*always-on*<br>— | Emit final `Segment`s. Use `speaker_hint` when present, otherwise fall back to per-segment max-overlap with pyannote turns. No CLI knob. | — | — |
| **[8] crossings telemetry**<br>*always-on*<br>⭐ | Count how many ASR segments overlap two pyannote speakers by more than 300 / 500 ms each. Surfaced in `01-meta.json.stages.merge`. Informs whether the opt-in fixes above are worth turning on by default. No render change. | — | — |

Legend: ⭐⭐⭐ visibly fixes a failure the user can point to in the
rendered transcript · ⭐⭐ fixes a subset of the same family, leaves
measurable residual · ⭐ no rendered difference, informs the next
decision.

The pipeline order is load-bearing: rearranging the steps either
breaks back-compat with prior calibration or silently drops a
refinement. See [ADR 0037](../adr/0037-merge-split-optimisation-pipeline.md)
for the reasoning and what would break under each plausible reordering.

## How the stage works

### Inputs

- `asr_segments: list[AsrSegment]` from `whisper_asr.transcribe()`.
  Each segment carries `start`, `end`, `content`, and (since 0.39.0)
  optional `words: list[Word]` with per-word `probability`. ASR ran
  with `word_timestamps=True` unconditionally.
- `turns: list[DiarTurn]` from `diarize_speakers.diarize()`. Each turn
  is `(start, end, speaker)` where `speaker` is a pyannote label like
  `"SPEAKER_00"`. Turns come from pyannote's `exclusive_diarization`
  view, so they do not overlap.

### Output

- `segments: list[Segment]` — one `Segment` per (possibly post-split)
  ASR fragment, with `speaker` set from pyannote.
- `telemetry: dict` — surfaced into `01-meta.json.stages.merge`; see
  the "Telemetry" section below.

### Algorithm

```
for seg in asr_segments:
    if seg crosses >=2 pyannote turns by more than threshold_ms
       AND --merge-split-on-boundary:
        word-by-word, assign each word to its argmax-overlap turn
        find naive cuts at every owner change
        for each naive cut, if --merge-split-snap-window-ms > 0:
            find the lowest-probability word within ±window of the
            cut whose probability is strictly below the threshold;
            shift the cut to BEFORE that word
            rewrite per-word owners on the moved side
        emit one fragment per consecutive same-owner group;
        each fragment carries a `speaker_hint` = majority-vote owner
    else:
        keep the segment unchanged

for fragment in (possibly split) segments:
    if fragment.speaker_hint is not None:
        speaker = fragment.speaker_hint        # snap overrides max-overlap
    else:
        speaker = argmax(time overlap with each pyannote turn)
    emit Segment(start, end, content, speaker)
```

The `speaker_hint` path matters specifically when the snap step shifts
a fragment's time span enough that the *time-overlap* max-overlap
would put it back on the original side of the pyannote boundary — the
hint encodes "we already decided who owns this fragment, do not second-
guess it." Without it the snap correction would be quietly reverted.

### Telemetry

`01-meta.json.stages.merge`:

```jsonc
{
  "strategy": "max_overlap",
  "total_asr_segments": N,
  "crossings_300ms": int,    // # segments overlapping >=2 speakers by >300 ms each
  "crossings_500ms": int,    // same with a 500 ms gate
  "split": {                 // present only when --merge-split-on-boundary
    "strategy": "split_on_boundary",
    "input_asr_segments": int,
    "output_asr_segments": int,
    "segments_split": int,
    "new_segments_produced": int,
    "fallback_char_split": int,    // chars-proportional fallback when no words
    "fragments_skipped_below_min": int,
    "min_segment_ms": int,
    "threshold_ms": int,
    "snap_window_ms": int,
    "snap_prob_threshold": float,
    "snaps_applied": int           // # naive cuts that actually moved
  }
}
```

The baseline crossings counters are computed *after* any split, so on
a split-on run they reflect the residual rate, not the original one.

### CLI knobs

| Flag | Default | Notes |
|---|---|---|
| `--merge-split-on-boundary` | off | Opt-in for the whole split mechanism. |
| `--merge-split-min-segment-ms` | 200 | Fragments shorter than this are dropped; if all would be dropped, the original segment is kept unchanged. |
| `--merge-split-threshold-ms` | 300 | A segment must overlap each of ≥2 speakers by more than this to qualify as boundary-crossing. |
| `--merge-split-snap-window-ms` | 0 (off) | Cap on how far a cut may snap from its naive position. |
| `--merge-split-snap-prob-threshold` | 0.7 | Snap only to words strictly below this `probability`. |
| `--merge-split-filler-bias-tie-ms` | 0 (off) | Near-tie overlap (ms) at which a filler word is reassigned to its `word.end` turn. |
| `--merge-split-filler-lang` | `none` | Which filler list to use. `en` → built-in English fillers; `none` disables. |
| `--merge-split-deadband-ms` | 0 (off) | Any word whose centre is within ±N ms of a pyannote boundary is reassigned to the downstream turn. |

### Why the optimisation order matters

The current order is `split_on_turn_boundary` → snap → `speaker_hint`
→ `merge`. It is not interchangeable, and each step exists because it
catches a failure mode of the previous one:

1. **Split runs before merge** because `merge` produces `Segment`,
   which has no `words` field. Once a multi-speaker `AsrSegment` is
   collapsed into a single `Segment` by max-overlap, the per-word
   evidence needed to undo that decision is gone. Splitting has to be
   a preprocessing pass over `AsrSegment`s, not a post-processing pass
   over `Segment`s.
2. **Snap runs inside split, between owner assignment and group
   emission**, because snap needs the naive cut indices to exist and
   needs to rewrite per-word owners before the groups are sliced out.
   Snapping after the groups are emitted means re-splitting already-
   emitted fragments — that is the same code with one extra round-trip.
3. **`speaker_hint` is set inside split (at group emission), not
   globally**, because the hint exists exactly when the snapped
   fragment's time span no longer agrees with its word-derived owner.
   For clean splits where snap did nothing, `speaker_hint=None` and
   `merge` falls back to max-overlap — keeping backwards compatibility
   with the no-words path and with all callers that build
   `AsrSegment`s by hand (tests, scripts).
4. **`merge` is always last** because it is the only step that emits
   `Segment` — the type the rest of the pipeline consumes. Every
   speaker decision must land here, either via `speaker_hint` or via
   max-overlap. Splitting the speaker decision across two stages turns
   debugging into archaeology.

**Adding a new optimisation:** put it inside `_split_segment_by_words`
in the right slot relative to the existing steps. The default slot
for a new word-level heuristic is **between naive owner assignment
and the snap step** — that is where per-word evidence is freshest and
hint propagation comes for free. If a candidate optimisation does not
fit in that slot, that is usually a signal it operates on the wrong
abstraction; consider whether it belongs upstream (`whisper_asr`,
`diarize_speakers`) or downstream (`proofread`, `render`) instead.

### Known limitations (open items, not bugs in this stage)

- **Whisper-hallucinated text on silence** (e.g. a sequence of `.` or
  a spurious `"Thank you."` over a long pause) lands here as a real
  `AsrSegment` and gets attributed by max-overlap to whichever
  pyannote label happens to be active. This stage cannot tell that
  the input was bogus.
- **pyannote mis-attribution of short confirmations** (`"Sure."`,
  `"Yeah, sounds good."`) — these are separate ASR segments, not
  cross-boundary cases. Their speaker comes straight from pyannote;
  if pyannote put them on the wrong cluster, this stage has nothing
  to correct.
- **Single straddling word at a clean turn change.** Addressed by the
  filler-bias and deadband refinements above (#177). The original case
  (`yes` back-channel) is fixed by `--merge-split-filler-bias-tie-ms`;
  the related case of a leading content word ("know", "the") swallowed
  by the previous turn is fixed by `--merge-split-deadband-ms`. Both
  default OFF until validated on a Ukrainian reference recording.
- **Non-filler content words at clean turn changes** that are *not*
  near a pyannote boundary still rely on max-overlap. Whisper-level
  tuning of `word_timestamps` granularity is the next lever — out of
  scope here.

## Updating this document

**Mandatory: any change to `src/voice/merge.py` must update this
document in the same PR.** That includes adding a new optimisation
step, changing a default, removing a flag, or noting a new failure
mode. The table at the top is the primary view contributors and
reviewers scan first; the prose below is where the why-and-how lives.

When adding a new optimisation:

1. Decide where it fits in the pipeline order — see "Why the
   optimisation order matters". Most new word-level heuristics belong
   inside `_split_segment_by_words`, between naive owner assignment
   and the snap step. If the new step cannot live in that order, that
   is the signal to discuss the design before writing code.
2. Add a row to the impact table with the appropriate emoji.
3. Document the algorithm in "How the stage works" and the new
   telemetry fields under "Telemetry".
4. Add the new CLI flag(s) to the "CLI knobs" table.
5. If shipping the change flips an existing default or invalidates a
   prior decision, also write or update the corresponding ADR.
