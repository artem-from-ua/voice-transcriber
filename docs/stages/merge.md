# `[7] merge` — assigning speakers to ASR text

`src/voice/merge.py` glues the two upstream signals together: ASR text
segments (from `[6] speech2text`) and pyannote diarization turns (from
`[3] diarize_speakers`). The default assignment is one speaker per ASR
segment by **maximum temporal overlap**; everything else in this file
is an opt-in correction for known failure modes of that default.

## Optimisations at a glance

| Impact | What it does | Wrong-merge (before) | After this optimisation |
|---|---|---|---|
| ⭐⭐⭐ | When two people talk over each other, split the transcript line at the moment the second person joins in. CLI: `--merge-split-on-boundary`. | <pre>Alice: ...about the field<br>Bob:   (silent)<br>Alice: pretty well yes yes i<br>       am lucky…</pre> | <pre>Alice: ...about the field<br>Bob:   pretty well<br>Alice: yes yes i am lucky…</pre> |
| ⭐⭐ | Diarization sometimes draws the speaker-change line one or two words off. Move it to the word the recogniser is least sure about — that is almost always where the speaker actually changes. CLI: `--merge-split-snap-window-ms 500`. | <pre>Alice: ...so yeah so you<br>       know<br>Bob:   the field pretty well<br>       yes<br>Alice: yes yes i am lucky…</pre> | <pre>Alice: ...so yeah<br>Bob:   so you know the field<br>       pretty well yes<br>Alice: yes yes i am lucky…</pre> |
| ⭐ | Count how often two speakers overlap inside one transcript line. Helps decide whether the fixes above are worth turning on by default. Numbers only — does not change the transcript. | <pre>(no rendered difference —<br>telemetry-only)</pre> | <pre>(no rendered difference —<br>telemetry-only)</pre> |

Legend: ⭐⭐⭐ visibly fixes a failure the user can point to in the
rendered transcript · ⭐⭐ fixes a subset of the same family, leaves
measurable residual · ⭐ no rendered difference, informs the next
decision.

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
- **Single straddling word at a clean turn change.** When a one-word
  filler (`"yes"`, `"okay"`) has a Whisper timestamp that overlaps
  both sides of a pyannote boundary by similar amounts and a high
  `probability` (no snap signal), the word goes to whichever side has
  the marginally larger overlap. Specific case: `[186.30-187.20] "yes"`
  with pyannote A-turn ending at 186.55 and B-turn starting
  at 187.02 ends up on A's side though acoustically it is B's
  back-channel confirmation. Not currently addressed.

## Updating this document

**Mandatory: any change to `src/voice/merge.py` must update this
document in the same PR.** That includes adding a new optimisation
step, changing a default, removing a flag, or noting a new failure
mode. The table at the top is the primary view contributors and
reviewers scan first; the prose below is where the why-and-how lives.

When adding a new optimisation:

1. Add a row to the impact table with the appropriate emoji.
2. Document the algorithm in "How the stage works" and the new
   telemetry fields under "Telemetry".
3. Add the new CLI flag(s) to the "CLI knobs" table.
4. If shipping the change flips an existing default or invalidates a
   prior decision, also write or update the corresponding ADR.
