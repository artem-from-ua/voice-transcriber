---
status: accepted
date: 2026-05-18
supersedes: []
---

# ADR 0036 — Text-similarity dedup for chunked ASR

## Context

ADR 0031 introduced chunked ASR with a structural `_dedup_overlap`:
chunk N+1 segments whose `start` lay inside the 5 s overlap window
with chunk N were dropped wholesale, no text inspection.
[PR #169 / issue #156](https://github.com/artem-from-ua/voice-transcriber/issues/156)
measured the result on the 48-min reference recording:
`material_rate = 0.50` (3 of 6 chunk boundaries lose words).

[Issue #159](https://github.com/artem-from-ua/voice-transcriber/issues/159)
attempted to fix this by snapping cuts to silence. Five variants
(V1 snap-start through V5 snap-end + cutoff-at-prev-end) all produced
the same `material_rate = 0.50`. The post-mortem ([ADR 0035](0035-asr-snap-to-silence.md),
rejected) identified two root causes:

1. **Geometric `_dedup_overlap` cannot distinguish a genuine duplicate
   from a legitimate continuation** when Whisper assigns ambiguous
   timestamps at the join. V4b under-trimmed (3× phrase repeat at B4).
   V5 over-trimmed (legitimate `"And again, our forecast product does
   not survive..."` dropped along with the duplicates).
2. Whisper's per-chunk segmentation drops short clauses near boundaries
   independent of cut placement (tracked by [issue #171](https://github.com/artem-from-ua/voice-transcriber/issues/171)).

Cause 1 is independent of snap and equally affects the strict-time
path that is currently in production: PR #156's `material_rate = 0.50`
breakdown was 3 missing, 0 duplicated — but the "missing" verdicts
included cases where structural dedup silently dropped both a duplicate
*and* the legitimate continuation that happened to share its overlap
window.

## Decision

Replace structural `_dedup_overlap` with a three-case text-similarity
reconciler that returns the **full updated** `accumulated` list
(original minus any superseded tail segments, plus kept `incoming`).

For each `incoming` segment with at least `min_token_count` tokens:

1. **Plain duplicate** — pairwise max Jaccard against any tail segment
   `>= jaccard_threshold` → drop incoming, keep tail.
2. **Superseding rewind** — pairwise max stays below the threshold,
   but the incoming segment's audio span overlaps one or more tail
   segments AND the Jaccard against the *union* of those overlapping
   tail tokens `>= supersede_aggregate_threshold` → drop the
   superseded tail segments, keep the incoming. This case catches
   Whisper re-emitting a longer, context-richer version of several
   short tail fragments — pairwise stays low because each fragment is
   short, but the aggregate signal exposes the rewind. Without it the
   rendered transcript shows both the short tail fragments and the
   long head segment back-to-back (visible duplication on the 48-min
   reference at B3/B4 after the first PR landed; see post-PR review).
3. **Legitimate continuation** — low pairwise, no audio overlap →
   keep incoming, stop scanning further `incoming` (boundary repeats
   live at the head, not the middle).

```
accumulated, incoming
  -> tail = [s for s in accumulated
             if s.end >= accumulated[-1].end - overlap_window_s]
  -> out_accumulated = list(accumulated)
  -> for each `seg in incoming`:
       seg_tokens = tokenize(seg.content)
       if len(seg_tokens) < min_token_count:
           keep seg, continue
       max_pairwise = max(jaccard(seg_tokens, t) for t in tail_tokens)
       if max_pairwise >= jaccard_threshold:
           drop seg, continue          # CASE 1
       overlap_idx = [i for i in tail_idx
                      if seg.start < tail[i].end
                      and seg.end > tail[i].start]
       if overlap_idx and jaccard(
              seg_tokens, union(tail_tokens[i] for i in overlap_idx)
          ) >= supersede_aggregate_threshold:
           drop tail[i] for i in overlap_idx   # CASE 2 (SUPERSEDE)
           keep seg, stop scanning
       else:
           keep seg, stop scanning      # CASE 3
  -> return out_accumulated + kept_incoming
```

The function returns the updated accumulated list directly; the
call-site in `transcribe()` rebinds `segments = _dedup_overlap(...)`.
This is a behavioural change from the original PR (which returned
only the kept-incoming list and never mutated accumulated); see the
post-PR walk-through under "Consequences" below.

Defaults (`src/voice/whisper_asr.py`):

- `ASR_DEDUP_OVERLAP_WINDOW_S = 30.0` — wider than the chunk overlap
  (5 s) on purpose. Whisper assigns segment timestamps inside its own
  30 s decode windows; the duplicate emitted in chunk N+1's head can
  land several seconds past the strict overlap cutoff. 30 s safely
  covers that drift without admitting unrelated earlier-chunk content.
- `ASR_DEDUP_JACCARD_THRESHOLD = 0.5` — empirically dropped the V4b B4
  three-fold repeat (Jaccard ≈ 0.62) while keeping legitimate
  continuation `"does not survive without our scheduling"` (Jaccard
  ≈ 0.38). `>=` semantics (a segment at exactly 0.5 drops).
- `ASR_DEDUP_MIN_TOKEN_COUNT = 3` — a 2-token backchannel like
  `"yeah okay"` near a turn change would otherwise self-match similar
  backchannels in the trailing context and disappear.
- `ASR_DEDUP_SUPERSEDE_AGG_THRESHOLD = 0.3` — lower than the pairwise
  threshold because the union over multiple tail fragments inflates
  the denominator. Calibrated from the 48-min reference's B3/B4
  rewinds (aggregate ≈ 0.45-0.55 against the merged tail tokens);
  setting it too high (e.g. 0.5) would miss the rewinds, setting it
  too low (e.g. 0.1) would supersede on coincidental vocabulary
  overlap from an unrelated topic that happens to be in the tail
  window. The audio-span overlap requirement guards against the
  latter even at low aggregate thresholds.

`ASR_CHUNK_OVERLAP_S = 5.0` is unchanged — it still drives the audio
slice geometry so Whisper has enough context on both sides of every
cut to produce the duplicate that text-dedup then removes. The dedup
*window* (`ASR_DEDUP_OVERLAP_WINDOW_S = 30 s`) is a separate knob
because the text-dedup logic doesn't need to match the audio overlap.

Helpers (`_tokenize`, `_jaccard`, `_TOKEN_RE`) are mirrored from
`scripts/asr-chunk-boundary-quality.py` rather than imported — keeps
the production module self-contained and avoids the script's heavy
deps. A code comment in both files marks them as a pair to keep in
sync.

## Consequences

**Wins (measured on 48-min reference, [`docs/benchmarks/asr-text-dedup.md`](../benchmarks/asr-text-dedup.md)):**

- `material_rate` dropped from 0.50 to **0.167** (1 of 6 boundaries
  missing) — a 3× improvement. The fixed boundaries are exactly the
  two cases where structural dedup was silently over-trimming:
  - B3 (cutoff 1905s): strict missing → text-dedup clean. The
    `"catch myself on this sometimes"` clause that strict baseline
    lost is now present in chunk N's tail.
  - B4 (cutoff 2380s): strict missing → text-dedup clean. The
    `"does not survive without our scheduling"` clause that strict
    fused into `"Forecast product scheduling product"` is now intact.
- B0 (already clean) is informationally richer — extra connecting
  segment `"So you mentioned you co-founded a company. Is"` survives.
- Zero ASR wall-clock cost: dedup overhead is microseconds (Jaccard on
  ~30 s of tokens). Full ASR pass on 48-min reference unchanged at
  ~247 s.

**Remaining limitation:**

- B2 (cutoff 1430s) stays missing. This is the "no usable silence in
  the snap window" / "Whisper-segmentation drop" case tracked by
  issue #171, orthogonal to dedup choice — no dedup mechanism can
  recover words Whisper never transcribed.

**Trade-offs:**

- Public signature of `_dedup_overlap` changed (no more `chunk_start_s`
  and `overlap_s` kwargs). This is internal to `whisper_asr.py` —
  `grep` confirms no external callers. The call-site simplified from
  `_dedup_overlap(segments, chunk_segments, chunk_start_s=..., overlap_s=...)`
  to `_dedup_overlap(segments, chunk_segments)`.
- A pathological recording with many legitimate short repeated phrases
  near chunk boundaries (e.g. a song with a chorus that happens to
  span the cut) could see false-positive drops. The `min_token_count`
  floor guards against the most common backchannel case
  (`"yeah okay"`); longer legitimate repeats remain a theoretical
  edge case for which no real example has surfaced in the corpus.

## Alternatives considered

- **Snap to silence (ADR 0035, rejected).** Tried 5 variants, all
  produced the same `material_rate = 0.50`. The root cause was in
  dedup logic, not cut placement — fixing dedup directly was the
  right layer.
- **Bump structural overlap from 5 s to 15-30 s.** Whisper would still
  emit duplicates near the wider boundary; structural dedup would
  still face the same "duplicate vs continuation" ambiguity, just
  shifted. Doesn't address cause 1.
- **Bigram / trigram similarity instead of unigram Jaccard.** More
  robust against rare-token collisions but adds tokenisation
  complexity and a noticeable per-call cost. Worth revisiting if
  unigram Jaccard ever produces a wrong drop on real audio; today's
  measurement gives zero false positives on the reference.
- **Embedding similarity** (e.g. via sentence-transformers). Would
  catch paraphrases that unigram Jaccard misses, but adds a model
  dependency and ~1 s per call. Massive overkill for boundary
  repetition where Whisper rewinds and produces near-identical text.

## Refs

- [Issue #172](https://github.com/artem-from-ua/voice-transcriber/issues/172)
  — proposal this ADR implements.
- [Issue #156](https://github.com/artem-from-ua/voice-transcriber/issues/156)
  / [PR #169](https://github.com/artem-from-ua/voice-transcriber/pull/169)
  — baseline measurement methodology, `material_rate = 0.50` baseline.
- [Issue #159](https://github.com/artem-from-ua/voice-transcriber/issues/159)
  — snap-to-silence attempt, rejected. Identified structural dedup as
  the root cause.
- [Issue #171](https://github.com/artem-from-ua/voice-transcriber/issues/171)
  — B2-class boundaries that no dedup mechanism can fix.
- [ADR 0031](0031-chunked-asr.md) — original chunked-ASR design (still
  in force; this ADR refines only the dedup mechanism).
- [ADR 0035 (rejected)](https://github.com/artem-from-ua/voice-transcriber/blob/feature/asr-snap-to-silence/docs/adr/0035-asr-snap-to-silence.md)
  — snap-to-silence post-mortem that motivated this ADR.
- [`docs/benchmarks/asr-text-dedup.md`](../benchmarks/asr-text-dedup.md)
  — full benchmark numbers backing this decision.
- `docs/measurements/issue-172/` — per-boundary judge marks.
