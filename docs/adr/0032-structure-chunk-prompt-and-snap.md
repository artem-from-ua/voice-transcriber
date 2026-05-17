---
status: accepted
date: 2026-05-17
supersedes: []
---

# ADR 0032 — Dedicated chunked-mode prompt + snap-to-speaker-boundary chunking for `speech_structure`

## Context

[Issue #151](https://github.com/artem-from-ua/voice-transcriber/issues/151) recorded a 76% rejection rate (13 of 17 chunks invalid) for `speech_structure` chunked-mode on the 48-min reference recording. Reconcile papered over the gap and produced 7 final sections, but the result included one ~25-minute mega-section ("Discussing AI skills and career aspirations") that was almost certainly an artefact of dropping chunks 8–11 in a row.

A diagnostic probe (`/tmp/structure_probe.py`, kept outside the source tree) re-ran the stage offline against the saved `07-segments-named.json` with the validator wrapped to log the raw LLM payload and the rejection reason. Every single rejection had the same cause: `section count 5 (or 6) outside [1, 4]`. No bad titles, no malformed JSON, no zero-length spans.

Two independent root causes surfaced:

### 1. Prompt mis-specification (acceptance rate)

`structure_system.md` (the only system prompt at the time) said:
- *"2 to 7 sections"* — the single-pass `MAX_SECTIONS` is 7, but `CHUNK_MAX_SECTIONS` is 4. The model faithfully followed the prompt and emitted 5+ sections on most chunks, where it was rejected by the chunked validator.
- *"The first section starts at 0 ms; the last ends at the dialogue end."* — geometrically wrong for non-leading chunks whose `total_start_ms` ≠ 0.

The same prompt was rendered for both paths because there was only one file.

### 2. Naive segment-index slicing (chunk quality)

`_chunk_segments(speech, chunk_size=35, overlap=2)` cuts strictly on segment index. If the cut lands inside one speaker's monologue, that monologue is split across two chunks. The downstream LLM sees two fragments of the same thought, neither complete, and is more likely to over-segment them.

## Decision

Both root causes fixed in one PR:

### Fix 1 — dedicated chunked prompts

New pair `structure_chunk_system.md` + `structure_chunk_user.md` mirroring the pattern established by [PR #148](https://github.com/artem-from-ua/voice-transcriber/pull/148) (`tldr_section` / `tldr_aggregate` / `tldr_final`). The chunked system prompt says:

- *"1 to 4 sections"* — matches `CHUNK_MAX_SECTIONS`.
- *"Use the exact `total_start_ms` and `total_end_ms` from the user message — do NOT use 0."* — fixes the geometry conflict.
- *"A fragment may legitimately contain a single topic — emit one section in that case rather than splitting artificially."* — leans into `CHUNK_MIN_SECTIONS=1`.

`_structure_chunk` and the `prompt_cache_session` prefix in `structure_dialog` both switched to `structure_chunk_system`. Single-pass path keeps `structure_system` unchanged.

### Fix 2 — snap-to-speaker-boundary in `_chunk_segments`

New parameters `snap_to_speaker_boundary: bool = True` and `max_snap_extension: int = 5`. After collecting `chunk_size` segments, the chunker continues forward while the next segment is the same speaker as the chunk's last segment, up to `max_snap_extension` extra segments. A natural turn change ends the extension; if none arrives within the cap, the cut happens anyway.

The cap prevents pathological single-monologue inputs from extending forever. 5 segments at ~10 s each is a half-minute leeway — covers most natural turn lengths on the reference recording.

### Why not bump `CHUNK_MAX_SECTIONS` to 5

A simpler one-line "fix" — push the validator to accept 5 sections — was rejected. It would push acceptance to ~100%, but:
- It does not address the prompt mis-specification ("starts at 0 ms" stays wrong).
- It does not address mid-monologue cuts.
- The right number for `CHUNK_MAX_SECTIONS` (and the other chunking knobs) is a benchmark-driven question, tracked in [#152](https://github.com/artem-from-ua/voice-transcriber/issues/152).

## Consequences

**Measured on the 48-min reference (`zendesk_spm_wfm_hm.wav`, 535 speech segments):**

| | v0.34.0 baseline | this ADR |
| --- | --- | --- |
| chunks emitted | 17 | **15** (snap consolidates mid-monologue splits) |
| chunks accepted | 4 | **9** |
| acceptance rate | **24%** | **60%** |
| 25-min mega-section | yes | **no** |
| final section count | 7 | 7 |

Final TL;DR section count is unchanged at 7 (capped by `MAX_SECTIONS`), but the **distribution** of those 7 sections is now driven by *9 accepted LLM views* of the dialogue instead of 4 — boundaries are picked from richer evidence. The "Discussing AI skills and career aspirations" 25-minute mega-section split into three more specific topics.

**Wins:**
- Acceptance rate **2.5×** the baseline.
- Wall-clock per stage drops ~12% from fewer chunks (15 vs 17 × ~15 s/chunk).
- The "log raw LLM payload + rejection reason" infrastructure from the probe is *not* committed but is documented in #152 as a `structure_probe.py` recipe — future regressions surface fast.

**Trade-offs:**
- 40% of chunks still reject. Every rejection is now exclusively `section count outside [1, 4]` — Qwen2.5-7B-4bit at `temperature=0.2` still over-segments dense topics regardless of "1 to 4" instruction. This is a *tuning* problem, not a *correctness* problem, and lives in #152. A "NEVER 5" capslock instruction was probed and rejected — it produced 6- and 7-section rejections instead of stable 5, no net gain.
- `snap_to_speaker_boundary=True` is the default. Tests on the existing `_long_segs` fixture broke because that fixture used a single speaker for all segments — `_long_segs` now alternates speakers so the snap stays out of its way. One new test covers each shape: natural boundary at `chunk_size`, monologue extending past `chunk_size` (snap extends), and monologue exceeding `max_snap_extension` (snap caps).

## Refs

- [Issue #151](https://github.com/artem-from-ua/voice-transcriber/issues/151) — root-cause analysis with the per-chunk evidence table
- [ADR 0018](0018-chunked-structure-dialog.md) — introduced the chunked path
- [Issue #152](https://github.com/artem-from-ua/voice-transcriber/issues/152) — benchmark-driven tuning of `CHUNK_MAX_SECTIONS` / `STRUCTURE_CHUNK_SIZE` / `STRUCTURE_CHUNK_OVERLAP`, parent of any future fix for the remaining 40% rejections
- `src/voice/prompts/structure_chunk_system.md` — the chunked-mode system prompt
- `src/voice/speech_structure.py:_chunk_segments` — snap-to-boundary chunker
