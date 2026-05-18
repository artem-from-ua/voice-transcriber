---
status: accepted
date: 2026-05-17
---

# ADR 0031 — Chunk Whisper ASR in Python for long recordings

## Context

[Issue #147](https://github.com/artem-from-ua/voice-transcriber/issues/147) recorded a silent SIGKILL during stage `[6] speech2text` on a 48-minute Ukrainian recording: no Python traceback, no `.ips` crash report, no `jetsam` log line — the kernel terminated the process before any Python-level shutdown hook could run. On the same reference 16 GB M1 Mac the 6-minute recording passes the stage cleanly. Long-form ASR was unreachable.

A focused memory probe added in v0.32.x (pre/peak around the `mlx_whisper.transcribe` call) gave the first concrete numbers:

| input | pre-transcribe MLX | peak MLX | elapsed |
| --- | --- | --- | --- |
| 6 min  | 3.08 GB | **7.17 GB** | 50.7 s |
| 48 min | 3.08 GB | **(SIGKILLed)** | n/a |

The 7.17 GB peak for a 6-minute clip already exceeds half of unified memory. The 8× longer input never reached the post-transcribe probe.

Inspection of `mlx_whisper.transcribe` showed that despite Whisper's own 30-second decode windows, **the log-mel spectrogram is computed for the entire input audio up front** in one allocation (`log_mel_spectrogram(audio, padding=N_SAMPLES)` at `mlx_whisper/transcribe.py:150`). Beyond that one allocation, the per-call working set scales with audio length: cross-window state plus segment accumulators stay resident through the full decode loop. The original wrapper passed the whole WAV path in one call, so there was no opportunity to bound the per-call working set.

## Decision

`whisper_asr.transcribe` slices long audio in Python and calls `mlx_whisper.transcribe` once per slice:

```
duration <= ASR_CHUNK_THRESHOLD_S (600 s)
  -> single call, pass the WAV path as before (preserves v0.32.0 behaviour)

duration > threshold
  -> load audio via mlx_whisper.audio.load_audio
  -> slice into chunks of ASR_CHUNK_SIZE_S (480 s) with
     ASR_CHUNK_OVERLAP_S (5 s) overlap
  -> per chunk: clear MLX cache, call mlx_whisper.transcribe(np.ndarray)
  -> per-chunk segments: shift timestamps by chunk start, drop any whose
     start falls inside the overlap with the previously-accumulated tail
  -> stitch into a single list[AsrSegment]
```

Each chunk passes an `np.ndarray` slice rather than a path, so the global mel allocation happens at chunk granularity. `mx.clear_cache()` plus `mx.reset_peak_memory()` between chunks gives each call a fresh MLX cache and a real per-chunk peak in the log.

Knob defaults are empirical against the 6-minute baseline above:

- `ASR_CHUNK_THRESHOLD_S = 600.0` — below this the existing single-call path runs unchanged.
- `ASR_CHUNK_SIZE_S = 480.0` — 8 minutes per chunk leaves headroom under the 7.17 GB peak we saw on a 6-minute call.
- `ASR_CHUNK_OVERLAP_S = 5.0` — Whisper's own 30-second windows live *inside* a `transcribe()` call, so chunks are not aligned to them. 5 s overlap absorbs word-level drift across the boundary without retranscribing meaningful audio.

Overlap deduplication is structural, not text-based: segments from chunk N+1 whose `start` lies before `chunk_start + overlap_s` are dropped wholesale. The previous chunk's tail already covers that range.

> Dedup mechanism superseded by [ADR 0036](0036-text-similarity-asr-dedup.md) (text-similarity Jaccard, `material_rate` 0.50 → 0.167 on the 48-min reference). Chunk geometry, threshold and overlap from this ADR stay in force; only how `_dedup_overlap` decides what to drop changes.

## Consequences

**Wins:**
- Hour-long recordings can complete ASR on the 16 GB reference machine without SIGKILL.
- Per-chunk MLX peak is bounded by chunk size, not input duration — adding more audio no longer raises the cliff.
- The probe lines `whisper_asr: <label> mlx_peak=…GB` are kept after the fix lands so a future regression surfaces as a number, not a reboot.

**Trade-offs:**
- For long recordings each chunk re-pays the per-call setup cost (mel allocation, decode-loop init). Wall-clock on a 48-min recording is N × (single-call time of an 8-min recording) plus stitch overhead, which is more than the unreachable single-call time it replaces. Acceptable: the alternative was "does not run."
- Cross-chunk text continuity is bounded by the 5 s overlap. Whisper already runs with `condition_on_previous_text=False` (ADR 0017), so we are not regressing prompt-conditioning across chunks — there was none to lose.
- Quality risk at boundaries: a sentence that straddles the cut-point may be transcribed twice with slight differences, with one copy dropped. The dedup uses chunk geometry, not text-similarity, so it can drop a genuine post-overlap segment if Whisper happens to assign a timestamp earlier than expected. The conservative 5 s overlap keeps this rare; if it becomes a problem we can switch to text-similarity dedup in a follow-up.

## Alternatives considered

- **Smaller Whisper model (e.g. `whisper-medium-mlx`).** Would buy memory headroom for single-pass long recordings, at a quality cost. Rejected for the same reason ADR 0017 picked `whisper-large-v3-mlx`: Ukrainian ASR quality lives at the top of the model size range and dropping a tier is visible to the user.
- **Stream the audio through mlx_whisper without loading it all.** mlx_whisper's API does not currently support streaming; the audio either comes as a path it loads itself, or as a fully-materialised `np.ndarray`/`mx.array`. Working around this would mean upstreaming a change, not a self-contained fix.
- **Aggressive `mx.clear_cache()` inside the existing single call.** Cannot reach into the per-window decode loop without forking mlx_whisper. Chunking at our layer achieves the same goal with code we control.
