---
status: accepted
date: 2026-05-11
---

# ADR 0004 — Use pyannote 3.1 for diarization instead of VibeVoice's built-in tags

## Context

VibeVoice-ASR returns each segment with a `Speaker` field (0/1/…). It would be tempting to skip a separate diarizer and use VibeVoice's labels directly. In practice those labels are unreliable:

- On Ukrainian recordings VibeVoice would assign one speaker to long stretches of a two-speaker conversation, especially after ~3 minutes.
- It cannot mark overlaps; it just picks one.
- The labels are not stable across re-runs.

pyannote 3.1, by contrast, is the de facto open-source diarizer and produces a clean timeline of `SPEAKER_XX` turns with timestamps. It runs in ~30 s on a 6-minute recording on M-series hardware.

## Decision

Run pyannote 3.1 in parallel with VibeVoice and use its `exclusive_diarization` (no-overlap) timeline as the source of truth for speaker labels. VibeVoice's own `Speaker` tag is kept on `AsrSegment.speaker_asr` for sanity checks but never reaches the rendered output.

## Consequences

- Two model loads instead of one; ~30 s extra wall-clock per run.
- Requires a Hugging Face account with three pyannote licenses accepted (see `troubleshooting.md`).
- Eliminates the long-tail mislabelling we saw with VibeVoice-only labelling.
- Merging is trivial (`merge.py`, max-overlap heuristic) because the diarization is exclusive.

## Alternatives considered

- **VibeVoice labels only**: rejected — too unreliable in our test recording.
- **WhisperX**: relies on pyannote anyway (same gated repos), and the embedded Whisper would replace VibeVoice with a generally weaker Ukrainian ASR.
- **NeMo diarization**: heavier setup, weaker on small-speaker scenarios in our spot-checks.
