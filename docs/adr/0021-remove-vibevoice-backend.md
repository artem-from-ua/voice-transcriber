---
status: accepted
date: 2026-05-12
---

# ADR 0021 — Remove the VibeVoice ASR backend

## Context

[ADR 0017](0017-whisper-asr-backend.md) (v0.20.0) added Whisper-large-v3-MLX as a second ASR backend behind `--asr-engine` and made it the default after an end-to-end comparison on a Ukrainian reference recording. VibeVoice-ASR stayed in the codebase as a legacy backend accessible via `--asr-engine vibevoice`, with `--asr-bits {4,5,6,8}`, `--asr-chunk-duration`, and `--asr-temperature` controlling its operating point.

Two versions later (v0.20.0 → v0.22.0), the dual-backend setup is paying maintenance cost without paying back:

1. **There is no operating point on the target hardware (16 GB Apple Silicon, M-series) where VibeVoice wins.** It is slower (3.4×), heavier on MLX memory (2.8×), and produces visibly worse Ukrainian text than Whisper on the reference recording. The two genuinely unique features VibeVoice has — in-band `[Silence]` / `[Music]` / `[Human Sounds]` markers and per-segment `speaker_asr` hints — are **not consumed** by the rest of the pipeline:
   - `speaker_asr` is a debug-only field. `merge.py` derives speaker labels exclusively from pyannote's max-overlap mapping. Stage 7+ never reads `speaker_asr`. The field only appears in `--dump-stages` artefacts.
   - In-band markers (`[Silence]` etc.) flow through `merge.py` with `speaker=None`, `proofread.py` skips them, `render.py` ignores them. LLM stages (`structure_dialog`, `tldr`) tolerate their absence on Whisper output without measurable quality loss (already noted in ADR 0017).
2. **The dual-backend surface has weight everywhere:** a separate backend module (`speech2text.py`, ~150 LOC), four CLI flags whose semantics only apply to VibeVoice, a direct dependency on `mlx-audio>=0.4.3` (which pulls in `miniaudio` and `sounddevice` we never use), an `if/elif` dispatcher in `pipeline.py`, a marker-pass-through branch in `merge.py`, marker-skip logic in `proofread.py`, and signature accommodation in `render.py`. Every ASR-adjacent change has to be reasoned about for two backends.

The motivating recurring question — "do we need to keep VibeVoice around as a fallback in case Whisper is bad on some recording?" — has the same answer it has had since v0.20.0: not on this hardware, not on this audio. `--asr-engine vibevoice` is not a one-flag fallback either: it requires a separate ~5–9 GB model download through LM Studio's GUI, into a different cache layout, before the flag does anything. It is a manual revert path, not a hot fallback.

## Comparison on the target hardware (16 GB Apple Silicon, M-series)

Numbers below are from ADR 0017's reference recording (`two-speakers-diar-test-ukr.m4a`, 6:13, two speakers, code-switching to English, swearing). The "Whisper wins on" column is what the row shows; columns are kept side-by-side so the trade-off is visible even when a future Whisper alternative is being evaluated.

| Aspect | Whisper-large-v3-MLX (default since v0.20.0) | VibeVoice-ASR-6bit (legacy) |
| --- | --- | --- |
| Stage 5 wall-clock | **55 s (6.8× realtime)** | 187 s (2.0× realtime) |
| Peak MLX memory during stage | **3.2 GB** | 9.0 GB |
| Peak system RAM during stage | **9.1 GB / 57 %** | 11.2 GB / 70 % |
| Model size on disk | ~3 GB (single weight) | 5–9 GB (per `--asr-bits` 4/5/6/8) |
| Model cache layout | `~/.cache/huggingface/hub/` (standard HF) | `~/.cache/lm-studio/models/` (LM Studio GUI flat layout) |
| Onboarding | `voice download-whisper` (one command) | LM Studio → Models → download `VibeVoice-ASR-{bits}bit` |
| Ukrainian fidelity (subjective) | **better** — preserves swearing and code-switched English verbatim | drifts to Russian phonetics on quiet turns; drops swearing; fuses code-switched words |
| Repetition-loop failure mode | not observed (`condition_on_previous_text=False` + library-internal temperature schedule) | observed on 4-bit; mitigated on 6/8-bit with `repetition_penalty=1.3` + `temperature=0.0` |
| Segments returned on reference | ~90 (finer-grained — better merge with pyannote turns) | ~39 (coarser — worse on fast turn-taking) |
| Runtime dependency | `mlx-whisper>=0.4.3` + `huggingface-hub>=1.14` | `mlx-audio>=0.4.3` (also pulls `miniaudio`, `sounddevice` we do not use) |

### Feature differences that matter — and whether the pipeline uses them

| Feature | VibeVoice | Whisper | Used by our pipeline? |
| --- | --- | --- | --- |
| In-band markers `[Silence]` / `[Music]` / `[Human Sounds]` | yes (as `Content` strings) | no | **No.** `merge.py` passes them through with `speaker=None`, `proofread.py` skips them, `render.py` ignores them. LLM stages do not depend on them (ADR 0017). |
| Per-segment `speaker_asr` hint (0/1/…) | yes (34 of 39 segments on reference) | always `None` | **No.** `merge.py` computes speaker via max-overlap with pyannote turns; `speaker_asr` only ever appeared in `--dump-stages` artefacts. |
| Tunable `chunk_duration` / `temperature` / `context` / `bitness` | yes, via four CLI flags | no — `mlx-whisper` bakes 30-s windows and runs its own temperature schedule internally | **Yes for VibeVoice — but no use case without it.** With VibeVoice gone, the flags lose meaning and are removed together. |
| Word-level timestamps | no | no (`word_timestamps=False`) | (Both give nothing here; not consumed anyway.) |
| In-ASR speaker diarization | partial (per-segment hint, unreliable on Ukrainian — see ADR 0004) | no | **No.** pyannote 3.1 is the sole source of speaker labels (ADR 0004). |
| Emotions / prosody / per-segment confidence | no | no | (Both give nothing.) |
| Trained single-pass context length | up to 60 min continuous | 30-s windows (chunked internally) | On recordings of minutes (not hours), Whisper's chunked path is more stable in practice than VibeVoice's long-context advantage was. The long-context win was theoretical and did not materialise on the target audio. |

**Conclusion.** On this hardware and this audio domain (Ukrainian + occasional English code-switching, runs of minutes to an hour, 16 GB Mac), Whisper dominates jointly on speed, memory, and quality. VibeVoice's unique features are unused. There is no operating point left to defend.

## Decision

Remove the VibeVoice ASR backend entirely. Whisper-large-v3-MLX becomes the only ASR backend.

**Removed surface:**

- CLI flags: `--asr-engine`, `--asr-bits`, `--asr-chunk-duration`, `--asr-temperature` (argparse errors on the next invocation that uses them — no aliases, pre-1.0 project).
- Direct dependency: `mlx-audio>=0.4.3` from `pyproject.toml`. `uv lock` also drops `miniaudio` and `sounddevice` as transitive deps that only `mlx-audio` pulled.
- Module: `src/voice/speech2text.py` deleted.
- `AsrSegment.speaker_asr` and `Segment.speaker_asr` fields (debug-only, no downstream consumer).
- Marker-handling branch in `merge.py` (`_is_marker`, `[...]` content pass-through).
- Marker-skip branch in `proofread.py` (`_is_marker`).
- `render_markdown(asr_label=…)` parameter (already non-rendering since v0.9.1; the signature now drops it).
- `pipeline.py` dispatcher (`if asr_engine == "vibevoice": …` / `elif "whisper": …`) becomes a direct call to `whisper_asr_module.transcribe(...)`.
- All tests under `tests/test_pipeline_ordering.py` that exercised the `asr_engine="vibevoice"` path.

**Preserved surface:**

- Whisper backend code (`src/voice/whisper_asr.py`) is unchanged except for moving `AsrError` from `speech2text.py` (now gone) into the file itself.
- `voice download-whisper` subcommand and the HF-cache verification (`_verify_model_cached`) keep their behaviour.
- ADR 0017 stays `accepted` as the historical record of *how* Whisper entered the project; this ADR records *why* VibeVoice was removed. The two are complementary, not superseding.

## Consequences

- **Breaking change for users who explicitly ran `--asr-engine vibevoice`.** They have two migration paths, both documented in CHANGELOG 0.23.0: stay on v0.22.0, or run `voice download-whisper` once and use the default. There is no scriptable bridge; this is intentional pre-1.0 housekeeping.
- **In-band markers are gone for good in this version.** If a future recording shows that absence of pause cues materially degrades structure or TL;DR output, regenerate them in `merge.py` from inter-segment gaps — see follow-up issue [#88](https://github.com/artem-from-ua/voice-transcriber/issues/88). That work is gated on observing the problem in real output, not on speculative completeness.
- **`speaker_asr` is gone from `AsrSegment` and `Segment`.** Dump artefacts (`03-asr.json`, `04-merge.json`, `07-segments-named.json`) lose the field. Downstream consumers never read it; no fix-up needed elsewhere.
- **LM Studio is no longer mentioned in the ASR onboarding path.** The README and `docs/models.md` no longer suggest it for ASR. It remains a perfectly fine way to manage *LLM* checkpoints (see ADR 0001 / 0006 for the LLM-side history), so the project doesn't actively oppose its use — it just doesn't require it.
- **Diagram update.** `docs/architecture.md` stage [5] now lists only Whisper. The PlantUML pre-commit hook regenerates the embedded SVG URL automatically; do not hand-edit the URL.
- **Documentation cleanup.** `docs/models.md` ASR section is rewritten around Whisper. The four ASR CLI flags vanish from `docs/cli.md`. ADRs 0004, 0006, 0007, 0011, 0013, 0014, 0015 keep their VibeVoice references intact — they are immutable historical snapshots.

## Alternatives considered

- **Keep VibeVoice behind `--asr-engine vibevoice` (status quo).** Rejected. Dual-backend support has weight in seven files, and nobody benefits — there is no operating point where it wins on this hardware. The "what if Whisper is bad on some recording" argument is theoretical; the manual revert path (downgrade to v0.22.0) covers it without keeping live code paths.
- **Remove the CLI flag but keep `speech2text.py` as a dormant module.** Rejected. Dead code accumulates rot faster than removed code, and the `mlx-audio` dependency would stay regardless. If we ever want VibeVoice back, `git log` is a perfectly fine source.
- **Remove VibeVoice and at the same time regenerate pause markers from `merge.py`.** Rejected for this PR. That is a separate feature whose value should be proven by an observed structure / TL;DR regression on real audio, not assumed. Tracked separately as issue [#88](https://github.com/artem-from-ua/voice-transcriber/issues/88).
- **Mark VibeVoice deprecated for one version, remove the version after.** Rejected. The project is pre-1.0; the deprecation dance is heavier than the actual fix, and the only people who would notice the warning are exactly the people for whom we already document the migration path in the CHANGELOG.
