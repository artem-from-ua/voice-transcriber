---
status: superseded
superseded_by:
  - 0021-remove-vibevoice-backend.md
date: 2026-05-12
---

# ADR 0017 — Whisper-large-v3-MLX as a second ASR backend behind `--asr-engine`

> **2026-05-12 update:** superseded by [ADR 0021](0021-remove-vibevoice-backend.md), which dropped the VibeVoice backend and the `--asr-engine` flag entirely in v0.23.0. The decision below — *add* Whisper *alongside* VibeVoice as a routable second engine — is therefore no longer the current design. This ADR is preserved as the historical record of *how* Whisper entered the project and the empirical comparison that justified making it the default.

## Context

Issue #25 has tracked the addition of Whisper as a second ASR backend since the project's early days. The motivation is empirical: VibeVoice-ASR (the current default) is multilingual on paper but drifts to Russian phonetics on Ukrainian conversations regularly enough that we want a second engine available even before a formal benchmark exists.

Three constraints shaped the scope of this ADR:

1. **No Ukrainian reference transcript yet.** Issue #49 (benchmark harness) and the human-corrected reference recording it needs are still open. Without WER numbers, this ADR cannot pick a new default ASR engine; the default-engine decision is explicitly deferred.
2. **Method copied from stone-scriber** (a sibling local project under `~/devel/stone-scriber`) — same author, same hardware target, same Ukrainian audio domain. Specifically: `mlx_whisper.transcribe(audio, path_or_hf_repo=..., language=..., condition_on_previous_text=False)`. No custom chunking — mlx-whisper handles 30-second windows internally — and `condition_on_previous_text=False` to avoid the repetition-loop failure mode Whisper is known for on long mono inputs.
3. **Onboarding download is a separate explicit step.** Users opt in to Whisper by running `voice download-whisper` once. `voice transcribe --asr-engine whisper` hard-fails with an actionable error if the model isn't cached, so a 3 GB fetch never starts in the middle of a transcription run.

The project's policy is MLX-first ([ADR 0002](0002-mlx-format-preference.md)) and in-process ([ADR 0006](0006-mlx-lm-over-lm-studio.md)). `mlx-community/whisper-large-v3-mlx` is MLX-native; `mlx-whisper` is an in-process library, no HTTP, no daemon. Both ADRs are satisfied.

## Decision

Add a Whisper backend behind a new `--asr-engine {vibevoice,whisper}` flag, with **`whisper` as the default**.

The default was switched after an empirical end-to-end comparison on the project's reference Ukrainian recording (`two-speakers-diar-test-ukr.m4a`, 6:13 of two speakers, code-switching to English, swearing). On a 16 GB M-series Mac:

| Metric | Whisper-large-v3-MLX | VibeVoice-ASR-6bit |
| --- | --- | --- |
| Stage 5 wall-clock | 55 s (6.8× realtime) | 187 s (2.0× realtime) |
| Peak MLX memory during stage | 3.2 GB | 9.0 GB |
| Peak system RAM during stage | 9.1 GB (57 %) | 11.2 GB (70 %) |
| Segments returned | 90 (finer-grained) | 39 (coarser) |
| `[Silence]` / `[Music]` markers | none | yes |
| Speaker hint per segment | none (`speaker_asr=None`) | 34 of 39 |
| Subjective Ukrainian fidelity | clearly better on this sample | weaker (drops mat words, fuses code-switched words) |

The decision is reversible at any time via `--asr-engine vibevoice`; this is not a binding "best engine" statement, just the better default for the typical user on the typical Mac. A formal WER benchmark against a human-corrected reference (issue #49) is still wanted to make this a defended-with-numbers decision rather than a defended-with-anecdotes one.

**Model:** `mlx-community/whisper-large-v3-mlx` (vanilla OpenAI Whisper, large-v3 architecture, MLX-converted weights). Other Whisper variants (Ukrainian fine-tunes — issue #52; large-v3-turbo — issue #54) are deferred to follow-up issues, to be evaluated once a reference recording is available.

**Backend module** (`src/voice/whisper_asr.py`):

```python
def transcribe(wav_path, *, language="uk", log=_noop_log) -> list[AsrSegment]:
    _verify_model_cached()          # AsrError if missing
    import mlx_whisper
    result = mlx_whisper.transcribe(
        str(wav_path),
        path_or_hf_repo=WHISPER_REPO_ID,
        language=language,
        condition_on_previous_text=False,
    )
    segments = _parse_segments(result)  # AsrSegment with speaker_asr=None
    del result
    free_mlx(log)                    # mirror of speech2text.py:149–152
    return segments
```

The trailing `free_mlx(log)` call is load-bearing on 16 GB Macs: without it, Whisper's ~3 GB MLX cache stays resident while gemma-3-12b loads for the LLM stages, and `structure_dialog` then hits a Metal OOM. The pattern is borrowed verbatim from `speech2text.py`, where the same problem was solved earlier for VibeVoice (see ADR 0006 / 0007 era discussions).

**Dispatcher** (`src/voice/pipeline.py`): `if options.asr_engine == "vibevoice": …` / `elif "whisper": …`. The same `processed_wav_path` (output of the clearspeech chain) feeds both backends — no per-engine preprocessing branching.

**Download** (`voice download-whisper` subcommand → `src/voice/download_whisper.py`): a thin wrapper over `huggingface_hub.snapshot_download(repo_id)`. The model lands in the standard HF cache (`~/.cache/huggingface/hub/models--mlx-community--whisper-large-v3-mlx/`), not the LM Studio cache that VibeVoice uses, because `mlx-whisper` expects the standard HF layout.

**Cache miss** (`whisper_asr._verify_model_cached`): `huggingface_hub.try_to_load_from_cache(repo_id, "config.json")`. Returns `None` ⇒ raise `AsrError` with the message "Whisper model ... is not in the HuggingFace cache. Run `voice download-whisper` once to fetch it (~3 GB)." No silent fetches inside `transcribe`.

**Speaker labels:** Whisper produces no speaker hint; merge already drives speaker assignment from pyannote turns via max-overlap (`src/voice/merge.py`), so `AsrSegment(speaker_asr=None)` flows through correctly. Stage 4 (clearspeech autogain) still uses pyannote turns as guard-rails — independent of which ASR runs next.

## Consequences

**Two new direct dependencies.** `mlx-whisper>=0.4.3` (the runtime) and `huggingface-hub>=1.14` (promoted from a transitive pyannote-audio dep to a direct one because we now call `snapshot_download` and `try_to_load_from_cache` ourselves). Both are wheels on PyPI; `mlx-whisper` brings in `numba` + `tiktoken` + `llvmlite` (audio decoding + tokeniser + JIT). Cold import of `whisper_asr` is fast because `mlx_whisper` is imported lazily inside `transcribe()`.

**Default-engine change is user-visible.** Anyone who relied on the v0.19.0 default behaviour will see a new ASR pipeline by default. The first run after upgrading needs `voice download-whisper` (~3 GB into the HF cache) or an explicit `--asr-engine vibevoice` to keep the old behaviour. The CHANGELOG entry calls this out under **Changed** and gives both migration paths.

**`speaker_asr` is now `null` by default.** Downstream code (`merge.py`) never used the field for its routing — speakers come from pyannote — so this is a no-op for transcript content. Dump artefacts (`03-asr.json`, `04-merge.json`, `07-segments-named.json`) will show `"speaker_asr": null` for every segment when Whisper is active.

**`[Silence]` / `[Music]` / `[Human Sounds]` markers disappear.** VibeVoice emits these in-band as `Content` strings starting with `[`; Whisper does not. The structure / TLDR LLM stages have to infer pacing from segment timestamps alone. In practice the LLM tolerates this; if it ever stops doing so, the fix is to re-emit pause markers from `merge.py` based on inter-segment gaps.

**Render label.** `render_markdown` now receives `asr_label=engine_label` (computed in `pipeline.run()`), so the final Markdown's metadata line reports the engine that produced it: `VibeVoice-ASR-6bit` or `Whisper-large-v3-MLX`.

**OOM during `structure_dialog` is a separate, pre-existing issue.** The end-to-end runs that informed this ADR uncovered a Metal OOM on 16 GB Macs when gemma-3-12b proceeds from `identify` into `structure_dialog`. It reproduces with both engines (Whisper just happens to leave less headroom). The fix is out of scope for this ADR — tracked as a follow-up. The PR ships the `free_mlx` cleanup so Whisper itself is not the cause; everything else (chunked structure prompts, smaller LLM for structure, pre-stage `mx.clear_cache()`) is a separate investigation.

**Diagram.** `docs/architecture.md` stage [5] now lists both backends explicitly (`<i><mlx-audio> VibeVoice-ASR</i>` and `<i><mlx-whisper> Whisper-large-v3</i>`); the prose under "Data flow" item 5 explains the dispatcher and links here.

## Alternatives considered

- **Auto-download on first cache miss inside `transcribe()`.** Rejected: a 3 GB network fetch in the middle of `voice transcribe` would be a startling failure mode if the user runs without `--verbose`. Explicit `voice download-whisper` separates onboarding from runtime.
- **LM Studio as the download front-end.** Rejected: LM Studio is primarily an LLM tool; Whisper is not in its first-class catalogue. Its cache layout (`~/.cache/lm-studio/models/<owner>/<repo>/`, flat) also doesn't match what `mlx-whisper` reads (`~/.cache/huggingface/hub/models--<owner>--<repo>/snapshots/<hash>/`, content-addressed). Using HF directly avoids a translation layer.
- **Ship a Ukrainian fine-tune as the first Whisper variant.** Rejected for the first PR scope. The Ukrainian fine-tunes on HuggingFace (Yehor's, ArtificialThinker's, …) are unevaluated by us and would compound two unknowns — "is Whisper better than VibeVoice on uk" plus "is fine-tune-X better than vanilla". Shipping vanilla first lets the benchmark (when it lands) compare apples-to-apples with the published Whisper claims; issues #52 and #54 track fine-tune and turbo variants as follow-ups against the vanilla baseline.
- **Re-purpose `--asr-bits` to accept `whisper`.** Rejected: `bits` semantically means quantisation level for one engine, not engine identity. Conflating the two would make the help text dishonest and the default migration awkward.
- **Single `voice` argparser handling everything with a positional command (no subparsers).** Already rejected by the existing structure — `voice transcribe …` is the only subcommand today, and adding `voice download-whisper …` is the natural extension.
