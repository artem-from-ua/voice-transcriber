# Changelog

All notable changes to this project will be documented in this file. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.26.0] — 2026-05-12

### Changed

- **Redesigned Markdown header — per-model AI table, lang_detect confidence, speaker source tags** ([#93](https://github.com/artem-from-ua/voice-transcriber/issues/93)).

  The flat chip strip is replaced by a structured header:
  - `🌐 Мова:` now appends `(user-specified)` when `--language` was given, or `(auto-detected, p=0.74, ru=0.24)` when `[4] lang_detect` ran. Runner-up shown only when ≥ 0.05.
  - `👥 Учасники:` is now a bulleted list. Each speaker shows its source tag: `(self-introduced)` when extracted by `[9] identify_speakers`'s LLM call, `(user-specified)` when the name came from `--names`, `(interactive)` when typed at the interactive prompt, or `(unidentified)` when none of the above.
  - `🤖 AI моделі:` is a Markdown table grouping stages by the model that ran them. The **first row for each model** is `model loaded in Xm Ys` — the cold-start load time. Subsequent rows list each stage with its elapsed time. Stages that did not run (disabled via `--no-*` or `--names`) are omitted from the table entirely.
  - `⏲️ Обробка:` now appends `(X% of duration)` — a quick speed indicator (`<100%` = faster than real-time).
  - Old `⏱️ AI-стадії: diarize=Xs · asr=Ys …` chip removed.

- **`render_markdown()` signature changed** (`render.py`). Old kwargs `models` and `timings` replaced by:
  - `stage_models: dict[str, str] | None` — stage name → model repo/label
  - `stage_timings: dict[str, float] | None` — stage name → elapsed seconds (includes `"total"` key)
  - `model_load_elapsed: dict[str, float] | None` — model repo/label → load seconds
  - `name_sources: dict[str, str] | None` — pyannote cluster label → source tag
  - `lang_detect_info: dict | None` — `{"top": (lang, p), "second": (lang, p)}` or `None`

- **`lang_detect.detect_language_on_longest_turn()` return type changed** from `str` to `LangDetectResult(language: str, probabilities: dict[str, float] | None)`. The `probabilities` field is `None` on all fallback paths (no turns, too-short turn, empty probs from Whisper). Callers must access `.language` for the ISO code.

- **`identify_speakers.identify_speakers()` return type changed** from `dict[str, str]` to `dict[str, NamedAssignment(name, source)]`. The `source` field is `"user-specified"`, `"self-introduced"`, `"interactive"`, or `"unidentified"`. The pipeline builds `name_sources` from this dict and passes it to render.

- **`diarize_speakers.diarize()` return type changed** from `list[DiarTurn]` to `tuple[list[DiarTurn], float]` — the second element is the pyannote model load elapsed seconds.

- **`whisper_asr.load_model()`** new public function. Returns `(model, load_elapsed_s)`. The pipeline calls it explicitly before `transcribe()` to measure Whisper cold-start separately from inference. `transcribe()` accepts a new `model` kwarg — when provided, weights are reused instead of reloaded.

- **`MlxLLM._last_load_s`** new attribute on `llm.MlxLLM`. Set in `load()` to the wall-clock seconds of the last `mlx_lm.load()` call. `pipeline.run()` accumulates these into `model_load_elapsed` per model spec.

### Internal

- `pipeline._llm_summary()` removed (replaced by `stage_models` dict passed to render).
- `pipeline._ensure_llm()` gains `model_load_elapsed: dict | None` kwarg; accumulates load time into it on each fresh load.

## [0.25.0] — 2026-05-12

### Changed

- **Module renames for snake_case consistency** ([#90](https://github.com/artem-from-ua/voice-transcriber/issues/90)). Six modules renamed:
  - `audiometa` → `audio_meta`
  - `clearspeech` → `clear_speech`
  - `diarize` → `diarize_speakers`
  - `identify` → `identify_speakers`
  - `structure` → `speech_structure`
  - `tldr` → `speech_summary`
  
  Corresponding test files, dump artifacts (`02-diarize_speakers.json`, `06-identify_speakers.json`, `08-speech_structure.json`, `09-speech_summary.txt`, `02b-clear_speech-*`), `breakdown_order` keys in `render.py`, progress-spinner labels, and all documentation updated in the same commit.
  
  **Breaking for direct importers of `voice` internals** (e.g. `from voice.identify import identify_speakers` → `from voice.identify_speakers import identify_speakers`). CLI flags (`--clearspeech-*`, `--no-tldr`, `--no-structure`, `--llm-{identify,structure,tldr}-model`) are unchanged. No known external consumers.

### Internal

- ADRs 0008–0022 that mentioned the old module names have a one-line postscript `> Note (v0.25.0)` pointing at the new names; bodies preserved as immutable historical record.

## [0.24.0] — 2026-05-12

### Added

- **New stage `[4] lang_detect` between diarize and clearspeech.** When `--language` is omitted, the pipeline picks the longest pyannote turn and calls Whisper-large-v3-MLX's `model.detect_language()` on its mel-spectrogram. The detected ISO language code becomes `effective_language` and flows through ASR, proofread, identify, structure, TL;DR and render. When `--language uk` (or any ISO code) is given explicitly, this stage is skipped entirely — no extra model load. See [ADR 0022](docs/adr/0022-asr-language-autodetect.md) for the design and the empirical reason (the first-30-second internal Whisper detect was unreliable on quiet or short opening segments and classified the project's Ukrainian reference recording as Russian, ruining the entire pipeline run).
- New module `voice.lang_detect` with public `detect_language_on_longest_turn(wav_path, turns) -> str`. Built on top of the same `mlx-whisper` package and Whisper-large-v3-MLX weights that the ASR stage uses — no new dependencies.

### Changed

- **`--language` default is now auto-detect via `[4] lang_detect`** (was hard-coded `"uk"`). For Ukrainian recordings the user-visible behaviour is unchanged in practice — Whisper-large-v3 reliably reports `uk` from a typical longest pyannote turn (~0.74 confidence on the reference recording, 3× margin over `ru`). For non-Ukrainian recordings the detected language now flows correctly through every stage instead of being silently overridden by `"uk"`. Explicit `--language uk` (or any other ISO code) still works as before and skips detection.
- **Pipeline stage numbering shifted by one** starting at clearspeech: clearspeech `[4]` → `[5]`, speech2text `[5]` → `[6]`, merge `[6]` → `[7]`, proofread `[7]` → `[8]`, identify `[8]` → `[9]`, structure `[9]` → `[10]`, tldr `[10]` → `[11]`, render `[11]` → `[12]`. Progress-label denominator changes from `[N/10]` to `[N/11]`. Markdown `AI-стадії:` header chips now include a new `langid=Xs` entry when the stage ran.
- `whisper_asr.transcribe()` signature: `language: str | None = None` (was `language: str = "uk"`). `pipeline.run()` always passes a concrete string from `effective_language`, so the behaviour is invisible from there; direct callers that omitted the kwarg now get mlx-whisper's auto-detect instead of a forced `"uk"` hint.
- ADR 0017 frontmatter updated to `status: superseded` (superseded by ADR 0021's full VibeVoice removal, which dropped the `--asr-engine` flag the original decision depended on). The body is preserved as the historical record. ADR index (`docs/adr/README.md`) shows the entry with strikethrough on both the number and title.

### Internal

- `PipelineOptions.language` type changed from `str` to `str | None`.
- `pipeline.run()` introduces a local `effective_language` variable computed between diarize and clearspeech, then forwards it to every downstream stage that previously read `options.language` directly (proofread, identify, structure ×2, TL;DR, render).
- New stage label in verbose log: `[4/11] Визначення мови` plus `lang_detect: top1 <lang> (p=…, ru p=…)` when running.

### Failure mode and recovery

- If pyannote returns no turns at all, or the longest turn is shorter than `MIN_TURN_DURATION_S = 2.0`, `lang_detect` returns the fallback `"uk"` and logs a hint to pass `--language` explicitly. The fallback prevents crashes on degenerate inputs at the cost of being wrong by default on very short non-Ukrainian recordings — those are exactly the cases where an explicit `--language` is required anyway.
- If `model.detect_language()` returns an empty probability list (defensive, never observed in practice), same fallback.

## [0.23.0] — 2026-05-12

### Removed (BREAKING)

- **VibeVoice ASR backend deleted in full.** With Whisper-large-v3-MLX winning empirically on every metric on the 16 GB Apple Silicon target (3.4× faster, 2.8× less MLX memory, better Ukrainian fidelity including swearing and code-switched English — see [ADR 0017](docs/adr/0017-whisper-asr-backend.md)) and VibeVoice's two unique features (in-band `[Silence]`/`[Music]`/`[Human Sounds]` markers, per-segment `speaker_asr` hints) going unused by downstream stages, there is no operating point left to defend. Full reasoning in [ADR 0021](docs/adr/0021-remove-vibevoice-backend.md).
  - Removed CLI flags: `--asr-engine`, `--asr-bits`, `--asr-chunk-duration`, `--asr-temperature`. argparse now rejects them.
  - Removed dependency: `mlx-audio>=0.4.3`. `uv lock` also drops `miniaudio` and `sounddevice` (transitive deps that only `mlx-audio` pulled).
  - Removed module: `src/voice/speech2text.py`.
  - Removed fields: `AsrSegment.speaker_asr` and `Segment.speaker_asr` (debug-only, no downstream consumer). `--dump-stages` artefacts no longer carry the field.
  - Removed marker pass-through: `merge.py` `_is_marker` branch, `proofread.py` `_is_marker` skip, `render.py` marker prose in docstrings. Stage 6+ now treat every Whisper segment uniformly.
  - **Migration:** users on the default since v0.20.0 (Whisper) — nothing to do. Users who passed `--asr-engine vibevoice` explicitly: downgrade to v0.22.0 to keep VibeVoice, or run `voice download-whisper` and use the default. There is no scriptable bridge.
- **`render_markdown(asr_label=…)` parameter removed** from the public signature. The header has not rendered it since v0.9.1; the parameter survived as a no-op for backward compatibility. Now gone.

### Changed

- `docs/architecture.md` stage [5] PlantUML node simplified to one backend (`Whisper-large-v3`). A new optional dashed blue edge `User → [5] speech2text` carries the `speech language (optional)` hint, mirroring the existing `User → [8] identify` edge for `unknown speaker ids`.
- `docs/models.md`, `docs/cli.md`, `docs/troubleshooting.md`, `docs/pipeline.md`, `README.md` rewritten around Whisper as the only ASR backend; LM Studio is no longer listed as a prerequisite for ASR.

### Internal

- `AsrError` moved from `speech2text.py` (gone) into `whisper_asr.py`.
- `pipeline.run()` dispatcher (`if asr_engine == "vibevoice": …` / `elif "whisper": …`) collapsed to a direct `whisper_asr.transcribe(...)` call.
- `tests/test_pipeline_ordering.py` parameterisations for `asr_engine="vibevoice"` and the cross-route assertions (`test_asr_engine_whisper_routes_to_whisper_module`, `test_asr_engine_vibevoice_does_not_invoke_whisper`) removed. `tests/test_cli.py` updated to assert argparse now rejects `--asr-engine`. `tests/test_whisper_asr.py` imports `AsrError` from `voice.whisper_asr` directly.

### Follow-up

- [#88](https://github.com/artem-from-ua/voice-transcriber/issues/88) tracks regenerating pause markers from inter-segment gaps in `merge.py` if a future recording shows that the absence of `[Silence]` cues materially degrades structure / TL;DR output. Gated on observing the problem, not on speculative completeness.

## [0.22.0] — 2026-05-12

### Changed (behaviour)

- **Default LLM is now `mlx-community/Qwen2.5-7B-Instruct-4bit`** (was `gemma-3-12b-it-qat-4bit`). On a 16 GB Mac the previous default could not finish `structure_dialog` on long inputs even with v0.21's chunking + per-stage cleanup; Qwen2.5-7B fits and produces multi-section structure reliably. See [ADR 0020](docs/adr/0020-default-llm-qwen25-7b.md) for the bench comparing 5 models. Migration: pass `--llm-model ~/.cache/lm-studio/models/mlx-community/gemma-3-12b-it-qat-4bit` to keep the v0.21 default. First-time setup: `huggingface-cli download mlx-community/Qwen2.5-7B-Instruct-4bit` (~4 GB).
- **Default sampling now matches Qwen2.5's recommended values:** `temperature=0.7`, `top_p=0.8`, `top_k=20`, `repetition_penalty=1.05`. These override the prompt-frontmatter values (which kept their previous per-stage values for `max_tokens`). Users running a different model should pass that model's recommended sampling explicitly — there is no auto-tuning per model. See `docs/troubleshooting.md` for per-model recommended values.

### Added

- `MlxLLM.model_path` now accepts either a filesystem path **or** a HuggingFace `org/repo` id. Repo ids are resolved via `huggingface_hub.try_to_load_from_cache(repo_id, "config.json")` — no network access; missing repos raise `LLMError` with the `huggingface-cli download …` command to run. Filesystem paths keep working unchanged for users with custom MLX checkpoints or the legacy LM Studio cache layout.
- Markdown header now lists the toolchain that produced the transcript (`Діаризація`, `ASR`, `LLM`) and the wall-clock timings (`Обробка: 5m43s`, `AI-стадії: diarize=25s · asr=53s · proofread=131s · …`). The `LLM` line collapses to a single model name when every enabled stage used the same one, otherwise shows `stage=model` chips. The timings make each saved transcript self-documenting and serve as a built-in benchmark record.

### Internal

- `MlxLLM._resolved_path` populated by `.load()` exposes the on-disk snapshot directory; `_ensure_llm` compares it across repo-id ↔ resolved-path pairs so two specs that point to the same model don't trigger a spurious cold reload.

## [0.21.0] — 2026-05-12

### Added

- `structure_dialog` now chunks long dialogues automatically. Recordings with more than ~60 speech segments are split into overlapping ~35-segment chunks (overlap 2), the LLM is called once per chunk, and the per-chunk section lists are reconciled into one contiguous list (edge-snap, drop overlap dupes, merge same-title neighbours, fuse the shortest pair if total exceeds 7 sections). Short recordings (under 60 segments) use the original single-pass path, unchanged byte-for-byte. See [ADR 0018](docs/adr/0018-chunked-structure-dialog.md) for the design and the per-call validator differences.
- Per-stage LLM model selection: `--llm-proofread-model`, `--llm-identify-model`, `--llm-structure-model`, `--llm-tldr-model`. Each defaults to `--llm-model` if unset, which itself defaults to `mlx-community/gemma-3-12b-it-qat-4bit`. The pipeline swaps models between stages only when the paths differ, paying one cold load per swap; consecutive stages on the same model share one resident instance. See [ADR 0019](docs/adr/0019-per-stage-llm-models.md). Default behaviour is unchanged from v0.20.0 when no per-stage flags are passed.
- `MlxLLM.log_memory` (off by default; enabled automatically under `--verbose`). Each `chat()` / `chat_json()` call prints `pre`, `peak`, `post-clear` MLX active/peak memory plus prompt/output token counts, with a best-effort `mx.reset_peak_memory()` before generation so per-call peaks are meaningful.
- `pipeline.run()` invokes `free_mlx(log)` (`gc.collect()` + `mx.clear_cache()`) after each enabled LLM stage, not only at the end of the run. This shrinks the working set the next stage sees, which materially helps `structure_dialog` after a long proofread pass.

### Changed

- Internal: `pipeline.run()` no longer instantiates one `MlxLLM` for all four stages. The new `_ensure_llm(current, want_path, …)` helper returns the resident instance when the requested path matches, otherwise unloads + cold-reloads. The `finally` block that closes the LLM is unchanged.
- Internal: `MlxLLM._stream_generate` now returns `(text, output_token_count)` so the memory-logging path can report token counts without double-counting via `on_token`.
- Internal: `structure._validate` takes optional `min_sections`, `max_sections`, `require_exact_bounds` parameters (defaults reproduce v0.20.0 behaviour) so the chunked path can reuse it with looser per-chunk bounds.

### Known issues

- Empirical per-stage model defaults remain TODO (issue #78 follow-up). v0.21.0 ships the mechanism; deciding which of gemma-3-{12b,4b,1b}, Qwen2.5-3B, Llama-3.2-3B should be the default for each stage requires a subjective quality benchmark on real recordings that has not yet been run. Users who want to try a smaller model on, say, proofread can do so today via `--llm-proofread-model`.
- On 16 GB Macs, chunked `structure_dialog` on gemma-3-12b is **still** OOM-prone even after the per-stage cleanup — a clean 12b enters chunk 1 OK (peak ~10.6 GB) but the allocator fragmentation accumulated across chunks tips chunk 2 over the Metal ceiling. The working configuration is to swap to a smaller model **either** on proofread (which clears the long-tail KV churn before structure) **or** on structure itself. See `docs/troubleshooting.md` for the exact CLI invocations.
- On `mlx-community/gemma-3-4b-it-qat-4bit` the `structure_dialog` chunked path frequently returns JSON that fails the per-chunk validator (sections out of bounds, wrong count). When that happens the pipeline gracefully falls back to a single section, but the section titling produced by 12b is more useful in practice. This is a model-choice trade-off, not a bug.
- Chunk-boundary section duplication is possible when the LLM names the same thread slightly differently across chunks ("обговорення задачі" vs "обговорення задач"). The same-title merge handles the exact-match case; the conservative fuzz-match case is intentionally not handled because false fusing is worse than near-duplicate titles. Acceptable for v0.21.0; will revisit if real long recordings show it as a recurring pain point.

## [0.20.0] — 2026-05-12

### Added

- Second ASR backend: `--asr-engine {vibevoice,whisper}` routes the speech2text stage to either `mlx-community/VibeVoice-ASR-{bits}bit` (legacy) or `mlx-community/whisper-large-v3-mlx` via `mlx-whisper` (new).
- New subcommand `voice download-whisper` pre-fetches the Whisper model into the standard HuggingFace cache (~3 GB). `voice transcribe --asr-engine whisper` fails fast with an actionable error if the model isn't cached — no implicit network fetches inside the pipeline.
- See [ADR 0017](docs/adr/0017-whisper-asr-backend.md) for the design, the empirical default-engine decision on a Ukrainian reference, and the method (copied from the sibling stone-scriber project: `condition_on_previous_text=False`, no custom chunking).

### Changed

- **Default `--asr-engine` is now `whisper`** (was implicitly `vibevoice` in v0.19.0). On our Ukrainian reference recording Whisper-large-v3-MLX is 3.4× faster than VibeVoice-ASR-6bit, uses 2.8× less MLX memory, and produces visibly more faithful Ukrainian text including code-switched English words and swearing. Pass `--asr-engine vibevoice` to keep the old behaviour. The first run after upgrading needs `voice download-whisper` (~3 GB).
- Whisper emits no speaker hint per ASR segment (`speaker_asr=None`), but stage 6 (merge) already drives speaker assignment from pyannote turns, so the merge / proofread / structure / tldr stages are unaffected. VibeVoice's `[Silence]` / `[Music]` / `[Human Sounds]` in-band markers do not appear when Whisper is active; the LLM stages tolerate this in practice.
- `render_markdown` now receives a computed `asr_label` (either `VibeVoice-ASR-{bits}bit` or `Whisper-large-v3-MLX`) so the final Markdown header reflects the engine that produced the transcript.
- Two new direct dependencies: `mlx-whisper>=0.4.3` and `huggingface-hub>=1.14` (the latter promoted from transitive to direct because we now call `snapshot_download` and `try_to_load_from_cache` ourselves).
- `docs/architecture.md` stage [5] PlantUML node now lists both backends (`VibeVoice-ASR` and `Whisper-large-v3`); the prose under data-flow item 5 explains the dispatcher.

### Known issues

- 16 GB Mac users may still hit a Metal OOM during `structure_dialog` (stage 9) regardless of which ASR engine is active — gemma-3-12b's KV growth at the end of `identify` plus a single-prompt structure pass on 90 segments can exceed available memory. Workaround: run with `--no-structure --no-tldr` to get a flat transcript. The root cause is tracked as a follow-up; it is not caused by either ASR engine but is more visible with Whisper because Whisper leaves less peak headroom upstream.

## [0.19.0] — 2026-05-12

### Changed (BREAKING)

- Renamed clearspeech effect `agc` → `autogain` for clarity. `agc` was internal jargon (automatic gain control); `autogain` reads as English and is unambiguous in chain strings. See [ADR 0016](docs/adr/0016-rename-agc-to-autogain.md) for the full rationale. Migration:
  - `--clearspeech-chain agc[,…]` → `--clearspeech-chain autogain[,…]` (also the new default).
  - `--clearspeech-agc-target-dbfs` → `--clearspeech-autogain-target-dbfs`.
  - `--clearspeech-agc-max-gain-db` → `--clearspeech-autogain-max-gain-db`.
  - `clearspeech(...)` kwargs `agc_turns` / `agc_target_dbfs` / `agc_max_gain_db` / `agc_crossfade_ms` → `autogain_*`.
  - `PipelineOptions.clearspeech_agc_*` → `PipelineOptions.clearspeech_autogain_*`.
  - Sibling-WAV suffix `<stem>.agc.wav` → `<stem>.autogain.wav` (downstream chain artefacts likewise: `<stem>.autogain.bandpass.wav`, etc.).
  - Dump-artefact filenames `02b-clearspeech-N-agc.wav` → `02b-clearspeech-N-autogain.wav`.

### Changed

- Architecture diagram (`docs/architecture.md`): edge labels `speaker turn boundaries` → `speaker timecodes` (on both `Diar → Clearspeech` and `Diar → Merge` edges), `text split into sections` → `split by topic sections` (on the `Structure → TLDR` edge).
- Historical ADRs (0007, 0009, 0010–0015) intentionally keep the legacy `agc` term — they are immutable snapshots of decisions made at the time.

## [0.18.0] — 2026-05-12

### Changed (internal)
- Aligned four pipeline stage labels with their module filenames so `find src/voice -name "<stage>.py"` works for any stage you see in the diagram or the progress log:
  - `src/voice/ffprobe.py` → `src/voice/audiometa.py` (public `extract_metadata()` unchanged).
  - `src/voice/asr.py` → `src/voice/speech2text.py` (public `transcribe()` unchanged).
  - Inline `_to_wav_16k_mono()` in `pipeline.py` extracted into a new `src/voice/transcode.py` module exporting `transcode()`.
  - Stage `[1]` progress label `audiotranscode` → `transcode`; stage `[5]` progress label `ASR (...)` → `speech2text (...)`.
- Architecture and pipeline docs (`docs/architecture.md`, `docs/pipeline.md`) updated to match the new module names; the high-level architecture diagram was retired in favour of the more informative pipeline-stages diagram.
- Public surface unchanged: CLI flags (`--asr-bits`, `--asr-temperature`, `--asr-chunk-duration`), dump artefact filename `03-asr.json`, `AsrSegment` type, `proofread.fix_asr_errors()` all keep their existing names — they are engineering identifiers, not stage presentation.

## [0.17.0] — 2026-05-12

### Added
- New clearspeech effect **dereverb** — per-pyannote-turn Lebart-Polack spectral subtraction (room reverberation removal). RT60 is estimated per turn via Schroeder backward integration on a 500–2000 Hz bandpassed envelope; the predicted late-reverberation power is then subtracted from the STFT magnitude before iSTFT. No new dependencies — `scipy.signal.stft`/`istft`/`hilbert` already available. **Default off; experimental opt-in.** Three new tuning knobs: `--clearspeech-dereverb-rt60-floor-ms` (default `300`), `--clearspeech-dereverb-subtract-factor` (default `1.0`), `--clearspeech-dereverb-crossfade-ms` (default `50`).
- `dereverb` is the first chain effect that *requires* `pyannote` turns to run — short or quiet stretches of audio don't carry enough decay information for the RT60 fit. Chains containing `dereverb` without an upstream diarisation step raise `ClearspeechError`.
- The `--clearspeech-chain` `--help` text now lists `dereverb` among the known effects and points to ADR 0015 for empirical caveats.

### Changed
- Pipeline stage `[4/10] Clearspeech (…)` accepts `dereverb` anywhere in the chain string; no other behavioural change.

### Empirical results
A four-run Metric A grid on the reference recording (RU-glyph drift rate):

| Chain | RU-glyph | Note |
| --- | --- | --- |
| `agc` (default, unchanged) | 5.1 % | best-balanced baseline |
| `dereverb,agc` | 7.1 % | pre-AGC; merges short Ukrainian turns into long Russian-leaning blocks |
| `agc,dereverb` | 10.5 % | post-AGC; AGC lifts reverb tails before subtraction can target them |
| `dereverb` (isolated) | 0.0 % | **Simpson's paradox** — drops ~9 % of Ukrainian text into `[Human Sounds]` markers, the 0 % is rate-not-volume |

Default chain stays `"agc"`. See [`docs/adr/0015-clearspeech-dereverb.md`](docs/adr/0015-clearspeech-dereverb.md) for the side-by-side transcript audit and the cross-engine-validation follow-up.

## [0.16.0] — 2026-05-12

### Added
- New clearspeech effect **denoise** (issue #48, E2d): FFT-domain spectral subtraction via `ffmpeg afftdn`, exposed by two knobs — `--clearspeech-denoise-noise-floor-db` (default `-25`) and `--clearspeech-denoise-reduction-db` (default `12`). No new Python dependency: the existing ffmpeg subprocess used by stage `[1]` is reused. **Default off; experimental opt-in.** Metric A on the reference recording shows denoise degrades ASR at every position measured — the bare-AGC default (5.1 % RU-glyph drift) stays the empirical best, and the best chain involving denoise lands at 8.2 % (post-AGC). Pre-AGC denoise is catastrophic (27.5 %, comparable to the `agc,presence`-without-bandpass foot-gun from ADR 0013). See [`docs/adr/0014-clearspeech-denoise.md`](docs/adr/0014-clearspeech-denoise.md) for the five-run grid, the backend bake-off (`noisereduce` was dropped on the listening A/B), and the OpenAI Whisper-rule verification.
- The `--clearspeech-chain` `--help` warning now flags two foot-guns: enabling `presence` without `bandpass` first (ADR 0013) and enabling `denoise` at all on Ukrainian content (ADR 0014).

### Changed
- Pipeline log line `[4/10] Clearspeech (…)` now includes `denoise` when present. No change to dump layout — denoise writes `02b-clearspeech-N-denoise.wav` per the existing per-effect numbering.

## [0.15.0] — 2026-05-11

### Added
- New clearspeech effect **presence** (issue #48, E2b): peaking EQ via the Robert Bristow-Johnson Audio EQ Cookbook biquad, applied zero-phase with `scipy.signal.filtfilt`. Boosts a configurable band around `center_hz` (default 3000) by `boost_db` (default +6, tuned by listening test) with sharpness controlled by `q` (default 1.0). Restores the 2–5 kHz consonant intelligibility band that distance attenuates for far-mic speakers. **Default off; experimental opt-in.** Metric A on the reference recording showed no language-drift improvement (best chain `agc,bandpass,presence` at 10.3 % vs default `agc` at 5.1 %); using presence **without** bandpass first catastrophically breaks ASR (63 % drift). See [`docs/adr/0013-clearspeech-presence-defaults.md`](docs/adr/0013-clearspeech-presence-defaults.md) for the full empirical table, the four-run Metric A grid, and the physical explanation.
- Free-order chains: any permutation of known effects (`agc`, `bandpass`, `presence`) is now valid. Duplicates and unknown names raise `ValueError`. See [`docs/adr/0012-clearspeech-chain-string-cli.md`](docs/adr/0012-clearspeech-chain-string-cli.md) for the CLI design.

### Changed (Breaking, pre-1.0)
- **CLI collapsed to a single chain string.** `--no-clearspeech-agc`, `--clearspeech-bandpass`, and `--no-clearspeech-bandpass` are removed. Their behaviour now lives in `--clearspeech-chain STR`. Migration:

  | v0.14.0 | v0.15.0 |
  | --- | --- |
  | (no clearspeech flag) | (no clearspeech flag — same: `agc` chain) |
  | `--no-clearspeech-agc` | `--clearspeech-chain ""` |
  | `--clearspeech-bandpass` | `--clearspeech-chain agc,bandpass` |
  | `--no-clearspeech-agc --clearspeech-bandpass` | `--clearspeech-chain bandpass` |
  | (not possible) | `--clearspeech-chain agc,bandpass,presence` |
  | (not possible) | `--clearspeech-chain presence,agc` (any order) |

  The default value of `--clearspeech-chain` is `"agc"` so callers that pass no clearspeech flag get the same behaviour as 0.14.0.

- **Internal chain gate removed.** `_PR1_ALLOWED_CHAINS` is replaced by `_validate_chain()` — chains are validated only for unknown effect names and duplicates; order is the user's call.

### Added (continued)
- Three new tuning flags for the presence effect: `--clearspeech-presence-center-hz HZ` (default `3000`), `--clearspeech-presence-boost-db DB` (default `3.0`), `--clearspeech-presence-q Q` (default `1.0`). Sane bounds enforced — `-24..24 dB`, `0.1..10 Q`, `0 < center < Nyquist`.

## [0.14.0] — 2026-05-11

### Added
- New chain-of-effects audio-cleanup stage **clearspeech** (issue #48, PR-1). The stage replaces the v0.13.0 `loudness_normalize` stage and runs an ordered chain of DSP effects between diarize and ASR. PR-1 ships two effects with a fixed canonical order `agc → bandpass`; free-order arrives in PR-2 once the presence-boost effect lands and reordering becomes meaningful. Each enabled effect writes a sibling WAV consumed by the next; ASR reads the output of the last applied effect, or the raw WAV when the chain is empty. See [`docs/adr/0010-clearspeech-chain.md`](docs/adr/0010-clearspeech-chain.md) for the design rationale.
- New effect **bandpass** (issue #48, E2a): Butterworth IIR order-4 via `scipy.signal.sosfiltfilt`, zero-phase. Cuts sub-vocal rumble and super-vocal noise so ASR sees a cleaner spectral envelope. Default off — experimental opt-in.
- New CLI group `clearspeech (audio cleanup for ASR)`:
  - `--no-clearspeech-agc` — skip the AGC step (default: AGC on, preserving v0.13.0 behaviour).
  - `--clearspeech-agc-target-dbfs FLOAT` (default `-20.0`).
  - `--clearspeech-agc-max-gain-db FLOAT` (default `16.0`).
  - `--clearspeech-bandpass` — opt into the new effect (default off).
  - `--clearspeech-bandpass-low-hz HZ` (default `150.0` — tuned by listening test, see [`docs/adr/0011-clearspeech-bandpass-defaults.md`](docs/adr/0011-clearspeech-bandpass-defaults.md)). Note: Metric A (Russian-glyph language drift) was flat in measurement, so bandpass stays opt-in.
  - `--clearspeech-bandpass-high-hz HZ` (default `5500.0` — tuned by listening test; the 16 kHz Nyquist of 8 kHz is a hard upper bound, issue #48's nominal 10 kHz is unreachable at this SR).
- New runtime dependency: `scipy>=1.14` for filter design.
- New dump artefact `02b-clearspeech-config.json` records the active chain, per-effect parameters, and AGC stats (turn count, RMS spread before/after, ceiling hits) for `--dump-stages` reproducibility.

### Changed (Breaking, pre-1.0)
- **CLI rename:** `--no-loudness-normalize` → `--no-clearspeech-agc`; `--loudness-target-dbfs` → `--clearspeech-agc-target-dbfs`; `--loudness-max-gain-db` → `--clearspeech-agc-max-gain-db`. No aliases; update scripts directly. Defaults are unchanged so existing invocations using only positional arguments behave the same as 0.13.0.
- **Dump layout:** `02b-normalized.wav` is gone. The new layout uses `02b-clearspeech-config.json` plus one `02b-clearspeech-N-<effect>.wav` per applied effect (e.g. `02b-clearspeech-1-agc.wav`, `02b-clearspeech-2-bandpass.wav`). Indices reflect the position in the *active* chain — disabling AGC moves bandpass to index 1.
- **Internal:** module `voice.audio_preprocess` removed; `loudness_normalize` is now `voice.clearspeech._apply_agc` (private). Pipeline stage label changed from `[4/10] Loudness-нормалізація` to `[4/10] Clearspeech (<chain>)`.
- Pipeline progress headers remain `[N/10]`. The chain-of-effects design absorbs new DSP work without growing the stage count, so future E2-experiment PRs will not renumber.

## [0.13.0] — 2026-05-11

### Changed
- **Renamed pipeline stage `postprocess` → `proofread`** (no behaviour change; rename only). Aligns with ADR 0005's "proof-read" terminology, which the codebase had already internalised in prose but not yet in identifiers.
  - **Breaking CLI:** `--no-postprocess` → `--no-proofread`. No deprecation alias — pre-1.0 project; update scripts directly.
  - **Breaking dump layout:** `05-postprocess.json` → `05-proofread.json` (when `--dump-stages DIR` is used).
  - Module `voice.postprocess` → `voice.proofread`. Prompt files `prompts/postprocess_{system,user}.md` → `prompts/proofread_{system,user}.md`. ADR `0005-segment-level-asr-postprocess.md` → `0005-segment-level-asr-proofread.md`.
  - Internal field `PipelineOptions.run_postprocess` → `run_proofread`.
- **Architecture diagram: stages colour-coded by category.** `docs/architecture.md` pipeline-stages PlantUML now annotates each box with the underlying library / model on extra lines, and fills the box with a category-specific pastel: light-salmon for subprocess (ffprobe/ffmpeg), light-sky-blue for pyannote diarization, light-green for VibeVoice ASR, plum for the four LLM stages (gemma-3-12b), light-grey for pure-Python stages (merge / render / normalize). Legend text uses dark grey `#404040` for visual consistency.

## [0.12.0] — 2026-05-11

### Added
- New pipeline stage **loudness normalization** between WAV conversion and ASR (issue #47). The pipeline now runs **diarize before ASR**: pyannote turns serve as per-segment guard-rails for RMS-based AGC, applied with 100 ms linear crossfades on segment boundaries (silent gaps left at unity gain). New module `voice.audio_preprocess.loudness_normalize(wav, turns, target_dbfs, max_gain_db, crossfade_ms)` returns a sibling `<stem>.normalized.wav` consumed by ASR. The motivating bug: VibeVoice-ASR drifts to Russian on quiet turns; equalising per-speaker levels measurably reduces the drift on the test recording.
- CLI flags: `--no-loudness-normalize` (stage is default-on), `--loudness-target-dbfs FLOAT` (default `-20.0`), `--loudness-max-gain-db FLOAT` (default `16.0` — tuned by listening test, see [`docs/adr/0007-loudness-normalize-defaults.md`](docs/adr/0007-loudness-normalize-defaults.md)), grouped under "loudness normalization" in `--help`.
- New runtime dependency: `soundfile>=0.12` for libsndfile-backed WAV I/O.
- `voice._dump.StageDumper.write_binary(name, src)` copies a binary artefact (file path or raw bytes) into the dump dir; used to dump `02b-normalized.wav`.

### Changed
- **Pipeline order: diarize now runs before ASR** (was: ASR → diarize). The order swap is required so the normalizer has its per-turn guard-rails before ASR sees the audio. With `--no-loudness-normalize` the pipeline produces an output equivalent to 0.11.0 on the same audio — ASR is deterministic and `merge()` is order-independent.
- `--dump-stages DIR` file layout renumbered to reflect the new order:
  - `02-diarize.json` (was `03-diarize.json`)
  - `02b-normalized.wav` (new, binary; only present when normalization stage runs)
  - `03-asr.json` (was `02-asr.json`)
  - `04-merge.json` … `09-tldr.txt` unchanged.
- Pipeline progress headers go from `[N/9]` to `[N/10]`.

## [0.11.0] — 2026-05-11

### Added
- New `--dump-stages DIR` CLI flag writes each stage's output as JSON for troubleshooting. Layout under DIR:
  - `01-meta.json` — `AudioMeta` (ffprobe)
  - `02-asr.json` — `list[AsrSegment]` (raw VibeVoice)
  - `03-diarize.json` — `list[DiarTurn]` (pyannote)
  - `04-merge.json` — `list[Segment]` after merge (no LLM touches yet)
  - `05-postprocess.json` — `list[Segment]` after the LLM proof-reader
  - `06-identify.json` — `{pyannote-label → name}` map
  - `07-segments-named.json` — `list[Segment]` with `.name` filled in
  - `08-structure.json` — `StructuredDialog` (sections + segments)
  - `09-tldr.txt` — raw Markdown TL;DR string
- `voice._dump.StageDumper` is a thin no-op when the flag isn't passed; the pipeline always calls `dumper.write(...)` so call sites don't branch.

## [0.10.0] — 2026-05-11

### Changed
- ASR `chunk_duration` raised from 15 s to **45 s**. VibeVoice was trained for up to 60-minute single-pass inputs and explicitly benefits from long context — short chunks were the dominant cause of language drift (Ukrainian → Russian / nonsense). At 45 s the KV cache stays comfortably under 2 GB.
- ASR `temperature` lowered from 0.1 to **0.0** (greedy decoding). The 0.1 was a band-aid against repetition loops; raising `repetition_penalty` from 1.2 to **1.3** prevents the same loops without the run-to-run quality lottery — output is now reproducible for the same input.

### Added
- `--asr-chunk-duration SECONDS` and `--asr-temperature T` CLI flags so users can experiment without editing constants.
- `asr.transcribe()` gains optional `chunk_duration` and `temperature` kwargs; `None` keeps the module defaults.

## [0.9.2] — 2026-05-11

### Added
- Header now lists the participants with their colour chips: `👥 **Учасники:** 🔵 Артем, 🟢 Остап`. Order matches first appearance; falls back to pyannote labels (`SPEAKER_00` etc.) when names weren't supplied. Skipped entirely when the recording has no detected speakers.

### Fixed
- `tldr.generate_tldr()` strips a leading `## TL;DR` (or `# TL;DR`, `**TL;DR**`, `**TL;DR:**`, lowercased / no-semicolon variants) from the LLM's reply. The renderer was already adding its own heading, so the model's extra one produced two `## TL;DR` stacked on top of each other.

## [0.9.1] — 2026-05-11

### Changed
- Markdown header: `📅 Початок` now prints as `2026-05-10 15:44 UTC` (normalised to UTC). `🌐 Мова` continues to show the raw ISO code passed in via `--language`.
- Removed lines from the header: `🏁 Кінець` (redundant with `Початок` + `Тривалість`) and `🎙️ Транскрипція` (the toolchain doesn't belong in the transcript itself). `render_markdown(asr_label=…)` is still accepted for backward-compatibility but is no longer rendered.
- Memory line in progress UI: `RAM 9.4 GB (59%) · MLX 8.0 GB`. Dropped `/16.0 GB` total (never changes) and `peak N.N GB` (only ratchets up — duplicates info from the active value).

## [0.9.0] — 2026-05-11

### Added
- `voice._progress.ProgressReporter` — `rich.Progress` wrapper with three context managers: `task(label, total)` for deterministic counters, `spinner(label)` for indeterminate ops, `token_counter(label)` for token-streaming LLM generation.
- TTY mode: in-place bars with elapsed/ETA, plus a persistent resource footer underneath updated once per second: `RAM 9.4/16.0 GB (59%) · MLX 8.0 GB · peak 9.6 GB`. RAM is read with `psutil.virtual_memory().used` which matches macOS Activity Monitor's "Memory Used" formula.
- Non-TTY mode (CI, Claude Code Bash tool, `2> log`): bars suppressed; instead a background heartbeat thread prints **one line every 15 s** while a stage is active — `· [6/9] postprocess: 23/64 (35%) elapsed 0:32 5 fixed · RAM 9.2/16 GB (57%) · MLX 8.0 GB`. Stage transitions (`✓ {label} in N.Ns`) print immediately, never blocked by the heartbeat interval.
- `voice._memory_stats.read_memory_snapshot()` returns a best-effort RAM + MLX snapshot; each field is `None` if its source is unavailable, and the formatter silently omits missing columns.
- `voice._progress.NullProgress` — drop-in no-op replacement so stage modules don't need to branch on `progress is None`.
- Stage modules (`postprocess`, `identify`, `structure`, `tldr`) and `pipeline` accept an optional `progress=` argument and emit per-stage UI.
- Model loads (VibeVoice, pyannote, Gemma) each run inside a `progress.spinner(...)`.

### Changed
- `MlxLLM.chat()` and `MlxLLM.chat_json()` accept an optional `on_token` callback. Internally they switch from `mlx_lm.generate` to `mlx_lm.stream_generate` so the progress reporter can advance once per token.
- `voice._memory.free_mlx()` now prefers `mx.clear_cache()` over the deprecated `mx.metal.clear_cache()`. Removes a deprecation warning on mlx ≥ 0.21.
- `rich` and `psutil` added as runtime dependencies.

## [0.8.0] — 2026-05-11

### Changed
- LLM call parameters (`temperature`, `max_tokens`, `top_p`, `repetition_penalty`, `response_format`) are now declared in each prompt's YAML frontmatter, not in Python constants. Tuning a stage now means editing a `.md` file.
- New `voice._prompts.load_prompt(name) -> PromptFile` returns a dataclass with `body`, `placeholders`, and `params`.
- New `voice._prompts.call_kwargs(name)` extracts the subset of `params` that maps to `MlxLLM.chat()` / `.chat_json()` kwargs. `response_format` is excluded — JSON mode is selected at the call site.
- `identify.py`, `postprocess.py`, `structure.py`, `tldr.py` no longer carry temperature/max_tokens literals; they pull them from `call_kwargs(prompt_name)`.

### Added
- Frontmatter parser handles YAML-style bracketed lists with unquoted elements (e.g. `placeholders: [language]`) — `ast.literal_eval` does not parse that, so we fall back to a comma-split path.
- `tests/test_prompts.py` exercises the new param schema (every system prompt declares `temperature` and `max_tokens`; `response_format ∈ {json_object, text}`).

## [0.7.0] — 2026-05-11

### Changed
- LLM runtime moved from LM Studio's HTTP API to in-process `mlx-lm`. Same MLX-quantised model files (`~/.cache/lm-studio/models/mlx-community/…`) are reused — no extra downloads. Eliminates the Metal allocator fragmentation that crashed LM Studio's long-context prompts after dozens of short postprocess calls.
- `voice.llm.LLMClient` replaced by `voice.llm.MlxLLM`. Public surface kept compatible: same `chat(messages, *, temperature, max_tokens, ...)` and `chat_json(messages, *, schema, …)` shape, plus `load()` / `close()` lifecycle.
- `chat_json()` now uses `lm-format-enforcer` as a logits processor against a JSON schema — output is guaranteed parseable on the first try, no retry loop.
- Pipeline lifecycle: ASR, diarization, and LLM model each load and free in sequence. The LLM is loaded once after merge and reused across postprocess, identify, structure, and TL;DR. MLX cache is cleared between calls.

### Removed
- `--llm-base-url` CLI flag (no HTTP server to point at).
- `LLMClient.unload_model()` — LM Studio-only concern.
- `httpx.MockTransport`-based tests; replaced by `monkeypatch` of `mlx_lm.load` / `mlx_lm.generate` in `tests/test_mlxllm.py`.

### Added
- `lm-format-enforcer` dependency for JSON-constrained generation.
- `mlx-lm` dependency (was previously only transitively via `mlx-audio`).
- `MlxLLM.health_check()` validates the model directory exists with `config.json`; raises an actionable error pointing to LM Studio's Models tab for the download.

## [0.6.4] — 2026-05-11

### Added
- `LLMClient.unload_model()` — best-effort call to LM Studio's native `/api/v1/models/unload` endpoint. Returns `True` on success, `False` on any failure; never raises.

### Changed
- Pipeline orchestrator now asks LM Studio to drop the LLM weights three times: right after the pre-flight health check (so VibeVoice has the full GPU during ASR), before the structure stage, and before TL;DR. The two long-context prompts otherwise inherit Metal allocator fragmentation from the dozens of short postprocess calls and run out of GPU memory on 16 GB Macs.

## [0.6.3] — 2026-05-11

### Changed
- LLM prompts moved out of the Python modules into individual `src/voice/prompts/*.md` files with YAML frontmatter. Substitution uses `<<placeholder>>` delimiters so JSON braces in the prompt body are preserved literally.
- New internal `voice._prompts` loader (`render`, `list_placeholders`) drives every stage; modules now hold only call parameters and safety checks.
- `docs/prompts.md` reorganised around the new file layout and the LM Studio response-format contract.

## [0.6.2] — 2026-05-11

### Changed
- `asr.transcribe()` and `diarize.diarize()` drop their model objects and trigger MLX/Metal/MPS cache eviction right after producing their outputs. On 16 GB Macs the three model weights (VibeVoice + pyannote + LM Studio's LLM) otherwise exceed unified memory.

## [0.6.1] — 2026-05-11

### Fixed
- `LLMClient.chat_json()` now sends `response_format={"type": "json_schema", ...}` instead of `"json_object"`. LM Studio's MLX runtime rejects the latter with `400 Bad Request`, which broke `structure_dialog()` end-to-end on the real test recording.
- `chat/completions` HTTP errors now include the server's response body (first 500 chars) in the `LLMError` message — previously the actionable detail (e.g. *"'response_format.type' must be 'json_schema' or 'text'"*) was hidden by `httpx`'s default `__str__`.

## [0.6.0] — 2026-05-11

### Added
- `pipeline.run()` — end-to-end orchestration from audio file to Markdown
- `cli` module — `voice transcribe <audio>` with all options from the design
- `render` module — final Markdown output with metadata, optional TL;DR, emoji-tagged speakers, pause markers, and sectioned dialogue
- `speaker_emojis` — fixed 9-emoji palette cycled past nine speakers
- `tldr` module — Markdown TL;DR generated by a local LLM (Ukrainian or English prompt)
- `structure` module — LLM-driven section structuring with strict layout validation
- `postprocess` module — per-segment ASR proof-reader with edit-distance safety net
- `identify` module — LLM-based speaker identification from self-introductions
- Core modules: `asr`, `diarize`, `merge`, `ffprobe`, `types`
- `llm` module — synchronous LM Studio client (`chat`, `chat_json` with one retry)

## [0.1.0] — 2026-05-10

### Added
- Initial uv-managed Python 3.12 package skeleton
