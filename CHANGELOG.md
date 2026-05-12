# Changelog

All notable changes to this project will be documented in this file. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
- CLI flags: `--no-loudness-normalize` (stage is default-on), `--loudness-target-dbfs FLOAT` (default `-20.0`), `--loudness-max-gain-db FLOAT` (default `16.0` — tuned by listening test, see [`docs/adr/0006-loudness-normalize-defaults.md`](docs/adr/0006-loudness-normalize-defaults.md)), grouped under "loudness normalization" in `--help`.
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
