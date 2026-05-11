# Changelog

All notable changes to this project will be documented in this file. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
