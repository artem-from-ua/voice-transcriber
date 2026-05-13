# Models

Three model families drive the pipeline: speaker diarization (pyannote), ASR (Whisper-large-v3-MLX), and a general-purpose LLM (`Qwen2.5-7B-Instruct-4bit` by default since v0.22.0; see [ADR 0020](adr/0020-default-llm-qwen25-7b.md)) covering identify_speakers / proofread / speech_structure / speech_summary.

Every transcript's Markdown header lists the exact models that produced it (`Діаризація`, `ASR`, `LLM` lines) plus wall-clock timings (`Обробка: 5m43s`, `AI-стадії: diarize_speakers=… · asr=… · proofread=… · …`) — saved transcripts double as benchmark records.

## ASR — `mlx-community/whisper-large-v3-mlx`

OpenAI Whisper large-v3 converted to MLX. Driven by `mlx-whisper` (in-process, no HTTP). Selected as the default in v0.20.0 after the empirical comparison documented in [ADR 0017](adr/0017-whisper-asr-backend.md); the legacy VibeVoice backend was removed entirely in v0.23.0 — see [ADR 0021](adr/0021-remove-vibevoice-backend.md) for the side-by-side numbers and the reasoning.

**Onboarding.** `voice download-whisper` fetches the model (~3 GB) into `~/.cache/huggingface/hub/`. `voice transcribe` fails fast with an actionable error if the cache is empty — there is no implicit network fetch inside the pipeline run.

**Operating settings.** `whisper_asr.transcribe` calls `mlx_whisper.transcribe(audio, path_or_hf_repo=..., language=..., condition_on_previous_text=False)`. The other knobs (`temperature` schedule, `compression_ratio_threshold`, `logprob_threshold`, `no_speech_threshold`, 30-second windowing) are handled internally by `mlx-whisper` — we accept its defaults rather than re-expose them as CLI flags. `condition_on_previous_text=False` is the one explicit override: it prevents the repetition-loop failure mode Whisper is known for on long mono inputs (method copied verbatim from the sibling `stone-scriber` project — see ADR 0017).

**Language.** Since v0.24.0 `--language` defaults to `None`. A new stage `[4] lang_detect` runs between diarize_speakers and clear_speech, picks the longest pyannote turn, and calls `model.detect_language()` on Whisper-large-v3-MLX over that turn's mel-spectrogram. The detected ISO code becomes the `language` hint for `mlx_whisper.transcribe()` at stage `[6] speech2text` and flows through every downstream LLM / render stage. Passing `--language uk` (or any ISO code) explicitly skips lang_detect entirely. See [ADR 0022](adr/0022-asr-language-autodetect.md) for why the longest pyannote turn is a strictly better signal than Whisper's first-30 s internal classifier.

**Speaker labels.** Whisper has no speaker-prediction head; every ASR segment carries no speaker hint. Speaker labels come entirely from pyannote turns at stage 6 (merge).

## Diarization — `pyannote/speaker-diarization-3.1`

We only consume `exclusive_diarization` (turns without overlap) because the merge stage is much simpler when a given moment belongs to exactly one speaker. The full `diarization` field is ignored.

Three Hugging Face repositories must be accepted with the user's account before pyannote can download weights:

- [`pyannote/speaker-diarization-3.1`](https://huggingface.co/pyannote/speaker-diarization-3.1)
- [`pyannote/segmentation-3.0`](https://huggingface.co/pyannote/segmentation-3.0)
- [`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1)

> The canonical copy of this list lives in [`troubleshooting.md`](troubleshooting.md) under *GatedRepoError*. If you change one, change both.

The token sits in `~/.cache/huggingface/token` (mode `600`). The pipeline never reads it from environment variables or settings files — see [ADR 0004](adr/0004-pyannote-for-diarization.md).

Inference runs on `mps` when available and falls back to `cpu` if the move fails. The `diarize_speakers` log line tells you which device was used.

## LLM — `mlx-community/Qwen2.5-7B-Instruct-4bit`

Default for every language task: identify_speakers, proofread, speech_structure, speech_summary. The model is loaded in-process via `mlx-lm` against an MLX-quantised checkpoint in the Hugging Face cache.

**Resolution.** `MlxLLM.model_path` accepts either:

- A Hugging Face `org/repo` id (the default — `mlx-community/Qwen2.5-7B-Instruct-4bit`). Resolved via `huggingface_hub.try_to_load_from_cache`; the snapshot directory under `~/.cache/huggingface/hub/models--<org>--<repo>/snapshots/<sha>/` becomes the path passed to `mlx_lm.load`.
- A filesystem path (e.g. `~/.cache/lm-studio/models/mlx-community/gemma-3-12b-it-qat-4bit`). Returned as-is. (deprecated path format — LM Studio cache; see [ADR 0006](adr/0006-mlx-lm-over-lm-studio.md) and [#116](https://github.com/artem-from-ua/voice-transcriber/issues/116))

If a repo id is not in the cache, the pipeline fails fast with the exact `huggingface-cli download <repo>` command to run — there are no implicit multi-GB fetches. Same policy as Whisper (see [ADR 0017](adr/0017-whisper-asr-backend.md)).

**Default sampling.** Qwen2.5-Instruct's official `generation_config.json` is the source: `temperature=0.7`, `top_p=0.8`, `top_k=20`, `repetition_penalty=1.05`. These are `PipelineOptions` defaults that the pipeline forwards into `MlxLLM.sampling_overrides`, taking precedence over the per-prompt frontmatter values in `src/voice/prompts/*.md` (which still drive `max_tokens`). Override per-run with `--llm-temperature`, `--llm-top-p`, `--llm-top-k`, `--llm-repetition-penalty`.

**Prompts** are tuned for a model that handles Ukrainian fluently, behaves well under JSON-schema-constrained generation (`lm-format-enforcer` as a logits processor in `MlxLLM.chat_json()`), and follows strict "output JSON only, no commentary" instructions.

**Switching models.** Pass `--llm-model <repo-id-or-path>` (or any of `--llm-{proofread,identify,structure,tldr}-model` for per-stage overrides — see [ADR 0019](adr/0019-per-stage-llm-models.md)). If you switch to a different model family, **also pass its recommended sampling parameters**: the v0.22.0 defaults are Qwen2.5-specific and produce subtly worse results on other models. Recommended values for the five we've benched:

| Model | temp | top_p | top_k | rep_penalty | Source |
|---|---:|---:|---:|---:|---|
| `mlx-community/Qwen2.5-7B-Instruct-4bit` | 0.7 | 0.8 | 20 | 1.05 | HF `generation_config.json` (default) |
| `mlx-community/Qwen3-4B-Instruct-2507-4bit` | 0.7 | 0.8 | 20 | — | local `generation_config.json` |
| `mlx-community/gemma-3-12b-it-qat-4bit` | 1.0 | 0.95 | 64 | — | Google recommendation |
| `mlx-community/gemma-3-4b-it-qat-4bit` | 1.0 | 0.95 | 64 | — | Google recommendation |
| `mlx-community/gemma-3-1b-it-qat-4bit` | 1.0 | 0.95 | 64 | — | Google recommendation |
| `mlx-community/Llama-3.2-3B-Instruct-4bit` | 0.6 | 0.9 | — | — | Meta `generation.py` defaults |

**Memory caveat.** `gemma-3-12b` (~8 GB resident) does not finish `structure_dialog` on a 16 GB Mac even with v0.21's chunking — third chunk OOMs on Metal. Use it only on 24 GB+ Macs; on 16 GB stay with `Qwen2.5-7B` (~4 GB) or smaller. See [ADR 0020](adr/0020-default-llm-qwen25-7b.md) for the comparison.

[ADR 0006](adr/0006-mlx-lm-over-lm-studio.md) covers why the pipeline runs `mlx-lm` in-process rather than against LM Studio's HTTP API; [ADR 0001](adr/0001-local-llm-via-lm-studio.md) records the earlier LM Studio decision and is now superseded by 0006 and 0020. [ADR 0002](adr/0002-mlx-format-preference.md) covers MLX vs GGUF on Apple Silicon.
