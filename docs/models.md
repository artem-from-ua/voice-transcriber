# Models

Three model families drive the pipeline: speaker diarization (pyannote), ASR (Whisper by default, VibeVoice as a legacy backend), and a general-purpose LLM (`Qwen2.5-7B-Instruct-4bit` by default since v0.22.0; see [ADR 0020](adr/0020-default-llm-qwen25-7b.md)) covering identify / proofread / structure / TL;DR.

Every transcript's Markdown header lists the exact models that produced it (`Діаризація`, `ASR`, `LLM` lines) plus wall-clock timings (`Обробка: 5m43s`, `AI-стадії: diarize=… · asr=… · proofread=… · …`) — saved transcripts double as benchmark records.

## ASR — `mlx-community/VibeVoice-ASR-Nbit`

VibeVoice is Microsoft's MLX-quantised speech model; it transcribes the audio and *also* produces speaker tags (0/1/…) inline, which we keep for sanity checks but always override with pyannote's labels at merge time.

Pipeline supports four bitness variants chosen via `--asr-bits`:

| Bits | Path | Size | Quality | Speed |
|------|------|------|---------|-------|
| 4 | `mlx-community/VibeVoice-ASR-4bit` | ~5 GB | acceptable; repetition loops are more common on Ukrainian | fastest |
| 5 | `mlx-community/VibeVoice-ASR-5bit` | ~6 GB | better than 4-bit, marginally slower | fast |
| **6 (default)** | `mlx-community/VibeVoice-ASR-6bit` | ~7 GB | the best speed/quality knee-point | moderate |
| 8 | `mlx-community/VibeVoice-ASR-8bit` | ~9 GB | most accurate, hardware-bound | slowest |

The tuning constants in `speech2text.DEFAULT_GEN_KWARGS` (`repetition_penalty=1.3`, `repetition_context_size=64`, `temperature=0.0`) keep VibeVoice from looping on Ukrainian fragments like "шо я… шо я… шо я…" while giving deterministic, run-to-run reproducible output. `chunk_duration=45 s` is the matching default: VibeVoice was trained on up to 60-minute single-pass inputs and explicitly benefits from long context — short chunks were the dominant cause of language drift (Ukrainian → Russian / nonsense). Both can be overridden per run with `--asr-temperature` and `--asr-chunk-duration`.

`speech2text.py` looks up the model under `~/.cache/lm-studio/models/<repo>`. There is no implicit download — if a bitness is missing, the pipeline tells you to install it via LM Studio.

## Diarization — `pyannote/speaker-diarization-3.1`

We only consume `exclusive_diarization` (turns without overlap) because the merge stage is much simpler when a given moment belongs to exactly one speaker. The full `diarization` field is ignored.

Three Hugging Face repositories must be accepted with the user's account before pyannote can download weights:

- [`pyannote/speaker-diarization-3.1`](https://huggingface.co/pyannote/speaker-diarization-3.1)
- [`pyannote/segmentation-3.0`](https://huggingface.co/pyannote/segmentation-3.0)
- [`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1)

> The canonical copy of this list lives in [`troubleshooting.md`](troubleshooting.md) under *GatedRepoError*. If you change one, change both.

The token sits in `~/.cache/huggingface/token` (mode `600`). The pipeline never reads it from environment variables or settings files — see [ADR 0004](adr/0004-pyannote-for-diarization.md).

Inference runs on `mps` when available and falls back to `cpu` if the move fails. The `diarize` log line tells you which device was used.

## LLM — `mlx-community/Qwen2.5-7B-Instruct-4bit`

Default for every language task: identify, proofread, structure, TL;DR. The model is loaded in-process via `mlx-lm` against an MLX-quantised checkpoint in the HuggingFace cache.

**Resolution.** `MlxLLM.model_path` accepts either:

- A HuggingFace `org/repo` id (the default — `mlx-community/Qwen2.5-7B-Instruct-4bit`). Resolved via `huggingface_hub.try_to_load_from_cache`; the snapshot directory under `~/.cache/huggingface/hub/models--<org>--<repo>/snapshots/<sha>/` becomes the path passed to `mlx_lm.load`.
- A filesystem path (e.g. `~/.cache/lm-studio/models/mlx-community/gemma-3-12b-it-qat-4bit`). Returned as-is.

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
