# Models

Two models drive the pipeline: a speech model (VibeVoice-ASR) and a general-purpose language model (Gemma 3 12B QAT 4-bit by default). The diarization model is a sub-pipeline of pyannote that we don't pick or tune.

## ASR — `mlx-community/VibeVoice-ASR-Nbit`

VibeVoice is Microsoft's MLX-quantised speech model; it transcribes the audio and *also* produces speaker tags (0/1/…) inline, which we keep for sanity checks but always override with pyannote's labels at merge time.

Pipeline supports four bitness variants chosen via `--asr-bits`:

| Bits | Path | Size | Quality | Speed |
|------|------|------|---------|-------|
| 4 | `mlx-community/VibeVoice-ASR-4bit` | ~5 GB | acceptable; repetition loops are more common on Ukrainian | fastest |
| 5 | `mlx-community/VibeVoice-ASR-5bit` | ~6 GB | better than 4-bit, marginally slower | fast |
| **6 (default)** | `mlx-community/VibeVoice-ASR-6bit` | ~7 GB | the best speed/quality knee-point | moderate |
| 8 | `mlx-community/VibeVoice-ASR-8bit` | ~9 GB | most accurate, hardware-bound | slowest |

The tuning constants in `asr.DEFAULT_GEN_KWARGS` (`repetition_penalty=1.2`, `repetition_context_size=64`, `temperature=0.1`) come from real-recording experiments where the 4-bit model otherwise looped on Ukrainian fragments like "шо я… шо я… шо я…". `chunk_duration=15` matched the same regime; longer chunks gave more loops, shorter chunks gave fragmented sentences.

`asr.py` looks up the model under `~/.cache/lm-studio/models/<repo>`. There is no implicit download — if a bitness is missing, the pipeline tells you to install it via LM Studio.

## Diarization — `pyannote/speaker-diarization-3.1`

We only consume `exclusive_diarization` (turns without overlap) because the merge stage is much simpler when a given moment belongs to exactly one speaker. The full `diarization` field is ignored.

Three Hugging Face repositories must be accepted with the user's account before pyannote can download weights:

- [`pyannote/speaker-diarization-3.1`](https://huggingface.co/pyannote/speaker-diarization-3.1)
- [`pyannote/segmentation-3.0`](https://huggingface.co/pyannote/segmentation-3.0)
- [`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1)

The token sits in `~/.cache/huggingface/token` (mode `600`). The pipeline never reads it from environment variables or settings files — see [ADR 0004](adr/0004-pyannote-for-diarization.md).

Inference runs on `mps` when available and falls back to `cpu` if the move fails. The `diarize` log line tells you which device was used.

## LLM — `mlx-community/gemma-3-12b-it-qat-4bit`

Default for every language task: identify, postprocess, structure, TL;DR. The pipeline does not require any specific model — anything OpenAI-compatible that LM Studio can serve will work — but the prompts are tuned for a model that:

- Handles Ukrainian fluently (Gemma 3 12B does; smaller multilingual models often regress to Russian).
- Honours `response_format={"type": "json_object"}` for structured tasks.
- Is willing to follow strict "output JSON only, no commentary" instructions.

Verified alternatives that fit within ~8.5 GB on disk and worked in tests:

- `mlx-community/gemma-3-12b-it-4bit` — same model, no QAT
- `mlx-community/gemma-2-9b-it-4bit` — older, smaller; faster but lower quality on Ukrainian

Override via `--llm-model "<repo_id>"` (and `--llm-base-url` if you run LM Studio on a different port or another machine).

[ADR 0001](adr/0001-local-llm-via-lm-studio.md) covers why LM Studio over ollama or `mlx-lm` directly; [ADR 0002](adr/0002-mlx-format-preference.md) covers MLX vs GGUF on Apple Silicon.
