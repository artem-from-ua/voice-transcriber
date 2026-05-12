# voice-transcriber

End-to-end local pipeline that turns an audio recording into a diarized Markdown transcript with a TL;DR — no cloud calls, no Anthropic API, no OpenAI. Speech recognition runs on VibeVoice-ASR (MLX), speaker diarization on pyannote 3.1, and all language tasks (speaker identification, ASR proof-reading, section structuring, TL;DR) run in-process via `mlx-lm` against a local MLX-quantised model.

```bash
uv run voice transcribe ~/recordings/meeting.m4a
```

The output is `~/recordings/meeting.md` with a metadata block, an optional TL;DR, and the dialogue split into thematic sections with emoji-tagged speakers.

## Prerequisites

- macOS on Apple Silicon
- [`ffmpeg`](https://ffmpeg.org/) in `PATH`
- [`uv`](https://github.com/astral-sh/uv) for dependency management
- [LM Studio](https://lmstudio.ai/) — for downloading models only; the server does not need to run. The pipeline loads the LLM in-process via `mlx-lm` and just reuses LM Studio's model cache at `~/.cache/lm-studio/models/`.
- A Hugging Face account with **accepted licenses** for the three gated pyannote repositories — see [`docs/troubleshooting.md`](docs/troubleshooting.md) under *GatedRepoError* for the exact list and instructions

## Setup

1. **Install dependencies**

   ```bash
   git clone git@github.com:artem-from-ua/voice-transcriber.git
   cd voice-transcriber
   uv sync
   ```

2. **Download models in LM Studio**

   In LM Studio → Models → search and download:
   - `mlx-community/VibeVoice-ASR-6bit` (default ASR; 4/5/8-bit variants supported via `--asr-bits`)
   - `mlx-community/gemma-3-12b-it-qat-4bit` (default LLM)

3. **Save your Hugging Face token**

   ```bash
   echo 'hf_xxx' > ~/.cache/huggingface/token
   chmod 600 ~/.cache/huggingface/token
   ```

4. **Download the Whisper ASR model** (default backend since v0.20.0)

   ```bash
   uv run voice download-whisper
   ```

   Fetches `mlx-community/whisper-large-v3-mlx` (~3 GB) into the standard HuggingFace cache. Required for `voice transcribe` to work out of the box. Pass `--asr-engine vibevoice` to fall back to the legacy backend (uses the VibeVoice model from step 2). See [ADR 0017](docs/adr/0017-whisper-asr-backend.md) for the rationale.

## Run

```bash
uv run voice transcribe path/to/audio.m4a              # default settings
uv run voice transcribe a.m4a --names "Artem,Ostap"    # override speaker names
uv run voice transcribe a.m4a --asr-engine vibevoice   # legacy backend, with in-band [Silence]/[Music] markers
uv run voice transcribe a.m4a --asr-engine vibevoice --asr-bits 8  # higher-quality VibeVoice variant
uv run voice transcribe a.m4a --no-tldr --no-structure # plain dialogue only
uv run voice transcribe a.m4a --verbose                # progress logs to stderr
```

## Onboarding

Each topic has one canonical home under [`docs/`](docs/). Read these in order before changing anything:

| Topic                              | Canonical source                                 |
| ---------------------------------- | ------------------------------------------------ |
| What runs end-to-end and in what order | [`docs/architecture.md`](docs/architecture.md), then [`docs/pipeline.md`](docs/pipeline.md) for per-stage detail |
| Every CLI flag                     | [`docs/cli.md`](docs/cli.md)                     |
| Shape of the produced Markdown     | [`docs/output-format.md`](docs/output-format.md) |
| ASR / diarization / LLM model defaults and overrides | [`docs/models.md`](docs/models.md) |
| LLM call contracts (temperature, JSON schema, safety nets) | [`docs/prompts.md`](docs/prompts.md) |
| Setup failures, gated repos, OOMs, repetition loops | [`docs/troubleshooting.md`](docs/troubleshooting.md) |
| Why each major choice was made     | [`docs/adr/README.md`](docs/adr/README.md)       |

If you find the same fact in two places under `docs/`, the canonical-source column wins; the other copy should link back rather than re-state.

## Development

```bash
uv run pytest -q     # unit tests; no live LLM or ffprobe required
```
