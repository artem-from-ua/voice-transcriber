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
- [LM Studio](https://lmstudio.ai/) installed (used only to download models into its cache; no server needed at runtime)
- A Hugging Face account with **accepted licenses** for:
  - [`pyannote/speaker-diarization-3.1`](https://huggingface.co/pyannote/speaker-diarization-3.1)
  - [`pyannote/segmentation-3.0`](https://huggingface.co/pyannote/segmentation-3.0)
  - [`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1)

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

## Run

```bash
uv run voice transcribe path/to/audio.m4a              # default settings
uv run voice transcribe a.m4a --names "Artem,Ostap"    # override speaker names
uv run voice transcribe a.m4a --asr-bits 8             # higher-quality ASR
uv run voice transcribe a.m4a --no-tldr --no-structure # plain dialogue only
uv run voice transcribe a.m4a --verbose                # progress logs to stderr
```

Full CLI reference: [`docs/cli.md`](docs/cli.md). Output format: [`docs/output-format.md`](docs/output-format.md). When something breaks: [`docs/troubleshooting.md`](docs/troubleshooting.md). Architecture overview: [`docs/architecture.md`](docs/architecture.md).

## Development

```bash
uv run pytest -q     # unit tests; no live LLM or ffprobe required
```

All architectural decisions live in [`docs/adr/`](docs/adr/).
