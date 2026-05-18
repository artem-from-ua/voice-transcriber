# voice-transcriber

End-to-end local pipeline that turns an audio recording into a diarized Markdown transcript with a TL;DR — no cloud calls, no Anthropic API, no OpenAI. Speech recognition runs on Whisper-large-v3-MLX, speaker diarization on pyannote 3.1, and all language tasks (speaker identification, ASR proofreading, section structuring, TL;DR) run in-process via `mlx-lm` against a local MLX-quantised model.

```bash
uv run voice transcribe ~/recordings/meeting.m4a
```

The output is `~/recordings/meeting.md` with a metadata block, an optional TL;DR, and the dialogue split into thematic sections with emoji-tagged speakers.

> [!NOTE]
> Jump to: [Prerequisites](#prerequisites) · [Setup](#setup) · [Run](#run) · [Onboarding](#onboarding) · [Development](#development)

## Prerequisites

- macOS on Apple Silicon
- [`ffmpeg`](https://ffmpeg.org/) in `PATH`
- [`uv`](https://github.com/astral-sh/uv) for dependency management
- [`huggingface-cli`](https://huggingface.co/docs/huggingface_hub/guides/cli) (ships with `huggingface_hub`, pulled in by `uv sync`) — for fetching the default LLM and Whisper ASR weights into `~/.cache/huggingface/hub/`. [LM Studio](https://lmstudio.ai/) is **optional** and only useful if you want to manage local LLM checkpoints through a GUI — its server never needs to run. (deprecated — see [ADR 0006](docs/adr/0006-mlx-lm-over-lm-studio.md); a first-class `voice models` CLI is tracked in [#116](https://github.com/artem-from-ua/voice-transcriber/issues/116))
- A Hugging Face account with **accepted licenses** for the three gated pyannote repositories — see [`docs/troubleshooting.md`](docs/troubleshooting.md) under *GatedRepoError* for the exact list and instructions

## Setup

1. **Install dependencies**

   ```bash
   git clone git@github.com:artem-from-ua/voice-transcriber.git
   cd voice-transcriber
   uv sync
   ```

2. **Save your Hugging Face token**

   ```bash
   echo 'hf_xxx' > ~/.cache/huggingface/token
   chmod 600 ~/.cache/huggingface/token
   ```

3. **Download the default models** (~7 GB total into the Hugging Face cache)

   ```bash
   uv run voice download-whisper                                          # ~3 GB
   huggingface-cli download mlx-community/Qwen2.5-7B-Instruct-4bit        # ~4 GB
   ```

   - `Whisper-large-v3-MLX` is the only ASR backend (see [ADR 0017](docs/adr/0017-whisper-asr-backend.md) (superseded) for how it was chosen, [ADR 0021](docs/adr/0021-remove-vibevoice-backend.md) for why it is now the sole backend).
   - `Qwen2.5-7B-Instruct-4bit` is the default LLM since v0.22.0 (see [ADR 0020](docs/adr/0020-default-llm-qwen25-7b.md)). It is the smallest model that produces multi-section structure reliably on a 16 GB Mac. Use `--llm-model <other-repo-or-path>` to swap in another MLX-format LLM.

## Run

```bash
uv run voice transcribe path/to/audio.m4a              # default: Whisper detects the language on the longest pyannote turn
uv run voice transcribe a.m4a --language uk            # skip auto-detect; pin language explicitly (ADR 0022)
uv run voice transcribe a.m4a --names "Alice,Bob"      # override speaker names
uv run voice transcribe a.m4a --no-tldr --no-structure # plain dialogue only
uv run voice transcribe a.m4a --safe-speech-topics health,drugs  # redact sensitive utterances
uv run voice transcribe a.m4a --user-context "Phone interview between two software engineers about ML deployments"  # seed every LLM stage with a per-run context line (ADR 0033)
uv run voice transcribe a.m4a --llm-proofread-model …  # smaller model on proofread, default on the rest
uv run voice transcribe a.m4a --verbose                # progress logs to stderr (also turns on per-LLM-call memory lines)
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
