# voice-transcriber

End-to-end local pipeline that turns an audio recording into a diarized Markdown transcript with a TL;DR — no cloud calls, no Anthropic API, no OpenAI. Speech recognition runs on Whisper-large-v3-MLX, speaker diarization on pyannote 3.1, and all language tasks (speaker identification, ASR proofreading, section structuring, TL;DR) run in-process via `mlx-lm` against a local MLX-quantised model.

```bash
uv run voice transcribe ~/recordings/meeting.m4a
```

The output is `~/recordings/meeting.md` with a metadata block, an optional TL;DR, and the dialogue split into thematic sections with emoji-tagged speakers.

> [!NOTE]
> Jump to: [Pipeline](#pipeline) · [Prerequisites](#prerequisites) · [Setup](#setup) · [Run](#run) · [Onboarding](#onboarding) · [Development](#development)

## Pipeline

<a href="docs/pipeline.md"><img src="https://www.plantuml.com/plantuml/svg/lLXjJoCt4Fw-ls9qV6bLDa02uKNfwdYB3WcaArnx7wuHSdOdmSgklR8TEAwgrFw7KtzCVyxzaewzcxTb7G879H2nyVXvnZFZn-EyrOOfCyxIm72J8jnA7cDe51CwLhoF2hxzzHLodcFA1G9P3r47UiH5pXJB89PPBWKASsNkQRh2s30nJ77Ev50fUVVXXiSZWLf3SsuenI6Av4Yg1DMnJAK2nqo3XFZGZiMZeP9ZaHIsQwVk9mWwsf970K0Yut56S-4FUW2uO6h81JHtwEbF-cEneNd5s3cjP-RLzhpzZdjupvA4Yo4KapiR9KbGWBoX5zFmgqOL1DCem9jEF_gqrIlM4Si4Enlzw7VHuL5dCCXo74QT3HwvgHyFVwCbxDbN5Q3P0pPEkBpEqB1vX5p9FiuWkbqVGgi72MUAVy5hdICReT8pdFoU2I4DXeuaWj8YB6OmmwK8MusjASq9VGvhUxl7fzt3Aj5s3rBA-7M2Xd8_uyZIp-7T3ITe5S77ufymfVXYUDkQ8f_Jr1CqiBQ_UNgwlra5t-b1d29JTWqOtypGjQs20QTyKA2C738kB6Ov6FFyFeAEZlfC93dJj7GTrOtd2ZMA61V7oqHPWmI5v1fzywbiyHPLVpBkrdHyvKjAkmV5Gfe6FpyvLeTNbxuuxVOWGpAtaCxsmZxvtDZh0TMH72XaoQ8JQHR1BtSecesTSFSeF1FQgB7DQkd2UwafLztkyN6xNKNTxm6dS1a-kAyfkLNGp-nq7_pMvYHuryMqiSAdmL-vHoy-RLhD1lYWr5Q2febUuERCyN6MCNN_DEmFvoUNNcpQvYILL8RJp5-wnsdg0ojG47uYWFPHQwSNHbOq3g0JMmazMDlUjAy82hq1kwKw6sxCkbWr-k6EtQv3jiteBMz1ez7uSM9T8Let4sdzsP5QiNjDrjRHNhFRNa-Oli744awKfOQqo8QJhJ4RBcMMkHL3beAaRQwelbTWBigudjVk2X0iG_tIhAu4wqXHQ6xuBOAHEO-Vo-idTffGGpos4s8f2gbxdFHx8liANALQ_4zadkIz0fMJanUb2kpUzcvyrplKy6srOVJkonAcJ0ffPMFAiq3po1R_JlJg8kX8Vn4I5unfLXwNAIiNI_0Dpz3sc7e9GRthpy6UkBe0RviIhyGkWQ-ARq1-UiFl8R6jeS1a2OixNcQ5Qd68f7swVxhLhP8KBLCD3Trfv51G9xsCQNxRLBAmCj5paX8KQDvL-TJP-Wjtt2pBqUIyIbaVqncbW_GIUZNQjPoXxJaNWrLCcE8jrzwdkBrmp5Pfgcxx5FZvXEKY5dQBbyKUpNHKnrPRvo2ikTTbdv8_uSHNcD5HdyogYNh31OVBRjApAvIx7oVCuFeIacIit9Look8SzoxBJ_aJJ1le4gcqzMAvKx0ktkNt6xbfP7EhBxvci663cAg5RlO0dJwBHmVbvPUc0dNQz0bT_wX_NyGzF2F5Jp0zWBcBz0JTzlCpkE8QjohSmgMn3gwu7QmATtv-KOKfJGlvZRvz_Tl-_lFlnAm6Ra5QAp15_Qm4Ox_We09Y7gGYhIiXpN8A3m9bireaDtXBR-Ci_Gy0" alt="Pipeline stages" height="600"></a>

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
