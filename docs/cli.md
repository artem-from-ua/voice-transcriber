# CLI reference

```
voice transcribe <audio> [options]
```

`voice` is a single-subcommand CLI registered via `pyproject.toml` (`project.scripts`). It's intended to be invoked through `uv run`.

## Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `audio` | path | required | Input audio file. Anything ffmpeg can decode (m4a, mp3, wav, flac, …). |

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--language` | str | `uk` | Conversation language. Affects identify / proofread / structure / tldr prompts. |
| `--asr-bits` | `{4,5,6,8}` | `6` | VibeVoice quantisation; see [`models.md`](models.md). |
| `--unknown-speaker` | `{ask,keep}` | `ask` | What to do when self-intro is missing. `ask` prompts on stdin; `keep` leaves `SPEAKER_XX`. |
| `--names` | `"A,B,..."` | – | Override automatic naming. Mapped to clusters in order of first appearance. Skips the LLM identify step. |
| `--datetime` | ISO 8601 | – | Override recording start time. Defaults to `ffprobe creation_time`, then `stat birthtime`, then `stat mtime`. |
| `--llm-model` | path | `~/.cache/lm-studio/models/mlx-community/gemma-3-12b-it-qat-4bit` | Filesystem path to an MLX model directory (anything `mlx_lm.load()` accepts). |
| `--output`, `-o` | path | `<audio>.md` | Output Markdown path. |
| `--no-proofread` | flag | off | Skip per-segment ASR proof-reading. |
| `--no-tldr` | flag | off | Skip TL;DR generation. |
| `--no-structure` | flag | off | Skip LLM-driven sectioning; output is one section "Розмова". |
| `-v`, `--verbose` | flag | off | Stream stage progress to stderr. |

## Examples

```bash
# 1) Default — full pipeline, asks for any unidentified speaker
uv run voice transcribe ~/recordings/standup.m4a -v

# 2) Skip LLM identify; force the order Artem → Ostap
uv run voice transcribe call.m4a --names "Artem,Ostap"

# 3) High-quality ASR pass, English transcript
uv run voice transcribe interview.wav --asr-bits 8 --language en

# 4) Minimal output (plain dialogue), no TL;DR, no sectioning
uv run voice transcribe quick.mp3 --no-tldr --no-structure --unknown-speaker keep

# 5) Custom output path and a different local LLM
uv run voice transcribe a.m4a \
  -o ~/notes/2026-05-11-call.md \
  --llm-model ~/.cache/lm-studio/models/mlx-community/gemma-3-12b-it-4bit
```

## Exit codes

- `0` — success; the path of the produced Markdown is printed on stdout
- `1` — any caught exception during the run; the message goes to stderr
- `2` — `argparse` rejected the invocation (e.g. invalid `--asr-bits`)
