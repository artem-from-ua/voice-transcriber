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
| `--language` | str (ISO) | auto-detect | Conversation language hint. Default: detected at stage `[4] lang_detect` on the longest pyannote turn via Whisper's `model.detect_language()` ([ADR 0022](adr/0022-asr-language-autodetect.md)). Pass an explicit value (e.g. `--language uk`) to skip detection — useful when the diarized recording has only very short turns or when you already know the language and want to save the 3-5 s detect step. The override flows through to ASR, proofread, identify_speakers, speech_structure, speech_summary and render. |
| `--unknown-speaker` | `{ask,keep}` | `ask` | What to do when self-intro is missing. `ask` prompts on stdin; `keep` leaves `SPEAKER_XX`. |
| `--names` | `"A,B,..."` | – | Override automatic naming. Mapped to clusters in order of first appearance. Skips the LLM identify step. |
| `--datetime` | ISO 8601 | – | Override recording start time. Defaults to `ffprobe creation_time`, then `stat birthtime`, then `stat mtime`. |
| `--llm-model` | path or HF repo-id | `mlx-community/Qwen2.5-7B-Instruct-4bit` | Hugging Face `org/repo` id **or** a filesystem path to an MLX model directory, used as the default for all four LLM stages. Repo ids are resolved via `huggingface_hub.try_to_load_from_cache`; missing repos fail fast with the `huggingface-cli download` command to run. See [ADR 0020](adr/0020-default-llm-qwen25-7b.md). |
| `--llm-temperature` | float | `0.7` (Qwen2.5 rec) | Sampling temperature applied to all LLM stages. Overrides the per-prompt frontmatter value. When switching `--llm-model`, also pass this and the other sampling flags to match that model's recommendation — see `docs/models.md`. |
| `--llm-top-p` | float | `0.8` (Qwen2.5 rec) | Nucleus-sampling cutoff applied globally. |
| `--llm-top-k` | int | `20` (Qwen2.5 rec) | Top-K sampling cutoff (0 disables). |
| `--llm-repetition-penalty` | float | `1.05` (Qwen2.5 rec) | Repetition penalty for plain-text stages (proofread, speech_summary). |
| `--llm-proofread-model` | path | inherits `--llm-model` | Per-stage override: model used for proofread only. The pipeline cold-reloads weights between stages whose paths differ; identical paths share one resident instance. See [ADR 0019](adr/0019-per-stage-llm-models.md). |
| `--llm-identify-model` | path | inherits `--llm-model` | Per-stage override for the speaker-identify stage. |
| `--llm-structure-model` | path | inherits `--llm-model` | Per-stage override for the section-structuring stage. |
| `--llm-tldr-model` | path | inherits `--llm-model` | Per-stage override for the TL;DR stage. |
| `--output`, `-o` | path | `<audio>.md` | Output Markdown path. |
| `--proofread` | flag | off (default since v0.29.0) | **Opt-in** per-segment ASR proof-reading via LLM. Default is OFF — measurement on Whisper output showed the stage hurts more than it helps on Ukrainian conversational speech (see [ADR 0026](adr/0026-proofread-default-off.md) and [`docs/benchmarks/proofread-hit-rate.md`](benchmarks/proofread-hit-rate.md)). Pre-v0.29.0 users of `--no-proofread` should simply drop the flag. |
| `--no-tldr` | flag | off | Skip TL;DR generation. |
| `--no-structure` | flag | off | Skip LLM-driven sectioning; output is one section "Розмова". |
| `--no-safe-speech` | flag | off | Skip sensitive-content redaction entirely. |
| `--llm-safe-speech-model` | path or HF repo-id | inherits `--llm-model` | Per-stage override: model used for safe_speech redaction only. |
| `--safe-speech-topics` | CSV | `health,drugs,alcohol` | Comma-separated list of topics to redact. Pass `""` to disable redaction. Examples: `health`, `drugs`, `alcohol`, `legal`, `finance`. |
| `--safe-speech-policy` | `placeholder` \| `drop` | `placeholder` | How to handle flagged utterances. `placeholder` replaces the range with `[muted, X.Xs]`; `drop` removes the segments silently (section header is kept). See [ADR 0024](adr/0024-safe-speech-defaults.md). |
| `-v`, `--verbose` | flag | off | Stream stage progress to stderr. |

## Examples

```bash
# 1) Default — full pipeline, asks for any unidentified speaker
uv run voice transcribe ~/recordings/standup.m4a -v

# 2) Skip LLM identify; force the order Artem → Ostap
uv run voice transcribe call.m4a --names "Artem,Ostap"

# 3) Force English transcript (skip the [4] lang_detect probe)
uv run voice transcribe interview.wav --language en

# 4) Minimal output (plain dialogue), no TL;DR, no sectioning
uv run voice transcribe quick.mp3 --no-tldr --no-structure --unknown-speaker keep

# 7) Redact health and drug mentions (placeholder policy)
uv run voice transcribe meeting.m4a --safe-speech-topics health,drugs

# 8) Drop sensitive segments entirely instead of replacing with [muted]
uv run voice transcribe meeting.m4a --safe-speech-topics health --safe-speech-policy drop

# 9) Disable redaction for a plain unfiltered transcript
uv run voice transcribe meeting.m4a --no-safe-speech

# 5) Custom output path and a different local LLM (with that model's recommended sampling)
uv run voice transcribe a.m4a \
  -o ~/notes/2026-05-11-call.md \
  --llm-model mlx-community/Qwen3-4B-Instruct-2507-4bit \
  --llm-temperature 0.7 --llm-top-p 0.8 --llm-top-k 20

# 6) Keep the v0.21 default LLM (gemma-3-12b) on a machine with ≥24 GB unified memory
# (deprecated path — LM Studio cache; see ADR 0006 and #116 for the managed-LLM CLI)
uv run voice transcribe a.m4a \
  --llm-model ~/.cache/lm-studio/models/mlx-community/gemma-3-12b-it-qat-4bit \
  --llm-temperature 1.0 --llm-top-p 0.95 --llm-top-k 64 --llm-repetition-penalty 1.0
```

## Exit codes

- `0` — success; the path of the produced Markdown is printed on stdout
- `1` — any caught exception during the run; the message goes to stderr
- `2` — `argparse` rejected the invocation (e.g. unknown flag, bad type)
