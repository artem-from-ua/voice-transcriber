# Troubleshooting

## "LLM model not found at …"

The pipeline expected an MLX checkpoint directory and found nothing (or no `config.json` inside).

- Open LM Studio → **Models** → search for the default model (`mlx-community/gemma-3-12b-it-qat-4bit`) and download it. The pipeline reads the same cache LM Studio uses; the LM Studio server itself does not need to be running.
- To point at a different local model directory, pass `--llm-model /path/to/mlx/checkpoint`.

## "VibeVoice-ASR-Nbit not found at …"

The model directory doesn't exist under `~/.cache/lm-studio/models/mlx-community/`.

- Download the matching variant in LM Studio (search "VibeVoice-ASR" → pick the bitness).
- If you already downloaded it manually, make sure it's at exactly that path; the pipeline only looks at LM Studio's cache.
- The default is 6-bit. To switch: `--asr-bits 4|5|6|8`.

## "HF token not found at ~/.cache/huggingface/token"

pyannote needs your Hugging Face token to fetch the model. Create the file:

```bash
echo 'hf_xxx' > ~/.cache/huggingface/token
chmod 600 ~/.cache/huggingface/token
```

The pipeline does not read the token from environment variables, `.env`, or `settings.local.json` — only from this file.

## `GatedRepoError: Access to model … is restricted`

Each pyannote repo needs the license to be accepted with your Hugging Face account. Open these three pages while logged in and click "Agree and access":

- `https://huggingface.co/pyannote/speaker-diarization-3.1`
- `https://huggingface.co/pyannote/segmentation-3.0`
- `https://huggingface.co/pyannote/speaker-diarization-community-1`

## Repetition loops in ASR output ("шо я… шо я… шо я…")

The model's greedy decoder got stuck. The pipeline already enables `repetition_penalty=1.2` and a small chunk size, which mostly fixes it. If it still happens:

- Try `--asr-bits 8` (less aggressive quantisation → fewer pathological logits).
- Re-encode the input to clean WAV first if it's a heavily compressed file (e.g. low-bitrate MP3).
- For investigation only: run the affected segment through the standalone `mlx_audio.stt.generate` CLI with `--verbose` to see the loop in the log.

## "LLM returned invalid JSON after 2 attempt(s)"

The LLM failed to produce valid JSON twice in a row for an identify or structure call.

- `identify`: the affected cluster will fall back to the `--unknown-speaker` policy. No crash, but no name.
- `structure`: the whole dialogue collapses to one fallback section called `Розмова`. No crash.
- Likely cause: the model isn't following the `response_format` hint. Swap to a more capable model: `--llm-model mlx-community/gemma-3-12b-it-4bit` (or QAT variant).

## Speaker still labelled `SPEAKER_00` / `SPEAKER_01`

Self-introduction wasn't detected in the first ~60 s of that cluster's speech.

- The simplest fix: `--names "Artem,Ostap"` — overrides the LLM entirely.
- Or rerun with `--unknown-speaker ask` (default) and type the name when prompted.
- Otherwise rerecord with a clear "Привіт, я X" at the start.

## "Out of memory" / Metal allocator errors

The pipeline serialises three model loads (VibeVoice → pyannote → LLM) so they are never co-resident, but a single model still has to fit. A 12B LLM 4-bit is ~8 GB; combined with an 8-bit ASR (~9 GB) loaded simultaneously by mistake, you would exceed a 16 GB Mac.

- Stick to the default 6-bit ASR.
- If LM Studio is still running with its own model loaded in the background, quit it — its server is no longer needed at runtime (only the model cache is).
- Downgrade to `mlx-community/gemma-2-9b-it-4bit` for the LLM (`--llm-model …`).

## ffmpeg / ffprobe not found

```text
RuntimeError: ffmpeg not found in PATH. Install ffmpeg.
```

`brew install ffmpeg`. The CLI invokes both binaries as subprocesses.

## Permission error writing the Markdown output

The default output path is `<audio_basename>.md` next to the input file. If that directory is read-only (e.g. inside a cloud-mounted folder), pass `-o /absolute/writable/path.md`.
