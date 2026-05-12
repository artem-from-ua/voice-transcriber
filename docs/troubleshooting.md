# Troubleshooting

## "LLM model not found at …" / "LLM model `…` is not in the HuggingFace cache"

The pipeline expected either an MLX checkpoint directory or a cached HuggingFace repo and found nothing.

- Fetch the default model once: `huggingface-cli download mlx-community/Qwen2.5-7B-Instruct-4bit` (~4 GB into `~/.cache/huggingface/hub/`). The pipeline resolves repo ids via that cache automatically.
- To use a model already extracted on disk (e.g. via LM Studio's GUI), pass `--llm-model /path/to/mlx/checkpoint` — any directory containing `config.json` plus the MLX weights works.
- See [ADR 0020](adr/0020-default-llm-qwen25-7b.md) for the resolver and `docs/models.md` for the table of recommended sampling per model.

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

The model's greedy decoder got stuck. The pipeline already enables `repetition_penalty=1.3` and a small chunk size, which mostly fixes it. If it still happens:

- Try `--asr-bits 8` (less aggressive quantisation → fewer pathological logits).
- Re-encode the input to clean WAV first if it's a heavily compressed file (e.g. low-bitrate MP3).
- For investigation only: run the affected segment through the standalone `mlx_audio.stt.generate` CLI with `--verbose` to see the loop in the log.

## JSON-constrained LLM call (`identify` / `structure`) still failed

Since v0.7.0 (ADR 0006) `MlxLLM.chat_json()` uses `lm-format-enforcer` as a logits processor against a JSON schema — output is guaranteed to be parseable on the first try, there is no retry loop. If a call still fails it means generation itself blew up (the model crashed mid-stream, hit an OOM, or the schema is unsatisfiable), not that the model returned text instead of JSON.

The pipeline degrades gracefully in both cases:

- `identify`: the affected cluster falls back to the `--unknown-speaker` policy (`ask` prompts on stdin; `keep` leaves `SPEAKER_XX`). No crash, but no name.
- `structure`: the whole dialogue collapses to one fallback section called `Розмова`. No crash.

If you see this repeatedly:

- Check for OOM / Metal allocator errors in the same run (see *"Out of memory" / Metal allocator errors* below) — exhaustion mid-generation is the most common cause.
- Try a different model: e.g. `--llm-model mlx-community/Qwen3-4B-Instruct-2507-4bit --llm-temperature 0.7 --llm-top-p 0.8 --llm-top-k 20` (see `docs/models.md` for the per-model recommended sampling values; passing the wrong values on the wrong model is a common cause of repeated `chat_json` validator misses).

## Speaker still labelled `SPEAKER_00` / `SPEAKER_01`

Self-introduction wasn't detected in the first ~60 s of that cluster's speech.

- The simplest fix: `--names "Artem,Ostap"` — overrides the LLM entirely.
- Or rerun with `--unknown-speaker ask` (default) and type the name when prompted.
- Otherwise rerecord with a clear "Привіт, я X" at the start.

## "Out of memory" / Metal allocator errors

The pipeline serialises three model loads (VibeVoice → pyannote → LLM) so they are never co-resident, but a single model still has to fit. A 12B LLM 4-bit is ~8 GB; combined with an 8-bit ASR (~9 GB) loaded simultaneously by mistake, you would exceed a 16 GB Mac.

- Stick to the default `Qwen2.5-7B-Instruct-4bit` LLM (~4 GB) — the project's default since v0.22.0 specifically because it fits on a 16 GB Mac end-to-end. See [ADR 0020](adr/0020-default-llm-qwen25-7b.md).
- Stick to the default Whisper ASR (~3 GB) or pick 6-bit VibeVoice. Avoid `--asr-bits 8`.
- If LM Studio is still running with its own model loaded in the background, quit it — its server is no longer needed at runtime (only the model cache is).
- Run with `--verbose` to print per-call MLX `pre`/`peak`/`post-clear` lines and prompt/output token counts. The numbers pinpoint which stage actually hits the ceiling.
- `gemma-3-12b` does **not** fit on a 16 GB Mac for the full pipeline. Even with v0.21's chunked structure + per-stage cleanup, the third chunked structure call OOMs once the allocator is fragmented from proofread. Stay with the default `Qwen2.5-7B` or upgrade to a 24 GB+ machine.
- When you switch `--llm-model` to a different family, also pass that model's recommended sampling — Qwen2.5's defaults (`temp=0.7, top_p=0.8, top_k=20, rep_penalty=1.05`) hurt structure output on gemma / Llama. See `docs/models.md` for the per-model table.

  ```bash
  # Qwen3-4B (smaller, ~2.5 GB, more sections in structure)
  voice transcribe a.m4a --llm-model mlx-community/Qwen3-4B-Instruct-2507-4bit \
      --llm-temperature 0.7 --llm-top-p 0.8 --llm-top-k 20

  # Llama-3.2-3B (smallest stable, only with rec params — default sampling breaks JSON)
  voice transcribe a.m4a --llm-model mlx-community/Llama-3.2-3B-Instruct-4bit \
      --llm-temperature 0.6 --llm-top-p 0.9
  ```

  If OOM still hits with `Qwen2.5-7B`, run with `--verbose` and file an issue with the memory trace so we can tune `STRUCTURE_CHUNK_SIZE`.

## ffmpeg / ffprobe not found

```text
RuntimeError: ffmpeg not found in PATH. Install ffmpeg.
```

`brew install ffmpeg`. The CLI invokes both binaries as subprocesses.

## Permission error writing the Markdown output

The default output path is `<audio_basename>.md` next to the input file. If that directory is read-only (e.g. inside a cloud-mounted folder), pass `-o /absolute/writable/path.md`.
