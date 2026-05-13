# Troubleshooting

## "LLM model not found at …" / "LLM model `…` is not in the Hugging Face cache"

The pipeline expected either an MLX checkpoint directory or a cached Hugging Face repo and found nothing.

- Fetch the default model once: `huggingface-cli download mlx-community/Qwen2.5-7B-Instruct-4bit` (~4 GB into `~/.cache/huggingface/hub/`). The pipeline resolves repo ids via that cache automatically.
- To use a model already extracted on disk (e.g. via LM Studio's GUI), pass `--llm-model /path/to/mlx/checkpoint` — any directory containing `config.json` plus the MLX weights works. (LM Studio GUI path is deprecated — see [ADR 0006](adr/0006-mlx-lm-over-lm-studio.md) and [#116](https://github.com/artem-from-ua/voice-transcriber/issues/116))
- See [ADR 0020](adr/0020-default-llm-qwen25-7b.md) for the resolver and `docs/models.md` for the table of recommended sampling per model.

## "Whisper model … is not in the Hugging Face cache"

`voice transcribe` checked `~/.cache/huggingface/hub/models--mlx-community--whisper-large-v3-mlx/` and found nothing.

- Run `voice download-whisper` once to fetch it (~3 GB). The pipeline never auto-downloads — that is intentional, so a 3 GB network fetch cannot start in the middle of a transcription run.

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

Whisper's decoder occasionally loops on near-silent or very repetitive audio. The pipeline already disables `condition_on_previous_text` (which is Whisper's biggest repetition-loop driver) and lets `mlx-whisper` run its internal temperature schedule, so this should be rare on Whisper.

- Re-encode the input to clean WAV first if it's a heavily compressed file (e.g. low-bitrate MP3).
- Check whether the clearspeech chain is dropping signal: try `--clearspeech-chain ""` to bypass preprocessing and see if the loop disappears.
- For investigation only: run `mlx_whisper.transcribe(...)` directly on the WAV with `verbose=True` to see the per-window log.

## Auto-detect picks the wrong language

Symptom: the transcript header shows `🌐 Мова: ru` (or `pl`, `de`, …) on what you know is a Ukrainian recording, the ASR text reads like transliterated Russian (`Подошёл` instead of `Подойшов`), and the LLM stages produce a TL;DR in the wrong language.

Cause: `[4] lang_detect` runs Whisper's `model.detect_language()` over the longest pyannote turn, which is normally a 5+ second stretch of one speaker with enough lexical content to disambiguate Slavic languages. When the longest turn is unusually short (sub-2-second sound effects, a single "ага", a clap), Whisper's classifier has too little signal and picks the wrong language. See [ADR 0022](adr/0022-asr-language-autodetect.md) for why the longest pyannote turn is normally the best signal we have.

Fix:

- Re-run with `--language uk` (or whichever ISO code matches the audio). This skips `[4] lang_detect` entirely and pins the language hint for ASR and all LLM/render stages.
- If the recording itself is short and you expect to use only one language, just always pass `--language` — it costs nothing and removes the 3-5 s detect step.

The pipeline never errors on a wrong detect — it produces an incorrect transcript instead. Re-run with `--language` is the only recovery; there is no smart fallback.

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

The pipeline serialises three model loads (Whisper → pyannote → LLM) so they are never co-resident, but a single model still has to fit. A 12B LLM 4-bit is ~8 GB; that is the realistic upper bound on a 16 GB Mac.

- Stick to the default `Qwen2.5-7B-Instruct-4bit` LLM (~4 GB) — the project's default since v0.22.0 specifically because it fits on a 16 GB Mac end-to-end. See [ADR 0020](adr/0020-default-llm-qwen25-7b.md).
- Whisper (~3 GB) is the only ASR backend since v0.23.0 — see [ADR 0021](adr/0021-remove-vibevoice-backend.md).
- If LM Studio is still running with its own model loaded in the background, quit it — its server is no longer needed at runtime (only the model cache is). (deprecated — see [ADR 0006](adr/0006-mlx-lm-over-lm-studio.md))
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

## Running long LLM-stage benchmarks from an agent / unattended

`voice transcribe` is a slow command (3–10 min for a 6-minute recording) whose only signal of life is the on-TTY `rich.Progress` bar and the per-call `llm.chat: …` log lines. When the run is kicked off **from an agent or wrapper that doesn't render the TTY** (Claude Code's Bash tool, a CI runner, a custom script that captures stdout), the user can't see the bar — there is no built-in heartbeat that surfaces "where am I in the pipeline" outside that bar.

The fix is to run the transcribe in the background and pair it with a small monitor that polls the `tee`'d log file every ~10 s and prints the latest progress marker. Two concurrent shells, one writes, one reads:

```bash
# 1. Kick off the transcribe in the background, tee'ing to a log file.
voice transcribe path/to/audio.m4a --proofread --dump-stages /tmp/run/ \
    --verbose 2>&1 | tee /tmp/run.log &

# 2. In a separate shell (or via the agent's parallel-process tool), poll
#    the log file. Emit one line every ~10 s with the latest marker, exit
#    when the foreground process is gone.
while pgrep -f "voice transcribe.*path/to/audio.m4a" >/dev/null 2>&1; do
  line=$(grep -E "Proofread:|✓ \[[0-9]+/13\]|Traceback|Error" /tmp/run.log \
         | tail -1)
  echo "[$(date +%H:%M:%S)] ${line:-(no marker yet)}"
  sleep 10
done
```

Rules of thumb for that monitor:

- **Match the markers, not the raw log.** `grep -E "Proofread:|✓ \[[0-9]+/13\]|Traceback|Error"` covers the per-stage `· [N/13]` heartbeats, the `✓ [N/13] … in Xs` completions, and the failure paths. Don't `tail -f` the whole log — it floods the agent's chat with per-token memory lines that contain no actionable progress.
- **Always cover failure signatures.** Silence is not success: a monitor that only matches `✓` lines stays mute through a Traceback. Include `Traceback|Error|AssertionError` so a crash surfaces immediately.
- **Use 10 s as the default cadence.** Faster than that floods the chat; slower than that leaves the user wondering whether the run is alive (the in-process LLM bar refreshes at ≥1 Hz on a real TTY, so 10 s is already an order of magnitude slower than the native UX).
- **Prefix the monitor description with `⌛️`** so the user can spot at a glance in the agent's chat which event lines come from a periodic timer vs. from real interactive work. Example: `⌛️ feature run progress 10s + completion`. The hourglass is reserved for these timer-driven monitors; don't use it for one-shot notifications.
- **Stop the monitor when the foreground process exits.** Don't leave a `tail -f` armed — it never returns on its own. The `pgrep`-based `while` loop above exits cleanly when `voice transcribe` finishes.

This same recipe applies to any other long-running LLM benchmark in the repo (`uv run python scripts/<...>` that loads a model and iterates over inputs). Anything that runs more than ~30 s without surface output should be paired with a monitor.
