# Working notes for AI assistants and contributors

This file collects project-specific conventions that are not obvious from the code alone. Keep it short; if a note grows long enough to need a heading of its own, move it into `docs/` and link to it from here.

## Running the pipeline interactively

`voice transcribe` is slow (3–10 minutes for a typical recording) and its on-purpose progress UI — `rich.Progress` bars on a TTY, a 15-second heartbeat otherwise — is the user's only signal that the run is still alive. **Never** silence it.

When you need to keep a log file but still want progress on screen, pipe through `tee` instead of redirecting:

```bash
voice transcribe path/to/audio.m4a --verbose 2>&1 | tee /tmp/run.log
```

Not this:

```bash
voice transcribe path/to/audio.m4a --verbose 2> /tmp/run.log    # silent for minutes
```

This applies equally to background invocations from agentic tools (Claude Code's Bash tool, scripts, etc.) — keep `tee` in the command so the user's terminal stays informative.

## Troubleshooting a regression

Run with `--dump-stages DIR` to write one JSON file per stage (`01-meta.json` … `09-tldr.txt`). Diffing `02-asr.json` against `05-postprocess.json` is the fastest way to tell whether a bad output came from the raw ASR or the LLM proof-reader. See [`docs/architecture.md`](docs/architecture.md) for the stage map.

## Dependencies

- Always install with `uv` (`uv add`, `uv run`), never with `pip` directly — the lockfile and `[tool.uv]` settings depend on it.
- Prefer MLX builds of models (`mlx-community/*`) on Apple Silicon. GGUF only when MLX is not available.

## Versioning

SemVer. The version bumps every PR with code changes (see `pyproject.toml` + `CHANGELOG.md`). Docs-only PRs do not bump.
