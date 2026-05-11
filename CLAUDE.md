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

## Pull-request hygiene

### Always base off `main`, never another feature branch

Stacked PRs (PR-B with `base=feature/PR-A`) have a quiet failure mode: when PR-A merges, GitHub marks PR-B as MERGED too, but its commits don't land on `main` — they live only inside the now-merged feature branch. We hit this exactly once (#6 → #7 dance) and lost the work for a few minutes. Avoid it: every PR opens against `main`.

If a change naturally needs another PR to land first, **wait for the dependency to merge** and only then open the next one. Smaller feature scope per PR is the price.

### Version bumps in parallel PRs

`pyproject.toml` is the natural merge-conflict file when two PRs each pick a "next" version. Two ways to keep this calm:

- Only **one** open PR at a time touches code in practice. If you're shipping multiple, hold the second's version bump until the first merges, then rebase and bump from the freshly-merged version.
- Or accept the conflict — `git checkout --theirs uv.lock && uv lock` plus a one-line edit to `pyproject.toml` is a 10-second resolution. Just don't be surprised when it happens.

We've also gone version skipping in past PRs (0.4.0 was reserved for a PR that landed second, so we shipped 0.5.0 after 0.3.0). That's fine for a pre-1.0 project but spell it out in the CHANGELOG entry so a future reader doesn't think a release went missing.

## Editing the architecture diagram

`docs/architecture.md` has two PlantUML diagrams whose edges encode the actual data-flow:

- **Solid red arrows = audio bytes**. **Solid blue = structured data**. **Solid light-green (`#7CCD7C`) = Markdown**. **Dotted = library/file dependency**.
- Stages are numbered `[1]…[10]` matching the runtime log labels (`[6/9] ASR-постобробка` etc.).

Before changing an edge, grep the code:

```bash
grep -nE "^\s*(asr_module|diarize_module|merge|postprocess|identify|structure|tldr|render)" src/voice/pipeline.py
```

The stage that *appears* to feed the next one in the file order isn't always the truth — for example, ASR and diarize both read the same `wav_path` written by ffmpeg, so their incoming edge is from `[2] WAV`, not from `pipeline`. Match the diagram to the code, not to intuition.

The PlantUML pre-commit hook regenerates the embedded SVG URL from the source block. Don't hand-edit the URL; edit the source and let the hook resync.
