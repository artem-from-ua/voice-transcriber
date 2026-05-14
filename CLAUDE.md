# Working notes for AI assistants and contributors

This file collects project-specific conventions that are not obvious from the code alone. Keep it short; if a note grows long enough to need a heading of its own, move it into `docs/` and link to it from here.

## Project goal

Every technical decision in this repo should serve the user's goal stated here. When two design options exist, the one that better satisfies these targets wins.

- **Audience and content.** A command-line tool that transcribes lively, friendly Ukrainian conversations — debates, arguments, dialogues — with English technical terms mixed in (`Hugging Face`, `Gradio`, `Claude Code`, etc.). The transcript must preserve speaker turns, the conversational tone, and code-switched English verbatim.
- **Quality over speed (within the hardware budget).** Given the hardware constraint below, quality is the primary lever. We pick the strongest model that fits and tune for fidelity, not throughput. Proofreading, structuring, and rendering all exist to make the final document trustworthy — never trade them away for a faster default.
- **Hardware floor.** Apple Silicon Mac (M1, 16 GB unified memory). Both CPU and the integrated Apple Silicon GPU share that 16 GB pool — MLX leans on the GPU heavily (peak MLX memory in logs == GPU-resident memory), so "GPU-first" is the default for any model that runs through MLX. What we do **not** assume is a discrete GPU, more than 16 GB unified memory, or workloads that need to push beyond what fits alongside ASR + LLM on this machine. MLX-first ([ADR 0002](docs/adr/0002-mlx-format-preference.md)), in-process ([ADR 0006](docs/adr/0006-mlx-lm-over-lm-studio.md)). All defaults must run on this machine without swap thrashing or OOM.
- **Speed floor: faster than real-time.** End-to-end transcription must finish faster than the audio's own duration on the reference hardware. A 6-minute recording finishes in under 6 minutes. This floor is set with future live / streaming audio processing in mind — when streaming lands, anything slower than real-time would back up forever. Stage-level wall-clock budgets (`docs/pipeline.md`) follow from this.
- **Output shape.** The result is a structured Markdown document, readable both by humans and by downstream LLMs: clear speaker labels, time-stamped sections, clean proper-noun handling, and a TL;DR. "Just the raw ASR" is not the product — the proofread + structure + summary stages are part of the deliverable.

When introducing a feature or changing a default, sanity-check it against this list. If it's a clear win on one axis at the cost of another (e.g. +10 % quality for +50 % wall-clock that breaks the real-time floor on M1/16 GB) — that's a regression, not an improvement.

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

Run with `--dump-stages DIR` to write one file per stage (`01-meta.json` … `10-speech_summary.txt`; `09-safe_speech-decisions.json` is added when `--safe-speech-topics` is active). Diffing `03-asr.json` against `05-proofread.json` is the fastest way to tell whether a bad output came from the raw ASR or the LLM proof-reader. See [`docs/architecture.md`](docs/architecture.md) for the stage map.

## Dependencies

- Always install with `uv` (`uv add`, `uv run`), never with `pip` directly — the lockfile and `[tool.uv]` settings depend on it.
- Prefer MLX builds of models (`mlx-community/*`) on Apple Silicon. GGUF only when MLX is not available.

## Pytest output

`uv run pytest -q` ends with two `DeprecationWarning` summary entries and a third `swigvarlink` line on the way out:

```
<frozen importlib._bootstrap>:488: DeprecationWarning: builtin type SwigPyPacked has no __module__ attribute
<frozen importlib._bootstrap>:488: DeprecationWarning: builtin type SwigPyObject has no __module__ attribute
sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute
```

These are **not from our code**. They come from a SWIG-generated C extension somewhere under our heavy ML deps (verified via `PYTHONWARNINGS=error::DeprecationWarning` against `import voice` and every `voice.*` module — all clean). SWIG hasn't yet caught up with Python 3.12's rule that built-in types must expose `__module__`. The warnings are emitted in C via `PyErr_WarnEx` *before* any pytest filter can intercept them, which is why `filterwarnings = ["ignore::DeprecationWarning"]` in `pyproject.toml` does nothing — we tried.

So the right reaction is: ignore them. Don't try to `filterwarnings` these specific three away (won't work for the C-level emission path), don't silence DeprecationWarnings globally either (would hide legitimate ones). They will disappear on their own when our transitive deps rebuild against a newer SWIG — the underlying bug is [swig/swig#2881](https://github.com/swig/swig/issues/2881), fixed in the `swig-4.4` milestone (May 2025).

## Benchmarks and decision documents

Every time a measurement run produces a **material result** — a wall-clock change worth bragging or worrying about, a quality regression caught or ruled out, a "before / after" delta that informs a product decision — that result must land in a documented benchmark **plus** a decision record. The format is fixed by [`docs/benchmarks/README.md`](docs/benchmarks/README.md) (six-section contract: input → expected output → evaluation → fixtures → real input → result & decision).

Concretely:

- **A new benchmark doc** under `docs/benchmarks/<name>.md` whenever a fresh question is being measured. Reuses the six-section contract.
- **An ADR** under `docs/adr/NNNN-<title>.md` when the measurement informs a product decision (default flipped, new API shipped, alternative rejected). The ADR links to the benchmark.
- **Both** when you tried something, got numbers, made a call. Numbers without a recorded decision rot fast — six months later nobody remembers what we decided to do with the +37 % speedup on stage X.

This applies whenever the result is significant enough that we'd want a future contributor (human or AI) to find it. Trivial micro-optimisations and one-off curiosity runs don't need a doc. If you're unsure, lean toward writing it — the cost is 15 min of writing, the cost of not writing is rediscovering the same lesson by repeating the same wrong attempt.

## Before running an LLM benchmark — ask the user to quiet the laptop

`voice transcribe` and any other benchmark that exercises mlx-lm on the M1/16 GB reference machine is sensitive to memory pressure and GPU contention from other apps. A few stale Safari tabs, a Slack call, or VSCode's Copilot Chat can easily push MLX into swap and add 30-50 % wall-clock — silently making the measurement worthless.

**Before kicking off any timed LLM run** (the `voice transcribe …` reference command, an ad-hoc probe script that loads the LLM, a re-measurement of a benchmark) ask the user to:

- Close Safari / Chrome (the biggest single offender — tabs eat unified memory).
- Quit any open IDEs that have AI assistants loaded (VSCode + Copilot, Cursor, JetBrains AI).
- Pause Slack/Zoom/Teams calls if any are running.
- Confirm the laptop is plugged in (battery throttling skews wall-clock).

Then wait for explicit confirmation ("закрив", "ready", "go") before invoking the run. Yes, this slows the loop down. The alternative is a benchmark number we can't trust, which is worse than no benchmark at all — see the false +27.9 % regression we recorded the first time around on issue #122 because the laptop wasn't quiet.

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

## Issue labels — MANDATORY when creating or editing issues

Every issue in this repo MUST satisfy all four rules below. This applies whether the issue is created by a human, by Claude on the human's behalf, or by automation. The full rationale lives in [ADR 0029](docs/adr/0029-issue-label-taxonomy.md) and the full reference (with worked examples) in [`docs/conventions.md`](docs/conventions.md) — the rules below are the operational subset.

**Mandatory labels:**

1. Exactly one `type:*`.
2. Exactly one `priority:*`.
3. At least one of `stage:*` or `area:*` (both axes can coexist on the same issue).
4. Soft maximum: 5 labels total per issue.

### `type:*` (exactly one)

- `type:bug` — something is broken or behaves incorrectly against documented expectations.
- `type:feature` — a new user-visible capability (new flag, new output, new behaviour the user can observe).
- `type:perf` — speed or memory improvement work, with or without a user-visible behavioural change.
- `type:docs` — changes only to `README.md`, `CLAUDE.md`, `docs/`, or in-code docstrings/comments.
- `type:refactor` — internal restructuring with no user-visible behaviour change and no perf claim.
- `type:test` — adding, fixing, or restructuring tests.
- `type:chore` — tooling, build, dependencies, repo hygiene, CI, label/issue housekeeping.

### `priority:*` (exactly one)

- `priority:critical` — the project is broken for users right now, or about to be. Fix before anything else.
- `priority:high` — should land in the next release cycle.
- `priority:medium` — default for most work. Use this if unsure. Do not leave priority unset.
- `priority:low` — nice to have.

### `stage:*` (at least one of `stage:*` or `area:*`)

One label per pipeline stage, keyed on the module name in `src/voice/`. Use `stage:*` when the issue is about that specific step's behaviour, parameters, or output.

- `stage:transcode` — `transcode.py`
- `stage:audio_meta` — `audio_meta.py`
- `stage:diarize_speakers` — `diarize_speakers.py`
- `stage:lang_detect` — `lang_detect.py`
- `stage:clear_speech` — `clear_speech.py`
- `stage:speech2text` — `whisper_asr.py`, `download_whisper.py`
- `stage:merge` — `merge.py`
- `stage:proofread` — `proofread.py`
- `stage:identify_speakers` — `identify_speakers.py`
- `stage:speech_structure` — `speech_structure.py`
- `stage:safe_speech` — `safe_speech.py`
- `stage:speech_summary` — `speech_summary.py`
- `stage:render` — `render.py`, `speaker_emojis.py`

### `area:*` (at least one of `stage:*` or `area:*`)

Cross-stage concerns. Use `area:*` when the issue spans multiple stages or lives in shared infrastructure that does not belong to a single stage.

- `area:llm` — shared LLM stack (`llm.py`, prompt cache, token accounting). Affects all five LLM-driven stages.
- `area:audio` — audio preprocessing in general (`transcode`, `silence`, `clear_speech`, loudness work).
- `area:cli` — command-line surface (`cli.py`, flags, progress reporting).
- `area:perf` — cross-stage performance infrastructure (benchmarks, profiling, memory budgets). NOT used for per-stage speedups — those use `type:perf` + `stage:*`.
- `area:models` — model lifecycle across the project: download, quantization, swap (ASR + diarization + LLM).
- `area:prompts` — LLM system prompts only. Does NOT include Whisper `initial_prompt` — that goes under `stage:speech2text`.
- `area:i18n` — language behaviour: Ukrainian/English code-switching, IT loanwords, transliteration.

### Disambiguation rules (settled — do not re-derive)

- **`type:perf` vs `area:perf`**: per-stage speedup → `type:perf` + `stage:X`. Cross-stage perf infrastructure (benchmark harness, profiler) → `type:feature` (or whatever fits) + `area:perf`.
- **`type:feature` vs `type:refactor`**: only `type:feature` if a user can observe the change. Internal restructuring → `type:refactor`. Renames (`agc` → `autogain`) are `type:refactor`, not `type:feature`.
- **Whisper `initial_prompt`** issues → `stage:speech2text` (+ `area:i18n` if about language). Never `area:prompts`.
- **`area:models` vs `area:llm`**: `area:models` covers all three model classes; `area:llm` is LLM-specific. Whisper quantization is `area:models` + `stage:speech2text`, not `area:llm`.
- **`kb-grooming` issues**: keep the `kb-grooming` label AND apply the standard 4 axes (typically `type:docs` + `area:*` + `priority:low`).
- **Built-in `bug`, `enhancement`, `documentation`**: do not use. GitHub restores them after deletion for UI compatibility, but this project uses `type:bug`, `type:feature`, `type:docs` exclusively.

### When creating an issue

1. Pick `type:*` from the issue's core nature (one of seven).
2. Pick `priority:*` from impact and urgency (one of four; default `medium`).
3. Map to `stage:*` if the issue is about a specific pipeline step; otherwise (or additionally) pick `area:*`.
4. Apply labels in one shot: `gh issue create --label "type:X,priority:Y,stage:Z" --title "..." --body "..."`.

### Scope

This taxonomy applies to **issues only**. PRs are not labelled — Conventional Commit prefixes in PR titles (`feat:`, `fix:`, `docs:`, `perf:`, `chore:`) carry the type signal.

### Issue title format — MANDATORY

Title must be self-describing without labels (labels are absent in email notifications, mobile views, search results, cross-repo references).

**Format:** `[CRITICAL ]<type>(<scope>): <functional subject>`

- `<type>` — bare word matching the `type:*` label: `feat`, `fix`, `perf`, `docs`, `refactor`, `test`, `chore`. Plus `epic` (parent issue grouping several trackers) and `research` (investigation/spike). `epic` and `research` are title-only; the underlying work is still classified by its base `type:*` label.
- `<scope>` — the **stage where the effect lands for the user**, NOT where the code change happens. A speedup whose code lives in `area:llm` but whose user-facing effect is on proofread is `perf(proofread): ...`. Use `area:*` value (`cli`, `models`, `i18n`, ...) when the effect is genuinely cross-stage. One scope per title.
- `<functional subject>` — what the user gets, NOT the implementation. "3x speedup via prompt cache reuse" beats "reuse mlx-lm prompt cache for proofread system prompt prefix". Implementation details go in the body.
- `CRITICAL` — optional modifier mirroring `priority:critical`. Only for issues that block users right now. Format: `CRITICAL fix(proofread): ...`.

**Length:** target 60-80 characters. Soft limit.

**Language:** English. Mixed-language quotes from ASR output (e.g. `"корище цей"`, `"HugginsFace"`) are fine as evidence inside the title.

**No redundancy:** never repeat the scope word in the subject. `perf(proofread): proofread is 3x faster` → `perf(proofread): 3x speedup via prompt cache reuse`.

**Examples:**

- `perf(proofread): 3x speedup via prompt cache reuse`
- `CRITICAL fix(proofread): English IT terms come out broken (HugginsFace, Gradle)`
- `fix(speech_structure): Metal OOM on 16 GB Macs during long recordings`
- `research(speech_structure): two-pass section detection for better boundaries`
- `feat(cli): install/remove LLM checkpoints from the command line`

For full reference (more worked before/after pairs, `kb-grooming` exception) see [`docs/conventions.md`](docs/conventions.md#issue-title-format).

## Editing the architecture diagram

`docs/architecture.md` has two PlantUML diagrams whose edges encode the actual data-flow:

- **Solid red arrows = audio bytes**. **Solid blue = structured data**. **Solid light-green (`#7CCD7C`) = Markdown**. **Dotted = library/file dependency**.
- Stages are numbered `[1]…[10]` matching the runtime log labels (`[6/9] ASR-постобробка` etc.).

Before changing an edge, grep the code:

```bash
grep -nE "^\s*(speech2text_module|diarize_module|merge|proofread|identify|structure|tldr|render)" src/voice/pipeline.py
```

The stage that *appears* to feed the next one in the file order isn't always the truth — for example, ASR and diarize both read the same `wav_path` written by ffmpeg, so their incoming edge is from `[2] WAV`, not from `pipeline`. Match the diagram to the code, not to intuition.

The PlantUML pre-commit hook regenerates the embedded SVG URL from the source block. Don't hand-edit the URL; edit the source and let the hook resync.
