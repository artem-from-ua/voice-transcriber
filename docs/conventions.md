# Project conventions

This document collects project-wide conventions that span multiple files or are not obvious from the code. For architecture see [`architecture.md`](architecture.md); for pipeline stage details see [`pipeline.md`](pipeline.md); for dated design decisions see [`adr/`](adr/).

## Issue labels

Issues use a 4-axis colon-prefixed taxonomy. Each axis has a fixed dictionary and applies its own mandatory rules.

| Axis | Mandatory | Cardinality |
| --- | --- | --- |
| `type:*` | yes | exactly 1 |
| `priority:*` | yes | exactly 1 |
| `stage:*` | one of `stage:*` or `area:*` must be present | 1 or more |
| `area:*` | one of `stage:*` or `area:*` must be present | 1 or more |

Soft limit: 3–5 labels per issue. Beyond that an issue is usually doing too much and is a candidate to split.

The full rationale is recorded in [ADR 0029](adr/0029-issue-label-taxonomy.md). The same taxonomy is repeated, in directive form, in the root [`CLAUDE.md`](../CLAUDE.md) under "Issue labels" so that AI assistants applying labels see the rules in their default context.

### `type:*` (exactly one)

| Label | Meaning |
| --- | --- |
| `type:bug` | Something is broken or behaves incorrectly against documented expectations. |
| `type:feature` | A new user-visible capability. |
| `type:perf` | Speed or memory improvement work, with or without user-visible behavioural change. |
| `type:docs` | Changes to `README.md`, `CLAUDE.md`, `docs/`, or in-code docstrings/comments only. |
| `type:refactor` | Internal restructuring with no user-visible behaviour change and no perf claim. |
| `type:test` | Adding, fixing, or restructuring tests. |
| `type:chore` | Tooling, build, dependencies, repo hygiene, CI, label/issue housekeeping. |

Color: all `type:*` use `#cccccc` (neutral grey), with one exception — `type:bug` uses `#d73a4a` (red) so broken-functionality issues stand out in lists. The prefix carries the meaning; the color carries the axis.

### `priority:*` (exactly one)

| Label | When to use | Color |
| --- | --- | --- |
| `priority:critical` | The project is in a broken state for users right now, or about to be. Fix before anything else. | `#d73a4a` (same red as `type:bug`) |
| `priority:high` | Should land in the next release cycle. Blocks something the user cares about. | `#d93f0b` |
| `priority:medium` | Default for most work. Worth doing; no specific deadline. | `#fbca04` |
| `priority:low` | Nice to have. Often labelled and then closed without action. | `#cccccc` |

Default if unsure: `priority:medium`. Do not leave priority unset.

### `stage:*` (at least one of `stage:*` or `area:*`)

One label per pipeline stage from [`pipeline.md`](pipeline.md), keyed on the module name in `src/voice/`. Use `stage:*` when the issue is about that specific step's behaviour, parameters, or output.

| Label | Module(s) |
| --- | --- |
| `stage:transcode` | `transcode.py` |
| `stage:audio_meta` | `audio_meta.py` |
| `stage:diarize_speakers` | `diarize_speakers.py` |
| `stage:lang_detect` | `lang_detect.py` |
| `stage:clear_speech` | `clear_speech.py` |
| `stage:speech2text` | `whisper_asr.py`, `download_whisper.py` |
| `stage:merge` | `merge.py` |
| `stage:proofread` | `proofread.py` |
| `stage:identify_speakers` | `identify_speakers.py` |
| `stage:speech_structure` | `speech_structure.py` |
| `stage:safe_speech` | `safe_speech.py` |
| `stage:speech_summary` | `speech_summary.py` |
| `stage:render` | `render.py`, `speaker_emojis.py` |

Color: all `stage:*` share `#0052cc` (saturated blue). A per-stage gradient was tried first but made label chips visually noisy when an issue carried multiple stages; one color is calmer to read.

### `area:*` (at least one of `stage:*` or `area:*`)

Cross-stage concerns. Use `area:*` when the issue spans multiple stages or lives in shared infrastructure that does not belong to a single stage.

| Label | Scope |
| --- | --- |
| `area:llm` | Shared LLM stack: `llm.py`, prompt cache, token accounting, model loading. Affects all five LLM-driven stages. |
| `area:audio` | Audio preprocessing in general: `transcode`, `silence`, `clear_speech`, loudness work. |
| `area:cli` | **Runtime** command-line surface of `voice transcribe`: `cli.py`, flags, arguments, progress reporting, terminal output, `_progress.py`. NOT for repo housekeeping or dev tooling — use `area:repo`. |
| `area:perf` | Cross-stage performance infrastructure: benchmarks, profiling, memory budgets, GPU contention. |
| `area:models` | Model lifecycle across the project: download, quantization, swap (ASR + diarization + LLM). |
| `area:prompts` | LLM system prompts in `src/voice/prompts/` and prompt-engineering work. Does NOT include Whisper `initial_prompt`. |
| `area:i18n` | Language behaviour: Ukrainian/English code-switching, IT loanwords, transliteration, localized output. |
| `area:repo` | Repository housekeeping that does NOT affect the runtime: label/issue conventions, ADR flow, dev tooling, GitHub config, CI metadata, kb-grooming reports, commit-message conventions. Use this for issues like "introduce label taxonomy", "align ADR frontmatter", "standardise commit-message style". |

Color: all `area:*` use `#74a892` (muted sage green). Saturated green (`#2da44e`) was tried first but competed too much with `stage:*` blue and `type:bug` red for attention; areas are categorisation, not alarm, so a calmer hue works better.

### `reason:*` — closing reasons (orthogonal axis, optional, only on closed issues)

GitHub's native close reasons (`completed`, `not planned`, `duplicate`) cover most cases at the issue-state level. `reason:*` labels add the specific *why* when "not planned" needs disambiguation, and keep the signal searchable in `gh issue list` filters.

| Label | Apply when closing as |
| --- | --- |
| `reason:duplicate` | Already tracked in another issue (use alongside GitHub's `duplicate` close reason). Link the canonical issue in the close comment. |
| `reason:invalid` | Out of scope, misunderstanding, or otherwise not a real issue. |
| `reason:wontfix` | Acknowledged but explicitly decided not to fix. The decision should be in the close comment. |

Color: all `reason:*` use `#cccccc` (neutral grey).

### GitHub built-in labels are NOT used

All nine GitHub built-in labels were deleted from this repo: `bug`, `enhancement`, `documentation`, `duplicate`, `invalid`, `wontfix`, `question`, `good first issue`, `help wanted`. The first three are replaced by `type:*`; the next three are replaced by `reason:*` plus GitHub's native close-reason mechanism; the last three (`question`, `good first issue`, `help wanted`) are not used at all (`question` issues should use `type:*` + the body to clarify; `good first issue` / `help wanted` are community-recruitment signals not relevant to this solo project).

If GitHub silently re-creates any of them after a UI action: delete again. They are not part of the taxonomy.

### `by:*` — issue source (orthogonal axis, optional)

Marks issues that originate from a non-human source — automation, periodic audits, bots. Orthogonal to the 4 main axes: a `by:*`-tagged issue still gets a full `type` + `priority` + `stage`/`area` set on top.

| Label | Source | Issues come from |
| --- | --- | --- |
| `by:kb-grooming` | Project | The `kb-grooming` automation's periodic documentation-health audits. |

The axis exists so we can add `by:dependabot`, `by:security-audit`, `by:lint-report` etc. later without re-engineering the taxonomy. If a `by:*` value applies, add it; the issue still needs the standard 4 axes.

Color: all `by:*` use `#cccccc` (neutral grey) — provenance is metadata, not a primary signal.

## Disambiguation rules

These are the cases that cause repeated triage friction. Settle them once here.

- **`type:perf` vs `area:perf`.** Use `type:perf` for any performance work (the *kind* of change). Use `area:perf` for cross-stage perf infrastructure (benchmark harnesses, profilers, memory dashboards — the *topic*). A speedup of one specific stage is `type:perf` + `stage:<that_stage>`, without `area:perf`. A new benchmark harness covering several stages is `type:feature` + `area:perf`.
- **`type:feature` vs `type:refactor`.** An issue is `type:feature` only if it adds a user-visible capability — new flag, new output, new behaviour the user can observe. Internal restructuring without behaviour change is `type:refactor`. "Rename clearspeech effect `agc` to `autogain`" is `type:refactor` (or `type:chore`), not `type:feature`, even though it touches the CLI.
- **`area:prompts` scope.** `area:prompts` is only for LLM system prompts (`src/voice/prompts/`, `_prompts.py`). The Whisper `initial_prompt` is a different mechanism and belongs to `stage:speech2text`, optionally with `area:i18n` if the prompt content is about language/loanwords.
- **`area:models` vs `area:llm`.** `area:models` is broader: it covers all three model classes (ASR via `mlx-whisper`, diarization via `pyannote-audio`, LLM via `mlx-lm`). LLM-specific quantization or loader work is both `area:models` and `area:llm`. A Whisper quantization issue is `area:models` + `stage:speech2text`, not `area:llm`.
- **`area:cli` vs `area:repo`.** `area:cli` is for issues that change what a user sees when running `voice transcribe` — flags, output, progress. `area:repo` is for issues about how *we* work with the repo — label schemes, ADR conventions, commit-message style, dev tooling, `scripts/` directory, kb-grooming reports. A `--dump-stages` flag tweak is `area:cli`. A new ADR frontmatter rule is `area:repo`. A convention for how `scripts/*.py` print status is `area:repo` (it's developer ergonomics, not end-user CLI). If in doubt, ask: "does this affect the end-user CLI experience?" → yes is `area:cli`, no is `area:repo`.
- **Initial / MVP issues.** Retrospective issues that describe the first version of something (e.g. "Implementation: end-to-end pipeline" closed when the project was first stood up) are `type:feature` + `area:repo`, even though the implementation touches runtime code. `type:feature` because the issue introduced a user-visible capability; `area:repo` because the issue itself is project-bootstrapping, not a per-stage change you'd reopen to extend. Treat them as historical bookmarks, not active scope.
- **Automation-generated issues** carry a `by:*` label (e.g. `by:kb-grooming`) on top of the standard 4 axes. `by:*` marks the source; it does not replace `type`/`priority`/`stage`/`area`. A kb-grooming report issue is typically `type:docs` + `priority:low` + `area:*` + `by:kb-grooming`.
- **Epic issues** spanning many stages: prefer `area:*` over listing 4+ `stage:*` labels. If only 2–3 stages are involved, list them; beyond that the issue is cross-stage by nature.

## Worked examples

| Issue (real or hypothetical) | Labels |
| --- | --- |
| #122 `perf(llm): reuse mlx-lm prompt cache for proofread system prompt prefix` | `type:perf`, `priority:high`, `stage:proofread`, `area:llm` |
| #137 `CRITICAL: proofread cannot recognise Ukrainian-phonetic English IT loanwords` | `type:bug`, `priority:critical`, `stage:proofread`, `area:i18n` |
| #80 `KB grooming report 2026-05-12: stale docs from v0.13–v0.22 churn` | `type:docs`, `priority:low`, `area:repo`, `by:kb-grooming` (no `stage:*` — docs cover multiple stages) |
| #134 `epic: per-run user-provided topic hints across ASR + proofread stages` | `type:feature`, `priority:medium`, `stage:speech2text`, `stage:proofread`, `area:i18n` |
| #78 `Investigate Metal OOM during structure_dialog on 16 GB Macs` | `type:bug`, `priority:high`, `stage:speech_structure`, `area:perf` |
| #116 `feature: CLI to install/remove project-supported LLM checkpoints for the local hardware` | `type:feature`, `priority:medium`, `area:cli`, `area:models` |

## Scope

This taxonomy applies to **issues only**. Pull requests are not labelled — `feat:`/`fix:`/`docs:` Conventional Commit prefixes in the title carry that signal. There is no `status:*` axis — GitHub's open/closed state plus milestones cover workflow state. Both choices are recorded in [ADR 0029](adr/0029-issue-label-taxonomy.md).

## Issue title format

A title must be self-describing without labels — labels are stripped in email notifications, mobile views, GitHub search results, and cross-repo references. The label set is for filtering; the title is for reading.

**Format:**

```
[CRITICAL ]<type>(<scope>): <functional subject>
```

- `<type>` — same word as the `type:*` label without the prefix: `feat`, `fix`, `perf`, `docs`, `refactor`, `test`, `chore`. Plus `epic` for parent issues that group several trackers, and `research` for investigation/spike issues. `epic` and `research` are title-only; they do not have corresponding `type:*` labels (the underlying work is still classified by its base type label).
- `<scope>` — the **stage where the effect lands for the user**, not the stage where the code change happens. A speedup whose code lives in shared `area:llm` but whose user-facing effect is on the proofread stage is `perf(proofread): ...`, not `perf(llm): ...`. Use `area:*` value when the effect is genuinely cross-stage (`cli`, `models`, `i18n`, etc.). One scope per title.
- `<functional subject>` — what the user (or downstream reader) gets, not the implementation. "3x speedup via prompt cache reuse" beats "reuse mlx-lm prompt cache for proofread system prompt prefix". Implementation details go in the body, not the title.
- `CRITICAL` — optional modifier prefix that mirrors `priority:critical`. Apply only for issues that block users right now. `CRITICAL fix(proofread): ...`.

**Length:** target 60-80 characters. Soft limit — exceed when a longer title genuinely communicates better. GitHub trims around 70-80 in list views.

**Self-containment beats brevity.** Labels are absent in email notifications, mobile views, GitHub search results, and cross-repo references. If trimming a word from the subject makes the title ambiguous *without* reading the labels, keep the word — even if the title exceeds the 60-80 char target. Example: `CRITICAL fix(proofread): English IT terms in Ukrainian speech come out broken (HugginsFace, Gradle)` is 87 chars but self-describing; trimming "in Ukrainian speech" would lose the actual context (code-switching), which `area:i18n` only hints at. Length target is a guideline; self-containment is mandatory.

**Language:** English. Repository artifacts (issues, PRs, commits, docs) are English even though spoken conversation is Ukrainian. Mixed-language quotes from real ASR output (e.g. `"корище цей"`, `"HugginsFace"`) are fine inside the title as evidence.

**Avoid redundancy with the scope, not with the labels.** Do not repeat the scope word in the subject: `perf(proofread): proofread takes 3x less time` → `perf(proofread): 3x speedup via prompt cache reuse` — the scope is already in the parentheses. But do NOT trim a word just because a label covers it: labels are not visible in many surfaces, so context labels-only-encode must stay in the subject.

### Worked examples

| Before | After | Why |
| --- | --- | --- |
| `perf(llm): reuse mlx-lm prompt cache for proofread system prompt prefix` | `perf(proofread): 3x speedup via prompt cache reuse` | Scope follows effect (proofread), not code location (llm). Subject states the user-facing outcome. |
| `CRITICAL: proofread cannot recognise Ukrainian-phonetic English IT loanwords` | `CRITICAL fix(proofread): English IT terms in Ukrainian speech come out broken (HugginsFace, Gradle)` | `CRITICAL` is now a modifier on the type, not a standalone prefix. Subject shows the broken output the user sees. "in Ukrainian speech" stays because the bug is specific to code-switching, not English ASR in general — that context lives only in `area:i18n` which is not visible in notifications. |
| `research(speech_structure): two-pass design — generate detailed sections first, then consolidate via second LLM pass` | `research(speech_structure): two-pass section detection for better boundaries` | Drops the implementation walkthrough; keeps the *what* and *why*. |
| `Investigate Metal OOM during structure_dialog on 16 GB Macs` | `fix(speech_structure): Metal OOM on 16 GB Macs during long recordings` | "Investigate" reads as a status, not a title; replace with `fix(...)` and state the failure mode directly. |
| `feature: CLI to install/remove project-supported LLM checkpoints for the local hardware` | `feat(cli): install/remove LLM checkpoints from the command line` | Standardise on `feat`, scope from area, drop "project-supported" / "for the local hardware" as inferable from context. |

### Automation-generated issues (`by:*`)

Issues created by automation — `by:kb-grooming` and any future `by:*` source — keep whatever title format the automation produces (e.g. `KB grooming report YYYY-MM-DD: ...`). Do not retro-rename them — the `by:*` label is enough provenance signal, and the original title preserves the run identity.
