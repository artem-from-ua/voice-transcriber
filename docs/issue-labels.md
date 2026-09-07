# Issue taxonomy

`voice-transcriber` is a command-line pipeline that turns Ukrainian conversational recordings into structured Markdown transcripts. Its issues are almost always scoped to one pipeline stage or to one piece of shared infrastructure, so the taxonomy is built to make that scoping cheap: two mandatory axes that classify every issue, and two scoping axes of which at least one always applies.

## Axes

| Axis | Prefix | Mandatory | Cardinality | Color | Applies to |
|---|---|---|---|---|---|
| type | `type:` | yes | exactly one | `#cccccc` | all |
| priority | `priority:` | yes | exactly one | gradient | all |
| stage | `stage:` | no | zero or more | `#0052cc` | all |
| area | `area:` | no | zero or more | `#52a373` | all |
| reason | `reason:` | no | zero or more | `#cccccc` | closed only |
| by | `by:` | no | zero or more | `#cccccc` | all |

**Scope:** issues only. PRs are not labeled — Conventional Commit prefixes in the PR title (`feat:`, `fix:`, `docs:`, `perf:`, `chore:`) carry the type signal instead.

## Cross-axis rules

- at-least-one: stage, area
- soft-limit: 5

Every issue carries at least one `stage:*` or `area:*`; both axes can coexist on the same issue. Beyond five labels an issue is usually doing too much and is a candidate to split. There is deliberately no `status:*` axis — GitHub's open/closed state and assignees already carry workflow state, and a parallel copy would have to be hand-synced with them.

## Values

### `type:*`

| Label | Description | Color |
|---|---|---|
| `type:bug` | Something is broken or behaves incorrectly against documented expectations. | `#b60205` |
| `type:feature` | A new user-visible capability (new flag, output, or behaviour the user can observe). | |
| `type:perf` | Speed or memory improvement, with or without user-visible behavioural change. | |
| `type:docs` | Changes only to README, CLAUDE.md, docs/, or in-code docstrings/comments. | |
| `type:refactor` | Internal restructuring with no user-visible behaviour change and no perf claim. | |
| `type:test` | Adding, fixing, or restructuring tests. | |
| `type:chore` | Tooling, build, deps, repo hygiene, CI, label/issue housekeeping. | |

`type:bug` is the one value that overrides the axis color: broken functionality shares the deep red of `priority:critical` so the two loudest signals look alike in a list.

### `priority:*`

| Label | Description | Color |
|---|---|---|
| `priority:critical` | Project is broken for users right now, or about to be. Fix before anything else. | `#b60205` |
| `priority:high` | Should land in the next release cycle. Blocks something the user cares about. | `#e8814a` |
| `priority:medium` | Default for most work. Worth doing; no specific deadline. | `#bfd62c` |
| `priority:low` | Nice to have. | `#cccccc` |

Priority is the one axis where color carries urgency rather than membership, hence the red → orange → lime → grey gradient. Use `priority:medium` when unsure — an approximate priority is better than an unset one.

### `stage:*`

<!-- source: modules path=src/voice ignore=cli,llm,pipeline,types,silence,whisper_asr,download_whisper,speaker_emojis -->

One label per pipeline stage, keyed on the module name in `src/voice/`. Use `stage:*` when the issue is about that specific step's behaviour, parameters, or output.

| Label | Description | Color |
|---|---|---|
| `stage:transcode` | transcode.py — ffmpeg to 16 kHz mono WAV. | |
| `stage:audio_meta` | audio_meta.py — ffprobe metadata extraction. | |
| `stage:diarize_speakers` | diarize_speakers.py — pyannote-audio diarization. | |
| `stage:lang_detect` | lang_detect.py — Whisper-based language detection. | |
| `stage:clear_speech` | clear_speech.py — autogain, bandpass, denoise chain. | |
| `stage:speech2text` | whisper_asr.py, download_whisper.py — ASR. | |
| `stage:merge` | merge.py — combine ASR segments with diarization turns. | |
| `stage:proofread` | proofread.py — LLM proofreading of ASR output. | |
| `stage:identify_speakers` | identify_speakers.py — LLM-based speaker naming. | |
| `stage:speech_structure` | speech_structure.py — LLM section detection. | |
| `stage:safe_speech` | safe_speech.py — LLM redaction of sensitive content. | |
| `stage:speech_summary` | speech_summary.py — LLM TL;DR generation. | |
| `stage:render` | render.py, speaker_emojis.py — final Markdown assembly. | |

All `stage:*` share one blue. A per-stage gradient was tried first and rejected: on issues carrying two or more stages the chips read as visual noise.

Modules listed in `ignore=` above have no stage of their own on purpose. `whisper_asr.py`, `download_whisper.py` and `speaker_emojis.py` are covered inside the descriptions of `stage:speech2text` and `stage:render`; `cli.py`, `llm.py`, `pipeline.py`, `types.py` and `silence.py` are infrastructure that belongs to the `area:*` axis. Underscore-prefixed modules (`_prompts.py`, `_progress.py`, …) are private helpers and are excluded before the ignore list is consulted, so they never need listing here.

### `area:*`

Cross-stage concerns. Use `area:*` when the issue spans several stages or lives in shared infrastructure that does not belong to a single stage.

| Label | Description | Color |
|---|---|---|
| `area:llm` | Shared LLM stack: llm.py, prompt cache, token accounting. Affects all five LLM-driven stages. | |
| `area:audio` | Audio preprocessing in general: transcode, silence, clear_speech, loudness work. | |
| `area:cli` | Runtime CLI of voice transcribe: flags, args, progress reporting, terminal output. | |
| `area:perf` | Cross-stage perf infrastructure: benchmarks, profiling, memory budgets. NOT per-stage speedups. | |
| `area:models` | Model lifecycle: download, quantization, swap (ASR + diarization + LLM). | |
| `area:prompts` | LLM system prompts. Does NOT include Whisper initial_prompt. | |
| `area:i18n` | Language behaviour: Ukrainian/English code-switching, IT loanwords, transliteration. | |
| `area:repo` | Repo housekeeping not affecting runtime: labels, ADRs, scripts/, dev tooling, kb-grooming reports. | |

### `reason:*`

Optional, applied only when closing an issue. GitHub's native close reasons cover most cases at the state level; these labels add the specific *why* when "not planned" needs disambiguation, and keep the signal searchable in `gh issue list` filters.

| Label | Description | Color |
|---|---|---|
| `reason:duplicate` | Closing reason: duplicate of another issue. | |
| `reason:invalid` | Closing reason: not a valid issue (out of scope, misunderstanding, etc). | |
| `reason:wontfix` | Closing reason: acknowledged but explicitly decided not to fix. | |

### `by:*`

Marks issues that originate from a non-human source. Orthogonal to the four main axes: a `by:*`-tagged issue still gets a full `type` + `priority` + `stage`/`area` set on top. The axis exists so `by:dependabot`, `by:security-audit` and similar can be added later without re-engineering the taxonomy.

| Label | Description | Color |
|---|---|---|
| `by:kb-grooming` | Filed by the kb-grooming automation (periodic documentation-health audit). | |

## Disambiguation rules

- **`type:perf` vs `area:perf`** — a per-stage speedup is `type:perf` + `stage:X`. Cross-stage perf infrastructure (benchmark harness, profiler) is `area:perf` plus whatever type fits.
- **`type:feature` vs `type:refactor`** — only `type:feature` when a user can observe the change. Internal restructuring is `type:refactor`; renames such as `agc` → `autogain` are `type:refactor`, not `type:feature`, even when they touch the CLI.
- **Whisper `initial_prompt`** — belongs to `stage:speech2text` (plus `area:i18n` when it is about language), never to `area:prompts`. That area covers LLM system prompts only.
- **`area:models` vs `area:llm`** — `area:models` covers all three model classes (ASR, diarization, LLM); `area:llm` is LLM-specific. Whisper quantization is `area:models` + `stage:speech2text`, not `area:llm`.
- **`area:cli` vs `area:repo`** — `area:cli` is what a user sees running `voice transcribe` (flags, output, progress). `area:repo` is how we maintain the repo (labels, ADRs, conventions, dev tooling, `scripts/`). Ask "does this affect the end-user CLI?" — no means `area:repo`. Do not default to `area:cli` just because the word "CLI" or "command" appears.
- **Meta-documentation** — issues about `CLAUDE.md`, ADRs, or `docs/conventions.md` go to `area:repo`. There is deliberately no value named after any single document: a value that fits one issue is taxonomy erosion.
- **Initial / MVP issues** — retrospective "Implementation: ..." issues that bootstrap the project are `type:feature` + `area:repo`. They are historical bookmarks, not active scope.
- **Epics spanning many stages** — prefer one `area:*` over listing four or more `stage:*`. With two or three stages involved, list them.
- **Catch-all guard** — repo housekeeping goes to `area:repo`, never to the nearest product-facing area. Without that discipline `area:cli` slowly turns into a bucket for everything.

## Worked examples

| Issue | Labels | Why |
|---|---|---|
| #190 `fix(speech_summary): TL;DR rendered in Chinese for cross-language input` | `type:bug`, `priority:medium`, `stage:speech_summary`, `area:prompts` | Broken output from one stage; the fix lives in that stage's system prompt. |
| #174 `research(speech2text): does bumping ASR_CHUNK_OVERLAP_S help boundary words` | `type:perf`, `priority:low`, `stage:speech2text`, `area:audio` | A tuning experiment on one stage; `research` is title-only, the label is the base type. |
| #191 `chore(repo): remove residual references to one specific language from docs` | `type:chore`, `priority:medium`, `area:repo` | Repo housekeeping with no runtime effect — `area:repo`, no stage. |
| #166 `refactor(cli): unify progress reporting protocol across all pipeline stages` | `type:refactor`, `priority:low`, `area:cli` | Touches every stage, so no `stage:*`; user-visible surface is the CLI, but nothing observable changes — refactor, not feature. |
| #161 `feat(proofread): auto-disable when diarize shows mostly long monologue-shaped turns` | `type:feature`, `priority:medium`, `stage:diarize_speakers`, `stage:proofread`, `area:cli` | Two stages genuinely involved, so both are listed; five labels is the soft limit, not a violation. |

## Title format

**Pattern:** `[CRITICAL ]<type>(<scope>): <subject>`

| Element | Source | Rule |
|---|---|---|
| `<type>` | the `type:*` value without its prefix | `feat`, `fix`, `perf`, `docs`, `refactor`, `test`, `chore` |
| `<scope>` | a `stage:*` or `area:*` value without its prefix | where the effect lands for the user, not where the code lives; exactly one per title |
| `<subject>` | — | what the user gets, not how it is implemented |
| `CRITICAL` | `priority:critical` | optional prefix modifier, only for issues blocking users right now |

`epic` and `research` are title-only types: they have no `type:*` label, and the underlying work is still classified by its base type.

**Scope is mandatory and singular.** A comma-separated scope (`feat(cli,llm,speech2text): ...`) is not allowed — when the effect is genuinely cross-stage, use the `area:*` value that covers it instead of listing stages. A missing scope is equally wrong: labels are invisible in email notifications, mobile views and search results, and the scope is what keeps the title self-contained there.

**Scope values come from the axis dictionaries**, with one exception. A scope that names a single file or document (`claude-md`) is not a taxonomy value — such issues take `area:repo` and the matching scope. Neither is a module deliberately left out of the axis: `pipeline`, `cli`, `llm` and the rest of the `ignore=` list are infrastructure, and an issue about them takes the `area:*` value that covers it.

The exception is a **proposed stage**: an issue that argues for a new pipeline stage may use that stage's future name as its scope before any label or module exists. The title is what a reader sees first, and naming the thing being proposed beats routing it through the nearest existing area. Its labels stay on the `area:*` axis until the stage ships, at which point the stage gets its own value, the scope becomes real, and the row below is removed. Every such scope is listed in the table below — one line per exception, so an unlisted scope stays an error.

**Length:** 60–80 characters, soft. Self-containment beats brevity — if trimming a word makes the title ambiguous without reading the labels, keep the word.

**Language:** English, like every repository artifact. Mixed-language quotes from real ASR output (`"корище цей"`, `"HugginsFace"`) are fine inside a title as evidence.

**Exempt:** issues carrying a `by:*` label keep whatever title the automation produced.

**Allowed scopes beyond the axis values:**

| Scope | Why |
|---|---|
| `followup` | Proposes a pipeline stage that does not exist yet; labels use a cross-cutting axis until it ships. |

## Legacy label mapping

| Old label | Action | New label | Why |
|---|---|---|---|
| `dependencies` | keep | | Applied by Dependabot to its own pull requests; not ours to manage. |
| `python:uv` | keep | | Applied by Dependabot for the `uv` dependency group. |

Dependabot is configured through the repository settings rather than a `dependabot.yml` in the tree, so the absence of `.github/` says nothing about whether it is active — it is (`automated-security-fixes` is enabled, and PR #192 carries both labels). These two labels appear on pull requests, which this taxonomy does not cover, so they are reported as undeclared and left alone.

The nine GitHub built-ins were deleted before this document existed: `bug`, `enhancement` and `documentation` are replaced by `type:*`; `duplicate`, `invalid` and `wontfix` by `reason:*` plus GitHub's native close reasons; `question`, `good first issue` and `help wanted` are community-recruitment signals not relevant to a solo project.

## GitHub built-in labels

Policy: **delete**. Exceptions kept: none. GitHub silently re-creates built-ins after some UI operations — the drift check watches for that.

---

<!-- issue-conventions:managed -->
> **Do NOT edit this file by hand.** Run `/issue-conventions:setup` to change the taxonomy and `/issue-conventions:relabel` to apply it to existing issues. Hand edits are honored — this document is the source of truth — but the plugin cannot guarantee that GitHub labels and this file agree until you re-run those commands.

| | |
|---|---|
| Config | `.claude-plugin/issue-conventions.json` |
| Plugin | `issue-conventions` v0.1.2 |
| Last synced with GitHub | 2026-09-07 |
<!-- /issue-conventions:managed -->
