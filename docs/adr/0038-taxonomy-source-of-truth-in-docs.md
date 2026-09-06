---
status: draft
date: 2026-09-07
gate: promote to accepted once the SessionStart hook has been observed to fire reliably across ~20 sessions, and the label rules have been applied to new issues without the CLAUDE.md copy present
supersedes: [0029]
---

# ADR 0038 — The issue taxonomy lives in `docs/issue-labels.md`, not in `CLAUDE.md`

## Context

[ADR 0029](0029-issue-label-taxonomy.md) settled *what* the taxonomy is: four axes (`type`, `priority`, `stage`, `area`) plus two optional ones (`reason`, `by`). That decision is unchanged and this record does not revisit it.

What ADR 0029 also decided — implicitly, because at the time there was no alternative — is *where the rules live and how they reach an AI assistant*. Its wording was explicit about the mechanism: values are "documented in `docs/conventions.md` and mirrored in `CLAUDE.md`", and the mitigation against dictionary drift was "colocation: the source of truth is `CLAUDE.md` + `conventions.md`". The reasoning was sound: an assistant applying labels must see the rules in its default context, and in 2026-05 the only way to guarantee that was to put them in `CLAUDE.md`, which is loaded into every session.

The cost of that mechanism is measurable. The `CLAUDE.md` section was 111 lines — roughly 45 % of the file — and it was loaded into **every** session in this repository, including the large majority that never touch an issue. It was also one of three copies of the same taxonomy (`CLAUDE.md`, `docs/conventions.md`, and the live GitHub labels), with no mechanical check that they agreed. They did not: `area:repo` was added as an eighth `area:*` value after ADR 0029 recorded seven, and neither the ADR nor the census in `conventions.md` was updated.

The `issue-conventions` plugin replaces the delivery mechanism with two parts that did not exist when ADR 0029 was written:

- a `SessionStart` hook that prints a three-line mandate ("invoke the guide skill before `gh issue create` …"), and only in repositories that have a plugin config;
- a guide skill that routes to a subagent, which reads the taxonomy document itself — so the dictionaries never enter the main session's context at all.

The mechanism is therefore *replaced*, not supplemented. Keeping the `CLAUDE.md` copy after installing the plugin would leave three sources of truth instead of one, which is the failure ADR 0029's colocation mitigation was aiming at in the first place.

## Decision

The taxonomy's source of truth is **`docs/issue-labels.md`**, in the machine-readable schema the `issue-conventions` plugin parses. Concretely:

- **`docs/issue-labels.md`** — new file. Full axis table, dictionaries with colors and descriptions, cross-axis rules, disambiguation rules, worked examples drawn from real issues, title format, legacy label mapping, and the GitHub built-in policy. Parsed by `parse-taxonomy.py`; verified against the live labels by `drift-check.sh`.
- **`docs/conventions.md`** — the ~170-line taxonomy body is replaced by a two-paragraph pointer. The file keeps its role as a container for project-wide conventions; the plugin owns one topic in the repo, not that whole file.
- **`CLAUDE.md`** — the 111-line directive section is removed, leaving a heading and a single sentence naming the document and the plugin. The trigger is the hook; the content is the document.
- **`.claude-plugin/issue-conventions.json`** — records the document path, this repo's ADR, and `titleFormat.rewriteExisting: false`.

When the document and the live GitHub labels disagree, **the document wins**: a hand edit is a legitimate way to change the taxonomy, and the plugin's job is to propagate it to GitHub, never the reverse.

Three title rules are made explicit in the document because the existing backlog violated them without the rules being written down anywhere: scope is mandatory (#83, #111 have none), scope is singular — no comma lists (#152, #154), and scope values come from the axis dictionaries, so a scope naming a single document (`claude-md`, #109) is not valid. Issues about `CLAUDE.md`, ADRs, or `conventions.md` take `area:repo`; no axis value is created for a single document, because a value that fits one issue is taxonomy erosion.

Existing titles are **not** rewritten. 91 of 98 already match the format; three of the seven exceptions carry `by:kb-grooming` and are exempt by design, and the remaining four are fixed by hand when touched.

## Alternatives considered

- **Keep the `CLAUDE.md` copy alongside the document.** Rejected: it is exactly the per-session context cost the plugin exists to remove, and it recreates the three-copies problem. The counter-argument — that the hook's reliability in this repo is not yet measured — is answered by the two-line pointer, which costs ~20 tokens instead of 111 lines while leaving a fallback path if the hook does not fire.
- **Remove the `CLAUDE.md` section entirely, with no pointer.** Rejected for now, for the reason above; revisit at the gate.
- **Put the machine-readable taxonomy directly into `docs/conventions.md`.** Rejected: the plugin's footer claims ownership of the file it manages ("do not edit by hand"), and `conventions.md` is meant to hold other conventions too. Regeneration would have to preserve unrelated human prose in the same file — a class of bug that loses data quietly — and the strict parser schema would break as soon as the next convention is appended.
- **Amend ADR 0029 in place.** Rejected: it is `accepted`, and by this project's convention an accepted ADR is an immutable record — superseded, not edited.
- **Drop the unused `stage:*` values** (`stage:transcode`, `stage:audio_meta`, `stage:lang_detect`, currently zero issues each). Rejected: they map to real modules in `src/voice/`. Zero usage means no issue has been filed about them yet, not that the value is dead; deleting them would also break the document's module-to-axis correspondence, which is what lets the drift check spot a stage with no label.
- **Take over the Dependabot labels** (`dependencies`, `python:uv`). Rejected: another tool applies them, and only to pull requests, which this taxonomy does not cover. Dependabot is configured through the repository settings rather than a `dependabot.yml` in the tree, so the missing `.github/` directory is not evidence that it is dormant — `automated-security-fixes` is enabled and PR #192 carries both labels. They are recorded as `keep` in the legacy mapping and are never auto-deleted; the plugin reports them as undeclared and leaves them alone.

## Consequences

**Positive:** one source of truth, mechanically checkable — `label-plan.sh` reports zero creates and zero updates against the live labels today, so the document provably matches GitHub. Every session in this repo loses 111 lines of always-loaded context. The dictionaries reach an assistant on demand, in a subagent, instead of by permanent residence in the main context. Drift between the document, the ADR, and GitHub becomes a reported divergence rather than something noticed by chance.

**Negative:** the rules are no longer visible by default — if the `SessionStart` hook fails to fire, an assistant sees only the two-line pointer and has to follow it. This is untested in this repository, which is why this record ships as a draft with the gate above. The plugin also becomes a dependency of the labelling workflow: without it, the document is still readable by a human, but nothing verifies it against GitHub.

TODO after the gate:
- Whether the hook actually fires across sessions, and what an assistant does when it does not.
- Whether the two-line pointer turned out to be load-bearing or dead weight — remove it if the hook proves reliable.
- The first real divergence the drift check catches, and whether the report was actionable.
