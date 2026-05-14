---
status: accepted
date: 2026-05-14
see_also: []
---

# 0029 — Issue label taxonomy: 4 axes (type, priority, stage, area)

## Context

Before this ADR, the repository had 11 GitHub labels — 9 built-in + 2 custom (`kb-grooming`, `performance`) — and used 5 of them at all: `enhancement` (39 issues), `documentation` (22), `kb-grooming` (22), `bug` (2), `performance` (1). The remaining ~76 of ~140 issues had no labels. Pull requests had no labels at all (0/140+).

This made several common operations impossible or unreliable:

- "Show me everything open about the `proofread` stage" — no way to filter without grepping titles.
- "What's outstanding on LLM infrastructure across stages?" — no cross-stage signal.
- "Sort the backlog by urgency" — no priority signal at all.
- "How did `proofread` regressions evolve over time?" — historical (closed) issues are unlabelled and unsearchable.

The labels in use were also under-defined: `enhancement` conflated `feature` (new user-visible capability) with `refactor` (internal restructuring), and `performance` had no description and only one consumer.

A labelling scheme is not architecture, so it does not need to be perfect — it needs to be cheap to apply, unambiguous in 95 % of cases, and stable enough that historical labels keep meaning their original thing a year later. The constraints driving this ADR:

- **Solo project, occasional contributors.** No team triage process. Whoever opens the issue applies labels; the same person (or an AI assistant working on their behalf) decides priority. So the rules must be readable and the dictionary small.
- **Pipeline-shaped product.** The product is a 13-stage pipeline. Most issues are scoped to one stage. The labelling should make that scoping cheap.
- **AI assistants apply labels.** Claude routinely creates and edits issues. The rules must be available in `CLAUDE.md` in directive form, not buried in a reference doc that the assistant might not load.
- **Cost of relabelling is one-time.** ~140 historical issues will be migrated once; after that the scheme has to pay for itself only on new issues.

## Decision

Four label axes, colon-prefixed. Each axis has a fixed dictionary; values are documented in [`docs/conventions.md`](../conventions.md) and mirrored in [`CLAUDE.md`](../../CLAUDE.md).

| Axis | Mandatory | Cardinality | Dictionary size |
| --- | --- | --- | --- |
| `type:*` | yes | exactly 1 | 7 |
| `priority:*` | yes | exactly 1 | 4 |
| `stage:*` | at least one of `stage:*` or `area:*` | 1 or more | 13 |
| `area:*` | at least one of `stage:*` or `area:*` | 1 or more | 7 |

Total label dictionary: 31 prefixed labels + the project-relevant built-ins (`good first issue`, `help wanted`, `question`, `duplicate`, `invalid`, `wontfix`) + `kb-grooming`. Soft limit of 3–5 labels per issue.

**Format chosen: colon prefix** (`type:bug`, `stage:proofread`). The alternative — slash (`type/bug`, Kubernetes-style) — URL-escapes to `type%2Fbug` in GitHub filter URLs, which is less readable; colon stays literal. Colon is the de-facto convention in modern open-source projects (Salt, Sane GitHub Labels guide, Robin's labelling guide).

**Color per axis, not per value.** All `type:*` share `#cccccc` (with `type:bug` as an exception in `#d73a4a` red so bugs stand out in lists). All `area:*` share `#2da44e`. All `stage:*` share `#0052cc`. All `by:*` share `#cccccc` (provenance is metadata, not primary signal). `priority:*` uses a red→grey gradient (the only axis where the gradient carries meaning — urgency). The prefix carries identity; the color carries the axis. A per-stage blue gradient was considered (upstream-light → downstream-dark) and rejected after first use: multi-stage issues read as visual noise; one calm color is easier to scan.

**Disambiguation rules** are part of the decision, not an afterthought:

- `type:perf` is the *kind* of work; `area:perf` is the cross-stage perf *topic*. A proofread speedup is `type:perf` + `stage:proofread`, not `area:perf`.
- `enhancement` semantics split: user-visible new capability → `type:feature`; internal restructuring → `type:refactor`. A rename like `agc` → `autogain` is `type:refactor`, not `type:feature`.
- Whisper `initial_prompt` is `stage:speech2text`, not `area:prompts`. `area:prompts` is exclusively for LLM system prompts.

**Issue title format** is decided alongside the labels: `[CRITICAL ]<type>(<scope>): <functional subject>`. The title carries the same `type` word as the `type:*` label (`feat`, `fix`, `perf`, ...) plus `epic` and `research` modifiers that are title-only. Scope follows the **user-facing effect**, not the code location. The subject describes the outcome, not the implementation. Full rules and worked examples in [`docs/conventions.md`](../conventions.md#issue-title-format).

Rationale: labels are stripped in email notifications, mobile views, GitHub search, and cross-repo references. Filtering uses labels; reading uses titles. Both need to work without the other.

**Out of scope, decided explicitly:**

- **No `status:*` axis.** GitHub's open/closed state, assignees, and milestones already cover workflow state. A `status:*` axis would have to be kept in sync with all three of those by hand. The marginal value (`status:blocked`, `status:needs-info`) doesn't pay back the upkeep on a project this size.
- **PRs are not labelled.** Conventional Commit prefixes in PR titles (`feat:`, `fix:`, `docs:`, `perf:`) already carry the `type:*` signal. The current state (0/140+ PRs labelled) is intentional, not a backlog item. Revisiting requires an auto-labeler driven by changed file paths; that's a separate decision when/if it's worth it.
- **No issue templates, no labeler.yml.** The repo has no `.github/` directory and this ADR doesn't add one. Templates are over-engineering for a solo project; an auto-labeler depends on PRs being labelled, which we just opted out of.

## Consequences

**Positive.**

- Cheap to filter: `gh issue list --label "stage:proofread" --state all` returns the full lifecycle of one stage's issues, including closed ones, with one command.
- Easy to budget: counting `priority:critical` + `priority:high` open issues gives a release-blocking backlog at a glance.
- Cross-stage work is visible: `area:llm` collects everything in the LLM substrate, not just whatever stage I happened to be working on at the time of writing.
- AI assistants get a single source of truth: `CLAUDE.md` has the full dictionary and the mandatory rules, so issue creation doesn't require a sidecar lookup.

**Negative / risks.**

- Mandatory `priority` forces a choice for every issue. The cure (default `priority:medium`) is documented but still risks the label losing meaning if everything ends up medium.
- Mandatory `stage` *or* `area` covers most cases but a few legitimately-cross-cutting issues (docs about the project as a whole, meta-issues about labels themselves) have an awkward fit. We accept this and use `area:cli` or `area:perf` as the closest match.
- The dictionary will drift if new stages or areas are added without updating the docs. The mitigation is colocation: the source of truth is `CLAUDE.md` + `conventions.md`, which are part of the regular doc-update flow.
- GitHub restores built-in labels (`bug`, `enhancement`, `documentation`) after deletion, so we cannot fully eliminate them. We address this by directive in `CLAUDE.md`: "ignore built-ins; use `type:*` exclusively".

**Migration debt, one-time.** ~140 issues (including closed) need to be relabelled by hand-with-AI: read title + body, assign labels per the rules above. Estimated 2–3 hours of wall-clock work in batches of 20. Tracked in issue [#140](https://github.com/artem-from-ua/voice-transcriber/issues/140).

**Future revisits.** This ADR can be superseded if (a) the project grows a team and `status:*` becomes load-bearing, or (b) PR volume justifies an auto-labeler. Either change should be a new ADR, not an in-place edit of this one.
