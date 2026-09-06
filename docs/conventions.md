# Project conventions

This document collects project-wide conventions that span multiple files or are not obvious from the code. For architecture see [`architecture.md`](architecture.md); for pipeline stage details see [`pipeline.md`](pipeline.md); for dated design decisions see [`adr/`](adr/).

## Issue labels and titles

Issues use a colon-prefixed label taxonomy with two mandatory axes (`type:*`, `priority:*`) and two scoping axes (`stage:*`, `area:*`) of which at least one always applies, plus a title format that keeps an issue readable where labels are invisible.

The full dictionaries, disambiguation rules, worked examples, and title format live in [`issue-labels.md`](issue-labels.md) — the machine-readable source of truth, applied by the `issue-conventions` plugin. The rationale behind the four axes is recorded in [ADR 0029](adr/0029-issue-label-taxonomy.md).
