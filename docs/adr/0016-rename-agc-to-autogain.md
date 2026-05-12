---
status: accepted
date: 2026-05-12
---

# ADR 0016 — Rename clearspeech effect `agc` to `autogain`

## Context

The first clearspeech effect (shipped in v0.13.0 as `loudness_normalize` and refactored into the chain dispatcher in v0.14.0 — see ADR 0010) carried the internal acronym `agc` ("automatic gain control"). The name leaked into every public-facing surface:

- The CLI literal users type: `--clearspeech-chain agc`.
- The tuning flags: `--clearspeech-agc-target-dbfs`, `--clearspeech-agc-max-gain-db`.
- The Python kwargs of `clearspeech()`: `agc_turns`, `agc_target_dbfs`, `agc_max_gain_db`, `agc_crossfade_ms`.
- The on-disk artefact suffix produced under `--dump-stages`: `<stem>.agc.wav`.

`agc` is jargon. A casual `--help` reader has to either know the acronym or guess. `autogain` reads as English, hints at the behaviour (per-segment level normalisation), and is no longer than the alternatives the team considered (`level`, `normalize`, `loudness`). It is also unambiguous in chain strings: `autogain,bandpass,presence` parses the same way as `agc,bandpass,presence`.

The project is pre-1.0 (we are bumping to `0.19.0` with this change). SemVer 2.0.0 §4 explicitly says the public API is not stable in `0.x.y`, so a breaking rename of the CLI surface does not require a `1.0.0` bump. The bump to `0.19.0` is MINOR and the CHANGELOG entry is labelled BREAKING so anyone scripting the CLI updates their flags.

## Decision

Rename `agc` to `autogain` across every public and internal surface in a single PR:

- Chain literal: `agc` → `autogain` (including the `--clearspeech-chain` default value).
- CLI flags: `--clearspeech-agc-*` → `--clearspeech-autogain-*`.
- `clearspeech()` kwargs: `agc_turns`, `agc_target_dbfs`, `agc_max_gain_db`, `agc_crossfade_ms` → `autogain_*`.
- `PipelineOptions` fields: `clearspeech_agc_target_dbfs`, `clearspeech_agc_max_gain_db` → `clearspeech_autogain_*`.
- Private effect implementation: `_apply_agc()` → `_apply_autogain()`.
- Sibling-WAV suffix: `<stem>.agc.wav` → `<stem>.autogain.wav` (and downstream `<stem>.autogain.bandpass.wav`, etc.).
- Log lines: `clearspeech.agc:` → `clearspeech.autogain:`.
- Architecture diagram label and prose in `docs/architecture.md`.

Older ADRs (0007, 0009, 0010, 0011, 0012, 0013, 0014, 0015) that mention `agc` or `AGC` are **left untouched**. ADRs are immutable snapshots of the decision at the time it was made; rewriting them would corrupt the historical record. A reader following the chain from ADR 0010 forward will eventually land here and learn the term changed.

## Consequences

**Breaking change for users of the CLI.** Anyone running `voice transcribe --clearspeech-chain agc,…` or `voice transcribe --clearspeech-agc-target-dbfs …` will get an `argparse` error after upgrading. The CHANGELOG entry under `[0.19.0]` lists the exact flag and value migrations.

**Breaking change for any external Python caller.** Anyone importing `voice.clearspeech.clearspeech()` and passing `agc_turns=` (kwargs are part of the public function signature) will get a `TypeError`. The project is pre-1.0, so this risk is accepted.

**Historical artefacts continue to use `agc`.** Old dump directories created under v0.18.0 and earlier still contain `02b-clearspeech-1-agc.wav`. Nothing reads those files programmatically — they are diagnostic — so no compatibility shim is needed.

**ADRs are the canonical history.** Following ADR 0010 (chain-of-effects design) → 0011 (bandpass defaults) → 0012 (chain-string CLI) → 0013 (presence defaults) → 0014 (denoise) → 0015 (dereverb) → 0016 (this rename) tells the full story. Search for "AGC" in older ADRs and you will find the original motivation; the new name lives only from this ADR forward.

## Alternatives considered

- **Keep the legacy CLI flags as deprecated aliases.** Rejected: the project has one known user (the author), and pre-1.0 explicitly allows breaking renames. Carrying deprecated aliases for several releases adds CLI-parser noise without protecting anyone.
- **Only rename the chain literal, leave Python identifiers as-is.** Rejected: split-vocabulary surfaces are worse than a clean rename. A reader bouncing between `--help` and `clearspeech()` source would have to translate.
- **Pick a more descriptive name (`per_turn_rms_normalize`, `loudness_per_speaker`).** Rejected as too long for a chain literal that users type by hand. `autogain` is the shortest readable English equivalent of the acronym.
