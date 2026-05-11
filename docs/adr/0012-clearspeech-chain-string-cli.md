---
status: accepted
date: 2026-05-11
---

# ADR 0012 — Single `--clearspeech-chain` string replaces per-effect toggles

## Context

ADR 0010 promised that **free-order chains** would arrive in PR-2 alongside the second optional effect (presence boost). The v0.14.0 release froze a tiny allowlist (`_PR1_ALLOWED_CHAINS`) so only `(), ("agc",), ("bandpass",), ("agc", "bandpass")` were valid, and the CLI used three independent flags — `--no-clearspeech-agc`, `--clearspeech-bandpass`, and its tuning siblings — to control which effects ran. That fit two effects with one canonical order; it stops fitting once the chain grows.

PR-2 adds **presence boost** as the third effect and removes the order gate. The plan revision on 2026-05-11 also re-litigated the CLI surface: should we keep per-effect on/off flags (with an *optional* `--clearspeech-chain` override), or collapse to a single chain-string flag where order *and* enablement live in one place?

The team picked the single-string option. This ADR records why.

## Decision

The chain-of-effects CLI is now controlled by **one string flag**:

```
--clearspeech-chain "agc,bandpass,presence"
```

Rules:

- Comma-separated effect names; whitespace tolerated. Default value: `"agc"` — matches v0.14.0 behaviour for callers that pass no clearspeech flag at all.
- Empty string `""` disables all preprocessing (ASR sees the raw WAV).
- Effect names must be drawn from `KNOWN_EFFECTS = {"agc", "bandpass", "presence"}`; unknown names raise `ValueError`.
- Duplicates raise `ValueError` — running the same biquad twice is a programmer error, not a feature.
- Any permutation of known effects is valid. The chain is executed in the listed order.

Per-effect **tuning** flags remain unchanged in spirit but pick up the new `presence` siblings:

```
--clearspeech-agc-target-dbfs / --clearspeech-agc-max-gain-db
--clearspeech-bandpass-low-hz / --clearspeech-bandpass-high-hz
--clearspeech-presence-center-hz / --clearspeech-presence-boost-db / --clearspeech-presence-q
```

These accept values even when the corresponding effect is not in the chain; they are simply ignored. This keeps the CLI parser stateless — no cross-flag validation rules — and lets users construct ad-hoc invocations like `--clearspeech-chain agc --clearspeech-bandpass-low-hz 200` without an error message about the second flag being "unused" (it will be picked up next run if `bandpass` is added to the chain).

The previous **on/off toggles are removed**: `--no-clearspeech-agc`, `--clearspeech-bandpass`, `--no-clearspeech-bandpass` are gone in v0.15.0.

## Recommended chain (not the default)

The physically motivated full chain is:

```
agc        → vyrivnyaty rivni za turn-boundaries (amplitudinial foundation)
bandpass   → cut sub-vocal rumble and super-vocal noise
presence   → boost the 2–5 kHz consonant band inside what remains
```

This is the order in which each effect's input assumptions hold: AGC needs raw audio to read per-turn RMS; bandpass benefits from amplitude already flattened; presence shapes the already-cleaned band.

But the default chain stays `"agc"` (only). `bandpass` and `presence` are experimental — neither has shown a Metric A win as of v0.15.0 — so they are opt-in. Users running the recommended full chain explicitly type `--clearspeech-chain agc,bandpass,presence`.

## Consequences

- **Breaking CLI in v0.15.0 (pre-1.0, acceptable).** The migration table from the CHANGELOG covers the four common invocations; the two flags `--no-clearspeech-agc` and `--clearspeech-bandpass` are removed without aliases.
- **No cross-flag validation.** A tuning flag for an effect that isn't in the chain is silently ignored — the parser does not error. Users discover this either via the chain not behaving as expected (no progress label change) or via the dump artefacts (the per-effect WAV is absent). Trade-off accepted: stateful CLI rules cost more readability than they save user confusion.
- **`_PR1_ALLOWED_CHAINS` gate removed.** `clearspeech.clearspeech()` now validates only that effect names are known and unique. Order is the user's call.
- **`KNOWN_EFFECTS` becomes a single source of truth.** Adding `de-ess` (PR-3) or `denoise` (PR-4) is one frozenset entry plus a dispatch arm plus a `_apply_<name>()` function — no CLI flag additions.
- **Adding a new effect costs less CLI surface.** PR-3 (de-esser) brings only its tuning flags. The discoverability hit is real (users must read `--help` to see all current effects); mitigated by listing them explicitly in the `--clearspeech-chain` help text.
- **Default still preserves v0.14.0 behaviour.** Users who never typed any clearspeech flag get the same result before and after the breaking change.

## Alternatives considered

- **Keep per-effect toggles + add `--clearspeech-chain` as an optional override.** The first draft of the PR-2 plan. Required four interaction rules ("override wins for order, toggles win for enablement, conflict raises, both empty defaults to canonical") and a new `ClearspeechError` subtype to report cross-flag mismatches. Rejected for surface complexity: every new effect doubles the number of consistency checks, and the discoverability gain (toggles are easier to find in `--help` than a comma-separated string) is small.
- **Drop the chain concept entirely, use `--clearspeech-only-bandpass` / `--clearspeech-with-presence` style flags.** Rejected: the order matters physically and the user must be able to express it. Pure on/off toggles can't.
- **YAML / JSON config file for the chain.** Rejected as overkill for a CLI tool whose other knobs all live on the command line. If a future PR introduces more knobs (per-segment tuning, conditional chains), the config-file conversation re-opens.
- **`--clearspeech-effect agc --clearspeech-effect bandpass` (repeatable flag).** Argparse-idiomatic but the order in argv becomes load-bearing in a way users don't expect from `--key value` syntax. Rejected: comma-separated string makes ordering explicit in one place.
- **Migrate via deprecation period (keep both surfaces for one release).** Rejected because pre-1.0 already permits breaking changes, and the codebase doesn't carry compatibility shims (see CLAUDE.md). The migration table in the CHANGELOG documents the four mechanical rewrites users need to make.
