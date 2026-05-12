---
status: accepted
date: 2026-05-11
---

# ADR 0010 — Audio cleanup as a chain-of-effects stage, not a series of pipeline stages

## Context

Issue #47 added one DSP step between diarisation and ASR — per-turn AGC, implemented as `voice.audio_preprocess.loudness_normalize`. Issue #48 plans four more steps (bandpass, presence boost, de-esser, denoise) plus a likely fifth in a follow-up issue (dereverb). The natural question while planning the first of those follow-ups (E2a bandpass) was: do these land as a series of additional pipeline stages, or absorb into a single stage with internal structure?

The initial plan went with "series of stages": `[5] spectral` slotted between the old `[4] normalize` and `[5] asr`, renumber everything downstream, ship default-off flags. That works for one effect but pays compound cost on every subsequent sub-experiment:

- Each PR renumbers progress labels (`[N/10] → [N/11] → [N/12] → …`), test assertions on those labels, and the architecture diagram. Five planned effects = five renumbering cycles.
- The pipeline diagram grows wider with each PR without a corresponding rise in conceptual clarity — five tiny boxes between diarise and ASR all do "clean the audio for ASR" and a reader has to look up each one to know what it does.
- A/B observability between effects ends up ad-hoc — each stage gets one dump artefact (`02b-normalized.wav`, then `02c-spectral.wav`, then `02d-…`), with no shared layout.
- The fixed *ordering* of effects becomes implicit in the pipeline source. The first plan revision spent a full section debating whether bandpass belonged before or after AGC; either answer required wiring it into a fixed slot in the file.

## Decision

Replace the single `loudness_normalize` stage with one stage called **`clearspeech`** that runs an ordered chain of DSP effects. PR-1 ships two: `agc` (the v0.13.0 logic, transplanted) and `bandpass` (issue #48 E2a). The chain extends in subsequent PRs.

API:

```python
clearspeech(
    wav_path,
    chain: tuple[str, ...],          # e.g. ("agc", "bandpass") or ()
    *,
    agc_turns,
    agc_target_dbfs, agc_max_gain_db, agc_crossfade_ms,
    bandpass_low_hz, bandpass_high_hz,
    log,
    dump=None,                       # hook called once per applied effect
) -> tuple[Path, dict]
```

Pipeline progress label stays `[4/10] Clearspeech (<chain>)`. Stage count stays at 10/10. Each enabled effect writes a sibling WAV (`<stem>.agc.wav` → `<stem>.agc.bandpass.wav` → …) and feeds the next; ASR reads the output of the last applied effect, or the raw WAV if the chain is empty.

For PR-1 the order is **canonically fixed** at `agc → bandpass` (validated by a frozenset of allowed chain tuples). Free-order chains arrive in PR-2 alongside the presence-boost effect, when reordering becomes physically meaningful — until then "free order" would only buy `(bandpass, agc)`, which is debatable on its merits and not worth a CLI surface yet.

## Consequences

- **Pipeline stage count is now stable** across all of issue #48 and its follow-ups. Adding `presence`, `de-ess`, `denoise`, `dereverb` is a chain extension, not a renumbering.
- **Per-effect observability is first-class.** `--dump-stages DIR` writes `02b-clearspeech-config.json` plus one `02b-clearspeech-N-<effect>.wav` per applied effect. Comparing the WAV after AGC to the WAV after bandpass is now exactly as cheap as `diff 03-asr.json 05-proofread.json` for LLM stages.
- **Breaking CLI rename in v0.14.0.** `--no-loudness-normalize` → `--no-clearspeech-agc`, `--loudness-target-dbfs` → `--clearspeech-agc-target-dbfs`, `--loudness-max-gain-db` → `--clearspeech-agc-max-gain-db`. No aliases — pre-1.0 project, update scripts directly.
- **Breaking dump layout.** `02b-normalized.wav` is gone; readers of dumped runs need to switch to the new layout.
- **Internal module reshuffle.** `voice.audio_preprocess` removed; its `loudness_normalize` is now `voice.clearspeech._apply_agc`. ADR 0007 retains its empirical conclusions verbatim — only the CLI / module surface changed.
- **Order constraint cost** lives in one place (the `_PR1_ALLOWED_CHAINS` frozenset) instead of being spread across `pipeline.py` argument-wiring. PR-2 can lift this gate in one targeted change.

## Alternatives considered

- **Separate `[5] spectral` stage (the plan up to 2026-05-11).** Rejected for the renumbering / observability / fixed-order reasons listed in *Context*. Cheap for E2a alone; expensive across the full E2a..E2e roadmap.
- **Two-phase: ship `spectral` standalone first, refactor to chain later.** Rejected: pays the rename + breaking-dump cost twice, with a 1–2 week intermediate state where the pipeline carries vestigial structure. The breaking change is the same; doing it once is strictly cheaper.
- **Variant C from planning — chain stage but no free-order in PR-1.** Adopted (this ADR). Free-order is a feature with no current customer (only two effects, canonical order is the only physically reasonable one). Lifting it in PR-2 alongside the presence effect, when reordering becomes meaningful, lets us decide the CLI surface for "chain spec" against a concrete use case instead of a hypothetical one.
- **Compose effects via a Unix-pipe-style external API.** Rejected: the chain is internal — every effect is a function in `clearspeech.py`, sharing the 16 kHz mono PCM_16 invariant. A subprocess pipe would multiply I/O cost and break atomic dumping. The chain-of-Python-callables model already gives us composability without any of those costs.
- **Make the chain a list of dataclass objects (`AgcEffect(target_dbfs=...)`, `BandpassEffect(low=...)`) instead of string tags + flat kwargs.** Cleaner long-term and probably the right move once the chain grows past four effects. Deferred to PR-2/3 — the flat-kwargs API is fine for two effects and avoids a second public-API churn in the same release.

## Note (v0.15.0)

PR-2 (`feature/clearspeech-presence`) added the `presence` effect and lifted the `_PR1_ALLOWED_CHAINS` gate — `clearspeech()` now accepts any permutation of known effects (`agc`, `bandpass`, `presence`); duplicates and unknown names raise `ValueError`. The CLI also collapsed from per-effect toggles to a single `--clearspeech-chain` string. See [ADR 0012](0012-clearspeech-chain-string-cli.md) for the new CLI shape and rationale. The chain-of-effects architecture decision from this ADR remains in force.

## Note (v0.16.0)

PR-3 (`feature/clearspeech-denoise`) added the `denoise` effect (FFT-domain spectral subtraction via `ffmpeg afftdn`). Default chain stays `"agc"` — no Metric A win at any tested position. The interesting finding for the chain abstraction is that physics intuition about effect ordering (denoise must run on raw signal, i.e. pre-AGC) lost to empirical measurement — pre-AGC denoise is catastrophic on this material (27.5 % RU drift), post-AGC denoise is "merely worse than nothing" (8.2 % vs 5.1 % baseline). See [ADR 0014](0014-clearspeech-denoise.md) for the five-run grid and the chain-position reasoning.

## Note (v0.17.0)

PR-4 (`feature/clearspeech-dereverb`) added the `dereverb` effect (per-pyannote-turn Lebart-Polack spectral subtraction; the first effect that genuinely needs diarisation turns to run). Default chain stays `"agc"`. The headline result is methodological: the isolated-dereverb Metric A run dropped to 0 % RU-glyph rate, which *looks* like a runaway win — but the side-by-side transcript audit shows it dropped ~9 % of Ukrainian content into non-speech markers. The 0 % is rate-not-volume — Simpson's paradox on the metric. The chain-of-effects architecture stays intact across five effects; what loosened is the trust in any one metric to validate an effect on its own. See [ADR 0015](0015-clearspeech-dereverb.md) for the full audit and the follow-up issue for cross-engine validation.
