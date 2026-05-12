---
status: accepted
date: 2026-05-11
---

> **Note (v0.25.0):** the module formerly called `diarize` is now `diarize_speakers`. Body of this ADR preserves the historical name.

# ADR 0009 — Single diarize pass on the raw WAV; do not re-diarize after normalization

## Context

After ADR 0007 introduced per-segment loudness normalization between diarize and ASR, a natural follow-up question: should diarize run a second time on the normalized WAV, in case the AGC stage produced cleaner turn boundaries on the previously-quiet speaker?

The argument for two passes:

- pyannote-3.1's VAD has lower confidence on quiet speech, especially when the speaker is far from the microphone. AGC fixes the amplitude part.
- Speaker embeddings can be biased by the room characteristics that dominate a faint signal. Equalising loudness might give cleaner embeddings, hence cleaner cluster assignments.
- Cleaner boundaries flow into `merge` (which joins ASR segments to pyannote turns) and downstream stages.

The argument against:

- pyannote-3.1 is trained on mixed-loudness data and is empirically robust to a moderate amplitude range. Our 16 dB max-gain operates inside the range it has already seen during training.
- A second diarize pass costs ~25 s on a 6-minute recording (measured: ADR 0007's e2e runs showed 25.5 s for one pass). That is ~5% of the current end-to-end pipeline time.
- Two-pass diarize introduces a closed loop (AGC bounds depend on diarize, second diarize depends on AGC). Closed loops are debugging tax: when the output is wrong, the cause can live in either pass or in the interaction.
- Issue #47 motivated the AGC stage by *ASR* drift, not by *diarize* problems. The first-pass boundaries already produced 111 turns and correctly identified two speakers across the test recording — no observed quality regression in `merge` or `proofread` that points back to boundary errors.

## Decision

Keep a **single diarize pass** on the raw WAV. The output of that single pass is used both as AGC guard-rails (input to `loudness_normalize`) and as the speaker timeline (input to `merge`).

## Consequences

- Pipeline stays linear: each stage runs once, in a known order, with no feedback edges.
- The normalized WAV is consumed only by ASR. Other downstream stages (`merge`, `proofread`, `identify`, `structure`, `tldr`) work off the raw-WAV diarize output and the ASR transcript of the normalized WAV.
- If a future recording surfaces an observable diarize-quality issue that AGC could plausibly fix, the decision can be revisited as a follow-up issue with the experimental procedure outlined under "When to revisit" below.

## When to revisit

Reopen this decision if any of:

1. A real recording produces speaker-cluster errors (two speakers merged into one cluster, one speaker split across clusters) that correlate with quiet stretches.
2. `merge`'s assignment of ASR segments to speaker labels is wrong in a way that traces back to incorrect pyannote turn boundaries — not to AGC artefacts inside those boundaries.
3. A future ASR engine (#25, #50) ends up with timing significantly different from VibeVoice's, and the `merge` overlap heuristic loses accuracy on quiet ranges.

If any of those happens, the experiment is well-defined:

- Run the pipeline twice with `--dump-stages` — once normally, once with a hacked second-pass diarize.
- Save both `02-diarize.json` outputs.
- Diff the turn boundaries: count turns whose boundaries moved by > 100 ms, count speakers that changed cluster.
- Inspect a manually-judged subset to see whether the moved boundaries are improvements or regressions.
- Decide based on measurement, not on theoretical priors.

## Alternatives considered

- **Two passes, always on.** Rejected: 5% pipeline-time tax for an unmeasured benefit, plus the closed-loop debugging cost.
- **Two passes, opt-in via CLI flag.** Rejected for now: an opt-in flag with no default expectation of when to use it pollutes the CLI surface. Better to keep the option as "a future experiment" than as a "flag nobody uses".
- **Run diarize *only* on the normalized WAV, not at all on the raw WAV.** Cannot work — AGC needs turn boundaries *as input*, so diarize on the normalized WAV is by definition not the first stage that needs boundaries. Logically circular.
