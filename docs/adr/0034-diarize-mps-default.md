---
status: accepted
date: 2026-05-18
---

# ADR 0034 — Keep `mps` as the diarization default; pyannote 3.1 runs fully on Apple GPU

## Context

[Issue #163](https://github.com/artem-from-ua/voice-transcriber/issues/163) opened a real question about a load-bearing default: `src/voice/diarize_speakers.py` selects `torch.device("mps")` whenever available, and the `Pipeline.from_pretrained(...).to(device)` call succeeds, so the startup log reads "Pipeline loaded in 1.5s on mps." That message tells us the `.to()` call did not raise — it does **not** tell us how much of the actual per-frame work executes on Apple GPU, because PyTorch's MPS backend silently routes unimplemented operators to CPU unless `PYTORCH_ENABLE_MPS_FALLBACK=1` is set and the caller listens for warnings. Pyannote 3.1's stack (sincnet, BLSTM, attention pooling, agglomerative clustering) is not uniformly MPS-supported on every torch release, so "MPS by name" could have meant "MPS for trivial ops + CPU for everything that matters."

If diarization were largely CPU-bound, the `to("mps")` call would still be pulling weights into unified memory for no benefit, costing budget that the downstream LLM stages need on a 16 GB Mac. So the question had real product weight.

The measurement was the only way to settle it. The full protocol and raw numbers live in [`docs/benchmarks/diarize-mps-vs-cpu.md`](../benchmarks/diarize-mps-vs-cpu.md); the JSON artefacts are under [`docs/measurements/163/`](../measurements/163/).

Three configurations were measured sequentially on the M1/16 GB reference machine with the laptop quieted per CLAUDE.md:

- **A — CPU** via `VOICE_DIARIZE_DEVICE=cpu` (a new env override added by this work specifically to make the measurement reproducible without monkey-patching).
- **B — MPS** with the current production code path.
- **C — MPS + `PYTORCH_ENABLE_MPS_FALLBACK=1`** with the harness capturing every fallback warning grouped by op name.

The headline numbers on a 6-minute two-speaker recording:

| config | diarize wall-clock | × real-time | fallback warnings |
| --- | --- | --- | --- |
| A cpu | 317.5 s | 0.85× | n/a |
| B mps | 23.3 s | 16.1× | 0 |
| C mps-trace | 23.2 s | 16.2× | 0 |

Turn counts were identical across all three configurations (111 turns). The 48-min recording confirmed the speedup scales — 218.7 s on MPS (13.2× real-time), again with zero fallback warnings. The CPU run on 48 min was deliberately skipped: the 6-min ratio is already conclusive and a 40-minute CPU run would have added no new information.

One harness subtlety worth recording for future readers: `diarize_speakers.diarize()` calls `free_torch_mps()` (which invokes `torch.mps.empty_cache()`) on its way out, so a post-call read of `torch.mps.current_allocated_memory()` always returns 0. The first MPS measurement failed its gate for exactly this reason. The harness now hooks `empty_cache` to capture the value at drain time and additionally reads `torch.mps.driver_allocated_memory()`, which survives `empty_cache()`. The Metal driver pool grew from 384 KB (baseline) to 4.16 MB during every MPS run — modest in absolute terms, but only occurs when GPU is exercised. Combined with the 13× speedup and the zero fallback count under explicit logging, the conclusion is unambiguous.

## Decision

Keep the default device selection in `src/voice/diarize_speakers.py` as `mps` when `torch.backends.mps.is_available()`. No production code path changes.

Two pieces of measurement infrastructure stay in the tree as permanent artefacts:

- **`VOICE_DIARIZE_DEVICE` env override** in `diarize_speakers.py` — set to `cpu` or `mps` to force the choice, otherwise auto-detect is unchanged. This exists so the bench can be replayed without monkey-patching when a future torch or pyannote release lands.
- **`scripts/diarize-device-bench.py`** harness — same purpose, plus its gate checks (non-zero Metal pool growth, zero fallback warnings) will catch a future regression as soon as someone runs it.

## Consequences

- The "MPS or not?" question is settled for torch 2.11.0 / pyannote 3.1: MPS is 13×–16× faster than CPU on this hardware with no silent fallback. The current default is correct.
- Pyannote diarize comfortably meets the project's "faster than real-time" floor (13× on 48 min) and leaves headroom for the LLM stages that run after it — there is no diarize-side pressure on the 16 GB memory budget.
- Future regressions are observable. Re-running `scripts/diarize-device-bench.py --config mps-trace --label short` against this benchmark's numbers will surface a slowdown, and a non-zero `fallback_warning_count` will name the regressing op.
- No follow-up issue for per-op fallback investigation — there is no fallback to investigate at the measured versions.

## Alternatives considered

- **Flip default to CPU "for memory headroom reasons."** Rejected: the measurement shows MPS uses *less* RSS (1.3 GB vs 3.0 GB on CPU) on top of being 13× faster. CPU would lose on both axes.
- **Keep `mps` default but add per-op CPU fallback for known-slow ops.** Rejected: no fallback is occurring. Adding manual op-by-op routing would be premature complexity for a problem that does not exist at these versions.
- **Drop the env override after the measurement.** Rejected: the marginal cost of three lines in `diarize_speakers.py` is negligible, and the override is exactly what makes this measurement repeatable next time torch or pyannote ships a major release.
