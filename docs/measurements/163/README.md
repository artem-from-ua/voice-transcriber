# Issue #163 — pyannote MPS vs CPU measurement artefacts

Raw JSON output from `scripts/diarize-device-bench.py`. Schema and run
instructions are in [`docs/benchmarks/diarize-mps-vs-cpu.md`](../../benchmarks/diarize-mps-vs-cpu.md).

File naming: `<config>-<wav-stem>.json` where `<config>` ∈ {cpu, mps, mps-trace}.

The aggregated decision is written back into the benchmark doc's
section 6, not here. This directory exists only so future contributors
can replay or audit the underlying numbers.
