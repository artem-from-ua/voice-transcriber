# Issue #163 — pyannote MPS vs CPU measurement artefacts

Raw JSON output from `scripts/diarize-device-bench.py`. Schema and run
instructions are in [`docs/benchmarks/diarize-mps-vs-cpu.md`](../../benchmarks/diarize-mps-vs-cpu.md).

File naming: `<config>-<label>.json` where `<config>` ∈ {cpu, mps, mps-trace}
and `<label>` is a fictional short name (`short` / `long`) supplied via
`--label`. Real wav stems are deliberately not used — local recordings
carry company / person names that must not enter git.

The aggregated decision is written back into the benchmark doc's
section 6, not here. This directory exists only so future contributors
can replay or audit the underlying numbers.
