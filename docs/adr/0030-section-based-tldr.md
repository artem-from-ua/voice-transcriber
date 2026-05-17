---
status: accepted
date: 2026-05-17
---

# ADR 0030 — Section-based TL;DR with recursive aggregation

## Context

[Issue #146](https://github.com/artem-from-ua/voice-transcriber/issues/146) recorded a hard reboot of the reference 16 GB M1 Mac during `voice transcribe` on a 48-minute Ukrainian recording. No kernel-panic `.ips` was produced (consistent with wired-memory exhaustion: the kernel dies before it can write a dump). The pipeline lost the full preceding ASR + proofread + structure work because the renderer runs last and the in-flight task output was not flushed to a persistent log.

A stage-by-stage audit of every LLM-driven step established that this was the only remaining stage feeding the entire transcript to the LLM in one prompt:

| Stage | Prompt shape | OOM risk on 48-min input |
| --- | --- | --- |
| `proofread` | one segment + 3 neighbours per call | safe |
| `identify_speakers` | per-speaker `window_s` snippet | safe |
| `speech_structure` | chunked since v0.22.0 ([ADR 0018](0018-chunked-structure-dialog.md)) | safe |
| `safe_speech` | per-section (bounded by `speech_structure` output) | safe |
| **`speech_summary` (TL;DR)** | **entire dialog joined into one user message** | **OOM** |

The single-prompt path also has no path to graceful failure: a process kill we could catch, but a kernel reboot kills the whole machine — every editor, browser, call in flight.

## Decision

`speech_summary.generate_tldr` now builds the TL;DR recursively, one LLM call per section, so that no single prompt ever contains the full transcript:

```
level 0:  per section in dialog.sections   -> tldr_section_{lang}
                                              (input: raw segments of that section)
level k:  groups of <= TLDR_FANOUT          -> tldr_aggregate_{lang}
                                              (input: previous level's TL;DRs;
                                              prompt explicitly says input is
                                              already-compressed summaries)
final:    1..TLDR_FANOUT items left        -> tldr_final_{lang}
                                              (output: canonical TL;DR format
                                              the renderer consumes)
```

Constants (`src/voice/speech_summary.py`):
- `TLDR_FANOUT = 7` — mirrors `speech_structure.MAX_SECTIONS`, so a realistic input never enters the aggregate level.
- `TLDR_MAX_LEVELS = 4` — pathological-input guard (`7^4 = 2401` initial sections).

Singleton groups at the aggregate level pass through unchanged — re-summarising a single TL;DR adds no information.

The system prompt for each level is identical across calls in that level, so each level wraps its loop in `MlxLLM.prompt_cache_session([system_msg])` (introduced in [ADR 0027](0027-prompt-cache-llm-stages.md)). Only the per-section user content varies.

### `--no-structure` fallback

When the user passes `--no-structure`, `speech_structure.structure_dialog` falls back to one synthetic `Section(title="Розмова"|"Conversation", start_ms=…, end_ms=…)` covering the whole dialog. With section-based TL;DR that would collapse straight back to the OOM-prone single-prompt path. The stage now detects this synthetic-fallback shape and **skips TL;DR with a logged warning** instead. Users who want a TL;DR must re-run without `--no-structure`.

We rejected the alternatives:
- *Force-run `speech_structure` only for TL;DR*: silently doubles LLM stage cost; surprising for `--no-structure` users who chose the flag specifically to skip LLM work.
- *Auto-fallback to the old single-prompt path*: not a fix; reintroduces the bug it was created to avoid.

### MLX memory instrumentation

Every TL;DR LLM call wraps `mx.clear_cache()` / `mx.get_active_memory()` / `mx.get_peak_memory()` / `mx.reset_peak_memory()` and logs a line like:

```
speech_summary: [12/13] TL;DR section 3/5 (Дзвінок) mlx_active_before=4.91GB peak=6.04GB
```

The peak resets between calls so per-section numbers are real, not cumulative. This gives concrete evidence that the new shape stays inside the unified-memory budget.

### Persistent verbose log next to the audio

When `--verbose` is passed, `voice transcribe` mirrors `stderr` to `<input_audio_path>.log` (e.g. `2026-05-15_zendesk_spm_wfm_hm.wav.log`). The mirror is line-buffered with `os.fsync()` after every newline so a kernel reboot mid-run leaves a usable log behind. The CLAUDE.md `tee` convention still applies for live progress; the persistent log is the *backup* the bug taught us to need.

## Consequences

**Wins:**
- TL;DR works on hour-long recordings on the 16 GB reference hardware without `--no-tldr` workarounds.
- Per-call MLX peaks are now visible in the log; future stages that grow their prompt past the budget will surface as numbers, not a reboot.
- The persistent `<input>.log` is a generic safety net for any future kernel-level crash, not just OOMs from this stage.

**Trade-offs:**
- N+1 LLM calls per run instead of one. For the realistic 6-min / `~3` sections case, that is 4 calls in place of 1 — `tldr` wall-clock grows roughly proportionally. The model swap and weight load amortise across calls so the marginal cost per call is smaller than the headline ratio suggests.
- Final TL;DR quality depends on how well the model can synthesise from already-compressed inputs at the final pass. The aggregate prompt explicitly forbids smoothing names/numbers/decisions away into generic phrasing, but no golden TL;DR fixture exists yet to measure this objectively — see the open research items in #146.
- `--no-structure` users no longer get a TL;DR. Documented in CHANGELOG; the workaround is to drop `--no-structure` (or to write structure-skip TL;DR support in a future PR).

## Alternatives considered

- **`--max-llm-tokens` global safety net (option C in #146).** Would replace TL;DR with a placeholder string when input exceeded a token budget, without changing the prompt shape. Useful as a defense-in-depth layer for any future stage, but on its own does not give the user a TL;DR for long recordings — only skips it gracefully. Kept on the table for a separate PR.
- **Smaller LLM via `--llm-tldr-model` (option D).** Loading a 3-4B model for TL;DR only would buy headroom, at a quality cost and with extra model-load wall-clock. A workaround, not a fix.
