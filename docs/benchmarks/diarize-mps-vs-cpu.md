# Diarize device benchmark — pyannote MPS vs CPU

Tracks [issue #163](https://github.com/artem-from-ua/voice/issues/163).

`diarize_speakers.py` requests `torch.device("mps")` when available and the
`Pipeline.from_pretrained(...).to(device)` call returns without raising.
That is necessary but not sufficient: PyTorch's MPS backend silently
routes unimplemented operators to CPU unless `PYTORCH_ENABLE_MPS_FALLBACK`
is set and the caller listens for `UserWarning`. Pyannote 3.1 layers
sincnet, BLSTM, attention pooling, and agglomerative clustering — not a
stack that is uniformly MPS-supported. This benchmark answers: **is
"diarize on MPS" really MPS, and is it faster than CPU?**

## 1. Input

- **Stage under test:** `voice.diarize_speakers.diarize()` end-to-end.
- **Code state:** branch `chore/diarize-mps-vs-cpu`, commit pinned in each
  JSON artefact (`torch_version` field captured automatically).
- **Audio:** the project's two reference recordings (paths are local,
  not in the repo, per the "no real names in git" rule):
  - 6-min two-speaker recording — fast iteration.
  - 48-min two-speaker recording — primary measurement (same fixture
    used to motivate [ADR 0031](../adr/0031-chunked-asr.md)).
- **Hardware:** M1 / 16 GB unified memory, laptop on AC power, foreground
  apps quieted per CLAUDE.md "quiet the laptop" rule.
- **Pyannote config:** default (`pyannote/speaker-diarization-3.1`),
  `num_speakers` passed as the known truth for the recording.

## 2. Expected output

For each audio × each configuration, a JSON artefact under
`docs/measurements/163/<config>-<wav-stem>.json` containing:

- `load_elapsed_s`, `diarize_elapsed_s`
- `peak_rss_bytes`, `peak_mps_allocated_bytes`
- `fallback_warning_count`, `fallback_ops` (map `op-name → count`)
- `mps_available`, `actually_on_gpu`, `gate_failures`

"Good" means: the three configs produce comparable `turns_count`
(diarization is non-deterministic but stable enough that turn counts
match within ±1), and the MPS configurations have non-zero
`peak_mps_allocated_bytes` so we can trust the "MPS" label.

## 3. Evaluation

Primary metric: **wall-clock `diarize_elapsed_s` per minute of audio**,
compared across three configurations:

| # | Config | Env | Question it answers |
|---|---|---|---|
| A | `cpu` | `VOICE_DIARIZE_DEVICE=cpu` | Ground-truth CPU baseline. |
| B | `mps` | `VOICE_DIARIZE_DEVICE=mps`, no fallback | Production default. Gate-checked. |
| C | `mps-trace` | `+ PYTORCH_ENABLE_MPS_FALLBACK=1` | What ops fall back, and how often. |

Secondary signals: peak `torch.mps.current_allocated_memory()` (must be
non-zero for B/C to be valid), per-op fallback-warning histogram from C.

### Gate (pre-condition for trusting B's number as "MPS")

Implemented in `scripts/diarize-device-bench.py:gate_check()`:

1. `torch.backends.mps.is_available()` must be True.
2. `peak_mps_allocated_bytes > 0` after the diarize call — proves the
   pipeline actually allocated GPU memory.
3. In config B (no explicit fallback env), `fallback_warning_count`
   must be 0. If non-zero, B is not "pure MPS" — escalate to C and
   compare A vs C only.

If B fails the gate, we report it explicitly in section 6 rather than
pretending the number is MPS.

## 4. Synthetic input / fixtures

None. Pyannote is opaque; synthetic audio would not exercise the same
op mix. We measure on real recordings.

## 5. Real test input

Local paths supplied at run time (see issue #163). Raw measurement
artefacts go to `docs/measurements/163/`.

Run order (sequentially — no parallel inference per CLAUDE.md):

```bash
for cfg in cpu mps mps-trace; do
  uv run python scripts/diarize-device-bench.py \
    --wav <local>/short-6min.wav --label short \
    --config $cfg --num-speakers 2 \
    2>&1 | tee /tmp/diarize-163-$cfg-short.log
done

for cfg in cpu mps mps-trace; do
  uv run python scripts/diarize-device-bench.py \
    --wav <local>/long-48min.wav --label long \
    --config $cfg --num-speakers 2 \
    2>&1 | tee /tmp/diarize-163-$cfg-long.log
done
```

`--label` is required and becomes the only audio identifier written to
the JSON artefact and to its filename. Use a fictional short name
(e.g. `short` / `long`) — never the real wav stem, which on private
recordings carries company / person names that must not enter git.

## 6. Result and decision

Hardware: M1 / 16 GB, torch 2.11.0, pyannote/speaker-diarization-3.1.
Laptop quieted per CLAUDE.md (Safari/IDE/Slack closed, AC power) before
each run. Configurations ran sequentially.

### 6-min recording (375 s, two speakers, `num_speakers=2` passed)

| config | load (s) | diarize (s) | × real-time | peak RSS (GB) | peak Metal driver pool (MB) | fallback warnings | on GPU? |
|---|---|---|---|---|---|---|---|
| A cpu        | 1.2 | 317.5 | **0.85×** | 3.0  | n/a | n/a | n/a |
| B mps        | 1.2 | 23.3  | **16.1×** | 1.3  | 4.1 | 0 | ✅ |
| C mps-trace  | 1.3 | 23.2  | **16.2×** | 1.3  | 4.1 | 0 | ✅ |

Turn count was 111 in all three configs (diarization output is identical).

### 48-min recording (2884 s, no `num_speakers` hint)

CPU was skipped — the 6-min ratio (13.7× speedup, identical turns) is
already conclusive, and a 48-min CPU run on M1 would take ~40 wall-clock
minutes for no new information.

| config | load (s) | diarize (s) | × real-time | peak RSS (GB) | peak Metal driver pool (MB) | fallback warnings | on GPU? |
|---|---|---|---|---|---|---|---|
| A cpu        | _skipped — see above_ | | | | | | |
| B mps        | 1.2 | 218.7 | **13.2×** | 1.4  | 4.1 | 0 | ✅ |
| C mps-trace  | 1.4 | 219.2 | **13.2×** | 1.4  | 4.1 | 0 | ✅ |

Turn count was 305 in both MPS configs.

### Verification: actually on GPU?

The first MPS run revealed a harness bug: `diarize()` calls
`free_torch_mps()` (`torch.mps.empty_cache()`) internally right before
returning, so a post-call read of `torch.mps.current_allocated_memory()`
is always 0. The harness now hooks `torch.mps.empty_cache` to capture
the value at drain time, and additionally reads
`torch.mps.driver_allocated_memory()` which survives `empty_cache()`.

The Metal driver pool grew from **384 KB** (pre-diarize baseline) to
**4.16 MB** in every MPS run — modest in absolute terms but consistent
and only happens when GPU is exercised. Combined with the 13× wall-clock
speedup over CPU, identical turn counts across configs, and the **zero
fallback warnings with `PYTORCH_ENABLE_MPS_FALLBACK=1`** explicitly
enabled, the conclusion is unambiguous: **pyannote 3.1 + torch 2.11
runs entirely on Apple GPU with no silent CPU fallback**.

The original concern from issue #163 — "we don't know whether MPS is
buying 5% or 50%" — answers as **13×**, with no per-op CPU fallback at
all.

### Decision

Per the issue's decision tree, this is the "MPS ≥ 2× CPU" branch: keep
the default as-is, no code change. Specifically:

- Default device selection in `src/voice/diarize_speakers.py` stays at
  `mps` when available.
- The `VOICE_DIARIZE_DEVICE` env override (added by this work) stays in
  the codebase as a permanent escape hatch so this measurement can be
  replayed in the future without monkey-patching.
- No new issue for per-op fallback investigation — there is no fallback
  to investigate at the measured torch/pyannote versions.

Pinned for future regressions: if a later torch or pyannote release
makes diarize slower, re-run `scripts/diarize-device-bench.py --config
mps-trace --label short` and compare against this baseline. A non-zero
`fallback_warning_count` will name the regressing op.

See [ADR 0034](../adr/0034-diarize-mps-default.md) for the formal record.
