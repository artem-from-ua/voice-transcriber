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

_To be filled in after the run._ Template:

### 6-min recording

| config | load (s) | diarize (s) | × real-time | peak RSS (GB) | peak MPS (GB) | fallback warnings | on GPU? |
|---|---|---|---|---|---|---|---|
| A cpu | … | … | … | … | n/a | n/a | n/a |
| B mps | … | … | … | … | … | … | … |
| C mps-trace | … | … | … | … | … | `{op: n}` | … |

### 48-min recording

| config | load (s) | diarize (s) | × real-time | peak RSS (GB) | peak MPS (GB) | fallback warnings | on GPU? |
|---|---|---|---|---|---|---|---|
| A cpu | … | … | … | … | n/a | n/a | n/a |
| B mps | … | … | … | … | … | … | … |
| C mps-trace | … | … | … | … | … | `{op: n}` | … |

### Decision

Per the issue's decision tree:

- MPS ≥ 2× CPU → close WONTFIX, no code change.
- MPS within 0.8–1.5× CPU → keep MPS to spare CPU working set; add a
  note in `diarize_speakers.py` linking back here.
- MPS slower than CPU → flip default to CPU in a follow-up `perf/` PR
  and record the `torch`/`pyannote` versions in a new ADR.
- ≥ 100 fallback warnings/min of audio → open follow-up issue with the
  op-name distribution.
