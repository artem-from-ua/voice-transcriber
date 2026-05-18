"""Whisper-large-v3-MLX ASR backend.

`transcribe()` runs `mlx-whisper` in-process against
`mlx-community/whisper-large-v3-mlx`. This is the only ASR backend in the
project since v0.23.0 — see ADR 0017 (how it was chosen) and ADR 0021 (why
the legacy VibeVoice backend was removed).

The model is read out of the HuggingFace cache that `huggingface_hub`
populates, because `mlx-whisper` expects the standard HF layout
(blobs/refs/snapshots) and resolves repo ids through it directly. The
model is never auto-downloaded inside `transcribe`; the user runs
`voice download-whisper` once at onboarding and we hard-fail with an
actionable error if the cache is empty.

For long recordings the wrapper slices the audio in Python and calls
`mlx_whisper.transcribe` once per slice with a fresh MLX cache between
calls — see ADR 0031 / issue #147. Whisper's own 30-second windows live
*inside* a transcribe() call; the slicing here is a layer above that,
bounding the per-call peak so an hour-long recording does not trip the
wired-memory ceiling on a 16 GB Mac.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Callable

from ._memory import free_mlx
from .types import AsrSegment


WHISPER_REPO_ID = "mlx-community/whisper-large-v3-mlx"

# ASR chunking knobs — see ADR 0031 (geometry) and ADR 0036 (dedup).
#
# On the M1/16 GB reference machine a 6-min input produces a 7.17 GB MLX
# peak with the model resident (~3 GB) + per-call working set scaling with
# audio duration. Empirical baseline:
#     6 min   -> peak 7.17 GB
#     ~10 min -> headroom still safe alongside other resident artefacts
#     48 min  -> SIGKILL during the call (predates peak instrumentation)
#
# 10 min chosen as a conservative single-pass ceiling. 8 min per chunk
# keeps each call comfortably inside the working envelope; 5 s overlap
# gives Whisper enough audio context on both sides of every cut for
# `_dedup_overlap` to recognise repeated phrases and remove them.
ASR_CHUNK_THRESHOLD_S = 600.0
ASR_CHUNK_SIZE_S = 480.0
ASR_CHUNK_OVERLAP_S = 5.0

# Text-similarity dedup knobs — see ADR 0036 / issue #172.
#
# How far back into `accumulated` to look for matches. 30 s is wider than
# the chunk overlap (5 s) on purpose: Whisper assigns segment timestamps
# inside its own 30-s decode windows, and the duplicate emitted in
# chunk N+1's head can land several seconds past the strict overlap
# cutoff with a long enough trailing context in chunk N's tail.
ASR_DEDUP_OVERLAP_WINDOW_S = 30.0
ASR_DEDUP_JACCARD_THRESHOLD = 0.5
ASR_DEDUP_MIN_TOKEN_COUNT = 3
# Aggregate (union-token) Jaccard threshold for the SUPERSEDE branch:
# when chunk N+1's head segment audio overlaps multiple accumulated tail
# segments, this catches the "Whisper re-emitted a longer, richer version
# of several short tail fragments" case (pairwise max stays low because
# each fragment is short, but the union shares most vocabulary). Drops
# the superseded tail segments and keeps the incoming one. 0.3 is lower
# than the pairwise threshold (0.5) because the union always inflates
# the denominator — see the V4b B3/B4 walk-through in the post-PR review
# of #172 for the empirical calibration.
ASR_DEDUP_SUPERSEDE_AGG_THRESHOLD = 0.3


class AsrError(RuntimeError):
    pass


def _noop_log(_msg: str) -> None:
    return None


def _verify_model_cached(repo_id: str = WHISPER_REPO_ID) -> None:
    """Raise AsrError if the Whisper model is not present in the HF cache.

    Uses `try_to_load_from_cache` which is the cheap, no-network lookup —
    `snapshot_download` would silently start a 3 GB fetch.
    """
    from huggingface_hub import try_to_load_from_cache
    from huggingface_hub.errors import CacheNotFound

    try:
        hit = try_to_load_from_cache(repo_id=repo_id, filename="config.json")
    except CacheNotFound:
        hit = None
    if hit is None:
        raise AsrError(
            f"Whisper model {repo_id!r} is not in the HuggingFace cache. "
            f"Run `voice download-whisper` once to fetch it (~3 GB)."
        )


def _parse_segments(result: dict) -> list[AsrSegment]:
    """Convert mlx-whisper's `{text, segments, language}` payload to AsrSegment[].

    Whisper has no speaker-prediction head; speaker labels are assigned
    downstream by `merge.py` via max-overlap with pyannote turns.
    """
    segments: list[AsrSegment] = []
    for seg in result.get("segments", []):
        content = (seg.get("text") or "").strip()
        if not content:
            continue
        segments.append(AsrSegment(
            start=float(seg.get("start", 0.0)),
            end=float(seg.get("end", 0.0)),
            content=content,
        ))
    return segments


def load_model(
    log: Callable[[str], None] = _noop_log,
) -> tuple[Any, float]:
    """Load Whisper-large-v3-MLX weights and return (model, load_elapsed_s).

    Separated from transcribe() so the pipeline can measure model load time
    independently from inference time and report both in the Markdown header.
    """
    _verify_model_cached()
    log(f"Whisper-large-v3-MLX from HF cache ({WHISPER_REPO_ID})")

    from mlx_whisper.load_models import load_model as _mlx_load_model

    t0 = time.perf_counter()
    model = _mlx_load_model(WHISPER_REPO_ID)
    load_elapsed = time.perf_counter() - t0
    log(f"Whisper loaded in {load_elapsed:.1f}s.")
    return model, load_elapsed


def _call_mlx_whisper(
    audio_input: Any,
    *,
    language: str | None,
    label: str,
    log: Callable[[str], None],
) -> tuple[dict, float, float]:
    """One mlx_whisper.transcribe call wrapped in MLX peak instrumentation.
    Returns (result_dict, mlx_peak_gb, elapsed_s)."""
    import mlx.core as mx
    import mlx_whisper

    mx.clear_cache()
    mx.reset_peak_memory()
    before_gb = mx.get_active_memory() / 1e9
    log(f"whisper_asr: {label} pre-transcribe mlx_active={before_gb:.2f}GB")

    t0 = time.perf_counter()
    result = mlx_whisper.transcribe(
        audio_input,
        path_or_hf_repo=WHISPER_REPO_ID,
        language=language,
        condition_on_previous_text=False,
    )
    elapsed = time.perf_counter() - t0
    peak_gb = mx.get_peak_memory() / 1e9
    log(
        f"whisper_asr: {label} post-transcribe "
        f"mlx_peak={peak_gb:.2f}GB elapsed={elapsed:.1f}s"
    )
    return result, peak_gb, elapsed


def _shift_segments(segments: list[AsrSegment], offset_s: float) -> list[AsrSegment]:
    if offset_s == 0.0:
        return segments
    return [
        AsrSegment(start=s.start + offset_s, end=s.end + offset_s, content=s.content)
        for s in segments
    ]


# Lowercase alphanumerics, Cyrillic + Latin + digits. Mirrors the regex
# in scripts/asr-chunk-boundary-quality.py (lines 101-113). If you change
# one, change the other so the benchmark and production agree on what
# counts as a token.
_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яҐґЄєІіЇї0-9']+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _jaccard(a: list[str], b: list[str]) -> float:
    """Set-Jaccard on token bags. 0.0 on empty input."""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def _dedup_overlap(
    accumulated: list[AsrSegment],
    incoming: list[AsrSegment],
    *,
    overlap_window_s: float = ASR_DEDUP_OVERLAP_WINDOW_S,
    jaccard_threshold: float = ASR_DEDUP_JACCARD_THRESHOLD,
    min_token_count: int = ASR_DEDUP_MIN_TOKEN_COUNT,
    supersede_aggregate_threshold: float = ASR_DEDUP_SUPERSEDE_AGG_THRESHOLD,
) -> list[AsrSegment]:
    """Reconcile chunk N+1's head against chunk N's tail and return the
    full updated `accumulated` list (not just kept `incoming`).

    Whisper's per-chunk 30 s decode windows can re-emit a phrase from the
    end of chunk N at the start of chunk N+1. Three patterns appear in
    practice:

      1. **Plain duplicate** — chunk N+1's head segment is text-similar
         to one or more `accumulated` tail segments and its audio span
         overlaps theirs. Drop the head segment, keep the tail.
      2. **Superseding rewind** — chunk N+1's head segment audio span
         *contains* one or more accumulated tail segments AND is
         text-similar to their union (Jaccard against the union >=
         `supersede_aggregate_threshold`). The longer head segment is
         Whisper's better, longer-context transcription of the same
         audio region. **Drop the superseded tail segments, keep the
         head.** Without this branch the rendered transcript shows
         both the short tail fragments and the long head segment back
         to back — visible duplication despite the pairwise text-only
         check passing (issue #172 post-PR review on the 48-min ref).
      3. **Legitimate continuation** — chunk N+1's head segment has
         low Jaccard against tail AND no audio span overlap. Keep it,
         stop scanning further `incoming` (boundary repeats live at the
         head, not the middle).

    For each `incoming` segment with at least `min_token_count` tokens:
      - pick the `accumulated` tail by `s.end >= last_end - overlap_window_s`
        (so a long segment crossing the boundary still participates)
      - identify tail segments whose audio span overlaps `seg`'s span
        (`seg.start < t.end` and `seg.end > t.start`)
      - compute pairwise max Jaccard against tail tokens
      - compute Jaccard against the *union* of overlapping tail tokens
        (the aggregate signal that catches "one head segment ate several
        tail fragments")
      - decide:
          * if pairwise max >= `jaccard_threshold` -> drop head (case 1)
          * elif aggregate against overlapped tail >= `supersede_aggregate_threshold`
            AND there are overlapped tail segs -> drop those tail segs,
            keep head (case 2)
          * else -> keep head, stop scanning (case 3)

    Short segments (fewer than `min_token_count` tokens) bypass the
    Jaccard checks — a backchannel like "yeah okay so" would otherwise
    self-match nearby backchannels at the join.

    Returns the full updated accumulated list: original accumulated minus
    any superseded tail segments, plus the kept `incoming` segments."""
    if not accumulated:
        return list(incoming)
    if not incoming:
        return list(accumulated)

    last_end = accumulated[-1].end
    tail_window_start = last_end - overlap_window_s

    # Working copy of accumulated we may mutate (drop superseded tails).
    # Use ids rather than indices because we never reorder, only drop.
    out_accumulated: list[AsrSegment] = list(accumulated)

    def _tail_indices() -> list[int]:
        return [
            i for i, s in enumerate(out_accumulated)
            if s.end >= tail_window_start
        ]

    kept_incoming: list[AsrSegment] = []
    dedup_done = False
    for seg in incoming:
        if dedup_done:
            kept_incoming.append(seg)
            continue
        seg_tokens = _tokenize(seg.content)
        if len(seg_tokens) < min_token_count:
            kept_incoming.append(seg)
            continue

        tail_idx = _tail_indices()
        if not tail_idx:
            kept_incoming.append(seg)
            dedup_done = True
            continue

        tail_tokens_each = [
            _tokenize(out_accumulated[i].content) for i in tail_idx
        ]
        max_pairwise = max(_jaccard(seg_tokens, t) for t in tail_tokens_each)

        if max_pairwise >= jaccard_threshold:
            # Case 1: plain duplicate. Drop incoming; keep scanning, the
            # next head segment may also be a duplicate.
            continue

        # Identify accumulated tail segments whose audio overlaps `seg`.
        overlap_idx = [
            i for i in tail_idx
            if seg.start < out_accumulated[i].end
            and seg.end > out_accumulated[i].start
        ]
        if overlap_idx:
            union_tokens: list[str] = []
            for i in overlap_idx:
                union_tokens.extend(_tokenize(out_accumulated[i].content))
            aggregate = _jaccard(seg_tokens, union_tokens)
            if aggregate >= supersede_aggregate_threshold:
                # Case 2: superseding rewind. Drop those tail segments,
                # keep the (longer, context-rich) incoming.
                drop_set = set(overlap_idx)
                out_accumulated = [
                    s for i, s in enumerate(out_accumulated)
                    if i not in drop_set
                ]
                kept_incoming.append(seg)
                dedup_done = True
                continue

        # Case 3: legitimate continuation.
        kept_incoming.append(seg)
        dedup_done = True

    out_accumulated.extend(kept_incoming)
    return out_accumulated


def transcribe(
    wav_path: str | Path,
    *,
    language: str | None = None,
    model: Any = None,
    log: Callable[[str], None] = _noop_log,
    progress_state: Callable[..., None] | None = None,
) -> list[AsrSegment]:
    """Run Whisper-large-v3-MLX on a WAV. Returns a list of AsrSegment.

    `language=None` lets mlx-whisper auto-detect on the first 30 s window.
    The pipeline normally precedes this with [4] lang_detect (ADR 0022),
    which provides a stronger detection signal from the longest pyannote
    turn, but direct callers that skip the diarize step can still get a
    best-effort transcript by passing None.

    When `model` is provided (pre-loaded via `load_model()`), weights are
    reused — mlx_whisper.transcribe accepts a model object directly.
    When `model` is None, mlx_whisper loads weights internally (backwards-
    compatible path for callers that skip the separate load step).

    For recordings longer than ASR_CHUNK_THRESHOLD_S the audio is sliced
    in Python and Whisper is called once per slice with a fresh MLX cache
    between calls. See ADR 0031 / issue #147.
    """
    # Lazy import: mlx_whisper pulls in numba/tiktoken/llvmlite, ~1 s warmup
    # we do not want to pay on every import of this module (e.g. in tests
    # that don't exercise the Whisper branch).
    import mlx_whisper

    if model is None:
        _verify_model_cached()
        log(f"Whisper-large-v3-MLX from HF cache ({WHISPER_REPO_ID})")

    # Probe audio duration with the same loader Whisper uses, so we make
    # the chunk/single-pass decision off the exact sample count Whisper
    # will see (no ffprobe round-trip, no extra file I/O).
    from mlx_whisper.audio import SAMPLE_RATE, load_audio

    audio = load_audio(str(wav_path))
    duration_s = len(audio) / SAMPLE_RATE

    if duration_s <= ASR_CHUNK_THRESHOLD_S:
        # Short audio: pass the path through verbatim (preserves existing
        # behaviour and lets mlx_whisper handle file I/O itself).
        del audio
        result, peak_gb, elapsed = _call_mlx_whisper(
            str(wav_path),
            language=language, label="single-pass",
            log=log,
        )
        log(f"ASR finished in {elapsed:.1f}s.")
        segments = _parse_segments(result)
        del result
        free_mlx(log)
        return segments

    # Long audio: slice in Python, call Whisper per slice, stitch.
    chunk_samples = int(ASR_CHUNK_SIZE_S * SAMPLE_RATE)
    overlap_samples = int(ASR_CHUNK_OVERLAP_S * SAMPLE_RATE)
    step = chunk_samples - overlap_samples
    n_chunks = (len(audio) + step - 1) // step
    log(
        f"whisper_asr: long audio ({duration_s:.0f}s) -> "
        f"{n_chunks} chunks of {ASR_CHUNK_SIZE_S:.0f}s "
        f"with {ASR_CHUNK_OVERLAP_S:.0f}s overlap"
    )

    segments: list[AsrSegment] = []
    overall_t0 = time.perf_counter()
    for chunk_idx in range(n_chunks):
        start_sample = chunk_idx * step
        end_sample = min(start_sample + chunk_samples, len(audio))
        if start_sample >= len(audio):
            break
        chunk_audio = audio[start_sample:end_sample]
        chunk_start_s = start_sample / SAMPLE_RATE
        label = f"chunk {chunk_idx + 1}/{n_chunks} @{chunk_start_s:.0f}s"

        if progress_state is not None:
            progress_state("chunk", chunk_idx, n_chunks)

        result, _peak_gb, _elapsed = _call_mlx_whisper(
            chunk_audio,
            language=language, label=label,
            log=log,
        )
        chunk_segments = _shift_segments(_parse_segments(result), chunk_start_s)
        # _dedup_overlap returns the full updated accumulated list
        # (original minus superseded tail segments plus kept incoming);
        # see the SUPERSEDE branch in its docstring.
        segments = _dedup_overlap(segments, chunk_segments)
        del result

    del audio
    log(f"ASR finished in {time.perf_counter() - overall_t0:.1f}s.")
    # Drop the Whisper weights and KV cache before downstream LLM stages —
    # on 16 GB Macs they otherwise sit alongside the LLM and trigger a
    # Metal OOM at structure_dialog.
    free_mlx(log)
    return segments
