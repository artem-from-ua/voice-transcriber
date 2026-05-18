"""Unit tests for the Whisper ASR backend.

Patches `mlx_whisper.transcribe`, `mlx_whisper.audio.load_audio` and
`huggingface_hub.try_to_load_from_cache` so the tests never touch the real
model or network. The assertions verify:
  - mlx-whisper segments are parsed into `AsrSegment`
  - missing-cache raises `AsrError` with an actionable message
  - the `language` kwarg flows through to mlx-whisper
  - `condition_on_previous_text=False` is locked in (matches stone-scriber)
  - short audio takes the single-pass path; long audio is chunked with
    timestamp offsets and overlap deduplication (#147 / ADR 0031).
"""

from __future__ import annotations

import sys
import types
from typing import Any

import numpy as np
import pytest

from voice import whisper_asr
from voice.whisper_asr import AsrError
from voice.types import AsrSegment


def _install_fake_mlx_whisper(
    monkeypatch,
    return_payload: dict | None = None,
    payloads: list[dict] | None = None,
    capture: dict | None = None,
    audio_duration_s: float = 60.0,
) -> dict:
    """Inject a stub `mlx_whisper` module so `import mlx_whisper` in the SUT
    picks it up without hitting the real package.

    Use `payloads=[{...}, {...}]` to return a different payload per chunk
    call. With `return_payload` the same payload is returned every call.
    Capture defaults to {} and is populated with `calls: list[(audio, kwargs)]`.
    """
    if capture is None:
        capture = {}
    capture.setdefault("calls", [])

    fake = types.ModuleType("mlx_whisper")
    fake_audio = types.ModuleType("mlx_whisper.audio")
    fake_audio.SAMPLE_RATE = 16000
    fake_audio.load_audio = lambda path: np.zeros(
        int(audio_duration_s * 16000), dtype=np.float32
    )
    fake.audio = fake_audio

    def fake_transcribe(audio, **kwargs):
        capture["calls"].append((audio, dict(kwargs)))
        # Back-compat: last call's audio/kwargs also stored at top level.
        capture["audio"] = audio
        capture["kwargs"] = kwargs
        if payloads is not None:
            idx = len(capture["calls"]) - 1
            return payloads[idx] if idx < len(payloads) else {"segments": []}
        return return_payload or {"segments": []}

    fake.transcribe = fake_transcribe
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake)
    monkeypatch.setitem(sys.modules, "mlx_whisper.audio", fake_audio)
    return capture


def _patch_cache_hit(monkeypatch) -> None:
    monkeypatch.setattr(
        whisper_asr,
        "_verify_model_cached",
        lambda *a, **kw: None,
    )


def test_transcribe_parses_mlx_whisper_segments(monkeypatch):
    capture: dict[str, Any] = {}
    _install_fake_mlx_whisper(
        monkeypatch,
        return_payload={
            "text": "  privet  ",
            "segments": [
                {"start": 0.0, "end": 1.5, "text": "  привіт  "},
                {"start": 1.5, "end": 3.0, "text": "як справи"},
                {"start": 3.0, "end": 3.1, "text": "   "},  # blank → dropped
            ],
            "language": "uk",
        },
        capture=capture,
        audio_duration_s=60.0,  # under ASR_CHUNK_THRESHOLD_S → single pass
    )
    _patch_cache_hit(monkeypatch)

    out = whisper_asr.transcribe("/tmp/a.wav")

    assert out == [
        AsrSegment(start=0.0, end=1.5, content="привіт"),
        AsrSegment(start=1.5, end=3.0, content="як справи"),
    ]
    # Single-pass path forwards the original file path, not a numpy slice.
    assert capture["audio"] == "/tmp/a.wav"


def test_transcribe_raises_when_model_not_cached(monkeypatch):
    # Force try_to_load_from_cache to return None so _verify_model_cached fails.
    fake_hf = types.ModuleType("huggingface_hub")
    fake_hf.try_to_load_from_cache = lambda **kw: None
    fake_errors = types.ModuleType("huggingface_hub.errors")
    fake_errors.CacheNotFound = type("CacheNotFound", (Exception,), {})
    fake_hf.errors = fake_errors
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", fake_errors)

    with pytest.raises(AsrError, match="voice download-whisper"):
        whisper_asr._verify_model_cached()


def test_transcribe_passes_language_to_mlx_whisper(monkeypatch):
    capture: dict[str, Any] = {}
    _install_fake_mlx_whisper(
        monkeypatch,
        return_payload={"segments": []},
        capture=capture,
        audio_duration_s=60.0,
    )
    _patch_cache_hit(monkeypatch)

    whisper_asr.transcribe("/tmp/a.wav", language="en")

    assert capture["kwargs"]["language"] == "en"


def test_transcribe_disables_condition_on_previous_text(monkeypatch):
    """Locks in the stone-scriber policy: each 30-second window decodes
    fresh, no prompt-conditioning from the previous window. Trades a bit of
    cross-window consistency for far fewer repetition loops on Ukrainian.
    """
    capture: dict[str, Any] = {}
    _install_fake_mlx_whisper(
        monkeypatch,
        return_payload={"segments": []},
        capture=capture,
        audio_duration_s=60.0,
    )
    _patch_cache_hit(monkeypatch)

    whisper_asr.transcribe("/tmp/a.wav")

    assert capture["kwargs"]["condition_on_previous_text"] is False
    assert capture["kwargs"]["path_or_hf_repo"] == whisper_asr.WHISPER_REPO_ID


# ---------------------------------------------------------------------------
# Chunking path for long audio (#147 / ADR 0031)
# ---------------------------------------------------------------------------

def test_long_audio_is_chunked_with_timestamp_offsets(monkeypatch):
    """A 1200-second clip > ASR_CHUNK_THRESHOLD_S triggers per-chunk
    transcribe() calls. Timestamps in the returned segments must be
    shifted by each chunk's start in the original audio, not local
    to that chunk."""
    capture: dict[str, Any] = {}
    # Each chunk returns one segment at local t=10s.
    _install_fake_mlx_whisper(
        monkeypatch,
        payloads=[
            {"segments": [{"start": 10.0, "end": 12.0, "text": f"chunk{i}"}]}
            for i in range(4)
        ],
        capture=capture,
        audio_duration_s=1200.0,  # 20 min — well over threshold
    )
    _patch_cache_hit(monkeypatch)

    out = whisper_asr.transcribe("/tmp/long.wav")

    # 1200 s / (480-5) step = 3 chunks (480+475+245). With overlap dedup,
    # all four hypothetical chunks would have been called; the math here
    # is `ceil(1200 / (480-5))` = 3.
    # Each call must hand a numpy slice (chunked path), not a path string.
    assert len(capture["calls"]) >= 2
    for audio, _ in capture["calls"]:
        assert isinstance(audio, np.ndarray)

    # The first chunk's segment keeps its local timestamp (offset 0).
    # The second chunk's segment is shifted by chunk-start in seconds.
    assert out[0].start == pytest.approx(10.0)
    # Step is (480 - 5) = 475 s. Second chunk starts at 475 s → seg at 485 s.
    assert out[1].start == pytest.approx(485.0)


def test_short_audio_skips_chunking(monkeypatch):
    """Audio at or under ASR_CHUNK_THRESHOLD_S uses one call with the
    original file path (no numpy slicing, preserves the v0.32.0 path)."""
    capture: dict[str, Any] = {}
    _install_fake_mlx_whisper(
        monkeypatch,
        return_payload={
            "segments": [{"start": 0.0, "end": 1.0, "text": "hi"}]
        },
        capture=capture,
        audio_duration_s=300.0,  # 5 min, well under threshold
    )
    _patch_cache_hit(monkeypatch)

    out = whisper_asr.transcribe("/tmp/short.wav")

    assert len(capture["calls"]) == 1
    assert capture["calls"][0][0] == "/tmp/short.wav"
    assert len(out) == 1


def test_overlap_dedup_drops_repeated_segments_at_boundary(monkeypatch):
    """Boundary repeats are detected by text similarity (ADR 0036).

    Chunk N's tail emits the phrase 'the forecast product helps with
    scheduling'. Chunk N+1's first segment repeats the same phrase (a
    boundary duplicate Whisper emits from its own 30 s decode window
    rewind) — Jaccard 1.0 → dropped. The next chunk N+1 segment is a
    fresh sentence with disjoint vocabulary — kept."""
    capture: dict[str, Any] = {}
    _install_fake_mlx_whisper(
        monkeypatch,
        payloads=[
            {"segments": [{
                "start": 400.0, "end": 405.0,
                "text": "the forecast product helps with scheduling",
            }]},
            {"segments": [
                {
                    "start": 2.0, "end": 7.0,
                    "text": "the forecast product helps with scheduling",
                },
                {
                    "start": 10.0, "end": 12.0,
                    "text": "completely fresh sentence here now",
                },
            ]},
        ],
        capture=capture,
        audio_duration_s=900.0,  # forces 2 chunks
    )
    _patch_cache_hit(monkeypatch)

    out = whisper_asr.transcribe("/tmp/long.wav")

    contents = [s.content for s in out]
    assert "the forecast product helps with scheduling" in contents
    assert "completely fresh sentence here now" in contents
    # The repeated phrase appears exactly once (from chunk N), not twice.
    assert contents.count("the forecast product helps with scheduling") == 1


# ---------------------------------------------------------------------------
# Unit tests for the text-similarity _dedup_overlap (ADR 0036 / issue #172)
# ---------------------------------------------------------------------------


def _seg(start: float, end: float, content: str) -> AsrSegment:
    return AsrSegment(start=start, end=end, content=content)


def test_dedup_drops_clear_duplicate():
    accumulated = [_seg(0.0, 5.0, "the forecast product helps with scheduling")]
    incoming = [_seg(5.0, 10.0, "the forecast product helps with scheduling")]
    out = whisper_asr._dedup_overlap(accumulated, incoming)
    assert out == []


def test_dedup_keeps_continuation():
    accumulated = [_seg(0.0, 5.0, "the forecast product helps with scheduling")]
    incoming = [_seg(5.0, 10.0, "completely unrelated topic about birds")]
    out = whisper_asr._dedup_overlap(accumulated, incoming)
    assert out == incoming


def test_dedup_keeps_short_segment_regardless():
    """Short segments (below min_token_count) never get Jaccard-checked.
    A two-token backchannel like 'yeah okay' must survive even when the
    same two tokens appear in the trailing accumulated context."""
    accumulated = [_seg(0.0, 5.0, "yeah okay so I guess we should move on")]
    incoming = [_seg(5.0, 6.0, "yeah okay")]
    out = whisper_asr._dedup_overlap(accumulated, incoming)
    assert out == incoming


def test_dedup_threshold_exact_drops():
    """Jaccard exactly at the threshold drops (>= comparison)."""
    # 6-token bag in tail; 4 of those + 2 new in incoming.
    # Intersection = 4, union = 8 → Jaccard = 0.5.
    accumulated = [_seg(0.0, 5.0, "alpha beta gamma delta epsilon zeta")]
    incoming = [_seg(5.0, 10.0, "alpha beta gamma delta omega lambda")]
    out = whisper_asr._dedup_overlap(
        accumulated, incoming, jaccard_threshold=0.5
    )
    assert out == []


def test_dedup_pairwise_not_aggregated():
    """One tail segment matches strongly; the others dilute the bag.
    Aggregating tail tokens into one bag would lower the Jaccard against
    the diluting union; the pairwise max-Jaccard rule still drops."""
    accumulated = [
        _seg(0.0, 5.0, "the forecast product helps with scheduling"),  # strong match target
        _seg(5.0, 10.0, "completely unrelated topic about ships"),
        _seg(10.0, 15.0, "another orthogonal sentence about clouds"),
        _seg(15.0, 20.0, "yet another disjoint phrase about rivers"),
    ]
    incoming = [_seg(20.0, 25.0, "the forecast product helps with scheduling")]
    out = whisper_asr._dedup_overlap(accumulated, incoming)
    assert out == []


def test_dedup_stops_scanning_after_first_keep():
    """Once a non-duplicate is kept, all following incoming segments are
    kept without further Jaccard checks — boundary repeats are at the
    head of chunk N+1, not in its middle."""
    accumulated = [_seg(0.0, 5.0, "alpha beta gamma delta epsilon")]
    incoming = [
        _seg(5.0, 10.0, "fresh continuation with new vocabulary"),  # keep
        _seg(10.0, 15.0, "alpha beta gamma delta epsilon"),  # would be dropped if scanned
    ]
    out = whisper_asr._dedup_overlap(accumulated, incoming)
    # Both kept — second one survives despite being a duplicate of tail,
    # because dedup stopped scanning after the first keep.
    assert out == incoming


def test_dedup_empty_accumulated_passes_through():
    incoming = [_seg(0.0, 5.0, "the forecast product helps with scheduling")]
    assert whisper_asr._dedup_overlap([], incoming) == incoming


def test_dedup_overlap_window_excludes_old_tail():
    """Accumulated segments older than overlap_window_s before the last
    accumulated end are not considered for matching — they were never
    near the chunk boundary."""
    accumulated = [
        _seg(0.0, 5.0, "the forecast product helps with scheduling"),  # too old
        _seg(100.0, 105.0, "completely orthogonal sentence about birds"),  # in window
    ]
    incoming = [_seg(105.0, 110.0, "the forecast product helps with scheduling")]
    out = whisper_asr._dedup_overlap(
        accumulated, incoming, overlap_window_s=30.0
    )
    # The old "forecast product" tail seg is outside the 30 s window
    # (last_end=105, cutoff=75); the in-window "orthogonal" seg has low
    # Jaccard → incoming kept.
    assert out == incoming
