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
from voice.types import AsrSegment, Word


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
    """Case 1: pairwise max >= threshold -> drop incoming, keep tail intact."""
    accumulated = [_seg(0.0, 5.0, "the forecast product helps with scheduling")]
    incoming = [_seg(5.0, 10.0, "the forecast product helps with scheduling")]
    out = whisper_asr._dedup_overlap(accumulated, incoming)
    assert out == accumulated  # tail kept, incoming dropped


def test_dedup_keeps_continuation():
    """Case 3: low pairwise + no audio overlap -> tail kept, incoming kept."""
    accumulated = [_seg(0.0, 5.0, "the forecast product helps with scheduling")]
    incoming = [_seg(5.0, 10.0, "completely unrelated topic about birds")]
    out = whisper_asr._dedup_overlap(accumulated, incoming)
    assert out == accumulated + incoming


def test_dedup_keeps_short_segment_regardless():
    """Short segments (below min_token_count) bypass the Jaccard checks
    and are kept; tail is also kept."""
    accumulated = [_seg(0.0, 5.0, "yeah okay so I guess we should move on")]
    incoming = [_seg(5.0, 6.0, "yeah okay")]
    out = whisper_asr._dedup_overlap(accumulated, incoming)
    assert out == accumulated + incoming


def test_dedup_threshold_exact_drops():
    """Jaccard exactly at the threshold drops the incoming (`>=` semantics);
    tail is kept intact."""
    # 6-token bag in tail; 4 of those + 2 new in incoming.
    # Intersection = 4, union = 8 → Jaccard = 0.5.
    accumulated = [_seg(0.0, 5.0, "alpha beta gamma delta epsilon zeta")]
    incoming = [_seg(5.0, 10.0, "alpha beta gamma delta omega lambda")]
    out = whisper_asr._dedup_overlap(
        accumulated, incoming, jaccard_threshold=0.5
    )
    assert out == accumulated  # incoming dropped


def test_dedup_pairwise_not_aggregated():
    """One tail segment matches strongly; aggregating tail tokens into one
    bag would lower the Jaccard against the diluting union, but the
    pairwise max-Jaccard rule against the strong-match seg still drops
    the incoming."""
    accumulated = [
        _seg(0.0, 5.0, "the forecast product helps with scheduling"),  # strong match target
        _seg(5.0, 10.0, "completely unrelated topic about ships"),
        _seg(10.0, 15.0, "another orthogonal sentence about clouds"),
        _seg(15.0, 20.0, "yet another disjoint phrase about rivers"),
    ]
    incoming = [_seg(20.0, 25.0, "the forecast product helps with scheduling")]
    out = whisper_asr._dedup_overlap(accumulated, incoming)
    assert out == accumulated  # incoming dropped


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
    # Tail kept; both incoming kept (second survives because scan stopped).
    assert out == accumulated + incoming


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
    # Jaccard AND no audio span overlap with incoming (its end=105 <=
    # incoming.start=105 strictly) -> kept. Result: both tail kept,
    # incoming kept too.
    assert out == accumulated + incoming


# ---------------------------------------------------------------------------
# SUPERSEDE branch (Case 2) — issue #172 post-PR review (option Д)
# ---------------------------------------------------------------------------


def test_dedup_supersedes_short_tail_with_longer_richer_incoming():
    """Whisper sometimes re-emits a longer, context-richer version of
    several short tail fragments — pairwise Jaccard against each fragment
    stays low (each fragment is too short to match), but the aggregate
    Jaccard against the union of overlapping tail tokens passes. SUPERSEDE
    drops the short tail fragments and keeps the longer incoming.

    Concrete example mirroring B3 from the 48-min reference (#172):
    tail has two short fragments 'and asking questions.' (3 tokens) +
    'And I need to catch myself.' (6 tokens); incoming is the long
    re-transcription 'people and asking questions and i need to catch
    myself on this sometimes...' (~16 tokens) whose audio span covers
    both tail fragments. Pairwise max ≈ 0.35 (below 0.5 threshold) but
    aggregate ≈ 0.55 (above 0.3 supersede threshold) AND audio spans
    overlap -> drop both tail fragments, keep the incoming."""
    accumulated = [
        _seg(1900.40, 1903.34, "and asking questions."),
        _seg(1903.34, 1905.00, "And I need to catch myself."),
    ]
    incoming = [
        _seg(
            1900.00, 1908.40,
            "people and asking questions and i need to catch myself on "
            "this sometimes um and i think like",
        ),
    ]
    out = whisper_asr._dedup_overlap(accumulated, incoming)
    # Both tail fragments superseded; only the long incoming survives.
    assert out == incoming


def test_dedup_does_not_supersede_when_no_audio_overlap():
    """A long incoming segment with moderate-but-not-pairwise-dup Jaccard
    and NO audio overlap with the tail is NOT a Whisper rewind — it's a
    coincidental partial repetition later in the recording. Even though
    the aggregate Jaccard might be above the supersede threshold, the
    SUPERSEDE branch requires audio span overlap (`seg.start < t.end` and
    `seg.end > t.start`); without it, we keep both."""
    # Use vocabulary where pairwise max stays below 0.5 against either
    # tail seg, but aggregate against the union is above 0.3.
    accumulated = [
        _seg(0.0, 3.0, "alpha beta gamma"),
        _seg(3.0, 5.0, "delta epsilon zeta eta"),
    ]
    incoming = [
        # Starts well after the tail ends — no audio overlap.
        _seg(
            10.0, 18.0,
            "alpha beta delta epsilon brand new fresh extra content here",
        ),
    ]
    out = whisper_asr._dedup_overlap(accumulated, incoming)
    # No supersede; tail and incoming both kept.
    assert len(out) == 3
    assert out[:2] == accumulated
    assert out[2:] == incoming


def test_dedup_supersede_only_drops_overlapping_tail_not_unrelated_tail():
    """Tail has one unrelated short fragment plus two short fragments the
    incoming supersedes. Only the overlapping ones are dropped; the
    unrelated tail seg survives."""
    accumulated = [
        _seg(100.0, 102.0, "completely unrelated chitchat about weather"),
        _seg(1900.40, 1903.34, "and asking questions."),
        _seg(1903.34, 1905.00, "And I need to catch myself."),
    ]
    incoming = [
        _seg(
            1900.00, 1908.40,
            "people and asking questions and i need to catch myself on "
            "this sometimes um and i think like",
        ),
    ]
    # Widen the overlap window so the older 'weather' seg is considered
    # tail too (it sits ~1800 s before the boundary).
    out = whisper_asr._dedup_overlap(
        accumulated, incoming, overlap_window_s=2000.0,
    )
    # The unrelated seg is in the tail window but has no audio overlap
    # with incoming -> kept. The two overlapping fragments are dropped.
    assert len(out) == 2
    assert out[0] == accumulated[0]
    assert out[1] == incoming[0]


# ---------------------------------------------------------------------------
# Word-level timestamps (issue #125)
# ---------------------------------------------------------------------------


def test_parse_segments_extracts_word_timestamps():
    payload = {
        "segments": [
            {
                "start": 0.0, "end": 2.0, "text": "hello world",
                "words": [
                    {"word": " hello", "start": 0.0, "end": 0.5, "probability": 0.99},
                    {"word": " world", "start": 0.6, "end": 1.2, "probability": 0.88},
                ],
            }
        ]
    }
    out = whisper_asr._parse_segments(payload)
    assert len(out) == 1
    assert out[0].words is not None
    assert len(out[0].words) == 2
    assert out[0].words[0] == Word(start=0.0, end=0.5, content="hello", probability=0.99)


def test_parse_segments_words_absent_yields_none():
    payload = {"segments": [{"start": 0.0, "end": 1.0, "text": "hi"}]}
    out = whisper_asr._parse_segments(payload)
    assert out[0].words is None


def test_shift_segments_shifts_words():
    seg = AsrSegment(
        start=1.0, end=2.0, content="hello",
        words=[Word(start=1.0, end=2.0, content="hello", probability=0.9)],
    )
    out = whisper_asr._shift_segments([seg], 10.0)
    assert out[0].start == 11.0 and out[0].end == 12.0
    assert out[0].words is not None
    assert out[0].words[0].start == 11.0 and out[0].words[0].end == 12.0
    assert out[0].words[0].probability == 0.9


def test_transcribe_passes_word_timestamps_to_mlx_whisper(monkeypatch):
    capture: dict[str, Any] = {}
    _install_fake_mlx_whisper(
        monkeypatch,
        return_payload={"segments": []},
        capture=capture,
        audio_duration_s=60.0,
    )
    _patch_cache_hit(monkeypatch)

    whisper_asr.transcribe("/tmp/a.wav")

    assert capture["kwargs"]["word_timestamps"] is True
