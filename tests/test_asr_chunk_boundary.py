"""Unit tests for scripts/asr-chunk-boundary-quality.py — issue #156.

Only the pure helpers are tested (tokenisation, jaccard, boundary
geometry, mark aggregation). The end-to-end CLI is exercised by hand
on real dumps; here we just lock down the math the script relies on
so a future contributor cannot silently break the methodology that
#159 and #168 are supposed to reuse.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


# scripts/asr-chunk-boundary-quality.py uses a hyphenated filename
# (matches the existing `scripts/proofread-classify.py` convention).
# Import it via importlib so the hyphen doesn't trip the module loader.
SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "scripts" / "asr-chunk-boundary-quality.py"
)


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("asr_chunk_boundary", SCRIPT_PATH)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------------------
# tokenize / jaccard / looks_truncated
# ---------------------------------------------------------------------------


def test_tokenize_handles_cyrillic_and_latin(mod):
    assert mod.tokenize("Привіт, Hugging Face!") == ["привіт", "hugging", "face"]


def test_tokenize_drops_punctuation(mod):
    assert mod.tokenize("So... that's it?") == ["so", "that's", "it"]


def test_jaccard_identical(mod):
    assert mod.jaccard(["a", "b"], ["a", "b"]) == 1.0


def test_jaccard_disjoint(mod):
    assert mod.jaccard(["a"], ["b"]) == 0.0


def test_jaccard_empty(mod):
    assert mod.jaccard([], []) == 0.0


def test_looks_truncated_short_fragment(mod):
    assert mod.looks_truncated("ce") is True


def test_looks_truncated_hyphen(mod):
    assert mod.looks_truncated("cer-") is True
    assert mod.looks_truncated("-nat") is True


def test_looks_truncated_short_but_legit(mod):
    assert mod.looks_truncated("і") is False
    assert mod.looks_truncated("is") is False


def test_looks_truncated_long_word(mod):
    assert mod.looks_truncated("проектний") is False


# ---------------------------------------------------------------------------
# compute_boundaries — mirrors the geometry in whisper_asr.transcribe
# ---------------------------------------------------------------------------


def test_compute_boundaries_short_audio_no_chunks(mod):
    # Below chunk size: a single chunk, no boundaries.
    assert mod.compute_boundaries(120.0, chunk_size_s=480.0, overlap_s=5.0) == []


def test_compute_boundaries_reference_48min(mod):
    # 2910s, 480s chunks with 5s overlap -> 7 chunks, 6 boundaries.
    # cutoffs = chunk_start_{N+1} + overlap = step*N + overlap
    cutoffs = mod.compute_boundaries(2910.0, chunk_size_s=480.0, overlap_s=5.0)
    assert cutoffs == [480.0, 955.0, 1430.0, 1905.0, 2380.0, 2855.0]


def test_split_around_matches_dedup_rule(mod):
    Segment = mod.Segment
    segs = [
        Segment(0.0, 5.0, "a"),
        Segment(5.0, 10.0, "b"),
        Segment(10.0, 15.0, "c"),
    ]
    # _dedup_overlap drops segments whose start < cutoff (kept in tail).
    tail, head = mod.split_around(segs, 10.0)
    assert [s.content for s in tail] == ["a", "b"]
    assert [s.content for s in head] == ["c"]


# ---------------------------------------------------------------------------
# aggregate — drives the final decision; must catch malformed marks
# ---------------------------------------------------------------------------


def _bs(idx: int, cutoff: float, **kw):
    base = {
        "boundary_idx": idx,
        "cutoff_s": cutoff,
        "gap_s": 0.0,
        "word_overlap_jaccard": 0.0,
        "rms_before": 0.0,
        "rms_after": 0.0,
    }
    base.update(kw)
    return base


def test_aggregate_all_clean(mod):
    boundaries = [_bs(i, float(i)) for i in range(5)]
    marks = [{"boundary_idx": i, "verdict": "clean", "note": ""} for i in range(5)]
    out = mod.aggregate(boundaries, marks)
    sb = out["summary_block"]
    assert sb["counts"] == {"clean": 5, "missing": 0, "duplicated": 0, "truncated": 0}
    assert sb["material_issues"] == 0
    assert sb["material_rate"] == 0.0
    assert sb["decision"] == "baseline-acceptable"


def test_aggregate_material(mod):
    boundaries = [_bs(i, float(i)) for i in range(10)]
    verdicts = ["clean"] * 7 + ["missing", "duplicated", "truncated"]
    marks = [{"boundary_idx": i, "verdict": v, "note": ""} for i, v in enumerate(verdicts)]
    out = mod.aggregate(boundaries, marks)
    sb = out["summary_block"]
    assert sb["material_issues"] == 3
    assert sb["material_rate"] == pytest.approx(0.30)
    assert sb["decision"] == "material"


def test_aggregate_rejects_missing_marks(mod):
    boundaries = [_bs(i, float(i)) for i in range(3)]
    marks = [{"boundary_idx": 0, "verdict": "clean", "note": ""}]
    with pytest.raises(ValueError, match="missing marks"):
        mod.aggregate(boundaries, marks)


def test_aggregate_rejects_invalid_verdict(mod):
    boundaries = [_bs(0, 0.0)]
    marks = [{"boundary_idx": 0, "verdict": "maybe", "note": ""}]
    with pytest.raises(ValueError, match="invalid verdict"):
        mod.aggregate(boundaries, marks)
