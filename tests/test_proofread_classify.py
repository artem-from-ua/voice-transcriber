"""Offline tests for the proofread-classify script.

Tests the pure classification function and the spot-check parser. The
script lives at ``scripts/proofread-classify.py`` (with a dash, not
importable directly), so we load it via ``importlib`` once at module
import time.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "proofread-classify.py"
_spec = importlib.util.spec_from_file_location("proofread_classify", _SCRIPT_PATH)
assert _spec and _spec.loader
pc = importlib.util.module_from_spec(_spec)
# Register before exec so @dataclass (and other introspection-heavy
# decorators) can resolve the module via sys.modules — see CPython
# dataclasses._is_type which looks up cls.__module__ in sys.modules.
sys.modules["proofread_classify"] = pc
_spec.loader.exec_module(pc)  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# classify_pair
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("asr, proof, expected", [
    # ---- unchanged ----
    ("Це звичайний український текст.", "Це звичайний український текст.", "unchanged"),
    ("Привіт", "Привіт", "unchanged"),
    # Whitespace differences alone collapse to "unchanged".
    ("Привіт   світ", "Привіт світ", "unchanged"),

    # ---- cosmetic ----
    # Punctuation-only differences.
    ("Так, було", "Так було", "cosmetic"),
    ("Привіт.", "Привіт!", "cosmetic"),
    # Case-only differences.
    ("так було", "Так було", "cosmetic"),

    # ---- proper_noun_fix (introduces or fixes Latin/CamelCase tokens) ----
    (
        "Я завантажив модель з хагінг фейсі",
        "Я завантажив модель з Hugging Face",
        "proper_noun_fix",
    ),
    (
        "Це працює через градіо бібліотеку",
        "Це працює через Gradio бібліотеку",
        "proper_noun_fix",
    ),
    (
        "Запустив CloudCop і все злетіло",
        "Запустив Claude Code і все злетіло",
        "proper_noun_fix",
    ),
    # Pure Latin terms preserved on one side count too.
    (
        "Я використовую гітхаб для проектів",
        "Я використовую GitHub для проектів",
        "proper_noun_fix",
    ),

    # ---- substantive_rewrite (Ukrainian → different Ukrainian, no Latin) ----
    (
        "Я думав що це працює нормально",
        "Я думав що це не працює нормально",
        "substantive_rewrite",
    ),
    (
        "вчора ми ходили в парк гуляти",
        "вчора ми ходили в магазин гуляти",
        "substantive_rewrite",
    ),
])
def test_classify_pair(asr: str, proof: str, expected: str) -> None:
    assert pc.classify_pair(asr, proof) == expected


# ---------------------------------------------------------------------------
# parse_spot_check
# ---------------------------------------------------------------------------


def test_parse_spot_check_happy_path():
    text = """\
# Spot-check

intro text

## Segment 5 — timestamp 00:42-00:51 (speaker: SPEAKER_00)

**ASR (merge):** foo
**Proofread:**  bar

- [x] better
- [ ] worse
- [ ] neutral

## Segment 12 — timestamp 01:00-01:05 (speaker: SPEAKER_01)

**ASR (merge):** baz
**Proofread:**  qux

- [ ] better
- [ ] worse
- [X] neutral
"""
    out = pc.parse_spot_check(text)
    assert out == {5: "better", 12: "neutral"}


def test_parse_spot_check_no_mark_raises():
    text = """\
## Segment 1

- [ ] better
- [ ] worse
- [ ] neutral
"""
    with pytest.raises(ValueError, match="Segment 1.*found 0"):
        pc.parse_spot_check(text)


def test_parse_spot_check_multiple_marks_raises():
    text = """\
## Segment 1

- [x] better
- [x] worse
- [ ] neutral
"""
    with pytest.raises(ValueError, match="Segment 1.*found 2"):
        pc.parse_spot_check(text)


# ---------------------------------------------------------------------------
# parse_judge_marks
# ---------------------------------------------------------------------------


def test_parse_judge_marks_happy_path():
    payload = json.dumps({
        "5": {"mark": "better", "rationale": "Hugging Face is correct"},
        "12": {"mark": "neutral", "rationale": "stylistic"},
    })
    out = pc.parse_judge_marks(payload)
    assert out == {
        5: {"mark": "better", "rationale": "Hugging Face is correct"},
        12: {"mark": "neutral", "rationale": "stylistic"},
    }


def test_parse_judge_marks_rejects_invalid_mark():
    payload = json.dumps({"1": {"mark": "maybe", "rationale": ""}})
    with pytest.raises(ValueError, match="invalid mark"):
        pc.parse_judge_marks(payload)


def test_parse_judge_marks_requires_mark_field():
    payload = json.dumps({"1": {"rationale": "..."}})
    with pytest.raises(ValueError, match="missing 'mark'"):
        pc.parse_judge_marks(payload)


# ---------------------------------------------------------------------------
# Round-trip: classify → spot-check.md → parse-marks
# ---------------------------------------------------------------------------


def test_classify_then_parse_marks_round_trip(tmp_path: Path):
    """Synthetic end-to-end: build a fake dump, classify, edit checkboxes,
    parse — assert agreement counts."""
    merge = [
        {"start": 0.0, "end": 1.0, "content": "Привіт світ", "speaker": "SPEAKER_00"},
        {"start": 1.0, "end": 2.5, "content": "хагінг фейсі це круто", "speaker": "SPEAKER_00"},
        {
            "start": 2.5, "end": 4.0,
            "content": "вчора ми ходили в парк гуляти", "speaker": "SPEAKER_01",
        },
    ]
    proof = [
        # unchanged
        {"start": 0.0, "end": 1.0, "content": "Привіт світ", "speaker": "SPEAKER_00"},
        # proper_noun_fix
        {"start": 1.0, "end": 2.5, "content": "Hugging Face це круто", "speaker": "SPEAKER_00"},
        # substantive_rewrite
        {
            "start": 2.5, "end": 4.0,
            "content": "вчора ми ходили в магазин гуляти", "speaker": "SPEAKER_01",
        },
    ]
    dump = tmp_path / "dump"
    dump.mkdir()
    (dump / "04-merge.json").write_text(json.dumps(merge), encoding="utf-8")
    (dump / "05-proofread.json").write_text(json.dumps(proof), encoding="utf-8")
    (dump / "01-meta.json").write_text(json.dumps({
        "path": "/x.m4a",
        "started_at": "2026-05-13T15:00:00",
        "ended_at": "2026-05-13T15:00:04",
        "duration_s": 4.0,
        "source": "fake",
        "stages": {"proofread": {"wall_clock_s": 1.2, "llm_calls": 2, "model": "test"}},
    }), encoding="utf-8")

    out_dir = tmp_path / "out"

    # classify
    pairs = pc.load_pairs(dump)
    cats = [p.category for p in pairs]
    assert cats == ["unchanged", "proper_noun_fix", "substantive_rewrite"]

    summary = pc.categorise(pairs)
    assert summary["total"] == 3
    assert summary["unchanged"] == 1
    assert summary["proper_noun_fix"] == 1
    assert summary["substantive_rewrite"] == 1
    assert summary["hit_rate_pct"] == round(100.0 * 2 / 3, 1)

    # Render spot-check, fill the single substantive_rewrite segment.
    out_dir.mkdir()
    spot = out_dir / "spot-check.md"
    spot_check_segments = [p for p in pairs if p.category == "substantive_rewrite"]
    spot.write_text(pc._render_spot_check(spot_check_segments), encoding="utf-8")

    text = spot.read_text(encoding="utf-8")
    filled = text.replace("- [ ] worse", "- [x] worse", 1)
    spot.write_text(filled, encoding="utf-8")

    human = pc.parse_spot_check(spot.read_text(encoding="utf-8"))
    assert human == {2: "worse"}
