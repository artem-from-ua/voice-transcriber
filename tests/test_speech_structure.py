"""Offline tests for section structuring."""

from __future__ import annotations

from voice.llm import LLMError
from voice.structure import (
    STRUCTURE_CHUNK_OVERLAP,
    STRUCTURE_CHUNK_SIZE,
    STRUCTURE_CHUNK_THRESHOLD,
    _chunk_segments,
    _merge_same_title,
    _reconcile_chunks,
    structure_dialog,
)
from voice.types import Section, Segment


class StubLLM:
    def __init__(self, payload=None, error=None):
        self.payload = payload
        self.error = error
        self.calls = 0

    def chat_json(self, *_args, **_kwargs):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.payload


class ChunkedStubLLM:
    """Returns a different payload per call, indexed by call number."""

    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = 0

    def chat_json(self, *_args, **_kwargs):
        idx = self.calls
        self.calls += 1
        item = self.payloads[idx]
        if isinstance(item, Exception):
            raise item
        return item


def _seg(start, end, content, speaker="SPEAKER_00", name="Артем"):
    return Segment(start=start, end=end, content=content, speaker=speaker, name=name)


def test_no_llm_returns_single_section():
    segs = [_seg(0.0, 10.0, "hello")]
    sd = structure_dialog(segs, llm=None, log=lambda _s: None)
    assert len(sd.sections) == 1
    assert sd.sections[0].title == "Розмова"
    assert sd.sections[0].start_ms == 0
    assert sd.sections[0].end_ms == 10000


def test_no_speech_returns_empty_sections():
    segs = [Segment(start=0.0, end=2.0, content="[Human Sounds]", speaker=None)]
    sd = structure_dialog(segs, llm=None, log=lambda _s: None)
    assert sd.sections == []
    assert sd.segments == segs


def test_valid_layout_accepted():
    segs = [
        _seg(0.0, 30.0, "intro"),
        _seg(30.0, 60.0, "topic A"),
        _seg(60.0, 90.0, "topic B"),
    ]
    payload = {"sections": [
        {"title": "Привітання", "start_ms": 0, "end_ms": 30000},
        {"title": "Тема А", "start_ms": 30000, "end_ms": 60000},
        {"title": "Тема Б", "start_ms": 60000, "end_ms": 90000},
    ]}
    llm = StubLLM(payload=payload)
    sd = structure_dialog(segs, llm=llm, log=lambda _s: None)
    assert [s.title for s in sd.sections] == ["Привітання", "Тема А", "Тема Б"]


def test_non_contiguous_falls_back():
    segs = [_seg(0.0, 60.0, "x")]
    payload = {"sections": [
        {"title": "A", "start_ms": 0, "end_ms": 20000},
        {"title": "B", "start_ms": 30000, "end_ms": 60000},  # gap
    ]}
    sd = structure_dialog(segs, llm=StubLLM(payload=payload), log=lambda _s: None)
    assert len(sd.sections) == 1
    assert sd.sections[0].title == "Розмова"


def test_too_few_sections_falls_back():
    segs = [_seg(0.0, 60.0, "x")]
    payload = {"sections": [{"title": "Only", "start_ms": 0, "end_ms": 60000}]}
    sd = structure_dialog(segs, llm=StubLLM(payload=payload), log=lambda _s: None)
    assert sd.sections[0].title == "Розмова"


def test_too_many_sections_falls_back():
    segs = [_seg(0.0, 800.0, "x")]
    payload = {"sections": [
        {"title": f"S{i}", "start_ms": i * 100000, "end_ms": (i + 1) * 100000}
        for i in range(8)
    ]}
    sd = structure_dialog(segs, llm=StubLLM(payload=payload), log=lambda _s: None)
    assert sd.sections[0].title == "Розмова"


def test_invalid_json_shape_falls_back():
    segs = [_seg(0.0, 30.0, "x")]
    sd = structure_dialog(segs, llm=StubLLM(payload=["not", "a", "dict"]), log=lambda _s: None)
    assert sd.sections[0].title == "Розмова"


def test_llm_error_falls_back():
    segs = [_seg(0.0, 30.0, "x")]
    sd = structure_dialog(
        segs, llm=StubLLM(error=LLMError("down")), log=lambda _s: None,
    )
    assert sd.sections[0].title == "Розмова"


def test_english_fallback_title():
    segs = [_seg(0.0, 10.0, "hi")]
    sd = structure_dialog(segs, llm=None, language="en", log=lambda _s: None)
    assert sd.sections[0].title == "Conversation"


def test_bounds_must_match_total_dialogue():
    segs = [_seg(0.0, 60.0, "x")]
    payload = {"sections": [
        {"title": "A", "start_ms": 0, "end_ms": 30000},
        {"title": "B", "start_ms": 30000, "end_ms": 50000},  # ends short
    ]}
    sd = structure_dialog(segs, llm=StubLLM(payload=payload), log=lambda _s: None)
    assert sd.sections[0].title == "Розмова"


# ---------------------------------------------------------------- chunking


def _long_segs(n: int) -> list[Segment]:
    """Synthesise n contiguous 10-second segments."""
    return [
        _seg(i * 10.0, (i + 1) * 10.0, f"line {i}")
        for i in range(n)
    ]


def test_short_dialog_below_threshold_uses_single_pass():
    """≤60-segment dialogues take the fast path: one LLM call."""
    n = STRUCTURE_CHUNK_THRESHOLD - 1
    segs = _long_segs(n)
    payload = {"sections": [
        {"title": "Початок", "start_ms": 0, "end_ms": n * 10000 // 2},
        {"title": "Кінець", "start_ms": n * 10000 // 2, "end_ms": n * 10000},
    ]}
    llm = StubLLM(payload=payload)
    sd = structure_dialog(segs, llm=llm, log=lambda _s: None)
    assert llm.calls == 1
    assert len(sd.sections) == 2


def test_long_dialog_above_threshold_uses_chunks():
    """≥THRESHOLD segments → multiple LLM calls, one per chunk."""
    n = STRUCTURE_CHUNK_THRESHOLD + 50  # forces at least 3 chunks
    segs = _long_segs(n)
    # Each chunk gets the same "two equal sections" template, valid per-chunk.
    # The reconcile step will merge sequential same-title sections.
    def _payload_for(_chunk_idx):
        # produce two non-overlapping sections per chunk based on chunk's bounds;
        # but we don't know exact bounds here — the validator is permissive
        # (require_exact_bounds=False), so we use generic spans the stub
        # sends back. The structure code passes total_start_ms/total_end_ms
        # of the chunk to the prompt; the LLM stub doesn't see them, so we
        # rely on the permissive per-chunk validator.
        return {"sections": [
            {"title": "Тема А", "start_ms": 0, "end_ms": 175_000},
            {"title": "Тема Б", "start_ms": 175_000, "end_ms": 350_000},
        ]}

    # Compute number of chunks deterministically from the public constants.
    step = STRUCTURE_CHUNK_SIZE - STRUCTURE_CHUNK_OVERLAP
    expected_chunks = 1 + max(0, (n - STRUCTURE_CHUNK_SIZE + step - 1) // step)

    llm = ChunkedStubLLM([_payload_for(i) for i in range(expected_chunks)])
    sd = structure_dialog(segs, llm=llm, log=lambda _s: None)
    assert llm.calls == expected_chunks, (llm.calls, expected_chunks)
    assert sd.sections, "expected at least one section after reconcile"
    # Coverage: first section starts at total_start_ms (0), last ends at total_end_ms.
    assert sd.sections[0].start_ms == 0
    assert sd.sections[-1].end_ms == n * 10000


def test_chunk_segments_overlap():
    """Chunks must duplicate `overlap` segments at every internal boundary."""
    segs = _long_segs(80)
    chunks = _chunk_segments(segs, chunk_size=30, overlap=2)
    assert len(chunks) >= 3
    # Each pair of adjacent chunks shares its last/first `overlap` segments.
    for prev, nxt in zip(chunks, chunks[1:]):
        assert prev[-2] is nxt[0] and prev[-1] is nxt[1], "overlap broken"


def test_chunk_segments_rejects_bad_args():
    import pytest
    segs = _long_segs(10)
    with pytest.raises(ValueError):
        _chunk_segments(segs, chunk_size=0, overlap=0)
    with pytest.raises(ValueError):
        _chunk_segments(segs, chunk_size=5, overlap=5)


def test_merge_same_title_collapses_adjacent_duplicates():
    sections = [
        Section(title="Привітання", start_ms=0, end_ms=10_000),
        Section(title="привітання", start_ms=10_000, end_ms=20_000),  # case-insens
        Section(title="Робота", start_ms=20_000, end_ms=40_000),
        Section(title="Робота", start_ms=40_000, end_ms=60_000),
    ]
    out = _merge_same_title(sections)
    assert [s.title for s in out] == ["Привітання", "Робота"]
    assert out[0].start_ms == 0 and out[0].end_ms == 20_000
    assert out[1].start_ms == 20_000 and out[1].end_ms == 60_000


def test_reconcile_chunks_snaps_edges_and_covers_full_range():
    """Sections from two chunks: reconciler must produce gap-free coverage."""
    chunk_a = [
        Section(title="Вступ", start_ms=0, end_ms=30_000),
        Section(title="Спільна тема", start_ms=30_000, end_ms=60_000),
    ]
    chunk_b = [
        Section(title="Спільна тема", start_ms=55_000, end_ms=90_000),  # overlap
        Section(title="Висновки", start_ms=90_000, end_ms=120_000),
    ]
    result = _reconcile_chunks(
        [chunk_a, chunk_b],
        total_start_ms=0,
        total_end_ms=120_000,
        max_sections=7,
    )
    assert result is not None
    assert result[0].start_ms == 0
    assert result[-1].end_ms == 120_000
    # No gaps, no overlaps.
    for prev, nxt in zip(result, result[1:]):
        assert nxt.start_ms == prev.end_ms, (prev, nxt)
    # Same-title neighbours merged.
    titles = [s.title for s in result]
    assert titles == ["Вступ", "Спільна тема", "Висновки"]


def test_reconcile_chunks_enforces_max_sections():
    """If chunks produced too many sections, reconcile fuses the shortest pairs."""
    chunks = [[
        Section(title=f"T{i}", start_ms=i * 10_000, end_ms=(i + 1) * 10_000)
        for i in range(12)
    ]]
    result = _reconcile_chunks(
        chunks,
        total_start_ms=0,
        total_end_ms=120_000,
        max_sections=7,
    )
    assert result is not None
    assert len(result) <= 7
    assert result[0].start_ms == 0
    assert result[-1].end_ms == 120_000


def test_reconcile_chunks_returns_none_on_empty_input():
    result = _reconcile_chunks(
        [],
        total_start_ms=0,
        total_end_ms=10_000,
        max_sections=7,
    )
    assert result is None


def test_long_dialog_all_chunks_fail_falls_back_to_single_section():
    """Every chunk returns invalid JSON → single 'Розмова' fallback."""
    n = STRUCTURE_CHUNK_THRESHOLD + 50
    segs = _long_segs(n)
    step = STRUCTURE_CHUNK_SIZE - STRUCTURE_CHUNK_OVERLAP
    expected_chunks = 1 + max(0, (n - STRUCTURE_CHUNK_SIZE + step - 1) // step)
    # All chunks emit a payload that fails the validator (zero sections).
    llm = ChunkedStubLLM([{"sections": []} for _ in range(expected_chunks)])
    sd = structure_dialog(segs, llm=llm, log=lambda _s: None)
    assert llm.calls == expected_chunks
    assert len(sd.sections) == 1
    assert sd.sections[0].title == "Розмова"


def test_long_dialog_some_chunks_fail_still_succeeds():
    """If ≥1 chunk returns a valid payload, reconcile uses what it has."""
    n = STRUCTURE_CHUNK_THRESHOLD + 50
    segs = _long_segs(n)
    step = STRUCTURE_CHUNK_SIZE - STRUCTURE_CHUNK_OVERLAP
    expected_chunks = 1 + max(0, (n - STRUCTURE_CHUNK_SIZE + step - 1) // step)
    valid = {"sections": [
        {"title": "Загальна тема", "start_ms": 0, "end_ms": 200_000},
    ]}
    payloads = [valid] + [{"sections": []} for _ in range(expected_chunks - 1)]
    llm = ChunkedStubLLM(payloads)
    sd = structure_dialog(segs, llm=llm, log=lambda _s: None)
    # Final coverage must still span the full dialogue thanks to the
    # edge-snap step in _reconcile_chunks.
    assert sd.sections[0].start_ms == 0
    assert sd.sections[-1].end_ms == n * 10000


def test_long_dialog_llm_error_in_chunk_skips_chunk():
    """LLMError on a single chunk does not abort the whole structure step."""
    n = STRUCTURE_CHUNK_THRESHOLD + 50
    segs = _long_segs(n)
    step = STRUCTURE_CHUNK_SIZE - STRUCTURE_CHUNK_OVERLAP
    expected_chunks = 1 + max(0, (n - STRUCTURE_CHUNK_SIZE + step - 1) // step)
    valid = {"sections": [
        {"title": "Робоче", "start_ms": 0, "end_ms": 200_000},
    ]}
    payloads = [LLMError("boom")] + [valid for _ in range(expected_chunks - 1)]
    llm = ChunkedStubLLM(payloads)
    sd = structure_dialog(segs, llm=llm, log=lambda _s: None)
    assert llm.calls == expected_chunks
    # We have at least one section.
    assert len(sd.sections) >= 1
    assert sd.sections[0].start_ms == 0
    assert sd.sections[-1].end_ms == n * 10000
