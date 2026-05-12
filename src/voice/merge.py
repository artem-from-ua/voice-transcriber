"""Combine ASR segments with pyannote diarization turns.

For each ASR segment, pick the pyannote speaker whose `exclusive_diarization`
turns have the most temporal overlap. Segments without any overlap with a
pyannote turn keep `speaker=None`.
"""

from __future__ import annotations

from collections import defaultdict

from .types import AsrSegment, DiarTurn, Segment


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def merge(asr_segments: list[AsrSegment], turns: list[DiarTurn]) -> list[Segment]:
    merged: list[Segment] = []
    for seg in asr_segments:
        content = seg.content.strip()
        by_spk: dict[str, float] = defaultdict(float)
        for t in turns:
            ov = _overlap(seg.start, seg.end, t.start, t.end)
            if ov > 0:
                by_spk[t.speaker] += ov

        best = max(by_spk.items(), key=lambda kv: kv[1])[0] if by_spk else None
        merged.append(Segment(
            start=seg.start, end=seg.end,
            content=content, speaker=best,
        ))
    return merged
