"""Shared dataclasses for pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AsrSegment:
    """A raw segment from the ASR backend (Whisper-large-v3-MLX)."""
    start: float
    end: float
    content: str


@dataclass
class DiarTurn:
    """A pyannote diarization turn (from `exclusive_diarization`)."""
    start: float
    end: float
    speaker: str  # e.g. "SPEAKER_00"


@dataclass
class Segment:
    """A merged segment: ASR content + pyannote speaker label."""
    start: float
    end: float
    content: str
    speaker: str | None  # pyannote label, or None when no overlap with any turn
    name: str | None = None  # filled in by identify.py


@dataclass
class Section:
    title: str
    start_ms: int
    end_ms: int


@dataclass
class StructuredDialog:
    sections: list[Section]
    segments: list[Segment]  # the same segments, in order


@dataclass
class AudioMeta:
    path: str
    started_at: str  # ISO 8601
    ended_at: str
    duration_s: float
    source: str  # "ffprobe creation_time" or "stat birthtime" or "stat mtime" or "cli override"
    stages: dict[str, dict[str, Any]] | None = None
