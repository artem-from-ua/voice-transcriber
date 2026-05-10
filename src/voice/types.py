"""Shared dataclasses for pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AsrSegment:
    """A raw segment from VibeVoice-ASR.

    `speaker_asr` is the speaker id assigned by VibeVoice itself (0/1/None).
    """
    start: float
    end: float
    content: str
    speaker_asr: int | None = None


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
    speaker: str | None  # pyannote label or None for silence/markers
    speaker_asr: int | None = None  # kept for debugging / sanity checks
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
