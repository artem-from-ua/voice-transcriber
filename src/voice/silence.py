"""Silence-event extraction from a segment list.

A silence event is a contiguous "quiet span" built from one or more of:
  - inter-segment ASR gaps (time between seg[i].end and seg[i+1].start)
  - [muted, ...] placeholder segments inserted by the safe_speech stage

Adjacent sources are merged into one event when no speaker-bearing segment
falls between them. Events shorter than `min_silence_s` are discarded.

The `kind` field reflects whether the span contained any redaction:
  "muted"  — at least one [muted, ...] segment was part of the chain
  "pause"  — pure gap(s) with no redacted content
"""

from __future__ import annotations

from dataclasses import dataclass

from .types import Segment

_MUTED_PREFIX = "[muted"


def _is_muted(seg: Segment) -> bool:
    return seg.speaker is None and seg.content.startswith(_MUTED_PREFIX)


def _format_ts(t: float) -> str:
    s = int(round(t))
    return f"{s // 3600:02d}:{(s // 60) % 60:02d}:{s % 60:02d}"


@dataclass
class SilenceEvent:
    start: float
    end: float
    kind: str  # "pause" or "muted"

    @property
    def duration(self) -> float:
        return self.end - self.start

    def format_md(self) -> str:
        """Render as a Markdown blockquote line with timestamp range."""
        label = "пауза" if self.kind == "pause" else "muted"
        return f"> _[{label} {_format_ts(self.start)}–{_format_ts(self.end)}]_"


def extract_silence_events(
    segments: list[Segment],
    min_silence_s: float,
) -> list[SilenceEvent]:
    """Return silence events from *segments* that are >= *min_silence_s* long.

    Iterates segments once. Builds events greedily: an event grows as long as
    the next element is either a muted placeholder or a gap followed by another
    muted placeholder (i.e. no real speaker content interrupts the chain).
    """
    events: list[SilenceEvent] = []

    # We look ahead to decide whether a gap "merges" into an ongoing event or
    # starts a new one. Work on (prev_end, seg) pairs plus a look-ahead.
    speaker_segs = [s for s in segments]  # full list for index access
    n = len(speaker_segs)

    i = 0
    prev_end: float | None = None  # end of the last speaker-bearing segment seen

    while i < n:
        seg = speaker_segs[i]

        if seg.speaker is not None:
            # Real speaker segment — check for a leading gap from prev_end.
            if prev_end is not None:
                gap = seg.start - prev_end
                if gap >= min_silence_s:
                    events.append(SilenceEvent(start=prev_end, end=seg.start, kind="pause"))
            prev_end = seg.end
            i += 1
            continue

        if not _is_muted(seg):
            # Speakerless but not muted (e.g. [Human Sounds]) — ignore silently.
            i += 1
            continue

        # --- muted segment: start (or extend) an event chain ---
        ev_start = seg.start
        ev_end = seg.end
        ev_kind = "muted"

        # Peek forward: absorb further muted segments and pure-gap chains that
        # are not interrupted by a speaker segment.
        j = i + 1
        while j < n:
            nxt = speaker_segs[j]
            if _is_muted(nxt):
                # Adjacent muted — absorb gap + placeholder.
                ev_end = nxt.end
                ev_kind = "muted"
                j += 1
                continue
            if nxt.speaker is None:
                # Non-muted speakerless (noise tag) — skip over it.
                j += 1
                continue
            # nxt is a real speaker segment.
            # Absorb a trailing gap only if the gap itself is large enough
            # and we are already inside a muted chain (so we extend the event
            # to cover the "tail silence" after the last muted block).
            trailing_gap = nxt.start - ev_end
            if trailing_gap >= min_silence_s:
                ev_end = nxt.start
            break

        # Extend the event backward to cover any leading gap from the previous
        # speaker segment.  We absorb the gap unconditionally: the muted chain
        # already justifies the event; the gap is just its natural "lead-in".
        if prev_end is not None and prev_end < ev_start:
            ev_start = prev_end

        if ev_end - ev_start >= min_silence_s:
            events.append(SilenceEvent(start=ev_start, end=ev_end, kind=ev_kind))

        # Advance past all consumed segments (up to j, the first unconsumed).
        # Update prev_end to ev_end so a subsequent gap is measured from there.
        prev_end = ev_end
        i = j

    # Final trailing gap (after the last muted block we advanced past) is
    # already handled by the leading-gap logic on the next muted or by the
    # speaker-segment path. Nothing more needed.
    return events
