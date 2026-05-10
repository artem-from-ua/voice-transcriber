"""Assign a stable emoji to each speaker name."""

from __future__ import annotations

EMOJI_PALETTE: list[str] = [
    "🔵",  # blue
    "🟢",  # green
    "🔴",  # red
    "🟡",  # yellow
    "🟣",  # purple
    "🟠",  # orange
    "🟤",  # brown
    "⚪️",  # white
    "⚫️",  # black
]


def assign_emojis(speakers_in_order: list[str]) -> dict[str, str]:
    """Map each speaker to an emoji, cycling through the palette."""
    out: dict[str, str] = {}
    for i, speaker in enumerate(speakers_in_order):
        if speaker in out:
            continue
        out[speaker] = EMOJI_PALETTE[i % len(EMOJI_PALETTE)]
    return out
