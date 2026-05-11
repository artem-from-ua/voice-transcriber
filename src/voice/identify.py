"""Assign human-readable names to pyannote speaker clusters.

For each cluster, take its first ~60 seconds of speech and ask the LLM whether
the speaker introduced themselves in that snippet. If yes, use the name.
Otherwise fall back according to `unknown_policy` (`ask` or `keep`).

`names_override` (CLI `--names "Артем,Остап"`) skips the LLM entirely and maps
clusters by order of first appearance in time.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from typing import Callable, Iterable, Literal

from ._prompts import render as render_prompt
from .llm import LLMError, MlxLLM
from .types import Segment


INTRO_WINDOW_S = 60.0


Policy = Literal["ask", "keep"]


def _first_appearance(segments: list[Segment]) -> dict[str, float]:
    first: dict[str, float] = {}
    for seg in segments:
        if seg.speaker is None:
            continue
        first.setdefault(seg.speaker, seg.start)
    return first


def _cluster_snippet(segments: list[Segment], speaker: str, window_s: float) -> str:
    """First `window_s` of speech for `speaker`, content concatenated."""
    first_appearance = next((s.start for s in segments if s.speaker == speaker), 0.0)
    cutoff = first_appearance + window_s
    parts = [
        s.content for s in segments
        if s.speaker == speaker and s.start < cutoff
    ]
    return " ".join(parts).strip()


def _validate_name(name: str | None) -> str | None:
    if not name:
        return None
    name = name.strip()
    if len(name) < 2:
        return None
    if not name[0].isalpha() or not name[0].isupper():
        return None
    return name


def _ask_llm_for_name(
    llm: MlxLLM,
    snippet: str,
    language: str,
) -> tuple[str | None, str]:
    messages = [
        {"role": "system", "content": render_prompt("identify_system", language=language)},
        {"role": "user", "content": render_prompt("identify_user", snippet=snippet)},
    ]
    try:
        payload = llm.chat_json(messages, temperature=0.1, max_tokens=64)
    except LLMError:
        return None, "low"

    if not isinstance(payload, dict):
        return None, "low"
    name = _validate_name(payload.get("name"))
    confidence = payload.get("confidence", "low")
    if confidence not in {"high", "medium", "low"}:
        confidence = "low"
    return name, confidence


def _resolve_conflicts(
    candidates: dict[str, tuple[str, str]],
    first_seen: dict[str, float],
) -> dict[str, str]:
    """If two clusters claim the same name, keep the higher-confidence one
    (earlier appearance breaks ties); the other becomes unidentified.
    """
    by_name: dict[str, list[str]] = defaultdict(list)
    for cluster, (name, _conf) in candidates.items():
        by_name[name].append(cluster)

    out: dict[str, str] = {}
    for name, clusters in by_name.items():
        if len(clusters) == 1:
            out[clusters[0]] = name
            continue
        order = {"high": 0, "medium": 1, "low": 2}
        winner = min(
            clusters,
            key=lambda c: (order[candidates[c][1]], first_seen.get(c, 0.0)),
        )
        out[winner] = name
    return out


def _ask_user_interactively(
    speaker: str,
    segments: list[Segment],
    out: Callable[[str], None] = lambda s: print(s, file=sys.stderr),
    read: Callable[[str], str] = input,
) -> str | None:
    samples = [s.content for s in segments if s.speaker == speaker][:3]
    out(f"\nCould not identify {speaker} automatically. First lines:")
    for s in samples:
        out(f"  • {s[:120]}")
    try:
        reply = read(f"Name for {speaker} (Enter to keep as-is): ").strip()
    except EOFError:
        reply = ""
    return reply or None


def identify_speakers(
    segments: Iterable[Segment],
    *,
    language: str = "uk",
    llm: MlxLLM | None = None,
    unknown_policy: Policy = "ask",
    names_override: list[str] | None = None,
    log: Callable[[str], None] = lambda s: print(s, file=sys.stderr),
    read_input: Callable[[str], str] = input,
) -> dict[str, str]:
    """Return mapping `pyannote_label → human_name`.

    Clusters with no name in the result map are left as their pyannote label.
    """
    segs = list(segments)
    first_seen = _first_appearance(segs)
    clusters = sorted(first_seen, key=first_seen.get)

    if names_override:
        return {c: names_override[i] for i, c in enumerate(clusters) if i < len(names_override)}

    if llm is None:
        log("identify: no LLM provided; falling back to unknown_policy.")
        candidates: dict[str, tuple[str, str]] = {}
    else:
        candidates = {}
        for cluster in clusters:
            snippet = _cluster_snippet(segs, cluster, INTRO_WINDOW_S)
            if not snippet:
                continue
            name, confidence = _ask_llm_for_name(llm, snippet, language)
            if name and confidence != "low":
                candidates[cluster] = (name, confidence)
                log(f"identify: {cluster} → {name} ({confidence})")

    mapping = _resolve_conflicts(candidates, first_seen)

    if unknown_policy == "ask":
        for cluster in clusters:
            if cluster in mapping:
                continue
            guess = _ask_user_interactively(cluster, segs, log, read_input)
            if guess:
                mapping[cluster] = guess

    return mapping
