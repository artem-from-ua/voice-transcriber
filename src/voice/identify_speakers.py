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
from dataclasses import dataclass
from typing import Callable, Iterable, Literal

from ._progress import NullProgress, ProgressReporter
from ._prompts import call_kwargs, render as render_prompt
from .llm import LLMError, MlxLLM
from .types import Segment


NameSource = Literal["user-specified", "self-introduced", "interactive"]


@dataclass(frozen=True)
class NamedAssignment:
    name: str
    source: NameSource


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
        payload = llm.chat_json(messages, **call_kwargs("identify_system"))
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
) -> dict[str, NamedAssignment]:
    """If two clusters claim the same name, keep the higher-confidence one
    (earlier appearance breaks ties); the other becomes unidentified.
    """
    by_name: dict[str, list[str]] = defaultdict(list)
    for cluster, (name, _conf) in candidates.items():
        by_name[name].append(cluster)

    out: dict[str, NamedAssignment] = {}
    for name, clusters in by_name.items():
        if len(clusters) == 1:
            out[clusters[0]] = NamedAssignment(name=name, source="self-introduced")
            continue
        order = {"high": 0, "medium": 1, "low": 2}
        winner = min(
            clusters,
            key=lambda c: (order[candidates[c][1]], first_seen.get(c, 0.0)),
        )
        out[winner] = NamedAssignment(name=name, source="self-introduced")
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
    progress: "ProgressReporter | None" = None,
) -> dict[str, NamedAssignment]:
    """Return mapping `pyannote_label → NamedAssignment`.

    Clusters with no name in the result map are left as their pyannote label
    by the pipeline — they do not appear in this dict at all.
    """
    segs = list(segments)
    first_seen = _first_appearance(segs)
    clusters = sorted(first_seen, key=first_seen.get)

    if names_override:
        return {
            c: NamedAssignment(name=names_override[i], source="user-specified")
            for i, c in enumerate(clusters)
            if i < len(names_override)
        }

    if llm is None:
        log("identify: no LLM provided; falling back to unknown_policy.")
        candidates: dict[str, tuple[str, str]] = {}
    else:
        candidates = {}
        reporter = progress if progress is not None else NullProgress()
        with reporter.task(
            "[9/11] Ідентифікація мовців", total=len(clusters)
        ) as advance:
            for cluster in clusters:
                snippet = _cluster_snippet(segs, cluster, INTRO_WINDOW_S)
                if not snippet:
                    advance(1)
                    continue
                name, confidence = _ask_llm_for_name(llm, snippet, language)
                if name and confidence != "low":
                    candidates[cluster] = (name, confidence)
                    log(f"identify: {cluster} → {name} ({confidence})")
                advance(1, suffix=f"{len(candidates)} named")

    mapping: dict[str, NamedAssignment] = _resolve_conflicts(candidates, first_seen)

    if unknown_policy == "ask":
        for cluster in clusters:
            if cluster in mapping:
                continue
            guess = _ask_user_interactively(cluster, segs, log, read_input)
            if guess:
                mapping[cluster] = NamedAssignment(name=guess, source="interactive")

    return mapping
