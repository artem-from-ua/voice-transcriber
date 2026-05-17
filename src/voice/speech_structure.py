"""Split a dialogue into thematic sections using the LLM.

The LLM gets a "script" of utterances with timestamps and pause markers and
returns a JSON list of sections (title + start_ms..end_ms). The output is
validated: ranges in bounds, contiguous, covering the whole dialogue. If the
LLM fails or returns an invalid layout, the fallback is a single section
titled "Розмова" / "Conversation".

For long dialogues (>= STRUCTURE_CHUNK_THRESHOLD segments) the script is
split into overlapping chunks and the LLM is called once per chunk. The
per-chunk results are reconciled (sort, merge same-title neighbours, fix
edges) into a single section list. This keeps each prompt small enough for
gemma-3-12b on a 16 GB Mac without OOM (see ADR 0018).
"""

from __future__ import annotations

import sys
from typing import Callable, Iterable

from ._progress import NullProgress, ProgressReporter
from ._prompts import call_kwargs, render as render_prompt, with_user_context
from .llm import LLMError, MlxLLM
from .types import Section, Segment, StructuredDialog


PAUSE_GAP_S = 3.0
MIN_SECTIONS = 2
MAX_SECTIONS = 7

# Long-dialogue chunking. Tuned for gemma-3-12b 4-bit on a 16 GB Mac:
# ~35 segments × ~50 tokens/segment ≈ 1.7K user prompt + 2K output budget,
# which keeps peak MLX comfortably under 12 GB after the proofread KV
# fragments have been cleared.
STRUCTURE_CHUNK_THRESHOLD = 60
STRUCTURE_CHUNK_SIZE = 35
STRUCTURE_CHUNK_OVERLAP = 2
# Per-chunk validator bounds: a chunk may legitimately have a single
# section (e.g. one topic for the whole chunk).
CHUNK_MIN_SECTIONS = 1
CHUNK_MAX_SECTIONS = 4


def _format_timestamp(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def _build_script(segments: list[Segment], pause_gap_s: float) -> str:
    """Render utterances + pause markers in a compact form for the LLM."""
    lines: list[str] = []
    prev_end: float | None = None
    for seg in segments:
        if prev_end is not None and seg.start - prev_end >= pause_gap_s:
            lines.append(f"[{_format_timestamp(prev_end)} — pause {seg.start - prev_end:.0f}s]")
        speaker = seg.name or seg.speaker or "?"
        lines.append(f"[{_format_timestamp(seg.start)}] {speaker}: {seg.content}")
        prev_end = seg.end
    return "\n".join(lines)


def _fallback_section(segments: list[Segment], language: str) -> list[Section]:
    title = "Conversation" if language.startswith("en") else "Розмова"
    if not segments:
        return [Section(title=title, start_ms=0, end_ms=0)]
    start_ms = int(segments[0].start * 1000)
    end_ms = int(segments[-1].end * 1000)
    return [Section(title=title, start_ms=start_ms, end_ms=end_ms)]


def _validate(
    payload: object,
    *,
    total_start_ms: int,
    total_end_ms: int,
    min_sections: int = MIN_SECTIONS,
    max_sections: int = MAX_SECTIONS,
    require_exact_bounds: bool = True,
) -> list[Section] | None:
    if not isinstance(payload, dict):
        return None
    raw = payload.get("sections")
    if not isinstance(raw, list) or not (min_sections <= len(raw) <= max_sections):
        return None

    sections: list[Section] = []
    for item in raw:
        if not isinstance(item, dict):
            return None
        title = item.get("title")
        start = item.get("start_ms")
        end = item.get("end_ms")
        if not isinstance(title, str) or not title.strip():
            return None
        if not isinstance(start, int) or not isinstance(end, int):
            return None
        if end <= start:
            return None
        sections.append(Section(title=title.strip(), start_ms=start, end_ms=end))

    sections.sort(key=lambda s: s.start_ms)
    if require_exact_bounds:
        if sections[0].start_ms != total_start_ms or sections[-1].end_ms != total_end_ms:
            return None
        for prev, nxt in zip(sections, sections[1:]):
            if nxt.start_ms != prev.end_ms:
                return None
    return sections


def _chunk_segments(
    speech: list[Segment],
    *,
    chunk_size: int,
    overlap: int,
    snap_to_speaker_boundary: bool = True,
    max_snap_extension: int = 5,
) -> list[list[Segment]]:
    """Split into overlapping batches. Last batch absorbs the remainder.

    `overlap` segments are duplicated between adjacent chunks so the LLM
    sees the tail of the previous topic when judging section boundaries.

    `snap_to_speaker_boundary` extends each chunk past `chunk_size` until
    the next segment starts a new speaker turn (relative to the chunk's
    last segment), so we never cut mid-monologue. The extension is capped
    at `max_snap_extension` segments — very long monologues are still cut
    if necessary, but most natural turns end well within the cap.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(f"overlap must be in [0, chunk_size), got {overlap}")
    if max_snap_extension < 0:
        raise ValueError(
            f"max_snap_extension must be non-negative, got {max_snap_extension}"
        )

    chunks: list[list[Segment]] = []
    step = chunk_size - overlap
    i = 0
    n = len(speech)
    while i < n:
        end = min(i + chunk_size, n)
        if snap_to_speaker_boundary and end < n:
            # Look at the speaker label of the would-be last segment;
            # keep extending while the next segment is the same speaker
            # (the cut would land mid-monologue) — up to max_snap_extension.
            last_speaker = speech[end - 1].speaker
            extension = 0
            while (
                end < n
                and extension < max_snap_extension
                and speech[end].speaker == last_speaker
            ):
                end += 1
                extension += 1
        chunk = speech[i:end]
        chunks.append(chunk)
        if end >= n:
            break
        i = end - overlap
    return chunks


def _merge_same_title(sections: list[Section]) -> list[Section]:
    """Collapse adjacent sections with identical (case-insensitive) titles.

    Chunks frequently emit the same topic name on both sides of a boundary
    when a single conversation thread spans them.
    """
    if not sections:
        return sections
    out: list[Section] = [sections[0]]
    for s in sections[1:]:
        prev = out[-1]
        if s.title.strip().lower() == prev.title.strip().lower():
            out[-1] = Section(
                title=prev.title, start_ms=prev.start_ms, end_ms=s.end_ms
            )
        else:
            out.append(s)
    return out


def _reconcile_chunks(
    chunk_sections: list[list[Section]],
    *,
    total_start_ms: int,
    total_end_ms: int,
    max_sections: int,
) -> list[Section] | None:
    """Stitch per-chunk section lists into one contiguous list.

    Algorithm:
    1. Concatenate all sections, sort by start_ms.
    2. For overlapping chunks, drop sections that are wholly contained in
       a previous section (overlap region duplicates).
    3. Force the first section to start at total_start_ms, the last to end
       at total_end_ms. Snap each section's start to the previous section's
       end (closing micro-gaps and overlaps).
    4. Merge same-title neighbours.
    5. If the final count exceeds max_sections, greedily merge the two
       shortest neighbours until it fits.
    6. Return None on degenerate inputs (no sections, zero-length spans).
    """
    flat: list[Section] = [s for chunk in chunk_sections for s in chunk]
    if not flat:
        return None
    flat.sort(key=lambda s: (s.start_ms, s.end_ms))

    # Drop sections wholly contained in the previous one (overlap dupes).
    dedup: list[Section] = []
    for s in flat:
        if dedup and s.end_ms <= dedup[-1].end_ms:
            continue
        dedup.append(s)
    if not dedup:
        return None

    # Snap edges. First → total_start, each subsequent.start → prev.end.
    snapped: list[Section] = []
    prev_end = total_start_ms
    for idx, s in enumerate(dedup):
        start = prev_end if idx == 0 else prev_end
        # If the next chunk's first section starts way later than prev_end
        # (gap), we still snap to prev_end — coverage trumps fidelity here.
        end = s.end_ms
        if end <= start:
            continue
        snapped.append(Section(title=s.title, start_ms=start, end_ms=end))
        prev_end = end

    if not snapped:
        return None
    # Force last section to cover the tail.
    last = snapped[-1]
    if last.end_ms < total_end_ms:
        snapped[-1] = Section(
            title=last.title, start_ms=last.start_ms, end_ms=total_end_ms
        )

    merged = _merge_same_title(snapped)

    while len(merged) > max_sections:
        # Find the pair of adjacent sections with the smallest combined
        # span and fuse them under the longer side's title.
        best_idx = 0
        best_span = None
        for i in range(len(merged) - 1):
            span = (merged[i].end_ms - merged[i].start_ms) + (
                merged[i + 1].end_ms - merged[i + 1].start_ms
            )
            if best_span is None or span < best_span:
                best_span = span
                best_idx = i
        a = merged[best_idx]
        b = merged[best_idx + 1]
        keep_title = a.title if (a.end_ms - a.start_ms) >= (b.end_ms - b.start_ms) else b.title
        merged = (
            merged[:best_idx]
            + [Section(title=keep_title, start_ms=a.start_ms, end_ms=b.end_ms)]
            + merged[best_idx + 2 :]
        )

    return merged


def _structure_single_pass(
    speech: list[Segment],
    *,
    llm: MlxLLM,
    language: str,
    total_start_ms: int,
    total_end_ms: int,
    on_token: Callable[[int], None] | None = None,
    user_context: str | None = None,
) -> list[Section] | None:
    """One LLM call for the whole dialogue. Returns validated sections or None."""
    script = _build_script(speech, PAUSE_GAP_S)
    messages = [
        {
            "role": "system",
            "content": with_user_context(
                render_prompt("structure_system", language=language),
                user_context,
                language=language,
            ),
        },
        {
            "role": "user",
            "content": render_prompt(
                "structure_user",
                total_start_ms=total_start_ms,
                total_end_ms=total_end_ms,
                script=script,
            ),
        },
    ]
    payload = llm.chat_json(
        messages,
        on_token=on_token,
        **call_kwargs("structure_system"),
    )
    return _validate(
        payload,
        total_start_ms=total_start_ms,
        total_end_ms=total_end_ms,
    )


def _structure_chunk(
    chunk: list[Segment],
    *,
    llm: MlxLLM,
    language: str,
    on_token: Callable[[int], None] | None = None,
    user_context: str | None = None,
) -> list[Section] | None:
    """One LLM call for a sub-range. Returns sections covering the chunk."""
    if not chunk:
        return None
    chunk_start_ms = int(chunk[0].start * 1000)
    chunk_end_ms = int(chunk[-1].end * 1000)
    script = _build_script(chunk, PAUSE_GAP_S)
    # Chunked path uses its own prompt — see issue #151 / ADR 0032. The
    # single-pass `structure_system` told the LLM "2 to 7 sections, start at
    # 0 ms" which is wrong for a non-leading chunk and over the
    # CHUNK_MAX_SECTIONS cap. `structure_chunk_system` says "1 to 4 sections,
    # use the supplied total_start_ms" so per-chunk acceptance jumps from
    # ~24% to >80% on long inputs.
    messages = [
        {
            "role": "system",
            "content": with_user_context(
                render_prompt("structure_chunk_system", language=language),
                user_context,
                language=language,
            ),
        },
        {
            "role": "user",
            "content": render_prompt(
                "structure_chunk_user",
                total_start_ms=chunk_start_ms,
                total_end_ms=chunk_end_ms,
                script=script,
            ),
        },
    ]
    payload = llm.chat_json(
        messages,
        on_token=on_token,
        **call_kwargs("structure_chunk_system"),
    )
    # Per-chunk validator is more permissive: a chunk may legitimately
    # contain a single topic, and edges may drift by a few ms because the
    # LLM rounds to nearby pause markers. require_exact_bounds=False lets
    # us reconcile across chunks ourselves.
    return _validate(
        payload,
        total_start_ms=chunk_start_ms,
        total_end_ms=chunk_end_ms,
        min_sections=CHUNK_MIN_SECTIONS,
        max_sections=CHUNK_MAX_SECTIONS,
        require_exact_bounds=False,
    )


def structure_dialog(
    segments: Iterable[Segment],
    *,
    llm: MlxLLM | None = None,
    language: str = "uk",
    user_context: str | None = None,
    log: Callable[[str], None] = lambda s: print(s, file=sys.stderr),
    progress: "ProgressReporter | None" = None,
) -> StructuredDialog:
    """Return a `StructuredDialog`. Falls back to a single section on error."""
    segs = list(segments)
    speech = [s for s in segs if s.speaker is not None]
    if not speech:
        return StructuredDialog(sections=[], segments=segs)

    total_start_ms = int(speech[0].start * 1000)
    total_end_ms = int(speech[-1].end * 1000)

    if llm is None:
        log("structure: no LLM provided; using single-section fallback")
        return StructuredDialog(sections=_fallback_section(segs, language), segments=segs)

    reporter = progress if progress is not None else NullProgress()

    if len(speech) < STRUCTURE_CHUNK_THRESHOLD:
        # Fast path: one call for the whole dialogue.
        try:
            with reporter.token_counter("[10/13] Структурування на секції") as advance:
                sections = _structure_single_pass(
                    speech,
                    llm=llm,
                    language=language,
                    total_start_ms=total_start_ms,
                    total_end_ms=total_end_ms,
                    on_token=advance,
                    user_context=user_context,
                )
        except LLMError as exc:
            log(f"structure: LLM error — {exc}; falling back to single section")
            return StructuredDialog(sections=_fallback_section(segs, language), segments=segs)

        if sections is None:
            log("structure: invalid layout — falling back to single section")
            return StructuredDialog(sections=_fallback_section(segs, language), segments=segs)

        log(f"structure: {len(sections)} section(s)")
        return StructuredDialog(sections=sections, segments=segs)

    # Chunked path. Each chunk gets its own LLM call. Failed chunks are
    # skipped — reconcile will paper over the gap. If *every* chunk fails,
    # fall back to a single section.
    chunks = _chunk_segments(
        speech,
        chunk_size=STRUCTURE_CHUNK_SIZE,
        overlap=STRUCTURE_CHUNK_OVERLAP,
    )
    log(
        f"structure: long dialogue ({len(speech)} segments) — "
        f"splitting into {len(chunks)} chunks"
    )
    chunk_results: list[list[Section]] = []
    # prompt_cache_session prefix MUST match what each per-chunk call
    # sends as its system message — otherwise the cache fingerprint
    # never matches and the session amortises nothing. We use the
    # chunked prompt here for the same reason _structure_chunk does
    # (see issue #151).
    structure_system_msg = {
        "role": "system",
        "content": with_user_context(
            render_prompt("structure_chunk_system", language=language),
            user_context,
            language=language,
        ),
    }
    try:
        with llm.prompt_cache_session(
            prefix_messages=[structure_system_msg]
        ), reporter.task(
            "[10/13] Структурування на секції", total=len(chunks)
        ) as advance:
            for idx, chunk in enumerate(chunks, start=1):
                try:
                    sections = _structure_chunk(
                        chunk, llm=llm, language=language,
                        user_context=user_context,
                    )
                except LLMError as exc:
                    log(f"structure: chunk {idx}/{len(chunks)} LLM error — {exc}")
                    advance(1, suffix=f"chunk {idx} failed")
                    continue
                if sections is None:
                    log(f"structure: chunk {idx}/{len(chunks)} invalid — skipping")
                    advance(1, suffix=f"chunk {idx} invalid")
                    continue
                chunk_results.append(sections)
                advance(1, suffix=f"chunk {idx} ok ({len(sections)} sec)")
    except LLMError as exc:
        log(f"structure: LLM error — {exc}; falling back to single section")
        return StructuredDialog(sections=_fallback_section(segs, language), segments=segs)

    if not chunk_results:
        log("structure: all chunks failed — falling back to single section")
        return StructuredDialog(sections=_fallback_section(segs, language), segments=segs)

    reconciled = _reconcile_chunks(
        chunk_results,
        total_start_ms=total_start_ms,
        total_end_ms=total_end_ms,
        max_sections=MAX_SECTIONS,
    )
    if reconciled is None:
        log("structure: reconcile failed — falling back to single section")
        return StructuredDialog(sections=_fallback_section(segs, language), segments=segs)

    log(f"structure: {len(reconciled)} section(s) across {len(chunks)} chunks")
    return StructuredDialog(sections=reconciled, segments=segs)
