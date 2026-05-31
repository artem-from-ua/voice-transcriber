"""Generate a Markdown TL;DR for a structured dialogue.

The TL;DR is built recursively, one LLM call per section, so that no single
prompt ever contains the full transcript. This is what keeps the stage
within the unified-memory budget on a 16 GB Mac — see issue #146 and
ADR 0030 for why the old single-prompt path was abandoned.

Pipeline shape (depth grows only for very long inputs):

    level 0:  per section in dialog.sections    -> tldr_section_{lang}
                                                   (input: raw segments)
    level k:  groups of <= TLDR_FANOUT          -> tldr_aggregate_{lang}
                                                   (input: previous TL;DRs)
    final:    1..TLDR_FANOUT items left         -> tldr_final_{lang}
                                                   (output: canonical TL;DR
                                                   format the renderer wants)

If `dialog.sections` is empty, TL;DR is skipped. The single synthetic
"Розмова" / "Conversation" fallback that `speech_structure` emits (when
the user passed `--no-structure`, when the LLM returned an invalid layout,
or — the case this guard was relaxed for — when the dialog is genuinely
a single topic) is *not* automatically skipped: short synthetic fallbacks
fit a single LLM call safely, so we treat the fallback like a real section
and run the per-section path. Long synthetic fallbacks would collapse back
to the OOM-prone single-prompt shape ADR 0030 was created to avoid, so we
still skip those with a logged warning. The cutoff mirrors
`speech_structure.STRUCTURE_CHUNK_THRESHOLD` — the same segment count
above which `speech_structure` itself stops trusting a single LLM call.

On any LLM failure for an individual section the stage drops that section
and continues. If the final pass fails it returns "" so the renderer omits
the section.
"""

from __future__ import annotations

import re
import sys
from typing import Callable

import mlx.core as mx

from ._progress import NullProgress, ProgressReporter
from ._prompts import call_kwargs, render as render_prompt, with_user_context
from .llm import LLMError, MlxLLM
from .silence import extract_silence_events
from .speech_structure import STRUCTURE_CHUNK_THRESHOLD
from .types import Section, Segment, StructuredDialog


# Recursion knobs. TLDR_FANOUT mirrors speech_structure.MAX_SECTIONS so on a
# realistic input (<= 7 sections) we go directly from per-section to final,
# with no aggregate level in between.
TLDR_FANOUT = 7
TLDR_MAX_LEVELS = 4  # 7^4 = 2401 sections — pathological inputs only.

# Maximum dialog length (in segments) for which we still run TL;DR over a
# synthetic single-section fallback. The bound is the same one
# `speech_structure` uses to decide that a single LLM call is safe — above
# it, the structure stage itself switches to chunked, so we cannot trust
# a single TL;DR prompt either.
SYNTHETIC_FALLBACK_TLDR_MAX_SEGMENTS = STRUCTURE_CHUNK_THRESHOLD


def _pick_prompt_name(language: str, kind: str) -> str:
    """`kind` is one of 'section', 'aggregate', 'final'."""
    suffix = "en" if language.lower().startswith("en") else "uk"
    return f"tldr_{kind}_{suffix}"


# The prompt asks the LLM not to emit a "TL;DR" heading because the renderer
# adds its own. The model regularly ignores that and starts the reply with
# "## TL;DR" / "# TL;DR" / "**TL;DR**". Strip it so we don't end up with two
# headings stacked in the output.
_LEADING_TLDR_RE = re.compile(
    r"\A\s*(?:#{1,6}\s*TL;?DR\s*\n+|\*\*TL;?DR:?\*\*\s*\n+)",
    flags=re.IGNORECASE,
)


def _strip_leading_tldr_heading(text: str) -> str:
    return _LEADING_TLDR_RE.sub("", text, count=1).lstrip("\n")


def _format_dialogue(segments: list[Segment], include_silence: bool = False) -> str:
    """Render speaker turns as `Name: text` lines, optionally with pauses."""
    lines: list[str] = []
    silence_before: dict[int, str] = {}

    if include_silence:
        from .render import MIN_SILENCE_S
        silence_events = extract_silence_events(segments, MIN_SILENCE_S)
        for ev in silence_events:
            for idx, seg in enumerate(segments):
                if seg.speaker is not None and seg.start >= ev.end:
                    silence_before[idx] = ev.format_md().lstrip("> _").rstrip("_")
                    break

    for i, seg in enumerate(segments):
        if i in silence_before:
            lines.append(silence_before[i])
        if seg.speaker is None:
            continue
        text = seg.content.strip()
        if not text:
            continue
        speaker = seg.name or seg.speaker
        lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


def _section_segments(dialog: StructuredDialog, section: Section) -> list[Segment]:
    return [
        seg for seg in dialog.segments
        if section.start_ms <= int(seg.start * 1000) < section.end_ms
    ]


def _is_synthetic_fallback(dialog: StructuredDialog) -> bool:
    """True if `dialog.sections` is the single synthetic section that
    `speech_structure._fallback_section` emits under `--no-structure` —
    i.e. one section spanning the full dialog. See `_fallback_section`
    in `src/voice/speech_structure.py`.
    """
    if len(dialog.sections) != 1 or not dialog.segments:
        return False
    only = dialog.sections[0]
    if only.title not in {"Розмова", "Conversation"}:
        return False
    expected_start = int(dialog.segments[0].start * 1000)
    expected_end = int(dialog.segments[-1].end * 1000)
    return only.start_ms == expected_start and only.end_ms == expected_end


def _format_blocks(blocks: list[tuple[str, str]]) -> str:
    """Render previously-computed TL;DRs as `### {title}\\n{body}` chunks."""
    parts: list[str] = []
    for title, body in blocks:
        parts.append(f"### {title}")
        parts.append(body.strip())
        parts.append("")
    return "\n".join(parts).rstrip()


def _log_mlx_peak(log: Callable[[str], None], label: str, before_gb: float) -> None:
    """Record per-call MLX memory peak in a format that survives a reboot
    when paired with the persistent <input>.log mirror set up by the CLI."""
    peak_gb = mx.get_peak_memory() / 1e9
    log(
        f"speech_summary: {label} mlx_active_before={before_gb:.2f}GB "
        f"peak={peak_gb:.2f}GB"
    )
    mx.reset_peak_memory()


def _llm_call(
    *,
    llm: MlxLLM,
    system_msg: dict[str, str],
    user_content: str,
    prompt_name: str,
    label: str,
    progress: ProgressReporter,
    log: Callable[[str], None],
    stage_num: str | None = None,
    stage_id: str | None = None,
) -> str | None:
    """One TL;DR LLM call with MLX peak logging. Returns None on LLMError."""
    mx.clear_cache()
    before_gb = mx.get_active_memory() / 1e9
    try:
        with progress.token_counter(
            label, stage_num=stage_num, stage_id=stage_id, kind="sub_step",
        ) as advance:
            text = llm.chat(
                [system_msg, {"role": "user", "content": user_content}],
                on_token=advance,
                **call_kwargs(prompt_name),
            )
    except LLMError as exc:
        log(f"speech_summary: LLM error in {label} — {exc}; skipping")
        _log_mlx_peak(log, label, before_gb)
        return None
    _log_mlx_peak(log, label, before_gb)
    return _strip_leading_tldr_heading(text.strip())


def _per_section_tldrs(
    dialog: StructuredDialog,
    *,
    llm: MlxLLM,
    language: str,
    include_silence: bool,
    progress: ProgressReporter,
    log: Callable[[str], None],
    user_context: str | None = None,
    stage_num: str | None = None,
    stage_id: str | None = None,
) -> list[tuple[str, str]]:
    """Level 0: one TL;DR per real section. Returns (title, body) pairs.

    The system prompt is identical for every section so we wrap the loop in
    a `prompt_cache_session` and only the per-section user content varies.
    """
    prompt_name = _pick_prompt_name(language, "section")
    system_msg = {
        "role": "system",
        "content": with_user_context(
            render_prompt(prompt_name), user_context, language=language,
        ),
    }

    results: list[tuple[str, str]] = []
    total = len(dialog.sections)
    with llm.prompt_cache_session(prefix_messages=[system_msg]):
        for idx, section in enumerate(dialog.sections, start=1):
            section_segs = _section_segments(dialog, section)
            if not section_segs:
                continue
            section_text = _format_dialogue(section_segs, include_silence=include_silence)
            if not section_text:
                continue
            user_content = f"### {section.title}\n{section_text}"
            label = f"[12/13] TL;DR section {idx}/{total} ({section.title})"
            body = _llm_call(
                llm=llm, system_msg=system_msg, user_content=user_content,
                prompt_name=prompt_name, label=label,
                progress=progress, log=log,
                stage_num=stage_num, stage_id=stage_id,
            )
            if body:
                results.append((section.title, body))
    return results


def _aggregate_level(
    blocks: list[tuple[str, str]],
    *,
    level: int,
    llm: MlxLLM,
    language: str,
    progress: ProgressReporter,
    log: Callable[[str], None],
    user_context: str | None = None,
    stage_num: str | None = None,
    stage_id: str | None = None,
) -> list[tuple[str, str]]:
    """One recursion step. Groups `blocks` into chunks of TLDR_FANOUT and
    produces one aggregated TL;DR per group. Singleton groups pass through
    unchanged — re-summarising a single TL;DR adds no information."""
    prompt_name = _pick_prompt_name(language, "aggregate")
    system_msg = {
        "role": "system",
        "content": with_user_context(
            render_prompt(prompt_name), user_context, language=language,
        ),
    }

    out: list[tuple[str, str]] = []
    groups = [
        blocks[i : i + TLDR_FANOUT]
        for i in range(0, len(blocks), TLDR_FANOUT)
    ]
    with llm.prompt_cache_session(prefix_messages=[system_msg]):
        for gi, group in enumerate(groups, start=1):
            if len(group) == 1:
                out.append(group[0])
                continue
            agg_title = f"{group[0][0]} → {group[-1][0]}"
            user_content = _format_blocks(group)
            label = f"[12/13] TL;DR level{level} group {gi}/{len(groups)}"
            body = _llm_call(
                llm=llm, system_msg=system_msg, user_content=user_content,
                prompt_name=prompt_name, label=label,
                progress=progress, log=log,
                stage_num=stage_num, stage_id=stage_id,
            )
            if body:
                out.append((agg_title, body))
    return out


def _final_pass(
    blocks: list[tuple[str, str]],
    *,
    llm: MlxLLM,
    language: str,
    progress: ProgressReporter,
    log: Callable[[str], None],
    user_context: str | None = None,
    stage_num: str | None = None,
    stage_id: str | None = None,
) -> str:
    """Produce the canonical-format TL;DR that the renderer consumes."""
    prompt_name = _pick_prompt_name(language, "final")
    system_msg = {
        "role": "system",
        "content": with_user_context(
            render_prompt(prompt_name), user_context, language=language,
        ),
    }
    user_content = _format_blocks(blocks)
    body = _llm_call(
        llm=llm, system_msg=system_msg, user_content=user_content,
        prompt_name=prompt_name, label="[12/13] TL;DR final",
        progress=progress, log=log,
        stage_num=stage_num, stage_id=stage_id,
    )
    return body or ""


def generate_tldr(
    dialog: StructuredDialog,
    *,
    llm: MlxLLM,
    language: str = "uk",
    include_silence: bool = False,
    user_context: str | None = None,
    log: Callable[[str], None] = lambda s: print(s, file=sys.stderr),
    progress: "ProgressReporter | None" = None,
    stage_num: str | None = None,
    stage_id: str | None = None,
) -> str:
    """Return a Markdown TL;DR string, or empty string when summarising is
    not viable (no segments, synthetic-fallback section under --no-structure,
    or the final LLM pass failed).

    Memory profile: each LLM call sees one section's worth of text (level 0)
    or up to `TLDR_FANOUT` short TL;DRs (level 1+ and final). No call ever
    contains the full transcript, which is what keeps the stage within the
    unified-memory budget on a 16 GB Mac — see ADR 0030.
    """
    reporter = progress if progress is not None else NullProgress()

    if not dialog.segments:
        return ""

    if not dialog.sections:
        log("speech_summary: skipping TL;DR — no sections to summarise.")
        return ""

    if _is_synthetic_fallback(dialog):
        if len(dialog.segments) > SYNTHETIC_FALLBACK_TLDR_MAX_SEGMENTS:
            log(
                "speech_summary: skipping TL;DR — synthetic single-section "
                f"fallback with {len(dialog.segments)} segments "
                f"(> {SYNTHETIC_FALLBACK_TLDR_MAX_SEGMENTS}) would risk OOM "
                "on a 16 GB Mac. Re-run with structure enabled to split "
                "the dialog into sections."
            )
            return ""
        log(
            "speech_summary: synthetic single-section fallback with "
            f"{len(dialog.segments)} segments — short enough to summarise "
            "as one section."
        )

    blocks = _per_section_tldrs(
        dialog,
        llm=llm, language=language,
        include_silence=include_silence,
        progress=reporter, log=log,
        user_context=user_context,
        stage_num=stage_num, stage_id=stage_id,
    )
    if not blocks:
        return ""

    level = 0
    while len(blocks) > TLDR_FANOUT and level < TLDR_MAX_LEVELS:
        level += 1
        blocks = _aggregate_level(
            blocks, level=level,
            llm=llm, language=language,
            progress=reporter, log=log,
            user_context=user_context,
            stage_num=stage_num, stage_id=stage_id,
        )
        if not blocks:
            return ""

    if len(blocks) > TLDR_FANOUT:
        log(
            f"speech_summary: TLDR_MAX_LEVELS={TLDR_MAX_LEVELS} reached "
            f"with {len(blocks)} blocks remaining; final pass will see all "
            f"of them (may exceed memory budget on huge inputs)."
        )

    return _final_pass(
        blocks, llm=llm, language=language,
        progress=reporter, log=log,
        user_context=user_context,
        stage_num=stage_num, stage_id=stage_id,
    )
