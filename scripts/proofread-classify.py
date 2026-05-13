#!/usr/bin/env python3
"""Classify proofread changes vs the merge stage; prepare judge inputs;
parse spot-check + judge marks into a final comparison table.

Used to answer issue #57 — "is the proofread stage still earning its
keep on Whisper output?". The pipeline already dumps 04-merge.json
(pre-proofread Segment[]) and 05-proofread.json (post-proofread). This
script reads both, classifies each segment-pair into one of four
categories (unchanged, cosmetic, proper_noun_fix, substantive_rewrite),
and prepares artefacts for human spot-check and Claude-Code text-only
judgment.

Output is intentionally plain text with no ANSI, no `\\r`, no emoji —
the script is meant to be readable when invoked through Claude Code's
Bash tool. The final per-subcommand summary is a JSON block between
`---SUMMARY-START---` / `---SUMMARY-END---` markers for reliable
machine parsing.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CATEGORIES = ("unchanged", "cosmetic", "proper_noun_fix", "substantive_rewrite")


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


_PUNCTUATION_RE = re.compile(r"[.,!?;:\"'()—–\-«»…]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_CAMEL_RE = re.compile(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b|\b[a-z]+[A-Z][a-z]+\b")


def _collapse_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _aggressive_normalize(s: str) -> str:
    return _collapse_ws(_PUNCTUATION_RE.sub("", s.lower()))


def _diff_tokens(a: str, b: str) -> list[str]:
    """Return tokens that differ between `a` and `b` (added/removed)."""
    a_tokens = a.split()
    b_tokens = b.split()
    out: list[str] = []
    sm = difflib.SequenceMatcher(a=a_tokens, b=b_tokens, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        out.extend(a_tokens[i1:i2])
        out.extend(b_tokens[j1:j2])
    return out


def classify_pair(asr_text: str, proofread_text: str) -> str:
    """Return one of CATEGORIES describing how `proofread_text` relates
    to `asr_text`. Pure function — text-only, no I/O."""
    a = _collapse_ws(asr_text)
    b = _collapse_ws(proofread_text)
    if a == b:
        return "unchanged"
    if _aggressive_normalize(a) == _aggressive_normalize(b):
        return "cosmetic"
    diff = _diff_tokens(a, b)
    if not diff:
        # Same tokens, different order/spacing past collapse — treat as cosmetic.
        return "cosmetic"
    for tok in diff:
        if _LATIN_RE.search(tok) or _CAMEL_RE.search(tok):
            return "proper_noun_fix"
    return "substantive_rewrite"


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


@dataclass
class SegmentPair:
    index: int
    start: float
    end: float
    speaker: str | None
    asr_text: str
    proofread_text: str
    category: str


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _seg_view(seg: dict[str, Any]) -> tuple[float, float, str, str | None]:
    return (
        float(seg["start"]),
        float(seg["end"]),
        str(seg.get("content", "")),
        seg.get("speaker"),
    )


def load_pairs(dump_dir: Path) -> list[SegmentPair]:
    merge_path = dump_dir / "04-merge.json"
    proof_path = dump_dir / "05-proofread.json"
    if not merge_path.exists():
        die(f"missing {merge_path}")
    if not proof_path.exists():
        die(f"missing {proof_path}")
    merge = _load_json(merge_path)
    proof = _load_json(proof_path)
    if not isinstance(merge, list) or not isinstance(proof, list):
        die("04-merge.json and 05-proofread.json must each be a JSON array")
    if len(merge) != len(proof):
        die(
            f"segment count mismatch: 04-merge has {len(merge)}, "
            f"05-proofread has {len(proof)}"
        )
    pairs: list[SegmentPair] = []
    for idx, (m, p) in enumerate(zip(merge, proof)):
        ms, me, mt, msp = _seg_view(m)
        ps, pe, pt, _psp = _seg_view(p)
        # Timestamps are an invariant — proofread cannot change them.
        if abs(ms - ps) > 1e-6 or abs(me - pe) > 1e-6:
            die(
                f"segment {idx} timestamp drift: merge={ms:.3f}-{me:.3f} "
                f"vs proofread={ps:.3f}-{pe:.3f}"
            )
        pairs.append(
            SegmentPair(
                index=idx,
                start=ms,
                end=me,
                speaker=msp,
                asr_text=mt,
                proofread_text=pt,
                category=classify_pair(mt, pt),
            )
        )
    return pairs


def load_stage_meta(dump_dir: Path, stage: str) -> dict[str, Any] | None:
    meta_path = dump_dir / "01-meta.json"
    if not meta_path.exists():
        return None
    meta = _load_json(meta_path)
    stages = meta.get("stages") if isinstance(meta, dict) else None
    if not isinstance(stages, dict):
        return None
    info = stages.get(stage)
    return info if isinstance(info, dict) else None


def categorise(pairs: list[SegmentPair]) -> dict[str, Any]:
    counts = {c: 0 for c in CATEGORIES}
    for p in pairs:
        counts[p.category] += 1
    total = len(pairs)
    changed = total - counts["unchanged"]
    hit_rate = round(100.0 * changed / total, 1) if total else 0.0
    return {"total": total, "hit_rate_pct": hit_rate, **counts}


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def info(msg: str) -> None:
    print(f"[info]  {msg}", flush=True)


def tick(msg: str) -> None:
    print(f"[tick]  {msg}", flush=True)


def done(msg: str) -> None:
    print(f"[done]  {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"[warn]  {msg}", flush=True)


def die(msg: str, code: int = 1) -> None:
    print(f"[error] {msg}", flush=True)
    sys.exit(code)


def emit_summary(payload: dict[str, Any]) -> None:
    print("---SUMMARY-START---", flush=True)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)
    print("---SUMMARY-END---", flush=True)


def _fmt_ts(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = seconds - minutes * 60
    return f"{minutes:02d}:{secs:05.2f}"


# ---------------------------------------------------------------------------
# classify subcommand
# ---------------------------------------------------------------------------


def cmd_classify(args: argparse.Namespace) -> None:
    dump_dir = Path(args.dump_dir).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    info(f"Reading 04-merge.json and 05-proofread.json from {dump_dir}")
    pairs = load_pairs(dump_dir)
    info(f"Classifying {len(pairs)} segment pair(s)")

    summary = categorise(pairs)
    stage_info = load_stage_meta(dump_dir, "proofread") or {}
    summary["wall_clock_s"] = stage_info.get("wall_clock_s")
    summary["llm_calls"] = stage_info.get("llm_calls")
    summary["model"] = stage_info.get("model")

    spot_check_segments = [p for p in pairs if p.category == "substantive_rewrite"]
    summary["spot_check_needed"] = len(spot_check_segments)

    categories_path = out_dir / "categories.json"
    categories_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    done(f"Wrote {categories_path}")

    spot_path = out_dir / "spot-check.md"
    spot_path.write_text(
        _render_spot_check(spot_check_segments), encoding="utf-8"
    )
    if spot_check_segments:
        warn(
            f"{len(spot_check_segments)} substantive_rewrite segment(s) "
            f"need manual spot-check; see {spot_path}"
        )
    else:
        done(f"No substantive_rewrite segments — spot-check not required.")

    emit_summary(summary)


def _render_spot_check(segments: list[SegmentPair]) -> str:
    lines: list[str] = [
        "# Proofread spot-check",
        "",
        "For each segment below, listen to the audio in the given timestamp range",
        "and mark exactly one of `better` / `worse` / `neutral`:",
        "",
        "- **better** — proofread fixed a real ASR error: ASR produced a non-word",
        "  (`корище`, `контролуси`, `твими`), a clear mishearing of a technical",
        "  term (`HugginsFace` → `Hugging Face`), or a grammatical impossibility,",
        "  and proofread restored what the speaker actually said.",
        "- **worse** — proofread changed text that did not need fixing, or moved",
        "  it further from what the speaker said. This includes:",
        "    - swapping a word for a synonym (e.g. `чуваки` → `хлопці`) when the",
        "      original was correctly transcribed;",
        "    - rewriting a colloquial form into a literary one or vice versa when",
        "      the ASR-rendered form already matched the speech;",
        "    - introducing a regional / non-standard variant when the ASR form",
        "      was the conventional one for the context (e.g. tech slang —",
        "      `продуктовий` → `продуктівий`).",
        "- **neutral** — difference is immaterial: punctuation, capitalisation,",
        "  near-equivalent demonstratives, or a guess on a garbled segment that",
        "  is plausible but cannot be verified from audio.",
        "",
        "Guiding principle: the goal of this tool is to capture **real, lively",
        "Ukrainian conversation as it was actually spoken** (CLAUDE.md \"Project",
        "goal\"). We do **not** want proofread to enforce literary norms,",
        "substitute synonyms, or 'clean up' how people talk. Proofread's only",
        "job is to fix ASR mishearings — anything beyond that is a regression.",
        "",
        "After filling marks, run:",
        "    python scripts/proofread-classify.py parse-marks \\",
        "        --marks-file <this-file> \\",
        "        --judge-marks docs/measurements/57/judge-marks.json \\",
        "        --output docs/measurements/57/final-table.md",
        "",
    ]
    if not segments:
        lines.append("_No substantive_rewrite segments — nothing to check._")
        return "\n".join(lines) + "\n"
    for seg in segments:
        speaker = seg.speaker if seg.speaker else "?"
        lines.extend([
            f"## Segment {seg.index} — timestamp {_fmt_ts(seg.start)}-{_fmt_ts(seg.end)} (speaker: {speaker})",
            "",
            f"**ASR (merge):** {seg.asr_text}",
            f"**Proofread:**  {seg.proofread_text}",
            "",
            "- [ ] better",
            "- [ ] worse",
            "- [ ] neutral",
            "",
        ])
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# prepare-judge subcommand
# ---------------------------------------------------------------------------


def cmd_prepare_judge(args: argparse.Namespace) -> None:
    dump_dir = Path(args.dump_dir).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    info(f"Reading dump from {dump_dir}")
    pairs = load_pairs(dump_dir)
    candidates = [p for p in pairs if p.category == "substantive_rewrite"]
    info(f"{len(candidates)} substantive_rewrite segment(s) selected for judging")

    out_path.write_text(_render_judge_input(candidates), encoding="utf-8")
    done(f"Wrote {out_path}")
    emit_summary({"segments_for_judge": len(candidates), "path": str(out_path)})


def _render_judge_input(segments: list[SegmentPair]) -> str:
    lines: list[str] = [
        f"# Judge input — {len(segments)} segment(s)",
        "",
        "For each segment below, return a mark in {better, worse, neutral}:",
        "",
        "- **better** — proofread fixed a real ASR error: ASR produced a non-word",
        "  (`корище`, `контролуси`, `твими`), a clear mishearing of a technical",
        "  term (`HugginsFace` → `Hugging Face`), or a grammatical impossibility,",
        "  and proofread restored what the speaker most plausibly said.",
        "- **worse** — proofread changed text that did not need fixing, or moved",
        "  it further from what the speaker said. This includes:",
        "    - swapping a word for a synonym (e.g. `чуваки` → `хлопці`) when the",
        "      original was correctly transcribed;",
        "    - rewriting a colloquial form into a literary one or vice versa when",
        "      the ASR-rendered form already matched the speech;",
        "    - introducing a regional / non-standard variant when the ASR form",
        "      was the conventional one for the context (e.g. tech slang).",
        "- **neutral** — difference is immaterial: punctuation, capitalisation,",
        "  near-equivalent demonstratives, or a guess on a garbled segment that",
        "  is plausible but cannot be verified from text alone.",
        "",
        "Guiding principle: the goal of this tool is to capture **real, lively",
        "Ukrainian conversation as it was actually spoken** (CLAUDE.md \"Project",
        "goal\"). We do **not** want proofread to enforce literary norms,",
        "substitute synonyms, or 'clean up' how people talk. Proofread's only",
        "job is to fix ASR mishearings — anything beyond that is a regression.",
        "",
        "You do NOT have access to the audio. Judge from text and context only.",
        "When the direction of a change (literary↔colloquial, dialect↔standard)",
        "cannot be verified without listening, prefer `neutral` over guessing.",
        "The human evaluator has audio access — your role is text-only judgment",
        "for calibration: an agreement metric will tell us whether text-only LLM",
        "judgment can be trusted for future automated measurements.",
        "",
        "Reply format (single JSON block, between markers — save as judge-marks.json):",
        "",
        "    ---JUDGE-MARKS-START---",
        "    {",
        '      "<segment-index>": {"mark": "better", "rationale": "..."},',
        "      ...",
        "    }",
        "    ---JUDGE-MARKS-END---",
        "",
    ]
    if not segments:
        lines.append("_No substantive_rewrite segments — nothing to judge._")
        return "\n".join(lines) + "\n"
    for seg in segments:
        speaker = seg.speaker if seg.speaker else "?"
        lines.extend([
            f"## Segment {seg.index} — timestamp {_fmt_ts(seg.start)}-{_fmt_ts(seg.end)} (speaker: {speaker})",
            "",
            f"**ASR (merge):** {seg.asr_text}",
            f"**Proofread:**  {seg.proofread_text}",
            "",
        ])
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# parse-marks subcommand
# ---------------------------------------------------------------------------


_SEGMENT_HEADER_RE = re.compile(r"^##\s+Segment\s+(\d+)\b", re.MULTILINE)
_CHECKBOX_RE = re.compile(r"^- \[([ xX])\]\s+(better|worse|neutral)\s*$", re.MULTILINE)


def parse_spot_check(text: str) -> dict[int, str]:
    """Parse a filled `spot-check.md`. Returns ``{segment_index: mark}``.

    Raises a ValueError if a segment has 0 or >1 marks.
    """
    # Split text by segment headers; first chunk is the file preamble.
    parts = re.split(r"(?m)^##\s+Segment\s+(\d+)\b[^\n]*\n", text)
    # parts == [preamble, idx1, body1, idx2, body2, ...]
    result: dict[int, str] = {}
    for i in range(1, len(parts), 2):
        idx = int(parts[i])
        body = parts[i + 1] if i + 1 < len(parts) else ""
        marks = [
            mark
            for state, mark in _CHECKBOX_RE.findall(body)
            if state in ("x", "X")
        ]
        if len(marks) == 0:
            raise ValueError(f"Segment {idx}: expected exactly one mark, found 0")
        if len(marks) > 1:
            raise ValueError(
                f"Segment {idx}: expected exactly one mark, found {len(marks)} ({marks})"
            )
        result[idx] = marks[0]
    return result


def parse_judge_marks(text: str) -> dict[int, dict[str, str]]:
    """Parse a judge-marks JSON document. Returns ``{segment_index: {mark, rationale}}``."""
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError("judge marks file must be a JSON object")
    out: dict[int, dict[str, str]] = {}
    for k, v in obj.items():
        try:
            idx = int(k)
        except (TypeError, ValueError) as e:
            raise ValueError(f"judge-marks: key '{k}' is not an int") from e
        if not isinstance(v, dict) or "mark" not in v:
            raise ValueError(f"judge-marks segment {k}: missing 'mark'")
        if v["mark"] not in ("better", "worse", "neutral"):
            raise ValueError(
                f"judge-marks segment {k}: invalid mark '{v['mark']}'"
            )
        out[idx] = {"mark": v["mark"], "rationale": v.get("rationale", "")}
    return out


def cmd_parse_marks(args: argparse.Namespace) -> None:
    marks_path = Path(args.marks_file).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve()

    info(f"Parsing {marks_path}")
    try:
        human = parse_spot_check(marks_path.read_text(encoding="utf-8"))
    except ValueError as e:
        die(str(e))

    judge: dict[int, dict[str, str]] = {}
    if args.judge_marks:
        judge_path = Path(args.judge_marks).expanduser().resolve()
        info(f"Parsing judge marks from {judge_path}")
        try:
            judge = parse_judge_marks(judge_path.read_text(encoding="utf-8"))
        except ValueError as e:
            die(str(e))
        # Validate that keys align (warn if not — they should be identical sets).
        only_human = set(human) - set(judge)
        only_judge = set(judge) - set(human)
        if only_human:
            warn(f"segments marked by human but not judge: {sorted(only_human)}")
        if only_judge:
            warn(f"segments marked by judge but not human: {sorted(only_judge)}")

    human_counts = _count_marks(list(human.values()))
    judge_counts = _count_marks([v["mark"] for v in judge.values()])

    agreement: dict[str, Any] | None = None
    if judge:
        common = set(human) & set(judge)
        agreed = sum(1 for i in common if human[i] == judge[i]["mark"])
        total = len(common)
        agreement = {
            "agreed": agreed,
            "total": total,
            "agreement_pct": round(100.0 * agreed / total, 1) if total else 0.0,
            "confusion": _confusion_table(human, judge),
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        _render_final_table(human, judge, human_counts, judge_counts, agreement),
        encoding="utf-8",
    )
    done(f"Wrote {out_path}")

    payload: dict[str, Any] = {
        "human_marks": human_counts,
        "judge_marks": judge_counts if judge else None,
        "agreement": agreement,
    }
    emit_summary(payload)


def _count_marks(values: list[str]) -> dict[str, int]:
    out = {"better": 0, "worse": 0, "neutral": 0}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return out


def _confusion_table(
    human: dict[int, str], judge: dict[int, dict[str, str]]
) -> dict[str, dict[str, int]]:
    table: dict[str, dict[str, int]] = {
        h: {j: 0 for j in ("better", "worse", "neutral")}
        for h in ("better", "worse", "neutral")
    }
    for idx, h_mark in human.items():
        if idx not in judge:
            continue
        j_mark = judge[idx]["mark"]
        table[h_mark][j_mark] += 1
    return table


def _render_final_table(
    human: dict[int, str],
    judge: dict[int, dict[str, str]],
    human_counts: dict[str, int],
    judge_counts: dict[str, int],
    agreement: dict[str, Any] | None,
) -> str:
    lines: list[str] = [
        "# Proofread spot-check — final table",
        "",
        "## Human marks (audio-grounded)",
        "",
        f"- better: {human_counts.get('better', 0)}",
        f"- worse:  {human_counts.get('worse', 0)}",
        f"- neutral: {human_counts.get('neutral', 0)}",
        "",
    ]
    if judge:
        lines.extend([
            "## Claude Code judge marks (text-only)",
            "",
            f"- better: {judge_counts.get('better', 0)}",
            f"- worse:  {judge_counts.get('worse', 0)}",
            f"- neutral: {judge_counts.get('neutral', 0)}",
            "",
        ])
        if agreement is not None:
            lines.extend([
                f"## Agreement",
                "",
                f"- Agreed on {agreement['agreed']}/{agreement['total']} segments "
                f"({agreement['agreement_pct']}%).",
                "",
                "Confusion table (rows = human, columns = judge):",
                "",
                "| human \\ judge | better | worse | neutral |",
                "|---|---|---|---|",
            ])
            conf = agreement["confusion"]
            for h in ("better", "worse", "neutral"):
                row = conf[h]
                lines.append(
                    f"| {h} | {row['better']} | {row['worse']} | {row['neutral']} |"
                )
            lines.append("")

        lines.extend([
            "## Per-segment marks",
            "",
            "| Segment | Human | Claude | Agree? | Claude rationale |",
            "|---|---|---|---|---|",
        ])
        for idx in sorted(set(human) | set(judge)):
            h = human.get(idx, "—")
            j_entry = judge.get(idx)
            j = j_entry["mark"] if j_entry else "—"
            r = (j_entry or {}).get("rationale", "")
            agree = "✅" if h == j and h != "—" and j != "—" else ("—" if "—" in (h, j) else "❌")
            lines.append(f"| {idx} | {h} | {j} | {agree} | {r} |")
        lines.append("")
        lines.extend([
            "> Disclaimer: the Claude Code judge is text-only; the human eval is",
            "> audio-grounded. Agreement here measures whether text-only LLM",
            "> judgment can be trusted for future automated measurements.",
        ])
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    c1 = sub.add_parser("classify", help="Classify segment pairs and write spot-check.md")
    c1.add_argument("--dump-dir", required=True, help="Pipeline --dump-stages directory")
    c1.add_argument("--output-dir", required=True, help="Where to write categories.json + spot-check.md")
    c1.set_defaults(func=cmd_classify)

    c2 = sub.add_parser("prepare-judge", help="Write judge-input.md for the LLM judge")
    c2.add_argument("--dump-dir", required=True, help="Pipeline --dump-stages directory")
    c2.add_argument("--output", required=True, help="Path for judge-input.md")
    c2.set_defaults(func=cmd_prepare_judge)

    c3 = sub.add_parser("parse-marks", help="Parse spot-check + judge marks into final-table.md")
    c3.add_argument("--marks-file", required=True, help="Filled spot-check.md")
    c3.add_argument("--judge-marks", default=None, help="Optional judge-marks.json")
    c3.add_argument("--output", required=True, help="Path for final-table.md")
    c3.set_defaults(func=cmd_parse_marks)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
