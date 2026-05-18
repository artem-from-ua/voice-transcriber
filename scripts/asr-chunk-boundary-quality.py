#!/usr/bin/env python3
"""Measure word-level quality at chunked-ASR boundaries.

Issue #156 — companion to PR #149 (ADR 0031). The chunked ASR path in
`src/voice/whisper_asr.py` slices long audio into 8-min chunks with a
5 s overlap and stitches `_dedup_overlap`-trimmed segments back together.
The dedup is structural (drop chunk N+1 segments with `start < cutoff`)
and never looks at text. This script asks: how often does that cut a
word in half, lose one, or duplicate one?

Three subcommands form a reusable measurement pipeline (#159 and #168
will call the same script on snap-modified dumps to produce comparable
numbers):

    extract       --dump-dir <D> --out <O>
                  -> O/boundaries.json with per-boundary text + RMS
                     features. No LLM, pure text + numpy on the WAV.

    judge-prompt  --boundaries O/boundaries.json --out O/judge-input.md
                  -> human-readable per-boundary windows ready for a
                     text-only judge. In this project the judge is
                     Claude reading the file in-session; the marks land
                     in O/judge-marks.json (hand-written, same schema).

    parse-marks   --boundaries O/boundaries.json --marks O/judge-marks.json
                  --out O/summary.md
                  -> final categorised table + `---SUMMARY-START---`
                     JSON block for machine parsing.

Output is plain text with no ANSI / `\\r` / emoji — friendly to agentic
shells. The script does not modify production code or run any inference.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


# Mirror the constants from src/voice/whisper_asr.py so the script can
# rebuild the cutoffs without importing the heavy module (mlx_whisper
# pulls in numba/tiktoken on import). If those change, update here.
ASR_CHUNK_SIZE_S = 480.0
ASR_CHUNK_OVERLAP_S = 5.0
SAMPLE_RATE = 16000

VERDICTS = ("clean", "missing", "duplicated", "truncated")

WINDOW_SEGMENTS = 5      # tail/head segments shown around each cutoff
RMS_HALFWIDTH_S = 0.20   # +/- 200 ms RMS window centred on cutoff


# ---------------------------------------------------------------------------
# Logging — no ANSI, no `\r`, easy to read in an agentic shell.
# ---------------------------------------------------------------------------


def info(msg: str) -> None:
    print(f"[info]  {msg}", flush=True)


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
# Pure helpers — tokenisation and word-overlap. Unit-tested below.
# ---------------------------------------------------------------------------


_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яҐґЄєІіЇї0-9']+")


def tokenize(text: str) -> list[str]:
    """Lowercase tokens, alphanumerics only. Cyrillic + Latin + digits."""
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def jaccard(a: list[str], b: list[str]) -> float:
    """Set-Jaccard on token bags. 0.0 on empty input."""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def looks_truncated(token: str) -> bool:
    """Heuristic: very short token (1-2 chars) or token ending/starting
    with a hyphen are likely word-fragments. Imperfect, intentionally
    biased toward false positives — the judge filters them."""
    if not token:
        return True
    if token.startswith("-") or token.endswith("-"):
        return True
    return len(token) <= 2 and token not in {
        # short but legitimate Ukrainian/English words
        "і", "у", "в", "з", "до", "на", "не", "це", "та", "як",
        "i", "a", "is", "it", "of", "to", "in", "on", "no", "ok", "um",
        "so", "we", "he", "be", "by", "an", "or", "as", "at", "if", "do",
    }


# ---------------------------------------------------------------------------
# Boundary geometry
# ---------------------------------------------------------------------------


@dataclass
class Segment:
    start: float
    end: float
    content: str


def load_segments(asr_json: Path) -> list[Segment]:
    raw = json.loads(asr_json.read_text(encoding="utf-8"))
    segs = [
        Segment(start=float(s["start"]), end=float(s["end"]), content=str(s.get("content", "")))
        for s in raw
    ]
    segs.sort(key=lambda s: s.start)
    return segs


def compute_boundaries(
    total_duration_s: float,
    *,
    chunk_size_s: float = ASR_CHUNK_SIZE_S,
    overlap_s: float = ASR_CHUNK_OVERLAP_S,
) -> list[float]:
    """Return cutoff timestamps (seconds) at which `_dedup_overlap`
    trims chunk N+1. Mirrors the geometry in `whisper_asr.transcribe`.

    For chunk_size=480 and overlap=5 on a 2910 s recording the
    cutoffs are: [480, 955, 1430, 1905, 2380, 2855].
    """
    chunk_samples = int(chunk_size_s * SAMPLE_RATE)
    overlap_samples = int(overlap_s * SAMPLE_RATE)
    step_samples = chunk_samples - overlap_samples
    total_samples = int(total_duration_s * SAMPLE_RATE)
    n_chunks = (total_samples + step_samples - 1) // step_samples
    cutoffs: list[float] = []
    for i in range(1, n_chunks):  # skip chunk 0 (no preceding chunk)
        chunk_start_s = (i * step_samples) / SAMPLE_RATE
        cutoffs.append(chunk_start_s + overlap_s)
    return cutoffs


def split_around(segments: list[Segment], cutoff: float) -> tuple[list[Segment], list[Segment]]:
    """Split segments by start < cutoff (kept from chunk N) vs >= cutoff
    (kept from chunk N+1 after dedup). Mirrors `_dedup_overlap` rule."""
    tail = [s for s in segments if s.start < cutoff]
    head = [s for s in segments if s.start >= cutoff]
    return tail, head


def window(segments: list[Segment], cutoff: float, *, k: int = WINDOW_SEGMENTS) -> tuple[list[Segment], list[Segment]]:
    """Last k segments before cutoff and first k after."""
    tail_all, head_all = split_around(segments, cutoff)
    return tail_all[-k:], head_all[:k]


# ---------------------------------------------------------------------------
# RMS — load WAV with numpy via mlx_whisper.audio (already a project dep).
# ---------------------------------------------------------------------------


def rms_around(wav_path: Path, cutoff_s: float, halfwidth_s: float = RMS_HALFWIDTH_S) -> tuple[float, float]:
    """Return (rms_before, rms_after) for a +/- halfwidth_s window split
    at cutoff_s. Loads the WAV lazily; mlx_whisper.audio.load_audio
    returns float32 mono at 16 kHz."""
    from mlx_whisper.audio import load_audio

    audio = load_audio(str(wav_path))
    sr = SAMPLE_RATE
    centre = int(cutoff_s * sr)
    width = int(halfwidth_s * sr)
    before = audio[max(0, centre - width):centre]
    after = audio[centre:centre + width]

    def _rms(arr: Any) -> float:
        if len(arr) == 0:
            return 0.0
        return float(math.sqrt(float((arr * arr).mean())))

    return _rms(before), _rms(after)


# ---------------------------------------------------------------------------
# extract subcommand
# ---------------------------------------------------------------------------


def cmd_extract(args: argparse.Namespace) -> None:
    dump_dir = Path(args.dump_dir).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    asr_path = dump_dir / "03-asr.json"
    if not asr_path.exists():
        die(f"missing {asr_path}")
    wav_candidates = sorted(dump_dir.glob("02b-clear_speech-*-autogain.wav"))
    wav_path = wav_candidates[-1] if wav_candidates else None
    if wav_path is None:
        warn(f"no 02b-clear_speech-*-autogain.wav under {dump_dir}; RMS features will be 0")

    segments = load_segments(asr_path)
    if not segments:
        die("03-asr.json has no segments")

    total_duration_s = max(s.end for s in segments)
    cutoffs = compute_boundaries(
        total_duration_s,
        chunk_size_s=args.chunk_size_s,
        overlap_s=args.overlap_s,
    )
    info(
        f"{len(segments)} segments, {total_duration_s:.1f}s total -> "
        f"{len(cutoffs) + 1} chunks, {len(cutoffs)} boundaries at "
        f"{[round(c, 1) for c in cutoffs]}"
    )

    records: list[dict[str, Any]] = []
    for idx, cutoff in enumerate(cutoffs):
        tail, head = window(segments, cutoff)
        tail_text = " ".join(s.content.strip() for s in tail)
        head_text = " ".join(s.content.strip() for s in head)
        tail_tokens = tokenize(tail_text)
        head_tokens = tokenize(head_text)
        last_token = tail_tokens[-1] if tail_tokens else ""
        first_token = head_tokens[0] if head_tokens else ""
        gap_s = (head[0].start - tail[-1].end) if (tail and head) else float("nan")

        if wav_path is not None:
            rms_before, rms_after = rms_around(wav_path, cutoff)
        else:
            rms_before = rms_after = 0.0

        records.append({
            "boundary_idx": idx,
            "cutoff_s": cutoff,
            "tail_segments": [asdict(s) for s in tail],
            "head_segments": [asdict(s) for s in head],
            "tail_text": tail_text,
            "head_text": head_text,
            "last_token": last_token,
            "first_token": first_token,
            "last_token_truncated_heuristic": looks_truncated(last_token),
            "first_token_truncated_heuristic": looks_truncated(first_token),
            "gap_s": gap_s,
            "word_overlap_jaccard": jaccard(tail_tokens, head_tokens),
            "rms_before": rms_before,
            "rms_after": rms_after,
        })

    out_path = out_dir / "boundaries.json"
    out_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    done(f"wrote {out_path}")
    emit_summary({
        "subcommand": "extract",
        "n_boundaries": len(records),
        "total_duration_s": total_duration_s,
        "cutoffs_s": cutoffs,
        "chunk_size_s": args.chunk_size_s,
        "overlap_s": args.overlap_s,
    })


def _json_default(o: Any) -> Any:
    if isinstance(o, float) and math.isnan(o):
        return None
    raise TypeError(f"cannot serialise {type(o).__name__}")


# ---------------------------------------------------------------------------
# judge-prompt subcommand
# ---------------------------------------------------------------------------


_JUDGE_PREAMBLE = """\
# ASR chunk-boundary judge input

For each boundary below, decide **one** verdict from:

- `clean` — head_text logically continues tail_text without repetition;
  no torn word; gap_s small or matches a natural pause.
- `missing` — sense breaks across the cutoff (a word or short phrase
  appears lost); often correlated with a moderate gap_s and high
  rms_around_cutoff (audio was loud but no token landed).
- `duplicated` — the same phrase appears at the end of tail_text and at
  the start of head_text; word_overlap_jaccard is usually noticeably
  above the baseline of other boundaries.
- `truncated` — last_token or first_token is a word fragment ("Cer" +
  "nat"), or punctuation/spacing leaves a half-word stranded.

After deciding, write `judge-marks.json` next to this file with one
entry per boundary:

    [
      {"boundary_idx": 0, "verdict": "clean", "note": "..."},
      ...
    ]

`note` is free text — short rationale, especially for non-`clean`
verdicts. Even when the verdict is `clean`, calling out a small risk
("gap 3.96 s — looks like natural pause, but RMS_after is high") helps
the next iteration of this measurement.

---
"""


def _render_boundary_block(rec: dict[str, Any]) -> str:
    cutoff = float(rec["cutoff_s"])
    lines: list[str] = [
        f"## Boundary {rec['boundary_idx']} @ cutoff={_fmt_ts(cutoff)} ({cutoff:.2f}s)",
        "",
        f"- gap_s: {rec['gap_s']}",
        f"- word_overlap_jaccard: {rec['word_overlap_jaccard']:.3f}",
        f"- last_token: `{rec['last_token']}` "
        f"(truncated_heuristic={rec['last_token_truncated_heuristic']})",
        f"- first_token: `{rec['first_token']}` "
        f"(truncated_heuristic={rec['first_token_truncated_heuristic']})",
        f"- rms_before: {rec['rms_before']:.5f}, rms_after: {rec['rms_after']:.5f}",
        "",
        "### tail (chunk N, last segments before cutoff)",
        "",
    ]
    for s in rec["tail_segments"]:
        lines.append(f"- `{_fmt_ts(s['start'])} - {_fmt_ts(s['end'])}` {s['content'].strip()}")
    lines.append("")
    lines.append("### head (chunk N+1, first segments at/after cutoff)")
    lines.append("")
    for s in rec["head_segments"]:
        lines.append(f"- `{_fmt_ts(s['start'])} - {_fmt_ts(s['end'])}` {s['content'].strip()}")
    lines.append("")
    return "\n".join(lines)


def cmd_judge_prompt(args: argparse.Namespace) -> None:
    boundaries = json.loads(Path(args.boundaries).read_text(encoding="utf-8"))
    blocks = [_render_boundary_block(rec) for rec in boundaries]
    body = _JUDGE_PREAMBLE + "\n".join(blocks)
    out_path = Path(args.out).expanduser().resolve()
    out_path.write_text(body, encoding="utf-8")
    done(f"wrote {out_path}")
    emit_summary({
        "subcommand": "judge-prompt",
        "n_boundaries": len(boundaries),
        "out": str(out_path),
    })


# ---------------------------------------------------------------------------
# parse-marks subcommand
# ---------------------------------------------------------------------------


MATERIAL_RATE_THRESHOLD = 0.10  # >= 10% missing/duplicated/truncated => material


def cmd_parse_marks(args: argparse.Namespace) -> None:
    boundaries = json.loads(Path(args.boundaries).read_text(encoding="utf-8"))
    marks = json.loads(Path(args.marks).read_text(encoding="utf-8"))
    summary = aggregate(boundaries, marks)

    out_path = Path(args.out).expanduser().resolve()
    out_path.write_text(_render_summary_md(summary), encoding="utf-8")
    done(f"wrote {out_path}")
    emit_summary(summary["summary_block"])


def aggregate(boundaries: list[dict[str, Any]], marks: list[dict[str, Any]]) -> dict[str, Any]:
    """Pure function: combine boundaries + marks into a counted summary.

    Raises ValueError on mismatch — every boundary must have exactly one
    mark, with a valid verdict from VERDICTS.
    """
    marks_by_idx = {int(m["boundary_idx"]): m for m in marks}
    boundary_indices = {int(b["boundary_idx"]) for b in boundaries}
    if set(marks_by_idx) != boundary_indices:
        missing = boundary_indices - set(marks_by_idx)
        extra = set(marks_by_idx) - boundary_indices
        raise ValueError(
            f"mark/boundary mismatch: missing marks for {sorted(missing)}, "
            f"extra marks for {sorted(extra)}"
        )

    counts = {v: 0 for v in VERDICTS}
    rows: list[dict[str, Any]] = []
    for b in boundaries:
        idx = int(b["boundary_idx"])
        m = marks_by_idx[idx]
        verdict = m["verdict"]
        if verdict not in VERDICTS:
            raise ValueError(f"boundary {idx}: invalid verdict {verdict!r}")
        counts[verdict] += 1
        rows.append({
            "boundary_idx": idx,
            "cutoff_s": float(b["cutoff_s"]),
            "verdict": verdict,
            "note": m.get("note", ""),
            "gap_s": b.get("gap_s"),
            "word_overlap_jaccard": b.get("word_overlap_jaccard"),
            "rms_before": b.get("rms_before"),
            "rms_after": b.get("rms_after"),
        })

    total = len(rows)
    material = counts["missing"] + counts["duplicated"] + counts["truncated"]
    material_rate = (material / total) if total else 0.0
    summary_block = {
        "subcommand": "parse-marks",
        "n_boundaries": total,
        "counts": counts,
        "material_issues": material,
        "material_rate": material_rate,
        "material_rate_threshold": MATERIAL_RATE_THRESHOLD,
        "decision": "material" if material_rate >= MATERIAL_RATE_THRESHOLD else "baseline-acceptable",
    }
    return {"rows": rows, "summary_block": summary_block}


def _render_summary_md(summary: dict[str, Any]) -> str:
    sb = summary["summary_block"]
    rows = summary["rows"]
    lines: list[str] = [
        "# ASR chunk-boundary quality — summary",
        "",
        f"- boundaries judged: {sb['n_boundaries']}",
        f"- counts: {sb['counts']}",
        f"- material issues (missing+duplicated+truncated): {sb['material_issues']}",
        f"- material rate: {sb['material_rate']:.3f} "
        f"(threshold {sb['material_rate_threshold']:.2f})",
        f"- decision: **{sb['decision']}**",
        "",
        "| # | cutoff | verdict | gap_s | jaccard | rms_before | rms_after | note |",
        "|---|--------|---------|-------|---------|------------|-----------|------|",
    ]
    for r in rows:
        gap = "" if r["gap_s"] is None else f"{r['gap_s']:.2f}"
        jac = "" if r["word_overlap_jaccard"] is None else f"{r['word_overlap_jaccard']:.3f}"
        rb = "" if r["rms_before"] is None else f"{r['rms_before']:.4f}"
        ra = "" if r["rms_after"] is None else f"{r['rms_after']:.4f}"
        note = (r["note"] or "").replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {r['boundary_idx']} | {_fmt_ts(r['cutoff_s'])} | {r['verdict']} | "
            f"{gap} | {jac} | {rb} | {ra} | {note} |"
        )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("extract", help="extract boundary features from a dump dir")
    pe.add_argument("--dump-dir", required=True)
    pe.add_argument("--out", required=True)
    pe.add_argument("--chunk-size-s", type=float, default=ASR_CHUNK_SIZE_S)
    pe.add_argument("--overlap-s", type=float, default=ASR_CHUNK_OVERLAP_S)
    pe.set_defaults(func=cmd_extract)

    pj = sub.add_parser("judge-prompt", help="render boundaries as a judge-readable Markdown")
    pj.add_argument("--boundaries", required=True)
    pj.add_argument("--out", required=True)
    pj.set_defaults(func=cmd_judge_prompt)

    pp = sub.add_parser("parse-marks", help="combine boundaries + judge marks into summary")
    pp.add_argument("--boundaries", required=True)
    pp.add_argument("--marks", required=True)
    pp.add_argument("--out", required=True)
    pp.set_defaults(func=cmd_parse_marks)

    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
