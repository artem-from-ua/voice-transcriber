"""`voice transcribe <audio>` entry-point."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime

from .pipeline import PipelineOptions, run


def _parse_datetime(s: str) -> datetime:
    try:
        return datetime.fromisoformat(s)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid ISO datetime: {s!r}") from exc


def _parse_names(s: str) -> list[str]:
    return [part.strip() for part in s.split(",") if part.strip()]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="voice",
        description="Local audio → diarized Markdown transcript pipeline.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("transcribe", help="Transcribe an audio file end-to-end.")
    t.add_argument("audio", help="Path to the input audio file.")
    t.add_argument(
        "--language", default=None, metavar="ISO",
        help=(
            "Conversation language hint (e.g. 'uk', 'en'). Default: detect "
            "at stage [4] lang_detect on the longest pyannote turn. Pass an "
            "explicit value to skip detection — useful when the diarized "
            "recording has only very short turns or when you already know "
            "the language and want to avoid the extra model load. The "
            "override flows through to all LLM / render stages. See ADR 0022."
        ),
    )
    t.add_argument(
        "--unknown-speaker", choices=("ask", "keep"), default="ask",
        help="When self-intro is missing: ask interactively or keep SPEAKER_XX.",
    )
    t.add_argument(
        "--names", type=_parse_names, default=None,
        metavar='"NAME1,NAME2"',
        help="Override automatic naming: comma-separated names in time order.",
    )
    t.add_argument(
        "--datetime", dest="datetime_override", type=_parse_datetime, default=None,
        help="Override recording start time (ISO 8601, e.g. 2026-05-10T15:44:02).",
    )
    t.add_argument(
        "--llm-model", default=None,
        help=(
            "Path to a local MLX model directory used as the default for all "
            "four LLM stages (default: "
            "~/.cache/lm-studio/models/mlx-community/gemma-3-12b-it-qat-4bit). "
            "Override per stage with --llm-{proofread,identify,structure,safe-speech,tldr}-model."
        ),
    )
    t.add_argument(
        "--llm-proofread-model", default=None,
        help=(
            "MLX model directory for the proofread stage. Overrides --llm-model "
            "for proofread only. The pipeline cold-reloads the model when "
            "consecutive stages request different paths."
        ),
    )
    t.add_argument(
        "--llm-identify-model", default=None,
        help="MLX model directory for the speaker-identify stage (overrides --llm-model).",
    )
    t.add_argument(
        "--llm-structure-model", default=None,
        help="MLX model directory for the section-structuring stage (overrides --llm-model).",
    )
    t.add_argument(
        "--llm-safe-speech-model", default=None,
        help="MLX model directory for the safe-speech redaction stage (overrides --llm-model).",
    )
    t.add_argument(
        "--llm-tldr-model", default=None,
        help="MLX model directory for the TL;DR stage (overrides --llm-model).",
    )
    t.add_argument(
        "--llm-temperature", type=float, default=None, metavar="T",
        help=(
            "Override sampling temperature for all LLM stages. Defaults come "
            "from each prompt's frontmatter (proofread/identify=0.1, "
            "structure=0.2, tldr=0.3). Pass the model's officially-recommended "
            "value to bench it on its preferred settings."
        ),
    )
    t.add_argument(
        "--llm-top-p", type=float, default=None, metavar="P",
        help="Override nucleus-sampling top_p for all LLM stages (default: 1.0).",
    )
    t.add_argument(
        "--llm-top-k", type=int, default=None, metavar="K",
        help="Override top_k for all LLM stages (default: 0 = disabled).",
    )
    t.add_argument(
        "--llm-repetition-penalty", type=float, default=None, metavar="R",
        help=(
            "Override repetition_penalty for plain-text LLM stages "
            "(proofread, tldr). Default: none. Some Qwen variants ship "
            "1.05 as their recommended value."
        ),
    )
    t.add_argument(
        "--output", "-o", default=None,
        help="Output Markdown path (default: <audio_basename>.md alongside the audio).",
    )
    t.add_argument("--no-proofread", action="store_true", help="Skip per-segment ASR proof-reading.")
    t.add_argument("--no-tldr", action="store_true", help="Skip TL;DR generation.")
    t.add_argument("--no-structure", action="store_true", help="Skip section structuring.")
    t.add_argument("--no-safe-speech", action="store_true", help="Skip sensitive-content redaction.")
    t.add_argument(
        "--render-min-silence-s", type=float, default=None, metavar="SECONDS",
        help=(
            "Minimum silence duration (seconds) to show as --- in the transcript. "
            "By default the threshold is computed automatically per section based on "
            "its duration (5 s for ≤60 s sections, up to 15 s for ≥5 min sections). "
            "Pass an explicit value to override for all sections."
        ),
    )
    t.add_argument(
        "--tldr-include-silence", action="store_true",
        help=(
            "Feed silence events into the TL;DR prompt (experimental). "
            "By default the TL;DR only sees speaker utterances. Enable this flag "
            "to include [пауза ...] / [muted ...] markers so the model is aware of "
            "long pauses. Use --dump-stages to compare output with and without."
        ),
    )

    ss = t.add_argument_group("safe-speech (sensitive-content redaction)")
    ss.add_argument(
        "--safe-speech-topics", default=None, metavar="TOPICS",
        help=(
            "Comma-separated list of sensitive topics to redact. "
            "Default: 'health,drugs,alcohol'. "
            "Pass '' (empty string) to disable redaction entirely. "
            "Examples: 'health,drugs,alcohol,legal,finance'."
        ),
    )
    ss.add_argument(
        "--safe-speech-policy", default="placeholder",
        choices=["placeholder", "drop"],
        help=(
            "How to handle redacted segments. "
            "'placeholder' (default): replace with [muted, X.Xs]. "
            "'drop': remove the segment silently (section headers remain)."
        ),
    )

    cs = t.add_argument_group("clearspeech (audio cleanup for ASR)")
    cs.add_argument(
        "--clearspeech-chain", type=str, default="autogain", metavar="EFFECTS",
        help=(
            "Comma-separated effect names in execution order. Known effects: "
            "autogain, bandpass, presence, denoise, dereverb. Default: 'autogain'. "
            "Use '' to disable preprocessing entirely. WARNING: enabling "
            "'denoise' degrades ASR on Ukrainian content even at its "
            "empirically best position (after autogain); see ADR 0014 for "
            "Metric A and the OpenAI Whisper-rule check. WARNING: using "
            "'presence' without 'bandpass' first catastrophically breaks ASR "
            "(see ADR 0013); enable bandpass whenever you enable presence. "
            "'dereverb' is per-turn (uses pyannote turns) — see ADR 0015 for "
            "Metric A."
        ),
    )
    cs.add_argument(
        "--clearspeech-autogain-target-dbfs", type=float, default=-20.0, metavar="DBFS",
        help="Target RMS for per-turn autogain (default: -20.0).",
    )
    cs.add_argument(
        "--clearspeech-autogain-max-gain-db", type=float, default=16.0, metavar="DB",
        help="Max boost per turn; quieter turns clipped at this gain (default: 16.0).",
    )
    cs.add_argument(
        "--clearspeech-bandpass-low-hz", type=float, default=150.0, metavar="HZ",
        help=(
            "Bandpass low cutoff in Hz (default: 150 — tuned by listening "
            "test, see ADR 0011)."
        ),
    )
    cs.add_argument(
        "--clearspeech-bandpass-high-hz", type=float, default=5_500.0, metavar="HZ",
        help=(
            "Bandpass high cutoff in Hz (default: 5500 — tuned by listening "
            "test, see ADR 0011)."
        ),
    )
    cs.add_argument(
        "--clearspeech-presence-center-hz", type=float, default=3_000.0, metavar="HZ",
        help=(
            "Presence boost center frequency in Hz (default: 3000, starting "
            "anchor — see ADR 0013 for listening-loop results)."
        ),
    )
    cs.add_argument(
        "--clearspeech-presence-boost-db", type=float, default=6.0, metavar="DB",
        help=(
            "Presence boost gain in dB (default: +6.0 — tuned by listening "
            "test, see ADR 0013)."
        ),
    )
    cs.add_argument(
        "--clearspeech-presence-q", type=float, default=1.0, metavar="Q",
        help="Presence boost Q / sharpness; higher = narrower peak (default: 1.0).",
    )
    cs.add_argument(
        "--clearspeech-denoise-noise-floor-db", type=float, default=-25.0, metavar="DB",
        help=(
            "ffmpeg afftdn noise-floor estimate in dB (default: -25, tuned by "
            "listening test, see ADR 0014)."
        ),
    )
    cs.add_argument(
        "--clearspeech-denoise-reduction-db", type=float, default=12.0, metavar="DB",
        help=(
            "ffmpeg afftdn subtraction amount in dB (default: 12)."
        ),
    )
    cs.add_argument(
        "--clearspeech-dereverb-rt60-floor-ms", type=float, default=300.0, metavar="MS",
        help=(
            "Per-turn RT60 estimate is clamped above this floor in ms (default: "
            "300). Lower lets the estimator pick shorter RT60s on already-dry "
            "segments; higher forces stronger dereverb everywhere."
        ),
    )
    cs.add_argument(
        "--clearspeech-dereverb-subtract-factor", type=float, default=1.0, metavar="ALPHA",
        help=(
            "How aggressively to subtract predicted late-reverb power "
            "(0.0 = no-op, 1.0 = full Lebart-Polack). Default: 1.0."
        ),
    )
    cs.add_argument(
        "--clearspeech-dereverb-crossfade-ms", type=float, default=50.0, metavar="MS",
        help="Crossfade duration at per-turn boundaries (default: 50).",
    )

    t.add_argument(
        "--dump-stages", dest="dump_stages_dir", default=None, metavar="DIR",
        help=(
            "Write each stage's output to DIR for troubleshooting "
            "(01-meta.json, 02-diarize_speakers.json, 02b-clear_speech-config.json, "
            "02b-clear_speech-N-<effect>.wav per applied effect — autogain, "
            "bandpass, presence — 03-asr.json, 04-merge.json, "
            "05-proofread.json, 06-identify_speakers.json, 07-segments-named.json, "
            "08-speech_structure.json, 09-speech_summary.txt). Disabled when omitted."
        ),
    )
    t.add_argument("-v", "--verbose", action="store_true", help="Verbose progress logs to stderr.")

    dl = sub.add_parser(
        "download-whisper",
        help="Download the Whisper ASR model into the HuggingFace cache.",
        description=(
            "Pre-fetch the Whisper model so `voice transcribe` can find it offline. "
            "Idempotent — re-running is a no-op once cached."
        ),
    )
    dl.add_argument(
        "--repo-id", default="mlx-community/whisper-large-v3-mlx", metavar="REPO_ID",
        help="HuggingFace repository id (default: %(default)s).",
    )
    dl.add_argument(
        "-v", "--verbose", action="store_true",
        help="Print download progress lines (default: silent).",
    )

    return p


def _parse_topics(raw: str | None) -> list[str] | None:
    """Parse --safe-speech-topics CSV into a list, or None for 'use built-in defaults'.

    Empty string → [] (disable redaction). None → None (built-in defaults).
    """
    if raw is None:
        return None
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return parts


def _opts_from_args(args: argparse.Namespace) -> PipelineOptions:
    return PipelineOptions(
        audio_path=args.audio,
        output_path=args.output,
        language=args.language,
        unknown_speaker=args.unknown_speaker,
        names_override=args.names,
        datetime_override=args.datetime_override,
        llm_model=args.llm_model,
        llm_proofread_model=args.llm_proofread_model,
        llm_identify_model=args.llm_identify_model,
        llm_structure_model=args.llm_structure_model,
        llm_safe_speech_model=args.llm_safe_speech_model,
        llm_tldr_model=args.llm_tldr_model,
        llm_temperature=args.llm_temperature,
        llm_top_p=args.llm_top_p,
        llm_top_k=args.llm_top_k,
        llm_repetition_penalty=args.llm_repetition_penalty,
        run_proofread=not args.no_proofread,
        run_safe_speech=not args.no_safe_speech,
        run_tldr=not args.no_tldr,
        run_structure=not args.no_structure,
        safe_speech_topics=_parse_topics(args.safe_speech_topics),
        safe_speech_policy=args.safe_speech_policy,
        clearspeech_chain=args.clearspeech_chain,
        clearspeech_autogain_target_dbfs=args.clearspeech_autogain_target_dbfs,
        clearspeech_autogain_max_gain_db=args.clearspeech_autogain_max_gain_db,
        clearspeech_bandpass_low_hz=args.clearspeech_bandpass_low_hz,
        clearspeech_bandpass_high_hz=args.clearspeech_bandpass_high_hz,
        clearspeech_presence_center_hz=args.clearspeech_presence_center_hz,
        clearspeech_presence_boost_db=args.clearspeech_presence_boost_db,
        clearspeech_presence_q=args.clearspeech_presence_q,
        clearspeech_denoise_noise_floor_db=args.clearspeech_denoise_noise_floor_db,
        clearspeech_denoise_reduction_db=args.clearspeech_denoise_reduction_db,
        clearspeech_dereverb_rt60_floor_ms=args.clearspeech_dereverb_rt60_floor_ms,
        clearspeech_dereverb_subtract_factor=args.clearspeech_dereverb_subtract_factor,
        clearspeech_dereverb_crossfade_ms=args.clearspeech_dereverb_crossfade_ms,
        dump_stages_dir=args.dump_stages_dir,
        verbose=args.verbose,
        render_min_silence_s=args.render_min_silence_s,
        tldr_include_silence=args.tldr_include_silence,
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.cmd == "transcribe":
        try:
            out = run(_opts_from_args(args))
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(out)
        return 0
    if args.cmd == "download-whisper":
        from . import download_whisper as download_whisper_module

        log = (lambda msg: print(msg, file=sys.stderr, flush=True)) if args.verbose else print
        try:
            download_whisper_module.download(args.repo_id, log=log)
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
