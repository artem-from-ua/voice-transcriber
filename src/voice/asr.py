"""VibeVoice-ASR wrapper.

Resolves the model directory by bitness (4/5/6/8), invokes mlx-audio's
generate_transcription, and parses the embedded JSON timeline returned in
`result.text`. The pipeline only ever consumes `AsrSegment` instances.

Default chunk and repetition-penalty settings come from the session where
those values were tuned against real Ukrainian dialogue audio.
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Callable

from mlx_audio.stt.generate import generate_transcription

from .types import AsrSegment


VIBEVOICE_REPOS: dict[int, str] = {
    4: "mlx-community/VibeVoice-ASR-4bit",
    5: "mlx-community/VibeVoice-ASR-5bit",
    6: "mlx-community/VibeVoice-ASR-6bit",
    8: "mlx-community/VibeVoice-ASR-8bit",
}

LM_STUDIO_MODELS_DIR = Path("~/.cache/lm-studio/models").expanduser()

DEFAULT_GEN_KWARGS: dict[str, object] = {
    "repetition_penalty": 1.2,
    "repetition_context_size": 64,
    "temperature": 0.1,
}
DEFAULT_CHUNK_DURATION = 15.0


class AsrError(RuntimeError):
    pass


def resolve_model_path(bitness: int) -> str:
    """Return the local model path for a given bitness, or raise if missing."""
    try:
        repo = VIBEVOICE_REPOS[bitness]
    except KeyError as exc:
        raise AsrError(f"Unsupported VibeVoice bitness {bitness!r}; pick one of {sorted(VIBEVOICE_REPOS)}") from exc
    local = LM_STUDIO_MODELS_DIR / repo
    if not (local / "config.json").is_file():
        raise AsrError(
            f"VibeVoice-ASR-{bitness}bit not found at {local}. "
            f"Open LM Studio → Models → download '{repo}'."
        )
    return str(local)


def _parse_text_payload(raw: str) -> list[dict]:
    """VibeVoice returns the timeline as a JSON-array string inside `text`.

    Sometimes generation truncates mid-segment; recover by trimming to the
    last closing brace and re-parsing.
    """
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        last = raw.rfind("}")
        if last == -1:
            raise
        return json.loads(raw[: last + 1] + "]")


def transcribe(
    wav_path: str | Path,
    *,
    bitness: int = 6,
    language: str = "uk",
    context: str | None = None,
    log: Callable[[str], None] = print,
) -> list[AsrSegment]:
    """Run VibeVoice-ASR on a WAV. Returns a list of AsrSegment."""
    model_path = resolve_model_path(bitness)
    log(f"VibeVoice-ASR-{bitness}bit at {model_path}")

    # generate_transcription writes its output to a file even when we want it
    # in memory; route it to a temp path and discard.
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        out_path = tmp.name
    try:
        t0 = time.time()
        result = generate_transcription(
            model=model_path,
            audio=str(wav_path),
            output_path=out_path.removesuffix(".json"),  # generate appends extension
            format="json",
            language=language,
            chunk_duration=DEFAULT_CHUNK_DURATION,
            context=context,
            gen_kwargs=DEFAULT_GEN_KWARGS,
            verbose=False,
        )
        log(f"ASR finished in {time.time() - t0:.1f}s.")
    finally:
        for candidate in (out_path, out_path + ".json"):
            try:
                Path(candidate).unlink(missing_ok=True)
            except OSError:
                pass

    raw_text = getattr(result, "text", "") or ""
    raw_segments = _parse_text_payload(raw_text)

    segments: list[AsrSegment] = []
    for seg in raw_segments:
        content = (seg.get("Content") or "").strip()
        if not content:
            continue
        segments.append(AsrSegment(
            start=float(seg.get("Start", 0.0)),
            end=float(seg.get("End", 0.0)),
            content=content,
            speaker_asr=seg.get("Speaker"),
        ))
    return segments
