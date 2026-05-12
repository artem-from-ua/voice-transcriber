"""Convert any audio file to 16 kHz mono PCM WAV via ffmpeg.

Stage [1] of the pipeline. The downstream diarize / clearspeech / ASR
stages all consume the WAV produced here, so the sample rate and channel
count are fixed (they match Whisper-large-v3's expected input).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable


def transcode(src: Path, dst: Path, log: Callable[[str], None]) -> None:
    """Convert any audio to 16 kHz mono PCM WAV via ffmpeg."""
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-i", str(src),
        "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(dst),
    ]
    log(f"[1/11] transcode → WAV 16 kHz mono")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg not found in PATH. Install ffmpeg.") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg failed: {exc.stderr.strip()}") from exc
