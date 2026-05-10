"""Extract recording start/end datetime + duration for an audio file.

Strategy (in order):
1. ffprobe `format.tags.creation_time` (iOS m4a, modern recorders set this)
2. `stat().st_birthtime` on macOS (file creation time)
3. `stat().st_mtime` (last modification)
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .types import AudioMeta


class FfprobeError(RuntimeError):
    pass


def _run_ffprobe(path: str | os.PathLike[str]) -> dict:
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        str(path),
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError as exc:
        raise FfprobeError("ffprobe not found in PATH. Install ffmpeg.") from exc
    except subprocess.CalledProcessError as exc:
        raise FfprobeError(f"ffprobe failed for {path}: {exc.stderr.strip()}") from exc
    return json.loads(res.stdout)


def get_duration_seconds(path: str | os.PathLike[str]) -> float:
    data = _run_ffprobe(path)
    try:
        return float(data["format"]["duration"])
    except (KeyError, ValueError) as exc:
        raise FfprobeError(f"Cannot read duration from {path}") from exc


def extract_metadata(
    path: str | os.PathLike[str],
    *,
    override_started_at: datetime | None = None,
) -> AudioMeta:
    """Return AudioMeta with started_at/ended_at as ISO-8601 strings (local TZ)."""
    p = Path(path)
    data = _run_ffprobe(p)
    fmt = data.get("format", {})
    duration = float(fmt.get("duration", 0.0))

    if override_started_at is not None:
        started = override_started_at
        source = "cli override"
    else:
        started, source = _resolve_start(p, fmt)

    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc).astimezone()
    ended = started.fromtimestamp(started.timestamp() + duration, tz=started.tzinfo)

    return AudioMeta(
        path=str(p),
        started_at=started.isoformat(timespec="seconds"),
        ended_at=ended.isoformat(timespec="seconds"),
        duration_s=duration,
        source=source,
    )


def _resolve_start(path: Path, fmt: dict) -> tuple[datetime, str]:
    tags = fmt.get("tags") or {}
    raw = tags.get("creation_time")
    if raw:
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            return dt, "ffprobe creation_time"
        except ValueError:
            pass

    st = path.stat()
    btime = getattr(st, "st_birthtime", None)
    if btime:
        return datetime.fromtimestamp(btime, tz=timezone.utc), "stat birthtime"
    return datetime.fromtimestamp(st.st_mtime, tz=timezone.utc), "stat mtime"
