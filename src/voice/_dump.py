"""Optional stage-artefact dumping for troubleshooting.

When `--dump-stages DIR` is passed to the CLI, the pipeline writes one
artefact per stage so the user can diff what each step produced. The dump
is purely diagnostic — disabling it changes nothing in the rendered output.

File layout:
    01-meta.json                  AudioMeta from ffprobe
    02-diarize.json               list[DiarTurn] (pyannote, run on the raw WAV)
    02b-clearspeech-config.json   chain + params + per-step stats
    02b-clearspeech-N-<eff>.wav   one WAV per applied effect, N = position in
                                  the active chain (e.g. 02b-clearspeech-1-autogain.wav,
                                  02b-clearspeech-2-bandpass.wav). Absent when
                                  the chain is empty / fully disabled.
    03-asr.json                   list[AsrSegment] (Whisper on cleaned WAV)
    04-merge.json                 list[Segment] after merge (no LLM touches yet)
    05-proofread.json             list[Segment] after the LLM proof-reader
    06-identify.json              {pyannote label → human name}
    07-segments-named.json        list[Segment] with .name filled in
    08-structure.json             StructuredDialog (sections + segments)
    09-tldr.txt                   raw Markdown TL;DR string
"""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path
from typing import Any


class StageDumper:
    """Write one artefact per stage into a target directory.

    If `target` is None the dumper is a no-op — call sites use a single
    `dumper.write(name, obj)` or `dumper.write_binary(name, src)` line
    without branching on `enabled()`.
    """

    def __init__(self, target: Path | None) -> None:
        self.target = target
        if target is not None:
            target.mkdir(parents=True, exist_ok=True)

    def enabled(self) -> bool:
        return self.target is not None

    def write(self, name: str, obj: Any) -> None:
        if self.target is None:
            return
        path = self.target / name
        if name.endswith(".json"):
            path.write_text(
                json.dumps(_to_json_safe(obj), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        else:
            path.write_text(str(obj), encoding="utf-8")

    def write_binary(self, name: str, src: str | Path | bytes) -> None:
        """Copy a binary artefact (file path or raw bytes) into the dump dir."""
        if self.target is None:
            return
        path = self.target / name
        if isinstance(src, (bytes, bytearray)):
            path.write_bytes(bytes(src))
        else:
            shutil.copyfile(src, path)


def _to_json_safe(obj: Any) -> Any:
    """Convert dataclasses and lists thereof into JSON-friendly primitives."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _to_json_safe(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(x) for x in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)
