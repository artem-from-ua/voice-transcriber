"""Offline tests for the stage-artefact dumper."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from voice._dump import StageDumper
from voice.types import AsrSegment, AudioMeta, Section, Segment, StructuredDialog


def test_disabled_dumper_is_a_no_op(tmp_path: Path):
    d = StageDumper(target=None)
    assert d.enabled() is False
    d.write("01-meta.json", {"anything": "goes"})
    assert list(tmp_path.iterdir()) == []


def test_dump_creates_directory(tmp_path: Path):
    target = tmp_path / "new" / "nested" / "dir"
    StageDumper(target=target)
    assert target.is_dir()


def test_dump_dataclass_as_json(tmp_path: Path):
    d = StageDumper(target=tmp_path)
    meta = AudioMeta(
        path="/tmp/foo.m4a",
        started_at="2026-05-10T15:44:02+00:00",
        ended_at="2026-05-10T15:50:16+00:00",
        duration_s=374.0,
        source="ffprobe creation_time",
    )
    d.write("01-meta.json", meta)
    out = json.loads((tmp_path / "01-meta.json").read_text(encoding="utf-8"))
    assert out["path"] == "/tmp/foo.m4a"
    assert out["duration_s"] == 374.0
    assert out["source"] == "ffprobe creation_time"


def test_dump_list_of_dataclasses(tmp_path: Path):
    d = StageDumper(target=tmp_path)
    segs = [
        AsrSegment(start=0.0, end=1.5, content="hi"),
        AsrSegment(start=1.5, end=3.0, content="bye"),
    ]
    d.write("02-asr.json", segs)
    out = json.loads((tmp_path / "02-asr.json").read_text(encoding="utf-8"))
    assert isinstance(out, list)
    assert len(out) == 2
    assert out[0]["content"] == "hi"
    assert out[1]["end"] == 3.0


def test_dump_dict_with_non_string_keys(tmp_path: Path):
    d = StageDumper(target=tmp_path)
    name_map = {"SPEAKER_00": "Артем", "SPEAKER_01": "Остап"}
    d.write("06-identify_speakers.json", name_map)
    out = json.loads((tmp_path / "06-identify_speakers.json").read_text(encoding="utf-8"))
    assert out == {"SPEAKER_00": "Артем", "SPEAKER_01": "Остап"}


def test_dump_nested_structured_dialog(tmp_path: Path):
    d = StageDumper(target=tmp_path)
    segs = [Segment(start=0, end=1, content="x", speaker="A", name="Sam")]
    sections = [Section(title="T", start_ms=0, end_ms=1000)]
    dialog = StructuredDialog(sections=sections, segments=segs)
    d.write("08-speech_structure.json", dialog)
    out = json.loads((tmp_path / "08-speech_structure.json").read_text(encoding="utf-8"))
    assert out["sections"][0]["title"] == "T"
    assert out["segments"][0]["name"] == "Sam"


def test_dump_plain_text_for_non_json_extension(tmp_path: Path):
    d = StageDumper(target=tmp_path)
    d.write("09-speech_summary.txt", "## Summary\n- one\n- two")
    text = (tmp_path / "09-speech_summary.txt").read_text(encoding="utf-8")
    assert text.startswith("## Summary")


@pytest.mark.parametrize("text", ["", "  \n  "])
def test_dump_empty_string_still_writes(tmp_path: Path, text: str):
    d = StageDumper(target=tmp_path)
    d.write("09-speech_summary.txt", text)
    assert (tmp_path / "09-speech_summary.txt").exists()


def test_dump_binary_copies_file(tmp_path: Path):
    target = tmp_path / "out"
    d = StageDumper(target=target)
    src = tmp_path / "in.wav"
    src.write_bytes(b"RIFF\x00\x01\x02fake-wav-bytes")
    d.write_binary("02b-normalized.wav", src)
    assert (target / "02b-normalized.wav").read_bytes() == src.read_bytes()


def test_dump_binary_writes_raw_bytes(tmp_path: Path):
    d = StageDumper(target=tmp_path)
    d.write_binary("02b-normalized.wav", b"\x00\x01\x02")
    assert (tmp_path / "02b-normalized.wav").read_bytes() == b"\x00\x01\x02"


def test_dump_binary_disabled_is_noop(tmp_path: Path):
    d = StageDumper(target=None)
    src = tmp_path / "x.wav"
    src.write_bytes(b"abc")
    d.write_binary("02b-normalized.wav", src)  # must not raise
