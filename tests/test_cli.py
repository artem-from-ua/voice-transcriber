"""Offline tests for CLI parsing (no live pipeline execution)."""

from __future__ import annotations

from datetime import datetime

import pytest

from voice.cli import _build_parser, _opts_from_args


def _parse(argv: list[str]):
    return _build_parser().parse_args(argv)


def test_minimal_invocation_defaults():
    ns = _parse(["transcribe", "/tmp/a.m4a"])
    opts = _opts_from_args(ns)
    assert opts.audio_path == "/tmp/a.m4a"
    assert opts.language == "uk"
    assert opts.asr_bits == 6
    assert opts.unknown_speaker == "ask"
    assert opts.names_override is None
    assert opts.datetime_override is None
    assert opts.run_postprocess is True
    assert opts.run_tldr is True
    assert opts.run_structure is True
    assert opts.verbose is False


def test_names_parsed_into_list():
    ns = _parse(["transcribe", "/tmp/a.m4a", "--names", "Артем,Остап,Богдан"])
    assert _opts_from_args(ns).names_override == ["Артем", "Остап", "Богдан"]


def test_datetime_override_parsed():
    ns = _parse(["transcribe", "/tmp/a.m4a", "--datetime", "2026-05-10T15:44:02"])
    assert _opts_from_args(ns).datetime_override == datetime(2026, 5, 10, 15, 44, 2)


def test_no_flags_disable_stages():
    ns = _parse([
        "transcribe", "/tmp/a.m4a",
        "--no-postprocess", "--no-tldr", "--no-structure",
    ])
    opts = _opts_from_args(ns)
    assert opts.run_postprocess is False
    assert opts.run_tldr is False
    assert opts.run_structure is False


def test_asr_bits_must_be_in_choices():
    with pytest.raises(SystemExit):
        _parse(["transcribe", "/tmp/a.m4a", "--asr-bits", "7"])


def test_unknown_speaker_choices():
    ns = _parse(["transcribe", "/tmp/a.m4a", "--unknown-speaker", "keep"])
    assert _opts_from_args(ns).unknown_speaker == "keep"
    with pytest.raises(SystemExit):
        _parse(["transcribe", "/tmp/a.m4a", "--unknown-speaker", "wat"])
