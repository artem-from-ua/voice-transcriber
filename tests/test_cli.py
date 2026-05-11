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
    # New ASR tuning knobs default to None — meaning "use module defaults
    # (chunk_duration=45, temperature=0.0)".
    assert opts.asr_chunk_duration is None
    assert opts.asr_temperature is None
    assert opts.unknown_speaker == "ask"
    assert opts.names_override is None
    assert opts.datetime_override is None
    assert opts.run_proofread is True
    assert opts.run_tldr is True
    assert opts.run_structure is True
    assert opts.clearspeech_agc is True
    assert opts.clearspeech_agc_target_dbfs == -20.0
    assert opts.clearspeech_agc_max_gain_db == 16.0
    assert opts.clearspeech_bandpass is False
    assert opts.clearspeech_bandpass_low_hz == 80.0
    assert opts.clearspeech_bandpass_high_hz == 7_900.0
    assert opts.verbose is False


def test_asr_tuning_knobs_parsed():
    ns = _parse([
        "transcribe", "/tmp/a.m4a",
        "--asr-chunk-duration", "60",
        "--asr-temperature", "0.2",
    ])
    opts = _opts_from_args(ns)
    assert opts.asr_chunk_duration == 60.0
    assert opts.asr_temperature == 0.2


def test_dump_stages_flag_parsed():
    ns = _parse(["transcribe", "/tmp/a.m4a", "--dump-stages", "/tmp/dump-dir"])
    assert _opts_from_args(ns).dump_stages_dir == "/tmp/dump-dir"


def test_dump_stages_defaults_to_none():
    ns = _parse(["transcribe", "/tmp/a.m4a"])
    assert _opts_from_args(ns).dump_stages_dir is None


def test_names_parsed_into_list():
    ns = _parse(["transcribe", "/tmp/a.m4a", "--names", "Артем,Остап,Богдан"])
    assert _opts_from_args(ns).names_override == ["Артем", "Остап", "Богдан"]


def test_datetime_override_parsed():
    ns = _parse(["transcribe", "/tmp/a.m4a", "--datetime", "2026-05-10T15:44:02"])
    assert _opts_from_args(ns).datetime_override == datetime(2026, 5, 10, 15, 44, 2)


def test_no_flags_disable_stages():
    ns = _parse([
        "transcribe", "/tmp/a.m4a",
        "--no-proofread", "--no-tldr", "--no-structure",
    ])
    opts = _opts_from_args(ns)
    assert opts.run_proofread is False
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


def test_no_clearspeech_agc_flag():
    ns = _parse(["transcribe", "/tmp/a.m4a", "--no-clearspeech-agc"])
    assert _opts_from_args(ns).clearspeech_agc is False


def test_clearspeech_agc_tuning_flags():
    ns = _parse([
        "transcribe", "/tmp/a.m4a",
        "--clearspeech-agc-target-dbfs", "-18",
        "--clearspeech-agc-max-gain-db", "15",
    ])
    opts = _opts_from_args(ns)
    assert opts.clearspeech_agc_target_dbfs == -18.0
    assert opts.clearspeech_agc_max_gain_db == 15.0


def test_clearspeech_bandpass_flags():
    ns = _parse([
        "transcribe", "/tmp/a.m4a",
        "--clearspeech-bandpass",
        "--clearspeech-bandpass-low-hz", "120",
        "--clearspeech-bandpass-high-hz", "7000",
    ])
    opts = _opts_from_args(ns)
    assert opts.clearspeech_bandpass is True
    assert opts.clearspeech_bandpass_low_hz == 120.0
    assert opts.clearspeech_bandpass_high_hz == 7_000.0
