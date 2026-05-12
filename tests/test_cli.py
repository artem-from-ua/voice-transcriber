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
    assert opts.unknown_speaker == "ask"
    assert opts.names_override is None
    assert opts.datetime_override is None
    assert opts.run_proofread is True
    assert opts.run_tldr is True
    assert opts.run_structure is True
    assert opts.clearspeech_chain == "autogain"
    assert opts.clearspeech_autogain_target_dbfs == -20.0
    assert opts.clearspeech_autogain_max_gain_db == 16.0
    assert opts.clearspeech_bandpass_low_hz == 150.0
    assert opts.clearspeech_bandpass_high_hz == 5_500.0
    assert opts.clearspeech_presence_center_hz == 3_000.0
    assert opts.clearspeech_presence_boost_db == 6.0
    assert opts.clearspeech_presence_q == 1.0
    assert opts.clearspeech_denoise_noise_floor_db == -25.0
    assert opts.clearspeech_denoise_reduction_db == 12.0
    assert opts.clearspeech_dereverb_rt60_floor_ms == 300.0
    assert opts.clearspeech_dereverb_subtract_factor == 1.0
    assert opts.clearspeech_dereverb_crossfade_ms == 50.0
    assert opts.verbose is False


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


def test_unknown_speaker_choices():
    ns = _parse(["transcribe", "/tmp/a.m4a", "--unknown-speaker", "keep"])
    assert _opts_from_args(ns).unknown_speaker == "keep"
    with pytest.raises(SystemExit):
        _parse(["transcribe", "/tmp/a.m4a", "--unknown-speaker", "wat"])


def test_clearspeech_chain_parsed():
    ns = _parse([
        "transcribe", "/tmp/a.m4a",
        "--clearspeech-chain", "autogain,bandpass,presence",
    ])
    assert _opts_from_args(ns).clearspeech_chain == "autogain,bandpass,presence"


def test_clearspeech_chain_empty_disables_preprocessing():
    ns = _parse(["transcribe", "/tmp/a.m4a", "--clearspeech-chain", ""])
    assert _opts_from_args(ns).clearspeech_chain == ""


def test_clearspeech_chain_reorder_allowed():
    ns = _parse([
        "transcribe", "/tmp/a.m4a",
        "--clearspeech-chain", "presence,autogain,bandpass",
    ])
    assert _opts_from_args(ns).clearspeech_chain == "presence,autogain,bandpass"


def test_clearspeech_autogain_tuning_flags():
    ns = _parse([
        "transcribe", "/tmp/a.m4a",
        "--clearspeech-autogain-target-dbfs", "-18",
        "--clearspeech-autogain-max-gain-db", "15",
    ])
    opts = _opts_from_args(ns)
    assert opts.clearspeech_autogain_target_dbfs == -18.0
    assert opts.clearspeech_autogain_max_gain_db == 15.0


def test_clearspeech_bandpass_tuning_flags():
    ns = _parse([
        "transcribe", "/tmp/a.m4a",
        "--clearspeech-bandpass-low-hz", "120",
        "--clearspeech-bandpass-high-hz", "7000",
    ])
    opts = _opts_from_args(ns)
    assert opts.clearspeech_bandpass_low_hz == 120.0
    assert opts.clearspeech_bandpass_high_hz == 7_000.0


def test_clearspeech_presence_tuning_flags():
    ns = _parse([
        "transcribe", "/tmp/a.m4a",
        "--clearspeech-presence-center-hz", "2500",
        "--clearspeech-presence-boost-db", "6",
        "--clearspeech-presence-q", "1.4",
    ])
    opts = _opts_from_args(ns)
    assert opts.clearspeech_presence_center_hz == 2500.0
    assert opts.clearspeech_presence_boost_db == 6.0
    assert opts.clearspeech_presence_q == 1.4


def test_clearspeech_denoise_tuning_flags():
    ns = _parse([
        "transcribe", "/tmp/a.m4a",
        "--clearspeech-denoise-noise-floor-db", "-30",
        "--clearspeech-denoise-reduction-db", "9",
    ])
    opts = _opts_from_args(ns)
    assert opts.clearspeech_denoise_noise_floor_db == -30.0
    assert opts.clearspeech_denoise_reduction_db == 9.0


def test_clearspeech_dereverb_tuning_flags():
    ns = _parse([
        "transcribe", "/tmp/a.m4a",
        "--clearspeech-dereverb-rt60-floor-ms", "500",
        "--clearspeech-dereverb-subtract-factor", "0.7",
        "--clearspeech-dereverb-crossfade-ms", "25",
    ])
    opts = _opts_from_args(ns)
    assert opts.clearspeech_dereverb_rt60_floor_ms == 500.0
    assert opts.clearspeech_dereverb_subtract_factor == 0.7
    assert opts.clearspeech_dereverb_crossfade_ms == 25.0


def test_asr_engine_flag_is_no_longer_accepted():
    """`--asr-engine` was removed in v0.23.0 along with the VibeVoice backend
    (see ADR 0021). argparse must reject it rather than silently ignore it.
    """
    with pytest.raises(SystemExit):
        _parse(["transcribe", "/tmp/a.m4a", "--asr-engine", "whisper"])


def test_minimal_invocation_per_stage_llm_models_default_to_none():
    ns = _parse(["transcribe", "/tmp/a.m4a"])
    opts = _opts_from_args(ns)
    assert opts.llm_model is None
    assert opts.llm_proofread_model is None
    assert opts.llm_identify_model is None
    assert opts.llm_structure_model is None
    assert opts.llm_tldr_model is None


def test_per_stage_llm_model_flags_parsed():
    ns = _parse([
        "transcribe", "/tmp/a.m4a",
        "--llm-model", "/tmp/big",
        "--llm-proofread-model", "/tmp/small",
        "--llm-identify-model", "/tmp/tiny",
        "--llm-structure-model", "/tmp/big",
        "--llm-tldr-model", "/tmp/medium",
    ])
    opts = _opts_from_args(ns)
    assert opts.llm_model == "/tmp/big"
    assert opts.llm_proofread_model == "/tmp/small"
    assert opts.llm_identify_model == "/tmp/tiny"
    assert opts.llm_structure_model == "/tmp/big"
    assert opts.llm_tldr_model == "/tmp/medium"


def test_download_whisper_subcommand_defaults():
    ns = _parse(["download-whisper"])
    assert ns.cmd == "download-whisper"
    assert ns.repo_id == "mlx-community/whisper-large-v3-mlx"
    assert ns.verbose is False


def test_download_whisper_subcommand_custom_repo():
    ns = _parse(["download-whisper", "--repo-id", "foo/bar", "-v"])
    assert ns.cmd == "download-whisper"
    assert ns.repo_id == "foo/bar"
    assert ns.verbose is True
