"""Offline test for ffprobe metadata extractor — uses a real audio sample."""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from voice.ffprobe import extract_metadata


SAMPLE = Path("~/Downloads/two-speakers-diar-test-ukr.m4a").expanduser()


@pytest.mark.skipif(not SAMPLE.is_file(), reason="reference audio not present")
def test_extracts_creation_time_from_m4a():
    m = extract_metadata(SAMPLE)
    assert m.source == "ffprobe creation_time"
    assert m.duration_s == pytest.approx(374.93, abs=0.1)
    assert m.started_at.startswith("2026-05-10")


@pytest.mark.skipif(not SAMPLE.is_file(), reason="reference audio not present")
def test_override_datetime_wins():
    forced = datetime(2099, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    m = extract_metadata(SAMPLE, override_started_at=forced)
    assert m.source == "cli override"
    assert m.started_at.startswith("2099-01-01")
