"""Offline test for audiometa metadata extractor — uses a real audio sample.

The sample is a local recording that is not part of the repository. Point
VOICE_TEST_SAMPLE_M4A at an .m4a to run these; otherwise they skip. The
expected duration and creation date come from that file, so they are read
from the environment too rather than pinned to one person's recording.
"""

import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from voice.audio_meta import extract_metadata


_sample_env = os.environ.get("VOICE_TEST_SAMPLE_M4A")
SAMPLE = Path(_sample_env).expanduser() if _sample_env else None

_EXPECTED_DURATION_S = float(os.environ.get("VOICE_TEST_SAMPLE_DURATION_S", "0") or 0)
_EXPECTED_DATE_PREFIX = os.environ.get("VOICE_TEST_SAMPLE_DATE", "")

_no_sample = SAMPLE is None or not SAMPLE.is_file()
_skip_reason = "set VOICE_TEST_SAMPLE_M4A to a local .m4a to run this test"


@pytest.mark.skipif(_no_sample, reason=_skip_reason)
def test_extracts_creation_time_from_m4a():
    m = extract_metadata(SAMPLE)
    assert m.source == "ffprobe creation_time"
    if _EXPECTED_DURATION_S:
        assert m.duration_s == pytest.approx(_EXPECTED_DURATION_S, abs=0.1)
    else:
        assert m.duration_s > 0
    if _EXPECTED_DATE_PREFIX:
        assert m.started_at.startswith(_EXPECTED_DATE_PREFIX)
    else:
        assert m.started_at


@pytest.mark.skipif(_no_sample, reason=_skip_reason)
def test_override_datetime_wins():
    forced = datetime(2099, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    m = extract_metadata(SAMPLE, override_started_at=forced)
    assert m.source == "cli override"
    assert m.started_at.startswith("2099-01-01")
