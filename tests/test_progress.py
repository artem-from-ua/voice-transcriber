"""Tests for the progress reporter.

`rich.Progress` is not invoked here — we run with a non-TTY console so the
`ProgressReporter` skips bar rendering and only emits its completion line.
The contracts under test:
- All three context managers work (task / spinner / token_counter)
- `advance()` callable accepts the expected shapes
- `NullProgress` is a drop-in replacement
- Heartbeat fires in non-TTY mode with the active task's state
"""

from __future__ import annotations

import time
from io import StringIO

from rich.console import Console

from voice._progress import NullProgress, ProgressReporter


def _silent_reporter(*, heartbeat_interval_s: float = 99.0) -> ProgressReporter:
    """Reporter pinned to a non-TTY console so it suppresses bars."""
    return ProgressReporter(
        console=Console(file=StringIO(), force_terminal=False),
        heartbeat_interval_s=heartbeat_interval_s,
    )


def test_task_advance_accepts_n_and_suffix():
    with _silent_reporter() as p:
        with p.task("work", total=3) as advance:
            advance(1)
            advance(1, suffix="halfway")
            advance(1, suffix="done")


def test_spinner_runs_to_completion():
    with _silent_reporter() as p:
        with p.spinner("loading"):
            pass


def test_spinner_set_state_renders_step_and_percent_in_heartbeat(capsys):
    """spinner() yields set_state(step, completed, total); heartbeat formats
    it as `embeddings: 27% (12/45)` and prefixes with the stage_id."""
    p = _silent_reporter(heartbeat_interval_s=0.1)
    with p:
        with p.spinner("diarize", stage_id="3/13 diarize_speakers") as set_state:
            set_state("embeddings", 12, 45)
            time.sleep(0.25)
    captured = capsys.readouterr().err
    assert "--> ⏳ [3/13 diarize_speakers]" in captured, captured
    assert "embeddings: 27% (12/45)" in captured, captured


def test_spinner_set_state_renders_unknown_total_as_question_marks(capsys):
    """When pyannote does not report total/completed, show `?% (?/?)`."""
    p = _silent_reporter(heartbeat_interval_s=0.1)
    with p:
        with p.spinner("diarize", stage_id="3/13 diarize_speakers") as set_state:
            set_state("segmentation")
            time.sleep(0.25)
    captured = capsys.readouterr().err
    assert "segmentation: ?% (?/?)" in captured, captured


def test_token_counter_accepts_deltas():
    with _silent_reporter() as p:
        with p.token_counter("streaming") as advance:
            for _ in range(5):
                advance(1)


def test_null_progress_task():
    with NullProgress() as p:
        with p.task("x", total=10) as advance:
            advance(1, suffix="ignored")


def test_null_progress_spinner():
    with NullProgress() as p:
        with p.spinner("y"):
            pass


def test_null_progress_token_counter():
    with NullProgress() as p:
        with p.token_counter("z") as advance:
            advance(3)


def test_reporter_is_idempotent_on_double_start():
    p = _silent_reporter()
    p.start()
    p.start()  # must not raise
    p.stop()
    p.stop()  # must not raise


def test_non_tty_reporter_does_not_render_bars():
    """The silent reporter should not create a rich.Progress."""
    p = _silent_reporter()
    p.start()
    assert p._progress is None, "no rich.Progress on non-TTY"
    p.stop()


def test_heartbeat_fires_on_non_tty(capsys):
    """A short-interval heartbeat must emit at least one line during a task."""
    p = _silent_reporter(heartbeat_interval_s=0.1)
    with p:
        with p.task("crunching", total=10) as advance:
            advance(3, suffix="ok")
            time.sleep(0.25)  # let the monitor wake up at least once
            advance(2)
            time.sleep(0.15)
    captured = capsys.readouterr().err
    assert "--> ⏳ [crunching]" in captured, captured
    # transitions (✓) and heartbeat arrows both present
    assert "✓ crunching" in captured


def test_heartbeat_silent_when_no_active_task(capsys):
    """No heartbeat lines when nothing is registered."""
    p = _silent_reporter(heartbeat_interval_s=0.1)
    with p:
        time.sleep(0.3)
    captured = capsys.readouterr().err
    # No heartbeat lines emitted (heartbeat prefix is "--> ⏳").
    assert "--> ⏳" not in captured


def test_monitor_thread_stops_cleanly():
    """stop() must join the heartbeat thread within its short timeout."""
    p = _silent_reporter(heartbeat_interval_s=0.05)
    p.start()
    time.sleep(0.1)
    p.stop()
    assert p._monitor_thread is None
