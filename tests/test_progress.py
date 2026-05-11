"""Tests for the progress reporter.

`rich.Progress` is not invoked here — we run with a non-TTY console so the
`ProgressReporter` skips bar rendering and only emits its completion line.
The contracts under test:
- All three context managers work (task / spinner / token_counter)
- `advance()` callable accepts the expected shapes
- `NullProgress` is a drop-in replacement
"""

from __future__ import annotations

from io import StringIO

from rich.console import Console

from voice._progress import NullProgress, ProgressReporter


def _silent_reporter() -> ProgressReporter:
    """Reporter pinned to a non-TTY console so it suppresses bars."""
    return ProgressReporter(console=Console(file=StringIO(), force_terminal=False))


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
