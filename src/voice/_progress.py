"""Progress and spinner reporting for the pipeline.

`ProgressReporter` wraps `rich.Progress` so each stage can announce itself
through one of three context managers:

- `.task(label, total)` for deterministic counters (postprocess, identify)
- `.spinner(label)` for indeterminate ops (model loads, diarization)
- `.token_counter(label)` for streaming-LLM generation (structure, tldr)

On a real terminal each context renders an in-place bar/spinner with
elapsed and ETA. When stderr is redirected (CI, `tee > log`) the same
context exits to a single line — `✓ {label} in N.Ns` — so logs stay
clean and ANSI-free.

A single `ProgressReporter` lives for the whole pipeline run and is
threaded through every stage that accepts a `progress=` kwarg. Stage
unit tests pass `None` and behaviour is identical.
"""

from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from typing import Callable, Iterator

__all__ = ["ProgressReporter", "NullProgress"]


from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


class ProgressReporter:
    """Manages rich.Progress for the duration of a pipeline run.

    Build one instance, pass it through stages, and call `start()`/`stop()`
    around the run (or use the context manager). If stderr is not a TTY,
    bars are suppressed entirely and only the completion line is printed —
    no ANSI escapes leak into the log file.
    """

    def __init__(self, *, console: Console | None = None) -> None:
        self._console = console or Console(stderr=True)
        self._is_tty = self._console.is_terminal
        self._progress: Progress | None = None

    def start(self) -> None:
        if self._progress is not None or not self._is_tty:
            return
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed:>4}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            TextColumn("{task.fields[suffix]}"),
            console=self._console,
            transient=False,
            refresh_per_second=10,
        )
        self._progress.start()

    def stop(self) -> None:
        if self._progress is not None:
            self._progress.stop()
            self._progress = None

    def __enter__(self) -> "ProgressReporter":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()

    # ------------------------------------------------------------------ tasks

    @contextmanager
    def task(self, label: str, *, total: int) -> Iterator[Callable[..., None]]:
        """A counter task — yields `advance(n=1, suffix=None)`."""
        started = time.perf_counter()
        task_id = None
        if self._progress is not None:
            task_id = self._progress.add_task(label, total=total, suffix="")

        def advance(n: int = 1, suffix: str | None = None) -> None:
            if task_id is None:
                return
            update_kwargs: dict = {"advance": n}
            if suffix is not None:
                update_kwargs["suffix"] = suffix
            self._progress.update(task_id, **update_kwargs)  # type: ignore[union-attr]

        try:
            yield advance
        finally:
            elapsed = time.perf_counter() - started
            if task_id is not None:
                # Force the bar to its final 100 %; rich will redraw once.
                self._progress.update(task_id, completed=total)  # type: ignore[union-attr]
            self._finalise(label, elapsed)

    @contextmanager
    def spinner(self, label: str) -> Iterator[None]:
        """Indeterminate spinner. Just shows elapsed time."""
        started = time.perf_counter()
        task_id = None
        if self._progress is not None:
            task_id = self._progress.add_task(label, total=None, suffix="")
        try:
            yield
        finally:
            elapsed = time.perf_counter() - started
            if task_id is not None:
                self._progress.remove_task(task_id)  # type: ignore[union-attr]
            self._finalise(label, elapsed)

    @contextmanager
    def token_counter(self, label: str) -> Iterator[Callable[[int], None]]:
        """A token-streaming task; yields `advance(tokens_delta)`."""
        started = time.perf_counter()
        task_id = None
        if self._progress is not None:
            task_id = self._progress.add_task(label, total=None, suffix="0 tokens")

        token_count = 0

        def advance(delta: int = 1) -> None:
            nonlocal token_count
            token_count += delta
            if task_id is None:
                return
            elapsed_now = time.perf_counter() - started
            tps = token_count / elapsed_now if elapsed_now > 0 else 0.0
            self._progress.update(  # type: ignore[union-attr]
                task_id,
                advance=delta,
                suffix=f"{token_count} tokens · {tps:.1f} t/s",
            )

        try:
            yield advance
        finally:
            elapsed = time.perf_counter() - started
            if task_id is not None:
                self._progress.remove_task(task_id)  # type: ignore[union-attr]
            tail = f"{token_count} tokens"
            self._finalise(label, elapsed, suffix=tail)

    # ------------------------------------------------------------------ helpers

    def _finalise(self, label: str, elapsed: float, *, suffix: str | None = None) -> None:
        """Print a one-line completion message that survives non-TTY logs."""
        line = f"  ✓ {label} in {elapsed:.1f}s"
        if suffix:
            line += f" ({suffix})"
        if self._is_tty and self._progress is not None:
            self._console.log(line)
        else:
            print(line, file=sys.stderr, flush=True)


class NullProgress:
    """Drop-in stand-in for `ProgressReporter` in tests and when no UI is wanted.

    Every context manager yields a no-op callable so stage modules can be
    written as if a real reporter is always present.
    """

    def start(self) -> None: ...
    def stop(self) -> None: ...

    def __enter__(self) -> "NullProgress":
        return self

    def __exit__(self, *_: object) -> None: ...

    @contextmanager
    def task(self, _label: str, *, total: int) -> Iterator[Callable[..., None]]:
        def _advance(_n: int = 1, suffix: str | None = None) -> None:
            return
        yield _advance

    @contextmanager
    def spinner(self, _label: str) -> Iterator[None]:
        yield

    @contextmanager
    def token_counter(self, _label: str) -> Iterator[Callable[[int], None]]:
        def _advance(_delta: int = 1) -> None:
            return
        yield _advance
