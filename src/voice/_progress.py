"""Progress and resource reporting for the pipeline.

`ProgressReporter` wraps `rich.Progress` so each stage announces itself
through one of three context managers:

- `.task(label, total)` for deterministic counters (proofread, identify)
- `.spinner(label)` for indeterminate ops (model loads, diarization)
- `.token_counter(label)` for streaming-LLM generation (structure, tldr)

On a real terminal each context renders an in-place bar/spinner with
elapsed and ETA, and a persistent footer underneath shows RAM and MLX
memory refreshed every second.

On a non-TTY (CI, Claude Code Bash tool, `2> log`), bars are suppressed
but a background heartbeat thread prints a single-line snapshot of the
active task plus memory every 5 seconds. Stage transitions
(`✓ {label} in N.Ns`) print immediately, never waiting for the next
heartbeat tick.

A single `ProgressReporter` lives for the whole pipeline run and is
threaded through every stage that accepts a `progress=` kwarg.
"""

from __future__ import annotations

import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
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

from ._memory_stats import read_memory_snapshot


HEARTBEAT_INTERVAL_S = 5.0
FOOTER_REFRESH_HZ = 1.0


@dataclass
class _ActiveTask:
    label: str
    started: float
    state_fn: Callable[[], str]  # returns "23/64 (35%) elapsed 0:32" or similar
    stage_num: str | None = None  # e.g. "3/13" — pipeline position
    stage_id: str | None = None  # e.g. "diarize_speakers" — module-level id


class ProgressReporter:
    """Manages rich.Progress for the duration of a pipeline run."""

    def __init__(
        self,
        *,
        console: Console | None = None,
        heartbeat_interval_s: float = HEARTBEAT_INTERVAL_S,
    ) -> None:
        self._console = console or Console(stderr=True)
        self._is_tty = self._console.is_terminal
        self._progress: Progress | None = None
        self._footer_task_id = None
        self._heartbeat_interval = heartbeat_interval_s

        self._active_tasks: list[_ActiveTask] = []
        self._tasks_lock = threading.Lock()

        self._stop_event = threading.Event()
        self._monitor_thread: threading.Thread | None = None

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> None:
        if self._monitor_thread is not None:
            return
        if self._is_tty:
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
            # Persistent footer at the bottom of the progress group.
            self._footer_task_id = self._progress.add_task(
                "[dim]resources[/]", total=None, suffix="",
            )

        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, name="voice-progress-monitor", daemon=True,
        )
        self._monitor_thread.start()

    def stop(self) -> None:
        if self._monitor_thread is None:
            return
        self._stop_event.set()
        self._monitor_thread.join(timeout=2.0)
        self._monitor_thread = None

        if self._progress is not None:
            if self._footer_task_id is not None:
                try:
                    self._progress.remove_task(self._footer_task_id)
                except Exception:  # noqa: BLE001 — defensive cleanup
                    pass
                self._footer_task_id = None
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

        completed = 0
        last_suffix = ""

        def advance(n: int = 1, suffix: str | None = None) -> None:
            nonlocal completed, last_suffix
            completed += n
            if suffix is not None:
                last_suffix = suffix
            if task_id is None:
                return
            update_kwargs: dict = {"advance": n}
            if suffix is not None:
                update_kwargs["suffix"] = suffix
            self._progress.update(task_id, **update_kwargs)  # type: ignore[union-attr]

        def state() -> str:
            elapsed = time.perf_counter() - started
            pct = int(round(100 * completed / total)) if total else 0
            extras = f" {last_suffix}" if last_suffix else ""
            return f"{completed}/{total} ({pct}%) elapsed {_fmt_elapsed(elapsed)}{extras}"

        active = _ActiveTask(label=label, started=started, state_fn=state)
        self._register(active)
        try:
            yield advance
        finally:
            elapsed = time.perf_counter() - started
            if task_id is not None:
                self._progress.update(task_id, completed=total)  # type: ignore[union-attr]
            self._unregister(active)
            self._finalise(label, elapsed)

    @contextmanager
    def spinner(
        self,
        label: str,
        *,
        stage_num: str | None = None,
        stage_id: str | None = None,
    ) -> Iterator[Callable[..., None]]:
        """Indeterminate spinner. Yields a `set_state(step, completed=None,
        total=None)` callback the stage can call to surface live progress
        (e.g. pyannote's segmentation → embeddings → clustering steps).

        Callers that don't need it can ignore the yielded value — backwards
        compatible with `with progress.spinner(x):`.

        When `stage_num` and `stage_id` are supplied, the heartbeat
        rendering uses them as `[stage_num] stage_id/step_name` — e.g.
        `[3/13] diarize_speakers/embeddings`.
        """
        started = time.perf_counter()
        task_id = None
        if self._progress is not None:
            task_id = self._progress.add_task(label, total=None, suffix="")

        step_state: dict[str, object] = {"name": "", "completed": None, "total": None}

        def set_state(
            step: str, completed: int | None = None, total: int | None = None,
        ) -> None:
            step_state["name"] = step
            step_state["completed"] = completed
            step_state["total"] = total
            if task_id is not None:
                self._progress.update(  # type: ignore[union-attr]
                    task_id,
                    suffix=_format_step_progress(step, completed, total, stage_id=stage_id),
                )

        def state() -> str:
            elapsed_part = f"elapsed {_fmt_elapsed(time.perf_counter() - started)}"
            name = step_state["name"]
            if not name:
                return elapsed_part
            step_text = _format_step_progress(
                str(name),
                step_state["completed"],  # type: ignore[arg-type]
                step_state["total"],  # type: ignore[arg-type]
                stage_id=stage_id,
            )
            return f"{step_text} · {elapsed_part}"

        active = _ActiveTask(
            label=label, started=started, state_fn=state,
            stage_num=stage_num, stage_id=stage_id,
        )
        self._register(active)
        try:
            yield set_state
        finally:
            elapsed = time.perf_counter() - started
            if task_id is not None:
                self._progress.remove_task(task_id)  # type: ignore[union-attr]
            self._unregister(active)
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

        def state() -> str:
            elapsed = time.perf_counter() - started
            tps = token_count / elapsed if elapsed > 0 else 0.0
            return (
                f"{token_count} tokens ({tps:.1f} t/s) elapsed {_fmt_elapsed(elapsed)}"
            )

        active = _ActiveTask(label=label, started=started, state_fn=state)
        self._register(active)
        try:
            yield advance
        finally:
            elapsed = time.perf_counter() - started
            if task_id is not None:
                self._progress.remove_task(task_id)  # type: ignore[union-attr]
            self._unregister(active)
            tail = f"{token_count} tokens"
            self._finalise(label, elapsed, suffix=tail)

    # ------------------------------------------------------------------ helpers

    def _register(self, task: _ActiveTask) -> None:
        with self._tasks_lock:
            self._active_tasks.append(task)

    def _unregister(self, task: _ActiveTask) -> None:
        with self._tasks_lock:
            try:
                self._active_tasks.remove(task)
            except ValueError:
                pass

    def _current_active(self) -> _ActiveTask | None:
        with self._tasks_lock:
            return self._active_tasks[-1] if self._active_tasks else None

    def _finalise(self, label: str, elapsed: float, *, suffix: str | None = None) -> None:
        """Print a one-line completion message that survives non-TTY logs."""
        line = f"  ✓ {label} in {elapsed:.1f}s"
        if suffix:
            line += f" ({suffix})"
        if self._is_tty and self._progress is not None:
            self._console.log(line)
        else:
            print(line, file=sys.stderr, flush=True)

    # ------------------------------------------------------------------ background monitor

    def _monitor_loop(self) -> None:
        """Update the TTY footer ~1 Hz; emit a non-TTY heartbeat every 5 s."""
        if self._is_tty:
            interval = 1.0 / FOOTER_REFRESH_HZ
        else:
            interval = self._heartbeat_interval

        while not self._stop_event.is_set():
            self._stop_event.wait(interval)
            if self._stop_event.is_set():
                break
            try:
                if self._is_tty:
                    self._refresh_footer()
                else:
                    self._emit_heartbeat()
            except Exception:  # noqa: BLE001 — telemetry must not crash the pipeline
                pass

    def _refresh_footer(self) -> None:
        if self._progress is None or self._footer_task_id is None:
            return
        snapshot = read_memory_snapshot()
        text = snapshot.format()
        self._progress.update(self._footer_task_id, suffix=text)

    def _emit_heartbeat(self) -> None:
        active = self._current_active()
        if active is None:
            return
        snapshot = read_memory_snapshot()
        memory_part = snapshot.format()
        suffix = f" · {memory_part}" if memory_part else ""
        prefix = f"[{active.stage_num}]" if active.stage_num else f"[{active.label}]"
        line = f"--> ⏳ {prefix} {active.state_fn()}{suffix}"
        print(line, file=sys.stderr, flush=True)


def _format_step_progress(
    step: str, completed: int | None, total: int | None,
    *, stage_id: str | None = None,
) -> str:
    """`diarize_speakers/embeddings: 51% (18/35)` if stage_id given and
    counters known; `embeddings: ?% (?/?)` if no stage_id or counters."""
    head = f"{stage_id}/{step}" if stage_id else step
    if completed is not None and total:
        pct = int(round(100 * completed / total))
        return f"{head}: {pct}% ({completed}/{total})"
    return f"{head}: ?% (?/?)"


def _fmt_elapsed(seconds: float) -> str:
    s = int(round(seconds))
    if s >= 3600:
        return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"
    return f"{s // 60}:{s % 60:02d}"


class NullProgress:
    """Drop-in stand-in for `ProgressReporter` in tests and when no UI is wanted."""

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
