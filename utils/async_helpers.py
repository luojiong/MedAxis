"""MedAxis — Async Helpers.

AsyncTaskRunner: a QThread wrapper for running heavy operations
(segmentation, resampling, AI calls, file I/O) off the GUI thread,
with progress reporting, cancellation and structured error propagation.
"""
from __future__ import annotations

import inspect
import logging
import threading
import traceback
from typing import Any, Callable, Optional

from PySide6.QtCore import QThread, Signal, Slot

logger = logging.getLogger(__name__)


class CancellationToken:
    """Thread-safe cooperative cancellation token.

    Long-running tasks should periodically call ``token.check()`` or
    inspect ``token.is_cancelled`` and abort promptly.
    """

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def check(self) -> None:
        """Raise CancelledError if cancellation has been requested."""
        if self._event.is_set():
            raise CancelledError("Operation cancelled.")

    def reset(self) -> None:
        self._event.clear()


class CancelledError(Exception):
    """Raised inside a task when its CancellationToken is cancelled."""


class AsyncTaskRunner(QThread):
    """Run a callable on a background QThread.

    Signals:
        progress(int, str)   — percent (0-100) and description
        progress_only(int)   — percent only
        finished_ok(object)  — task result on success
        failed(str)          — error message on failure
        cancelled()          — task was cancelled

    Usage:
        runner = AsyncTaskRunner(heavy_function, arg1, arg2, kwarg=value)
        runner.progress.connect(on_progress)
        runner.finished_ok.connect(on_done)
        runner.failed.connect(on_error)
        runner.start()
        ...
        runner.cancel_task()   # cooperative cancellation
    """

    progress = Signal(int, str)
    progress_only = Signal(int)
    finished_ok = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        fn: Callable[..., Any],
        *args: Any,
        description: str = "",
        cancellable: bool = True,
        parent=None,
        **kwargs: Any,
    ) -> None:
        super().__init__(parent)
        self._fn = fn
        self._args = args
        self._kwargs = kwargs
        self._description = description
        self.token: CancellationToken = CancellationToken() if cancellable else None  # type: ignore[assignment]
        self._result: Any = None

        if self.token is not None and self._accepts_token(fn):
            # Inject token into task kwargs if the task accepts one.
            self._kwargs.setdefault("cancellation_token", self.token)

    @staticmethod
    def _accepts_token(fn: Callable[..., Any]) -> bool:
        """Check whether fn can receive a 'cancellation_token' keyword."""
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            return False
        if "cancellation_token" in sig.parameters:
            return True
        return any(p.kind == inspect.Parameter.VAR_KEYWORD
                   for p in sig.parameters.values())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @Slot()
    def cancel_task(self) -> None:
        """Request cooperative cancellation of the running task."""
        if self.token is not None:
            logger.info("Cancellation requested for: %s",
                        self._description or self._fn.__name__)
            self.token.cancel()

    @property
    def result(self) -> Any:
        """Result of the task once finished_ok has been emitted."""
        return self._result

    def report_progress(self, percent: int, message: str = "") -> None:
        """Convenience method callable from within the task."""
        percent = max(0, min(100, int(percent)))
        if message:
            self.progress.emit(percent, message)
        else:
            self.progress_only.emit(percent)

    # ------------------------------------------------------------------
    # QThread body
    # ------------------------------------------------------------------
    def run(self) -> None:  # noqa: D401 — QThread override
        name = self._description or getattr(self._fn, "__name__", repr(self._fn))
        logger.debug("AsyncTask started: %s", name)
        try:
            if self.token is not None:
                self.token.check()
            self._result = self._fn(*self._args, **self._kwargs)
            if self.token is not None and self.token.is_cancelled:
                self.cancelled.emit()
                logger.info("AsyncTask cancelled: %s", name)
                return
            self.finished_ok.emit(self._result)
            logger.debug("AsyncTask finished: %s", name)
        except CancelledError:
            self.cancelled.emit()
            logger.info("AsyncTask cancelled: %s", name)
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            logger.error("AsyncTask failed: %s\n%s", name, tb)
            self.failed.emit(f"{name}: {exc}")


def run_async(
    fn: Callable[..., Any],
    *args: Any,
    on_done: Optional[Callable[[Any], None]] = None,
    on_error: Optional[Callable[[str], None]] = None,
    on_progress: Optional[Callable[[int, str], None]] = None,
    description: str = "",
    parent=None,
    **kwargs: Any,
) -> AsyncTaskRunner:
    """Convenience helper: build, wire and start an AsyncTaskRunner.

    The caller must keep a reference to the returned runner to avoid
    premature garbage collection of the QThread.
    """
    runner = AsyncTaskRunner(fn, *args, description=description,
                             parent=parent, **kwargs)
    if on_done is not None:
        runner.finished_ok.connect(on_done)
    if on_error is not None:
        runner.failed.connect(on_error)
    if on_progress is not None:
        runner.progress.connect(on_progress)
    runner.start()
    return runner
