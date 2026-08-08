"""
Progress monitor with cancellation support for long-running algorithms.
"""
try:
    from PySide6.QtCore import QObject, Signal
except ImportError:  # pragma: no cover - exercised in headless tooling
    class QObject:
        def __init__(self, parent=None):
            self._parent = parent

    class Signal:
        def __init__(self, *_args, **_kwargs):
            self._slots = []

        def connect(self, slot):
            if slot not in self._slots:
                self._slots.append(slot)

        def emit(self, *args):
            for slot in list(self._slots):
                slot(*args)


class ProgressMonitor(QObject):
    """Emits progress signals during algorithm execution. Supports cancellation."""

    progress_changed = Signal(float, str)    # (0.0-1.0, status_message)
    finished = Signal()
    cancelled = Signal()
    error_occurred = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._progress = 0.0
        self._message = ""
        self._is_cancelled = False
        self._is_running = False

    @property
    def progress(self) -> float:
        return self._progress

    @property
    def message(self) -> str:
        return self._message

    @property
    def is_cancelled(self) -> bool:
        return self._is_cancelled

    @property
    def is_running(self) -> bool:
        return self._is_running

    def start(self, message: str = ""):
        self._is_running = True
        self._is_cancelled = False
        self._progress = 0.0
        self._message = message
        self.progress_changed.emit(0.0, message)

    def update(self, progress: float, message: str = ""):
        self._progress = min(1.0, max(0.0, progress))
        if message:
            self._message = message
        self.progress_changed.emit(self._progress, self._message)

    def cancel(self):
        self._is_cancelled = True
        self._is_running = False
        self.cancelled.emit()

    def complete(self, message: str = "Complete"):
        self._progress = 1.0
        self._message = message
        self._is_running = False
        self.progress_changed.emit(1.0, message)
        self.finished.emit()

    def error(self, message: str):
        self._is_running = False
        self._message = message
        self.error_occurred.emit(message)

    def check_cancelled(self) -> bool:
        """Called by algorithm implementations to check if cancelled."""
        return self._is_cancelled
