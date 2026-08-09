"""E2E test: open a real DICOM folder through the full UI stack.

Boots the real AppController + MainWindow (visible window), submits a DICOM
folder through the same code path the toolbar Open button uses
(``controller.open_file(path)``), waits for the background loader, verifies
the expected slice count, saves a screenshot and exits.

Usage:
    python scripts/e2e_open_folder.py [folder] [expected_slices]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from app.app_controller import AppController

FOLDER = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    Path.home() / "Desktop/workspace/测试数据CT534层/测试数据CT534层"
)
EXPECTED_SLICES = int(sys.argv[2]) if len(sys.argv) > 2 else 534
SHOT = Path(__file__).resolve().parent.parent / "build-deps" / "e2e_loaded.png"

state = {"error": None, "volume": None}

app = QApplication(sys.argv)
app.setOrganizationName("MedAxis")
app.setApplicationName("MedAxis")

controller = AppController()
controller.initialize()
controller.show()


def on_load_error(message: str) -> None:
    state["error"] = f"load_error: {message}"
    app.exit(2)


def poll() -> None:
    if controller.current_volume is not None:
        state["volume"] = controller.current_volume
        app.exit(0)
    if state["error"] is not None:
        app.exit(2)


fm = getattr(controller, "file_manager", None)
if fm is not None:
    try:
        fm.load_error.connect(on_load_error)
    except Exception:  # noqa: BLE001
        pass

# Give the window a moment, then drive the exact same entry point the
# toolbar "Open" button / MCP open_file tool uses.
QTimer.singleShot(1500, lambda: controller.open_file(str(FOLDER)))

poll_timer = QTimer()
poll_timer.timeout.connect(poll)
poll_timer.start(500)

# Hard timeout: 534 slices / 270 MB may take a while on first read.
QTimer.singleShot(300_000, lambda: (state.update(error="TIMEOUT"), app.exit(3)))

rc = app.exec()
volume = state["volume"]

if volume is not None:
    array = getattr(volume, "array", None)
    shape = tuple(array.shape) if array is not None else None
    spacing = getattr(volume, "spacing_mm", None)
    print(f"E2E_OK volume={getattr(volume, 'name', '')!r} shape={shape} spacing={spacing}")
    if shape is not None and len(shape) == 3 and shape[0] != EXPECTED_SLICES:
        print(f"E2E_SLICE_MISMATCH expected={EXPECTED_SLICES} got={shape[0]}")
        rc = 4
    try:
        controller.main_window.grab().save(str(SHOT))
        print(f"screenshot: {SHOT}")
    except Exception as exc:  # noqa: BLE001
        print(f"screenshot failed: {exc}")
else:
    print(f"E2E_FAILED: {state['error']}")

sys.exit(rc)
