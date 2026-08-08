"""MedAxis — Medical Imaging Platform Entry Point."""
import os
import sys

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication

from app.app_controller import AppController


# Native-only smoke test (packaged builds): verify the compiled modules load
# without creating a GUI / OpenGL context. Exit code 0 == all modules OK.
if os.environ.get("MEDAXIS_SMOKE_NATIVE"):
    from core.native_extensions import NativeExtensionRegistry

    status = NativeExtensionRegistry().status()
    ok = all(s.available for s in status.values())
    print("NATIVE:", {name: s.available for name, s in status.items()}, flush=True)
    for name, s in status.items():
        if not s.available:
            print(f"  {name} error: {s.error}", flush=True)
    sys.exit(0 if ok else 1)


def main():
    # High DPI support
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    app.setOrganizationName("MedAxis")
    app.setApplicationName("MedAxis")
    app.setApplicationVersion("0.1.0")

    # Smoke-test mode for packaged builds (CI): auto-quit after 3 seconds.
    if os.environ.get("MEDAXIS_SMOKE"):
        QTimer.singleShot(3000, app.quit)
    
    # Load dark theme
    theme_path = os.path.join(os.path.dirname(__file__), "..", "resources", "styles", "dark_theme.qss")
    if os.path.exists(theme_path):
        with open(theme_path, "r", encoding="utf-8") as f:
            app.setStyleSheet(f.read())
    
    # Init controller
    controller = AppController()
    controller.initialize()
    controller.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
