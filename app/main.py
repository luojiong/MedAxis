"""MedAxis — Medical Imaging Platform Entry Point."""
import sys
import os
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from app.app_controller import AppController

def main():
    # High DPI support
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    app.setOrganizationName("MedAxis")
    app.setApplicationName("MedAxis")
    app.setApplicationVersion("0.1.0")
    
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
