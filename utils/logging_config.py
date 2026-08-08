"""MedAxis — Logging Configuration.

Structured logging with file + console handlers and optional integration
of VTK / ITK native error callbacks into the Python logging system.
"""
from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from typing import Optional

LOG_DIR = os.path.join(os.path.expanduser("~"), ".medaxis", "logs")
LOG_FILE = os.path.join(LOG_DIR, "medaxis.log")

CONSOLE_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
FILE_FORMAT = "%(asctime)s %(levelname)-8s [%(threadName)s] %(name)s (%(filename)s:%(lineno)d): %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def setup_logging(level: int = logging.INFO, log_file: Optional[str] = None) -> logging.Logger:
    """Configure root logging with console + rotating file handlers.

    Safe to call multiple times; only configures once.
    """
    global _configured
    root = logging.getLogger()
    if _configured:
        return root

    root.setLevel(level)

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(logging.Formatter(CONSOLE_FORMAT, DATE_FORMAT))
    root.addHandler(console)

    # File handler (rotating, 5 MB x 5 backups)
    file_path = log_file or LOG_FILE
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            file_path, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(FILE_FORMAT, DATE_FORMAT))
        root.addHandler(file_handler)
    except OSError:
        root.warning("Could not create log file handler at %s", file_path)

    # Quieten noisy third-party loggers
    for noisy in ("urllib3", "httpx", "httpcore", "PIL", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True
    setup_native_callbacks()
    return root


# ----------------------------------------------------------------------
# VTK / ITK native error callback integration
# ----------------------------------------------------------------------
_vtk_logger = logging.getLogger("vtk")
_itk_logger = logging.getLogger("itk")


def _vtk_error_handler(obj, _event: str) -> None:
    """vtkOutputWindow error callback -> Python logging."""
    try:
        text = obj.GetText() if hasattr(obj, "GetText") else str(obj)
    except Exception:  # noqa: BLE001
        text = "Unknown VTK error"
    _vtk_logger.error("%s", text)


def setup_native_callbacks() -> None:
    """Route VTK/ITK native error output into Python logging.

    Both integrations are optional; failures are logged at debug level
    and never raise.
    """
    # --- VTK ---
    try:
        import vtk  # type: ignore

        output_window = vtk.vtkFileOutputWindow()
        log_dir = os.path.dirname(LOG_FILE)
        os.makedirs(log_dir, exist_ok=True)
        output_window.SetFileName(os.path.join(log_dir, "vtk_errors.log"))
        output_window.SetDisplayModeToNever()

        error_observer = vtk.vtkOutputWindow()
        error_observer.AddObserver(
            "ErrorEvent", lambda o, e: _vtk_error_handler(o, e)
        )
        error_observer.AddObserver(
            "WarningEvent",
            lambda o, e: _vtk_logger.warning("%s", o.GetText() if hasattr(o, "GetText") else ""),
        )
        vtk.vtkOutputWindow.SetInstance(error_observer)
        _vtk_logger.debug("VTK error callbacks hooked.")
    except ImportError:
        _vtk_logger.debug("VTK not available; skipping VTK callback hookup.")
    except Exception:  # noqa: BLE001
        _vtk_logger.debug("Failed to hook VTK callbacks.", exc_info=True)

    # --- ITK ---
    try:
        import itk  # type: ignore

        def _itk_message_callback(message: str) -> None:
            text = str(message).strip()
            if not text:
                return
            low = text.lower()
            if "error" in low or "exception" in low:
                _itk_logger.error("%s", text)
            elif "warning" in low:
                _itk_logger.warning("%s", text)
            else:
                _itk_logger.info("%s", text)

        # itk (wrapping >= 5.3) exposes itk.OutputWindow / itkLogger
        if hasattr(itk, "OutputWindow"):
            try:
                itk.OutputWindow.add_message_callback(_itk_message_callback)
                _itk_logger.debug("ITK message callbacks hooked.")
            except Exception:  # noqa: BLE001
                _itk_logger.debug("Could not attach ITK message callback.", exc_info=True)
    except ImportError:
        _itk_logger.debug("ITK not available; skipping ITK callback hookup.")
    except Exception:  # noqa: BLE001
        _itk_logger.debug("Failed to hook ITK callbacks.", exc_info=True)


def get_logger(name: str) -> logging.Logger:
    """Convenience accessor for module loggers."""
    return logging.getLogger(name)
