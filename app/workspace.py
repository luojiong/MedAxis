"""MedAxis — Workspace Manager.

Manages the multi-patient / multi-project workspace: opening, saving and
closing projects, tracking the active project and persisting the recent
projects list via QSettings.
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional

from PySide6.QtCore import QObject, QSettings, Signal, Slot

logger = logging.getLogger(__name__)

MAX_RECENT_PROJECTS = 10


class Workspace(QObject):
    """Manages open projects and the recent-projects history."""

    # Emitted when the active project changes (may be None when closed).
    project_changed = Signal(object)
    # Emitted after the active project has been saved. Payload: path.
    project_saved = Signal(str)

    def __init__(self, controller: QObject, parent: Optional[QObject] = None) -> None:
        super().__init__(parent or controller)
        self._controller = controller
        self._settings = QSettings("MedAxis", "MedAxis")

        self.active_project = None          # Project | None
        self.open_projects: List[object] = []  # all open projects (tabs)
        self.recent_projects: List[str] = self._load_recent()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @Slot(str)
    def open_project(self, path: str) -> Optional[object]:
        """Open a project file and make it the active project."""
        path = os.path.abspath(path)
        if not os.path.exists(path):
            logger.error("Project file not found: %s", path)
            if hasattr(self._controller, "report_error"):
                self._controller.report_error(f"Project file not found: {path}")
            return None

        # Already open? Just activate it.
        for proj in self.open_projects:
            if getattr(proj, "path", None) == path:
                self._set_active(proj)
                return proj

        project = self._load_project_file(path)
        if project is None:
            return None

        self.open_projects.append(project)
        self._add_recent(path)
        self._set_active(project)
        logger.info("Opened project: %s", path)
        return project

    @Slot()
    def save_project(self) -> bool:
        """Save the active project to disk."""
        if self.active_project is None:
            logger.warning("save_project() called with no active project.")
            return False

        path = getattr(self.active_project, "path", None)
        try:
            if hasattr(self.active_project, "save"):
                self.active_project.save(path)
            self._add_recent(path)
            self.project_saved.emit(path)
            logger.info("Saved project: %s", path)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to save project: %s", path)
            if hasattr(self._controller, "report_error"):
                self._controller.report_error(f"Failed to save project: {exc}")
            return False

    @Slot()
    def close_project(self) -> None:
        """Close the active project."""
        if self.active_project is None:
            return

        project = self.active_project
        if project in self.open_projects:
            self.open_projects.remove(project)

        if hasattr(project, "close"):
            try:
                project.close()
            except Exception:  # noqa: BLE001
                logger.exception("Error while closing project.")

        next_project = self.open_projects[-1] if self.open_projects else None
        self._set_active(next_project)
        logger.info("Closed project: %r", getattr(project, "path", project))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _set_active(self, project: Optional[object]) -> None:
        self.active_project = project
        self.project_changed.emit(project)

    def _load_project_file(self, path: str) -> Optional[object]:
        """Load a project from disk.

        Uses ProjectManager if available, otherwise falls back to a
        minimal dict-based project so the workspace is usable before the
        full Project model exists.
        """
        pm = getattr(self._controller, "project_manager", None)
        if pm is not None:
            if hasattr(pm, "load_project"):
                return pm.load_project(path)
            if hasattr(pm, "load"):
                return pm.load(path)

        # Minimal fallback project representation.
        import json
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to parse project file: %s", path)
            if hasattr(self._controller, "report_error"):
                self._controller.report_error(f"Failed to open project: {exc}")
            return None

        class _FallbackProject(dict):
            pass

        proj = _FallbackProject(data)
        proj.path = path
        proj.save = lambda p=None: self._save_fallback(proj, p or path)
        return proj

    @staticmethod
    def _save_fallback(project: dict, path: str) -> None:
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump(dict(project), f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Recent projects persistence
    # ------------------------------------------------------------------
    def _load_recent(self) -> List[str]:
        value = self._settings.value("workspace/recent_projects", [])
        if isinstance(value, str):
            value = [value]
        return [str(p) for p in value] if value else []

    def _add_recent(self, path: str) -> None:
        if not path:
            return
        if path in self.recent_projects:
            self.recent_projects.remove(path)
        self.recent_projects.insert(0, path)
        self.recent_projects = self.recent_projects[:MAX_RECENT_PROJECTS]
        self._settings.setValue("workspace/recent_projects", self.recent_projects)

    def clear_recent(self) -> None:
        self.recent_projects = []
        self._settings.setValue("workspace/recent_projects", [])
