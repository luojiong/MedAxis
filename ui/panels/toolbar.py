"""Main toolbar for the MedAxis imaging platform.

Provides :class:`MedAxisToolBar`, a :class:`QToolBar` populated with the
standard tool buttons (file, edit, interaction, annotation and navigation
tools).  Every button is backed by a :class:`QAction` so that it can also be
placed in menus and driven by keyboard shortcuts.

The toolbar is intentionally decoupled from any concrete controller: call
:meth:`MedAxisToolBar.attach_controller` once an ``AppController`` is
available and the relevant actions will be wired to controller slots.
"""

from __future__ import annotations

from typing import Dict, Optional

from PySide6.QtCore import Signal
from PySide6.QtGui import QAction, QActionGroup, QIcon, QKeySequence
from PySide6.QtWidgets import QToolBar, QWidget


class MedAxisToolBar(QToolBar):
    """Toolbar carrying all primary MedAxis tools.

    Signals
    -------
    tool_changed(str)
        Emitted whenever the active interaction tool changes.  The payload is
        one of ``"pointer"``, ``"paint"``, ``"erase"``, ``"fill"``,
        ``"distance"``, ``"angle"``, ``"zoom"``, ``"pan"``.
    """

    tool_changed = Signal(str)

    #: Logical tool names in display order (used to build button groups).
    TOOL_NAMES = (
        "open", "save", "undo", "redo",
        "pointer", "paint", "erase", "fill",
        "distance", "angle",
        "zoom", "pan", "reset",
    )

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("MedAxis Tools", parent)
        self.setObjectName("MedAxisToolBar")
        self.setMovable(False)
        self.setIconSize(self.iconSize())

        self.actions_by_name: Dict[str, QAction] = {}
        self._tool_group = QActionGroup(self)
        self._tool_group.setExclusive(True)

        self._build_actions()
        self._layout_actions()

        # Default tool.
        self.actions_by_name["pointer"].setChecked(True)

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    def _make_action(
        self,
        name: str,
        text: str,
        checkable: bool = False,
        shortcut: Optional[str] = None,
        tooltip: Optional[str] = None,
    ) -> QAction:
        action = QAction(text, self)
        action.setObjectName(f"action_{name}")
        action.setCheckable(checkable)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        action.setToolTip(tooltip or text)
        if checkable:
            self._tool_group.addAction(action)
        self.actions_by_name[name] = action
        return action

    def _build_actions(self) -> None:
        a = self.actions_by_name
        a["open"] = self._make_action("open", "Open", shortcut="Ctrl+O",
                                      tooltip="Open DICOM / image data")
        a["save"] = self._make_action("save", "Save", shortcut="Ctrl+S",
                                      tooltip="Save current session")
        a["undo"] = self._make_action("undo", "Undo", shortcut="Ctrl+Z")
        a["redo"] = self._make_action("redo", "Redo", shortcut="Ctrl+Y")

        # Interaction tools (exclusive group).
        a["pointer"] = self._make_action("pointer", "Pointer", checkable=True,
                                         shortcut="Esc",
                                         tooltip="Select / navigate")
        a["paint"] = self._make_action("paint", "Paint", checkable=True,
                                       shortcut="P", tooltip="Paint label")
        a["erase"] = self._make_action("erase", "Erase", checkable=True,
                                       shortcut="E", tooltip="Erase label")
        a["fill"] = self._make_action("fill", "Fill", checkable=True,
                                      shortcut="F", tooltip="Flood fill label")

        # Measurement tools (exclusive with the rest).
        a["distance"] = self._make_action("distance", "Distance", checkable=True,
                                          shortcut="D", tooltip="Measure distance")
        a["angle"] = self._make_action("angle", "Angle", checkable=True,
                                       shortcut="A", tooltip="Measure angle")

        # Navigation tools.
        a["zoom"] = self._make_action("zoom", "Zoom", checkable=True,
                                      shortcut="Z", tooltip="Zoom tool")
        a["pan"] = self._make_action("pan", "Pan", checkable=True,
                                     tooltip="Pan tool")
        a["reset"] = self._make_action("reset", "Reset", shortcut="Home",
                                       tooltip="Reset all views")

        # Emit tool_changed whenever an exclusive tool is toggled on.
        for name in ("pointer", "paint", "erase", "fill", "distance",
                     "angle", "zoom", "pan"):
            a[name].toggled.connect(lambda checked, n=name:
                                    checked and self.tool_changed.emit(n))
        a["reset"].triggered.connect(lambda: self.tool_changed.emit("reset"))

    def _layout_actions(self) -> None:
        a = self.actions_by_name
        self.addAction(a["open"])
        self.addAction(a["save"])
        self.addAction(a["undo"])
        self.addAction(a["redo"])
        self.addSeparator()
        self.addAction(a["pointer"])
        self.addAction(a["paint"])
        self.addAction(a["erase"])
        self.addAction(a["fill"])
        self.addSeparator()
        self.addAction(a["distance"])
        self.addAction(a["angle"])
        self.addSeparator()
        self.addAction(a["zoom"])
        self.addAction(a["pan"])
        self.addAction(a["reset"])

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def set_tool(self, name: str) -> None:
        """Programmatically activate the tool called *name*."""
        action = self.actions_by_name.get(name)
        if action is not None and action.isCheckable():
            action.setChecked(True)
            self.tool_changed.emit(name)

    def current_tool(self) -> str:
        checked = self._tool_group.checkedAction()
        if checked is None:
            return "pointer"
        return checked.objectName().removeprefix("action_")

    def set_icons(self, icons: Dict[str, QIcon]) -> None:
        """Apply a mapping of tool name -> QIcon."""
        for name, icon in icons.items():
            action = self.actions_by_name.get(name)
            if action is not None:
                action.setIcon(icon)

    def attach_controller(self, controller) -> None:
        """Wire toolbar actions to an ``AppController``.

        Connections are guarded so that a partially-implemented controller
        does not break the UI.
        """
        def _connect(attr, signal_or_slot, is_slot=True):
            target = getattr(controller, attr, None)
            if target is None:
                return
            if is_slot:
                signal_or_slot.connect(target)

        a = self.actions_by_name
        _connect("open_file", a["open"].triggered)
        _connect("save_file", a["save"].triggered)
        _connect("undo", a["undo"].triggered)
        _connect("redo", a["redo"].triggered)
        _connect("reset_views", a["reset"].triggered)
        _connect("set_tool", self.tool_changed)

        # Keep toolbar state in sync with controller-driven tool changes.
        notify = getattr(controller, "tool_changed", None)
        if notify is not None:
            try:
                notify.connect(self.set_tool)
            except Exception:
                pass
