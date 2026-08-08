"""Wire the scripts subsystem (console/editor/recorder/hooks) into the UI."""
from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import QAction

from scripts.action_recorder import recorder
from scripts.hook_system import bind_controller_signals
from scripts.script_manager import ScriptManager


def attach_scripts(controller, window) -> None:
    """Inject the medaxis API, hooks and recorder into the main window."""
    try:
        from scripts.medaxis_api import MedAxisAPI

        api = MedAxisAPI(controller)
        window.script_console.append_context({"medaxis": api, "app": window})
        window.script_editor.set_console(window.script_console)
        bind_controller_signals(controller)

        # Recorder listens to the same signals.
        controller.volume_loaded.connect(lambda v: recorder.on_volume_loaded(v))
        controller.label_created.connect(lambda lab: recorder.on_label_created(lab))

        window.script_manager = ScriptManager()
    except Exception:
        import logging

        logging.getLogger(__name__).exception("script subsystem attach failed")


def build_script_menu(window) -> None:
    """Add Script actions to the Tools menu (console toggle, editor, recorder)."""
    tools_menu = window._tools_menu
    if tools_menu is None:
        return

    tools_menu.addSeparator()

    editor_action = QAction("&Script Editor", window)
    editor_action.setCheckable(True)
    editor_action.setChecked(False)
    editor_action.toggled.connect(
        lambda on: window.script_editor_dock.setVisible(on))
    tools_menu.addAction(editor_action)

    record_action = QAction("Record Actions (macro)", window)
    record_action.setCheckable(True)
    record_action.toggled.connect(lambda on: _toggle_recorder(window, on))
    tools_menu.addAction(record_action)

    templates_menu = tools_menu.addMenu("Script &Templates")
    window._script_templates_menu = templates_menu
    window._script_editor_action = editor_action
    window._record_action = record_action


def populate_templates_menu(window) -> None:
    """Populate the Script Templates menu from the built-in library."""
    menu = getattr(window, "_script_templates_menu", None)
    if menu is None:
        return
    menu.clear()
    manager: Optional[ScriptManager] = getattr(window, "script_manager", None)
    if manager is None:
        return
    for name, code in manager.templates().items():
        action = QAction(name, window)
        action.triggered.connect(
            lambda _=False, n=name, c=code: window.script_editor.set_code(c))
        menu.addAction(action)


def _toggle_recorder(window, on: bool) -> None:
    if on:
        recorder.start()
        window.statusBar().showMessage("Recording actions…", 3000)
    else:
        script = recorder.stop()
        window.script_editor.set_code(script)
        window.statusBar().showMessage("Recording stopped; script in editor", 5000)
