"""Multi-viewport container for MedAxis.

:class:`ViewContainer` arranges up to four medical viewports (coronal,
axial, sagittal :class:`SliceView` instances and a 3-D :class:`VolumeView`)
in a configurable grid and manages:

* layout modes: ``1up``, ``2up_h``, ``2up_v``, ``4up`` (default), ``1plus3``;
* active-viewport tracking with a blue highlight border;
* linked crosshair / scroll propagation between slice views.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from PySide6.QtCore import QEvent, Signal
from PySide6.QtWidgets import QGridLayout, QWidget

from ..widgets.slice_view import SliceView
from ..widgets.volume_view import VolumeView


class ViewContainer(QWidget):
    """Grid of medical viewports with layout modes and linked navigation.

    Signals
    -------
    active_view_changed(QWidget)
        Emitted with the viewport widget that became active.
    layout_changed(str)
        Emitted with the new layout mode name.
    """

    active_view_changed = Signal(QWidget)
    layout_changed = Signal(str)

    LAYOUT_MODES = ("1up", "2up_h", "2up_v", "4up", "1plus3")

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("ViewContainer")

        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(2, 2, 2, 2)
        self._grid.setSpacing(2)

        self._views: Dict[str, QWidget] = {}
        self._mode = "4up"
        self._active: Optional[QWidget] = None
        self._cursor: List[float] = [0.0, 0.0, 0.0]
        self._suppress_link = False

        self._build_views()
        self._apply_layout(self._mode)

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def _build_views(self) -> None:
        # Viewport roles -> widgets (spec: TL=coronal, TR=axial,
        # BL=sagittal, BR=3D).
        self._views["coronal"] = SliceView("coronal", self)
        self._views["axial"] = SliceView("axial", self)
        self._views["sagittal"] = SliceView("sagittal", self)
        self._views["3d"] = VolumeView(self)

        for name, view in self._views.items():
            view.installEventFilter(self)
            if isinstance(view, SliceView):
                view.crosshair_moved.connect(
                    lambda pos, v=view: self._on_crosshair_moved(v, pos))
                view.slice_changed.connect(
                    lambda idx, orientation, v=view:
                    self._on_slice_changed(v, idx, orientation))

        self.set_active_view(self._views["axial"])

    # ------------------------------------------------------------------ #
    # Layout management
    # ------------------------------------------------------------------ #
    def set_layout_mode(self, mode: str) -> None:
        """Switch layout: ``1up``, ``2up_h``, ``2up_v``, ``4up``, ``1plus3``."""
        if mode not in self.LAYOUT_MODES:
            raise ValueError(f"Unknown layout mode: {mode}")
        self._mode = mode
        self._apply_layout(mode)
        self.layout_changed.emit(mode)

    def layout_mode(self) -> str:
        return self._mode

    def _clear_grid(self) -> None:
        for view in self._views.values():
            self._grid.removeWidget(view)
            view.setParent(None)

    def _apply_layout(self, mode: str) -> None:
        self._clear_grid()
        v = self._views

        if mode == "1up":
            self._place({v["axial"]: (0, 0, 1, 1)})
        elif mode == "2up_h":
            self._place({
                v["axial"]: (0, 0, 1, 1),
                v["coronal"]: (0, 1, 1, 1),
            })
        elif mode == "2up_v":
            self._place({
                v["axial"]: (0, 0, 1, 1),
                v["sagittal"]: (1, 0, 1, 1),
            })
        elif mode == "1plus3":
            self._place({
                v["axial"]: (0, 0, 3, 1),       # big on the left
                v["coronal"]: (0, 1, 1, 1),
                v["sagittal"]: (1, 1, 1, 1),
                v["3d"]: (2, 1, 1, 1),
            })
        else:  # "4up" (default)
            self._place({
                v["coronal"]: (0, 0, 1, 1),
                v["axial"]: (0, 1, 1, 1),
                v["sagittal"]: (1, 0, 1, 1),
                v["3d"]: (1, 1, 1, 1),
            })

        # Hide any view not in the current arrangement.
        for view in self._views.values():
            if view.parent() is None:
                view.hide()

        # Keep the active view visible; otherwise activate the first shown.
        if self._active is None or self._active.parent() is None:
            for view in self._views.values():
                if view.parent() is not None:
                    self.set_active_view(view)
                    break

    def _place(self, placement: Dict[QWidget, tuple]) -> None:
        for view, (row, col, rspan, cspan) in placement.items():
            view.setParent(self)
            self._grid.addWidget(view, row, col, rspan, cspan)
            view.show()

    # ------------------------------------------------------------------ #
    # View access
    # ------------------------------------------------------------------ #
    def get_view(self, name: str) -> Optional[QWidget]:
        return self._views.get(name)

    def slice_views(self) -> List[SliceView]:
        return [v for v in self._views.values() if isinstance(v, SliceView)]

    def active_view(self) -> Optional[QWidget]:
        return self._active

    def active_slice_view(self) -> Optional[SliceView]:
        if isinstance(self._active, SliceView):
            return self._active
        for view in self.slice_views():
            if view.isVisible():
                return view
        return None

    def set_active_view(self, view: QWidget) -> None:
        """Highlight *view* as the active viewport (blue border)."""
        if view is self._active:
            return
        if self._active is not None:
            self._active.setObjectName("viewport")
            self._active.setStyleSheet("")
        self._active = view
        view.setObjectName("viewport-active")
        view.setStyleSheet(
            "QWidget#viewport-active { border: 2px solid #2f7fd1; }")
        self.active_view_changed.emit(view)

    # ------------------------------------------------------------------ #
    # Data distribution
    # ------------------------------------------------------------------ #
    def set_volume(self, volume) -> None:
        """Push a volume into every visible viewport."""
        for view in self._views.values():
            if isinstance(view, SliceView):
                view.set_volume(volume)
            else:
                view.set_volume(volume)
        # Centre the linked cursor.
        first = self.slice_views()[0]
        self._cursor = list(first.get_cursor())
        for view in self.slice_views():
            view.set_cursor(self._cursor)

    def attach_label_editor(self, editor) -> None:
        """Attach a label editor to every slice viewport."""
        for view in self.slice_views():
            view.attach_label_editor(editor)

    def reset_all_views(self) -> None:
        for view in self._views.values():
            view.reset_view()

    def to_state(self) -> dict:
        """Serialize navigation and display state for project persistence."""
        state = {"layout": self._mode, "cursor": list(self._cursor), "views": {}}
        for name, view in self._views.items():
            if isinstance(view, SliceView):
                state["views"][name] = {
                    "slice_index": view.slice_index(),
                    "window": view.get_window_level()[0],
                    "level": view.get_window_level()[1],
                    "zoom": view.zoom_factor(),
                }
        return state

    def restore_state(self, state: dict) -> None:
        """Restore state produced by :meth:`to_state` after a volume load."""
        if not state:
            return
        layout = state.get("layout")
        if layout in self.LAYOUT_MODES:
            self.set_layout_mode(layout)
        for name, values in state.get("views", {}).items():
            view = self._views.get(name)
            if not isinstance(view, SliceView):
                continue
            view.set_slice_index(int(values.get("slice_index", 0)), emit=False)
            view.set_window_level(float(values.get("window", 1.0)), float(values.get("level", 0.0)))
        cursor = state.get("cursor")
        if isinstance(cursor, (list, tuple)) and len(cursor) == 3:
            self._cursor = [float(value) for value in cursor]
            for view in self.slice_views():
                view.set_cursor(self._cursor, emit=False)

    # ------------------------------------------------------------------ #
    # Linked navigation
    # ------------------------------------------------------------------ #
    def _on_crosshair_moved(self, source: SliceView, pos) -> None:
        if self._suppress_link:
            return
        self._suppress_link = True
        try:
            self._cursor = [float(pos[0]), float(pos[1]), float(pos[2])]
            # Align every view's slice to pass through the cursor.
            for view in self.slice_views():
                if view is source:
                    view.set_cursor(self._cursor, emit=False)
                    continue
                axis = view.normal_axis()
                target_world = self._cursor[axis]
                self._move_view_to_world(view, axis, target_world)
                view.set_cursor(self._cursor, emit=False)
        finally:
            self._suppress_link = False

    def _on_slice_changed(self, source: SliceView, index: int,
                          _orientation: str) -> None:
        if self._suppress_link:
            return
        self._suppress_link = True
        try:
            axis = source.normal_axis()
            # The cursor follows the scrolled plane.
            extent = source._image.GetExtent() if source._image else (0,) * 6
            origin = source._image.GetOrigin() if source._image else (0,) * 3
            spacing = source._image.GetSpacing() if source._image else (1,) * 3
            self._cursor[axis] = (origin[axis]
                                  + (extent[2 * axis] + index) * spacing[axis])
            for view in self.slice_views():
                if view is source:
                    view.set_cursor(self._cursor, emit=False)
                    continue
                other_axis = view.normal_axis()
                self._move_view_to_world(view, other_axis,
                                         self._cursor[other_axis])
                view.set_cursor(self._cursor, emit=False)
        finally:
            self._suppress_link = False

    @staticmethod
    def _move_view_to_world(view: SliceView, axis: int,
                            world_pos: float) -> None:
        """Scroll *view* so its plane passes through *world_pos*."""
        if view._image is None:
            return
        origin = view._image.GetOrigin()
        spacing = view._image.GetSpacing()
        extent = view._image.GetExtent()
        index = round((world_pos - origin[axis]) / spacing[axis]) \
            - extent[2 * axis]
        view.set_slice_index(index, emit=False)

    # ------------------------------------------------------------------ #
    # Qt event filter: track the active viewport.
    # ------------------------------------------------------------------ #
    def eventFilter(self, watched, event) -> bool:  # noqa: N802
        if event.type() in (QEvent.MouseButtonPress, QEvent.FocusIn):
            if watched in self._views.values():
                self.set_active_view(watched)
        return super().eventFilter(watched, event)
