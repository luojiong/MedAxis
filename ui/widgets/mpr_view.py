"""Multi-Planar Reformation (MPR) view for MedAxis.

:class:`MPRView` extends :class:`SliceView` with an arbitrary (oblique)
reslice plane instead of the three canonical orientations.  It integrates a
``vtkResliceCursor`` (shared between linked views) and a
``vtkResliceCursorWidget`` for interactive plane adjustment, and provides
three-view crosshair linking helpers.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

from PySide6.QtCore import Signal

from vtkmodules.vtkInteractionWidgets import (
    vtkResliceCursor, vtkResliceCursorLineRepresentation, vtkResliceCursorWidget,
)

from .slice_view import SliceView

Vec3 = Tuple[float, float, float]


def _orthonormalize(normal: Sequence[float]) -> Tuple[Vec3, Vec3, Vec3]:
    """Build an orthonormal basis (row_x, row_y, normal) from *normal*."""
    nx, ny, nz = normal
    length = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
    n = (nx / length, ny / length, nz / length)

    # Pick a helper vector not parallel to the normal.
    helper = (0.0, 0.0, 1.0) if abs(n[2]) < 0.9 else (0.0, 1.0, 0.0)

    def cross(a, b):
        return (a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0])

    u = cross(helper, n)
    lu = math.sqrt(sum(c * c for c in u)) or 1.0
    u = tuple(c / lu for c in u)
    v = cross(n, u)
    return u, v, n


class MPRView(SliceView):
    """SliceView variant supporting arbitrary oblique reslice planes.

    Signals
    -------
    plane_changed(tuple, tuple)
        Emitted with ``(normal, point)`` whenever the reslice plane changes
        (user interaction or programmatic call).
    """

    plane_changed = Signal(tuple, tuple)

    def __init__(self, parent=None) -> None:
        super().__init__(orientation="axial", parent=parent)
        self.setObjectName("MPRView")

        self._reslice_cursor: Optional[vtkResliceCursor] = None
        self._cursor_widget: Optional[vtkResliceCursorWidget] = None
        self._plane_index = 0

    # ================================================================== #
    # Reslice cursor integration
    # ================================================================== #
    def set_reslice_cursor(self, cursor: vtkResliceCursor) -> None:
        """Share a ``vtkResliceCursor`` with linked MPR views.

        The cursor must already have an image set
        (``cursor.SetImage(image)``).
        """
        self._reslice_cursor = cursor

        if self._cursor_widget is None:
            rep = vtkResliceCursorLineRepresentation()
            widget = vtkResliceCursorWidget()
            widget.SetInteractor(self._iren)
            widget.SetRepresentation(rep)
            self._cursor_widget = widget
            widget.AddObserver("EndInteractionEvent",
                               self._on_cursor_interaction)
        self._cursor_widget.SetResliceCursor(cursor)
        self._cursor_widget.On()
        self._sync_from_cursor()

    def _on_cursor_interaction(self, _obj, _event) -> None:
        """User dragged the plane widget; resync and notify linked views."""
        self._sync_from_cursor()
        if self._reslice_cursor is not None:
            normal = self._reslice_cursor.GetNormal(self._plane_index)
            point = self._reslice_cursor.GetCenter()
            self.plane_changed.emit(tuple(normal), tuple(point))

    def _sync_from_cursor(self) -> None:
        if self._reslice_cursor is None:
            return
        plane = self._reslice_cursor.GetPlane(self._plane_index)
        self._axes_matrix.DeepCopy(plane)
        self._reslice.Update()
        self._update_crosshair_actor()
        self.render()

    # ================================================================== #
    # Plane control
    # ================================================================== #
    def set_plane(self, normal: Sequence[float], origin: Sequence[float],
                  emit: bool = True) -> None:
        """Orient the reslice plane to *normal* passing through *origin*."""
        u, v, n = _orthonormalize(normal)
        self._axes_matrix.Identity()
        for c in range(3):
            self._axes_matrix.SetElement(0, c, u[c])
            self._axes_matrix.SetElement(1, c, v[c])
            self._axes_matrix.SetElement(2, c, n[c])
            self._axes_matrix.SetElement(c, 3, origin[c])
        # vtkMatrix4x4 stores rows; set origin column properly.
        for r in range(3):
            self._axes_matrix.SetElement(r, 3, origin[r])

        if self._reslice_cursor is not None:
            self._reslice_cursor.SetNormal(*n)
            self._reslice_cursor.SetCenter(*origin)
            self._sync_from_cursor()
        else:
            self._reslice.Update()
            self._update_crosshair_actor()
            self.render()

        if emit:
            self.plane_changed.emit(tuple(n), tuple(origin))

    def get_plane(self) -> Tuple[Vec3, Vec3]:
        """Return the current ``(normal, point)`` of the reslice plane."""
        m = self._axes_matrix
        normal = (m.GetElement(2, 0), m.GetElement(2, 1), m.GetElement(2, 2))
        point = (m.GetElement(0, 3), m.GetElement(1, 3), m.GetElement(2, 3))
        return normal, point

    def set_plane_index(self, index: int) -> None:
        """Select which cursor plane (0/1/2) this view follows."""
        self._plane_index = int(index) % 3
        self._sync_from_cursor()

    # ------------------------------------------------------------------ #
    # Overridden navigation: move the cursor centre along the normal
    # ------------------------------------------------------------------ #
    def set_slice_index(self, index: int, emit: bool = True) -> None:
        if self._reslice_cursor is not None and self._image is not None:
            normal, point = self.get_plane()
            spacing = self._image.GetSpacing()
            center = list(self._reslice_cursor.GetCenter())
            step = (index - getattr(self, "_slice_index", index))
            for i in range(3):
                center[i] += normal[i] * step * spacing[0]
            self._reslice_cursor.SetCenter(*center)
            self._sync_from_cursor()
            self._slice_index = int(index)
            self._update_info_text()
            if emit:
                self.slice_changed.emit(index, "oblique")
        else:
            super().set_slice_index(index, emit)

    # ================================================================== #
    # Three-view crosshair linking
    # ================================================================== #
    @staticmethod
    def link_views(views: Sequence["MPRView"]) -> None:
        """Cross-link crosshairs between MPR views.

        When the user moves the crosshair in one view the others follow.
        """
        for view in views:
            for other in views:
                if other is not view:
                    try:
                        view.crosshair_moved.disconnect(
                            other._on_linked_crosshair)
                    except Exception:
                        pass
                    view.crosshair_moved.connect(other._on_linked_crosshair)

    @staticmethod
    def share_cursor(views: Sequence["MPRView"],
                     cursor: vtkResliceCursor) -> None:
        """Give every view in *views* the same ``vtkResliceCursor`` and
        assign each view to a different cursor plane."""
        for i, view in enumerate(views[:3]):
            view._plane_index = i
            view.set_reslice_cursor(cursor)
