"""Interactive label (mask) editor for MedAxis slice views.

:class:`LabelEditor` is embedded in a :class:`SliceView` and provides the
segmentation editing toolset:

Paint, Erase, Fill (flood fill), Lasso, Rectangle, Scissors and
Island-remove.  Brush size (mm), brush shape (Sphere / Box / Gaussian) and a
3-D depth (number of adjacent slices affected) are adjustable; *threshold
paint* restricts painting to voxels inside the current window/level range.

Every stroke is applied live to the label's voxel array for immediate
overlay feedback and then recorded as an :class:`EditLabelCommand` pushed to
the application ``QUndoStack``.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QUndoCommand
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QHBoxLayout, QLabel, QSlider,
    QToolButton, QVBoxLayout, QWidget,
)

TOOLS = ("pointer", "paint", "erase", "fill", "lasso", "rectangle",
         "scissors", "island_remove")

BRUSH_SHAPES = ("Sphere", "Box", "Gaussian")


class EditLabelCommand(QUndoCommand):
    """Undoable voxel edit on a label volume.

    Parameters
    ----------
    label:
        Label object exposing a 3-D numpy array as ``label.data``.
    flat_indices:
        Flat (C-order) indices into ``label.data`` that changed.
    old_values / new_values:
        Voxel values before / after the edit.
    already_applied:
        The editor applies edits live for instant feedback; when True the
        first ``redo()`` (triggered by ``QUndoStack.push``) is a no-op.
    """

    def __init__(self, label, flat_indices: np.ndarray,
                 old_values: np.ndarray, new_values: np.ndarray,
                 description: str = "Edit label",
                 already_applied: bool = True) -> None:
        super().__init__(description)
        self._label = label
        self._indices = np.asarray(flat_indices, dtype=np.int64)
        self._old = np.asarray(old_values)
        self._new = np.asarray(new_values)
        self._applied = already_applied

    # -- QUndoCommand interface ---------------------------------------- #
    def redo(self) -> None:
        if self._applied:
            self._applied = False
            return
        flat = np.asarray(self._label.data).ravel()
        flat[self._indices] = self._new
        self._notify()

    def undo(self) -> None:
        flat = np.asarray(self._label.data).ravel()
        flat[self._indices] = self._old
        self._notify()

    def _notify(self) -> None:
        refresh = getattr(self._label, "refresh", None)
        if callable(refresh):
            try:
                refresh()
            except Exception:
                pass


class LabelEditor(QWidget):
    """Label editing toolbar + interaction logic bound to a slice view.

    Signals
    -------
    label_modified(object)
        Emitted with the label object after any successful edit.
    tool_changed(str)
        Emitted when the active tool changes.
    """

    label_modified = Signal(object)
    tool_changed = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("LabelEditor")

        self._host = None                     # SliceView we edit in
        self._label = None                    # active LabelData
        self._tool = "pointer"
        self._undo_stack = None

        # Brush / stroke state.
        self._brush_mm = 5.0
        self._brush_shape = "Sphere"
        self._depth_slices = 1
        self._threshold_paint = False
        self._stoking = False
        self._pending: Dict[int, Tuple[int, int]] = {}
        self._last_voxel: Optional[Tuple[int, int, int]] = None
        self._rect_start: Optional[Tuple[int, int, int]] = None
        self._lasso_points: List[Tuple[int, int]] = []

        self._build_ui()

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        self._tool_buttons: Dict[str, QToolButton] = {}
        self._tool_group = QButtonGroup(self)
        self._tool_group.setExclusive(True)

        row1 = QHBoxLayout()
        for tool in TOOLS:
            btn = QToolButton(self)
            btn.setText(tool.replace("_", " ").title())
            btn.setCheckable(True)
            btn.setToolButtonStyle(Qt.ToolButtonTextOnly)
            self._tool_group.addButton(btn)
            self._tool_buttons[tool] = btn
            row1.addWidget(btn)
        row1.addStretch(1)
        self._tool_buttons["pointer"].setChecked(True)
        self._tool_group.buttonClicked.connect(self._on_tool_button)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Brush:", self))
        self._brush_slider = QSlider(Qt.Horizontal, self)
        self._brush_slider.setRange(1, 60)
        self._brush_slider.setValue(int(self._brush_mm))
        self._brush_slider.valueChanged.connect(
            lambda v: self._set_brush_mm(float(v)))
        row2.addWidget(self._brush_slider)
        self._brush_label = QLabel(f"{self._brush_mm:.0f} mm", self)
        row2.addWidget(self._brush_label)

        row2.addWidget(QLabel("Shape:", self))
        self._shape_combo = QComboBox(self)
        self._shape_combo.addItems(BRUSH_SHAPES)
        self._shape_combo.currentTextChanged.connect(self._on_shape_changed)
        row2.addWidget(self._shape_combo)

        row2.addWidget(QLabel("Depth:", self))
        self._depth_slider = QSlider(Qt.Horizontal, self)
        self._depth_slider.setRange(1, 9)
        self._depth_slider.setValue(self._depth_slices)
        self._depth_slider.setToolTip("Number of adjacent slices affected")
        self._depth_slider.valueChanged.connect(self._on_depth_changed)
        row2.addWidget(self._depth_slider)
        self._depth_label = QLabel("1", self)
        row2.addWidget(self._depth_label)

        self._threshold_check = QCheckBox("Threshold paint", self)
        self._threshold_check.setToolTip(
            "Only paint voxels within the current window/level range")
        self._threshold_check.toggled.connect(self._on_threshold_toggled)
        row2.addWidget(self._threshold_check)
        row2.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addLayout(row1)
        layout.addLayout(row2)

    # ------------------------------------------------------------------ #
    # Configuration
    # ------------------------------------------------------------------ #
    def set_host(self, view) -> None:
        """Bind to a :class:`SliceView` (called by
        ``SliceView.attach_label_editor``)."""
        self._host = view

    def set_undo_stack(self, stack) -> None:
        self._undo_stack = stack

    def set_label(self, label) -> None:
        """Set the active label to edit (needs ``.data`` numpy 3-D array)."""
        self._label = label

    def label(self):
        return self._label

    def set_tool(self, name: str) -> None:
        if name not in TOOLS:
            return
        self._tool = name
        btn = self._tool_buttons.get(name)
        if btn is not None and not btn.isChecked():
            btn.setChecked(True)
        if self._host is not None:
            if name in ("paint", "erase"):
                self._host.show_brush_cursor((0, 0, 0), self._brush_mm / 2)
            else:
                self._host.hide_brush_cursor()
        self.tool_changed.emit(name)

    def tool(self) -> str:
        return self._tool

    def consume_left_button(self) -> bool:
        """True when the current tool intercepts left mouse events."""
        return self._tool != "pointer"

    def stroke_active(self) -> bool:
        return self._stoking

    # ------------------------------------------------------------------ #
    # Interaction entry points (called by the host SliceView)
    # ------------------------------------------------------------------ #
    def on_left_press(self, x: int, y: int) -> None:
        if self._label is None or self._host is None:
            return
        voxel = self._host._display_to_voxel(x, y)
        if voxel is None:
            return

        self._pending = {}
        if self._tool in ("paint", "erase"):
            self._stoking = True
            self._stamp_at(voxel)
        elif self._tool in ("rectangle", "scissors"):
            self._stoking = True
            self._rect_start = voxel
        elif self._tool == "lasso":
            self._stoking = True
            self._lasso_points = [self._voxel_in_plane(voxel)]
        elif self._tool == "fill":
            self._flood_fill(voxel)
        elif self._tool == "island_remove":
            self._remove_island(voxel)

    def on_mouse_move(self, x: int, y: int) -> None:
        if not self._stoking or self._host is None:
            # Update brush cursor position even when not stroking.
            if self._host is not None and self._tool in ("paint", "erase"):
                world = self._host._display_to_world(x, y)
                self._host.show_brush_cursor(world, self._brush_mm / 2)
            return
        voxel = self._host._display_to_voxel(x, y)
        if voxel is None:
            return

        if self._tool in ("paint", "erase"):
            if self._last_voxel is not None:
                for v in self._interpolate(self._last_voxel, voxel):
                    self._stamp_at(v)
            else:
                self._stamp_at(voxel)
            self._last_voxel = voxel
        elif self._tool == "lasso":
            self._lasso_points.append(self._voxel_in_plane(voxel))

    def on_left_release(self, x: int, y: int) -> None:
        if not self._stoking:
            return
        self._stoking = False
        self._last_voxel = None

        if self._host is None:
            return
        if self._tool in ("rectangle", "scissors") and self._rect_start:
            voxel = self._host._display_to_voxel(x, y)
            if voxel is not None:
                value = 0 if self._tool == "scissors" else self._paint_value()
                self._apply_rectangle(self._rect_start, voxel, value)
            self._rect_start = None
        elif self._tool == "lasso" and len(self._lasso_points) >= 3:
            self._fill_polygon(self._lasso_points, self._paint_value())
            self._lasso_points = []

        self._commit()

    # ------------------------------------------------------------------ #
    # Editing primitives
    # ------------------------------------------------------------------ #
    def _paint_value(self) -> int:
        if self._tool == "erase":
            return 0
        value = getattr(self._label, "value", 1)
        return int(value) if value else 1

    def _set_voxel(self, ijk: Tuple[int, int, int], value: int) -> None:
        """Set one voxel, recording old/new for the undo command."""
        data = np.asarray(self._label.data)
        k, j, i = ijk[2], ijk[1], ijk[0]
        if not (0 <= k < data.shape[0] and 0 <= j < data.shape[1]
                and 0 <= i < data.shape[2]):
            return
        if self._threshold_paint and value != 0:
            vol = self._host.get_volume_array()
            if vol is not None:
                window, level = self._host.get_window_level()
                intensity = float(vol[k, j, i])
                if not (level - window / 2 <= intensity <= level + window / 2):
                    return
        old = int(data[k, j, i])
        if old == value:
            return
        data[k, j, i] = value
        flat_index = np.ravel_multi_index((k, j, i), data.shape)
        self._pending.setdefault(int(flat_index), (old, value))

    def _stamp_at(self, center: Tuple[int, int, int]) -> None:
        """Stamp the brush at *center* across the configured depth."""
        if self._host is None:
            return
        spacing = self._host.get_volume_spacing()
        axis = self._host.normal_axis()
        in_plane = [s for a, s in enumerate(spacing) if a != axis]
        mean_sp = float(np.mean(in_plane)) or 1.0
        radius_vox = max(self._brush_mm / (2.0 * mean_sp), 0.5)
        # Gaussian brush is effectively a tighter sphere.
        if self._brush_shape == "Gaussian":
            radius_vox *= 0.7

        value = self._paint_value()
        half_depth = self._depth_slices // 2
        r_int = int(np.ceil(radius_vox))

        for d_slice in range(-half_depth, half_depth + 1):
            c = list(center)
            c[axis] += d_slice
            for dj in range(-r_int, r_int + 1):
                for di in range(-r_int, r_int + 1):
                    dist2 = (di * di + dj * dj) ** 0.5
                    if self._brush_shape == "Sphere" or \
                            self._brush_shape == "Gaussian":
                        if dist2 > radius_vox:
                            continue
                    # Box accepts everything inside the bounding box.
                    planes = [a for a in range(3) if a != axis]
                    v = list(c)
                    v[planes[0]] += di
                    v[planes[1]] += dj
                    self._set_voxel((v[0], v[1], v[2]), value)
        self._refresh_overlay()

    def _apply_rectangle(self, start, end, value: int) -> None:
        axis = self._host.normal_axis()
        planes = [a for a in range(3) if a != axis]
        half_depth = self._depth_slices // 2
        for d_slice in range(-half_depth, half_depth + 1):
            c_axis = start[axis] + d_slice
            lo0, hi0 = sorted((start[planes[0]], end[planes[0]]))
            lo1, hi1 = sorted((start[planes[1]], end[planes[1]]))
            for a in range(lo0, hi0 + 1):
                for b in range(lo1, hi1 + 1):
                    v = [0, 0, 0]
                    v[axis] = c_axis
                    v[planes[0]] = a
                    v[planes[1]] = b
                    self._set_voxel((v[0], v[1], v[2]), value)
        self._refresh_overlay()

    def _voxel_in_plane(self, voxel) -> Tuple[int, int]:
        axis = self._host.normal_axis()
        planes = [a for a in range(3) if a != axis]
        return (voxel[planes[0]], voxel[planes[1]])

    def _fill_polygon(self, polygon: List[Tuple[int, int]], value: int) -> None:
        """Scanline fill of a closed polygon on the current slice."""
        axis = self._host.normal_axis()
        planes = [a for a in range(3) if a != axis]
        slice_index = self._host.slice_index()
        ext = self._host._image.GetExtent()

        # Convert slice coord along normal axis to array index.
        axis_index = ext[2 * axis] + slice_index

        rows = sorted({p[1] for p in polygon})
        if not rows:
            return
        n = len(polygon)
        for row in range(min(rows), max(rows) + 1):
            xs = []
            for idx in range(n):
                x0, y0 = polygon[idx]
                x1, y1 = polygon[(idx + 1) % n]
                if y0 == y1:
                    continue
                if not (min(y0, y1) <= row < max(y0, y1)):
                    continue
                xs.append(x0 + (row - y0) * (x1 - x0) / (y1 - y0))
            xs.sort()
            for pair in range(0, len(xs) - 1, 2):
                xa, xb = int(np.floor(xs[pair])), int(np.ceil(xs[pair + 1]))
                for col in range(xa, xb + 1):
                    v = [0, 0, 0]
                    v[axis] = axis_index
                    v[planes[0]] = col
                    v[planes[1]] = row
                    self._set_voxel((v[0], v[1], v[2]), value)
        self._refresh_overlay()

    # -- 2-D slice helpers --------------------------------------------- #
    def _slice_array(self):
        """Return (2-D numpy view of current slice, in-plane axes)."""
        axis = self._host.normal_axis()
        planes = [a for a in range(3) if a != axis]
        data = np.asarray(self._label.data)
        ext = self._host._image.GetExtent()
        axis_index = ext[2 * axis] + self._host.slice_index()
        if axis == 2:
            return data[axis_index], planes
        if axis == 1:
            return data[:, axis_index, :], planes
        return data[:, :, axis_index], planes

    def _plane_to_voxel(self, r: int, c: int) -> Tuple[int, int, int]:
        axis = self._host.normal_axis()
        planes = [a for a in range(3) if a != axis]
        ext = self._host._image.GetExtent()
        v = [0, 0, 0]
        v[axis] = ext[2 * axis] + self._host.slice_index()
        v[planes[0]] = c
        v[planes[1]] = r
        return (v[0], v[1], v[2])

    def _flood_fill(self, seed: Tuple[int, int, int]) -> None:
        """Flood-fill the 2-D connected region under *seed*."""
        slice_arr, planes = self._slice_array()
        r = seed[planes[1]]
        c = seed[planes[0]]
        if not (0 <= r < slice_arr.shape[0] and 0 <= c < slice_arr.shape[1]):
            return
        target = int(slice_arr[r, c])
        value = self._paint_value()
        if target == value:
            return
        h, w = slice_arr.shape
        queue = deque([(r, c)])
        seen = {(r, c)}
        while queue:
            cr, cc = queue.popleft()
            self._set_voxel(self._plane_to_voxel(cr, cc), value)
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen \
                        and int(slice_arr[nr, nc]) == target:
                    seen.add((nr, nc))
                    queue.append((nr, nc))
        self._refresh_overlay()
        self._commit()

    def _remove_island(self, seed: Tuple[int, int, int]) -> None:
        """Remove the connected island (non-zero region) under *seed*."""
        slice_arr, planes = self._slice_array()
        r = seed[planes[1]]
        c = seed[planes[0]]
        if not (0 <= r < slice_arr.shape[0] and 0 <= c < slice_arr.shape[1]):
            return
        target = int(slice_arr[r, c])
        if target == 0:
            return
        h, w = slice_arr.shape
        queue = deque([(r, c)])
        seen = {(r, c)}
        while queue:
            cr, cc = queue.popleft()
            self._set_voxel(self._plane_to_voxel(cr, cc), 0)
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen \
                        and int(slice_arr[nr, nc]) == target:
                    seen.add((nr, nc))
                    queue.append((nr, nc))
        self._refresh_overlay()
        self._commit()

    # ------------------------------------------------------------------ #
    # Commit / undo
    # ------------------------------------------------------------------ #
    def _commit(self) -> None:
        if not self._pending or self._label is None:
            self._pending = {}
            return
        indices = np.array(list(self._pending.keys()), dtype=np.int64)
        old = np.array([v[0] for v in self._pending.values()])
        new = np.array([v[1] for v in self._pending.values()])
        self._pending = {}

        command = EditLabelCommand(
            self._label, indices, old, new,
            description=f"{self._tool.title()} on "
                        f"{getattr(self._label, 'name', 'label')}",
            already_applied=True)
        if self._undo_stack is not None:
            self._undo_stack.push(command)
        self.label_modified.emit(self._label)

    def _refresh_overlay(self) -> None:
        if self._host is not None and self._label is not None:
            self._host.refresh_overlay(self._label)

    @staticmethod
    def _interpolate(a: Tuple[int, int, int], b: Tuple[int, int, int]):
        """Yield voxels along the segment a->b (inclusive) for gap-free
        brush strokes."""
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        dz = b[2] - a[2]
        steps = int(max(abs(dx), abs(dy), abs(dz))) or 1
        for s in range(steps + 1):
            t = s / steps
            yield (round(a[0] + dx * t), round(a[1] + dy * t),
                   round(a[2] + dz * t))

    # ------------------------------------------------------------------ #
    # UI slots
    # ------------------------------------------------------------------ #
    def _on_tool_button(self, button) -> None:
        for name, btn in self._tool_buttons.items():
            if btn is button:
                self.set_tool(name)
                break

    def _set_brush_mm(self, mm: float) -> None:
        self._brush_mm = mm
        self._brush_label.setText(f"{mm:.0f} mm")

    def _on_shape_changed(self, text: str) -> None:
        self._brush_shape = text

    def _on_depth_changed(self, value: int) -> None:
        self._depth_slices = int(value)
        self._depth_label.setText(str(value))

    def _on_threshold_toggled(self, checked: bool) -> None:
        self._threshold_paint = bool(checked)
