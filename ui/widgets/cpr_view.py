"""Curved Planar Reformation (CPR) view for MedAxis.

:class:`CPRView` displays the straightened projection of a volume along an
arbitrary curve (e.g. a vessel centreline).  The CPR image is built by
sampling the volume along the curve with a ``vtkProbeFilter``; the view also
exposes the curve as polydata so a reference line can be drawn on linked MPR
views, and offers a slab-thickness adjustment.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QSlider, QVBoxLayout, QWidget,
)

from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkImageData, vtkPolyData
from vtkmodules.vtkFiltersCore import vtkProbeFilter
from vtkmodules.vtkFiltersGeneral import vtkSplineFilter
from vtkmodules.vtkRenderingCore import vtkImageActor, vtkRenderer
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.util import numpy_support

# Rendering back-ends (side-effect imports).
import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
import vtkmodules.vtkInteractionStyle  # noqa: F401


class CPRView(QWidget):
    """Straightened Curved Planar Reformation viewer.

    Signals
    -------
    path_changed(object)
        Emitted with the resampled centreline ``vtkPolyData`` whenever the
        curve changes (use this to draw the reference line on MPR views).
    thickness_changed(float)
        Emitted when the sampling thickness is adjusted (millimetres).
    """

    path_changed = Signal(object)
    thickness_changed = Signal(float)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("CPRView")

        self._image: Optional[vtkImageData] = None
        self._path: Optional[vtkPolyData] = None
        self._resampled_path: Optional[vtkPolyData] = None
        self._thickness_mm = 10.0
        self._num_samples = 256
        self._mode = "Straightened"

        # ------------------------------------------------------------- #
        # VTK rendering
        # ------------------------------------------------------------- #
        self._renderer = vtkRenderer()
        self._renderer.SetBackground(0.07, 0.07, 0.09)
        self._iren = QVTKRenderWindowInteractor(self)
        self._iren.Initialize()
        self._iren.GetRenderWindow().AddRenderer(self._renderer)

        self._cpr_image: Optional[vtkImageData] = None
        self._image_actor = vtkImageActor()
        self._renderer.AddActor(self._image_actor)

        # ------------------------------------------------------------- #
        # Controls
        # ------------------------------------------------------------- #
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Mode:", self))
        self._mode_combo = QComboBox(self)
        self._mode_combo.addItems(["Straightened", "Stretched"])
        self._mode_combo.currentTextChanged.connect(self._on_mode_changed)
        controls.addWidget(self._mode_combo)

        controls.addWidget(QLabel("Thickness:", self))
        self._thickness_slider = QSlider(Qt.Horizontal, self)
        self._thickness_slider.setRange(1, 60)
        self._thickness_slider.setValue(int(self._thickness_mm))
        self._thickness_slider.valueChanged.connect(self._on_thickness_changed)
        controls.addWidget(self._thickness_slider, 1)
        self._thickness_label = QLabel(f"{self._thickness_mm:.0f} mm", self)
        controls.addWidget(self._thickness_label)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._iren, 1)
        layout.addLayout(controls)

    # ================================================================== #
    # Public API
    # ================================================================== #
    def set_volume(self, volume) -> None:
        """Set the source volume (``vtkImageData`` or object with an
        ``image_data`` attribute)."""
        image = volume
        if not isinstance(volume, vtkImageData):
            image = getattr(volume, "image_data", None)
            if image is None:
                raise TypeError("volume must be vtkImageData or expose "
                                "'image_data'")
        self._image = image
        self.rebuild()

    def set_path(self, path: vtkPolyData) -> None:
        """Set the centreline curve used for the CPR."""
        self._path = path
        self.rebuild()

    def set_thickness(self, mm: float) -> None:
        """Adjust the perpendicular sampling thickness in millimetres."""
        self._thickness_mm = float(mm)
        self._thickness_slider.blockSignals(True)
        self._thickness_slider.setValue(int(mm))
        self._thickness_slider.blockSignals(False)
        self._thickness_label.setText(f"{mm:.0f} mm")
        self.rebuild()
        self.thickness_changed.emit(self._thickness_mm)

    def thickness(self) -> float:
        return self._thickness_mm

    def get_path_polydata(self) -> Optional[vtkPolyData]:
        """Resampled centreline (use to draw reference lines on MPR views)."""
        return self._resampled_path

    def attach_reference_view(self, view) -> None:
        """Draw the CPR reference line on another view.

        *view* is expected to expose ``add_polydata_overlay(polydata, color)``
        (e.g. :class:`SliceView` / :class:`MPRView`).
        """
        if self._resampled_path is not None and hasattr(
                view, "add_polydata_overlay"):
            view.add_polydata_overlay(self._resampled_path, (1.0, 0.8, 0.0))

    # ================================================================== #
    # CPR construction
    # ================================================================== #
    def rebuild(self) -> None:
        """Recompute and display the straightened CPR image."""
        if self._image is None or self._path is None:
            return

        # 1) Resample the curve at even spacing.
        spline = vtkSplineFilter()
        spline.SetInputData(self._path)
        spline.SetSubdivideToLength()
        spline.SetLength(1.0)
        spline.Update()
        resampled = spline.GetOutput()
        self._resampled_path = resampled
        n_pts = resampled.GetNumberOfPoints()
        if n_pts < 2:
            return
        self.path_changed.emit(resampled)

        points = np.array([resampled.GetPoint(i) for i in range(n_pts)])

        # 2) Tangents along the path.
        tangents = np.gradient(points, axis=0)
        norms = np.linalg.norm(tangents, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        tangents /= norms

        # 3) Stable perpendicular "up" direction along the path.
        up = np.array([0.0, 0.0, 1.0])
        normals = np.cross(tangents, up)
        degenerate = np.linalg.norm(normals, axis=1) < 1e-3
        normals[degenerate] = np.cross(tangents[degenerate],
                                       np.array([1.0, 0.0, 0.0]))
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normals /= norms

        # 4) Build the sampling grid: width along path, height across it.
        spacing = float(np.mean(self._image.GetSpacing()))
        height = max(int(self._thickness_mm / spacing), 2)
        width = self._num_samples

        # Sample positions (height*width points).
        offsets = (np.arange(height) - height / 2.0) * spacing
        grid = (points[:, None, :]
                + normals[:, None, :] * offsets[None, :, None])
        flat = grid.reshape(-1, 3)

        # 5) Probe the volume at every grid position.
        probe_points = vtkPoints()
        probe_points.SetNumberOfPoints(flat.shape[0])
        arr = numpy_support.numpy_to_vtk(flat, deep=True)
        probe_points.SetData(arr)
        probe_pd = vtkPolyData()
        probe_pd.SetPoints(probe_points)

        probe = vtkProbeFilter()
        probe.SetInputData(probe_pd)
        probe.SetSourceData(self._image)
        probe.Update()

        scalars = probe.GetOutput().GetPointData().GetScalars()
        if scalars is None:
            return
        values = numpy_support.vtk_to_numpy(scalars)
        if self._mode == "Stretched":
            # Stretched mode keeps per-sample arc-length spacing (identity).
            pass
        slab = values.reshape(width, height)

        # 6) Wrap into an image for display.
        cpr = vtkImageData()
        cpr.SetDimensions(width, height, 1)
        cpr.SetSpacing(1.0, 1.0, 1.0)
        cpr.SetOrigin(0.0, 0.0, 0.0)
        out_arr = numpy_support.numpy_to_vtk(slab.astype(np.float32).ravel(),
                                             deep=True)
        cpr.GetPointData().SetScalars(out_arr)
        self._cpr_image = cpr

        self._image_actor.GetMapper().SetInputData(cpr)
        lo, hi = cpr.GetScalarRange()
        self._image_actor.SetWindow(max(hi - lo, 1.0))
        self._image_actor.SetLevel((hi + lo) / 2.0)
        self._renderer.ResetCamera()
        self.render()

    def render(self) -> None:
        self._iren.GetRenderWindow().Render()

    # ------------------------------------------------------------------ #
    # Slots
    # ------------------------------------------------------------------ #
    def _on_thickness_changed(self, value: int) -> None:
        self._thickness_mm = float(value)
        self._thickness_label.setText(f"{value} mm")
        self.rebuild()
        self.thickness_changed.emit(self._thickness_mm)

    def _on_mode_changed(self, text: str) -> None:
        self._mode = text
        self.rebuild()

    def close(self) -> None:  # pragma: no cover - Qt lifecycle
        try:
            self._iren.close()
        except Exception:
            pass
        super().close()
