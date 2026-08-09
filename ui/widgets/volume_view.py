"""3D volume rendering view for MedAxis.

:class:`VolumeView` embeds a :class:`QVTKRenderWindowInteractor` and renders
a volume with ``vtkGPUVolumeRayCastMapper``.  It ships with transfer-function
presets, an orientation cube + XYZ axis indicator, mesh overlay support and a
small transfer-function editor dialog.

Interaction model
-----------------
======================  =======================================
Input                   Action
======================  =======================================
Left button drag        Rotate
Middle button drag      Pan
Right button drag       Zoom
Scroll wheel            Zoom
R                       Reset camera
======================  =======================================
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDialog, QFormLayout, QComboBox, QDialogButtonBox, QSlider, QVBoxLayout, QWidget, QDoubleSpinBox,
)
from PySide6.QtCore import Qt

from vtkmodules.vtkCommonDataModel import (
    vtkImageData, vtkPiecewiseFunction, vtkPolyData,
)
from vtkmodules.vtkFiltersCore import vtkFlyingEdges3D
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkRenderingAnnotation import vtkAnnotatedCubeActor, vtkAxesActor
from vtkmodules.vtkRenderingCore import (
    vtkActor, vtkColorTransferFunction, vtkPolyDataMapper, vtkPropAssembly,
    vtkRenderer, vtkVolume, vtkVolumeProperty,
)
from vtkmodules.vtkInteractionWidgets import vtkOrientationMarkerWidget
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

# Rendering back-ends (side-effect imports).
import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
import vtkmodules.vtkRenderingVolumeOpenGL2  # noqa: F401
import vtkmodules.vtkInteractionStyle  # noqa: F401

try:
    from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkGPUVolumeRayCastMapper
except ImportError:  # pragma: no cover - very old VTK builds
    from vtkmodules.vtkRenderingVolume import \
        vtkFixedPointVolumeRayCastMapper as vtkGPUVolumeRayCastMapper


# ---------------------------------------------------------------------- #
# Transfer-function presets
# ---------------------------------------------------------------------- #
def _build_preset(name: str, lo: float, hi: float,
                  opacity_scale: float = 1.0
                  ) -> Tuple[vtkColorTransferFunction, vtkPiecewiseFunction]:

    color = vtkColorTransferFunction()
    opacity = vtkPiecewiseFunction()
    span = max(hi - lo, 1e-6)

    def t(frac: float) -> float:
        return lo + frac * span

    if name == "CT Bone":
        color.AddRGBPoint(t(0.00), 0.0, 0.0, 0.0)
        color.AddRGBPoint(t(0.35), 0.30, 0.10, 0.05)
        color.AddRGBPoint(t(0.60), 0.90, 0.85, 0.60)
        color.AddRGBPoint(t(1.00), 1.00, 1.00, 0.95)
        opacity.AddPoint(t(0.00), 0.00)
        opacity.AddPoint(t(0.30), 0.00)
        opacity.AddPoint(t(0.55), 0.15 * opacity_scale)
        opacity.AddPoint(t(1.00), 0.85 * opacity_scale)
    elif name == "CT Soft Tissue":
        color.AddRGBPoint(t(0.00), 0.0, 0.0, 0.0)
        color.AddRGBPoint(t(0.30), 0.55, 0.25, 0.15)
        color.AddRGBPoint(t(0.70), 0.90, 0.60, 0.45)
        color.AddRGBPoint(t(1.00), 1.00, 0.95, 0.85)
        opacity.AddPoint(t(0.00), 0.00)
        opacity.AddPoint(t(0.25), 0.00)
        opacity.AddPoint(t(0.60), 0.25 * opacity_scale)
        opacity.AddPoint(t(1.00), 0.70 * opacity_scale)
    elif name == "CT Lung":
        color.AddRGBPoint(t(0.00), 0.1, 0.1, 0.3)
        color.AddRGBPoint(t(0.50), 0.6, 0.6, 0.8)
        color.AddRGBPoint(t(1.00), 1.0, 1.0, 1.0)
        opacity.AddPoint(t(0.00), 0.00)
        opacity.AddPoint(t(0.40), 0.05 * opacity_scale)
        opacity.AddPoint(t(1.00), 0.60 * opacity_scale)
    else:  # Grayscale
        color.AddRGBPoint(t(0.00), 0.0, 0.0, 0.0)
        color.AddRGBPoint(t(1.00), 1.0, 1.0, 1.0)
        opacity.AddPoint(t(0.00), 0.00)
        opacity.AddPoint(t(0.50), 0.20 * opacity_scale)
        opacity.AddPoint(t(1.00), 0.80 * opacity_scale)
    return color, opacity


PRESET_NAMES = ("CT Bone", "CT Soft Tissue", "CT Lung", "Grayscale")


class TransferFunctionDialog(QDialog):
    """Small dialog to pick a TF preset, scalar range and opacity scale.

    Signals via :meth:`result`: call :meth:`values` after ``exec_`` returns
    ``QDialog.Accepted``.
    """

    def __init__(self, scalar_range: Tuple[float, float] = (0.0, 1.0),
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Transfer Function")

        form = QFormLayout(self)

        self._preset = QComboBox(self)
        self._preset.addItems(PRESET_NAMES)
        form.addRow("Preset:", self._preset)

        self._lo = QDoubleSpinBox(self)
        self._lo.setRange(-1e9, 1e9)
        self._lo.setDecimals(1)
        self._lo.setValue(scalar_range[0])
        form.addRow("Range min:", self._lo)

        self._hi = QDoubleSpinBox(self)
        self._hi.setRange(-1e9, 1e9)
        self._hi.setDecimals(1)
        self._hi.setValue(scalar_range[1])
        form.addRow("Range max:", self._hi)

        self._opacity = QSlider(Qt.Horizontal, self)
        self._opacity.setRange(10, 300)
        self._opacity.setValue(100)
        form.addRow("Opacity %:", self._opacity)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def values(self) -> Tuple[str, float, float, float]:
        """Return ``(preset_name, lo, hi, opacity_scale)``."""
        return (self._preset.currentText(), self._lo.value(),
                self._hi.value(), self._opacity.value() / 100.0)


class VolumeView(QWidget):
    """GPU volume-rendered 3D view.

    Signals
    -------
    mesh_count_changed(int)
        Number of mesh overlays currently shown.
    """

    mesh_count_changed = Signal(int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("VolumeView")
        self.setFocusPolicy(Qt.StrongFocus)

        self._image: Optional[vtkImageData] = None
        self._meshes: Dict[str, vtkActor] = {}

        # ------------------------------------------------------------- #
        # VTK pipeline
        # ------------------------------------------------------------- #
        self._renderer = vtkRenderer()
        self._renderer.SetBackground(0.0, 0.0, 0.0)

        self._iren = QVTKRenderWindowInteractor(self)
        self._iren.setStyleSheet("background-color: #000000;")
        self._iren.Initialize()
        self._style = vtkInteractorStyleTrackballCamera()
        self._iren.SetInteractorStyle(self._style)
        self._iren.GetRenderWindow().AddRenderer(self._renderer)

        self._mapper = vtkGPUVolumeRayCastMapper()
        self._mapper.SetBlendModeToComposite()
        if hasattr(self._mapper, "SetAutoAdjustSampleDistances"):
            self._mapper.SetAutoAdjustSampleDistances(True)

        self._volume = vtkVolume()
        self._volume.SetMapper(self._mapper)
        self._volume_property = vtkVolumeProperty()
        self._volume_property.SetInterpolationTypeToLinear()
        self._volume_property.ShadeOn()
        self._volume_property.SetAmbient(0.2)
        self._volume_property.SetDiffuse(0.7)
        self._volume_property.SetSpecular(0.2)
        self._volume.SetProperty(self._volume_property)
        self._renderer.AddVolume(self._volume)

        self._build_orientation_marker()
        self._install_observers()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._iren)
        self.setStyleSheet("background-color: #000000;")

    # ------------------------------------------------------------------ #
    def _build_orientation_marker(self) -> None:
        cube = vtkAnnotatedCubeActor()
        cube.SetXPlusFaceText("R")
        cube.SetXMinusFaceText("L")
        cube.SetYPlusFaceText("A")
        cube.SetYMinusFaceText("P")
        cube.SetZPlusFaceText("S")
        cube.SetZMinusFaceText("I")
        cube.SetFaceTextScale(0.65)

        axes = vtkAxesActor()
        axes.SetShaftTypeToCylinder()
        axes.AxisLabelsOff()

        assembly = vtkPropAssembly()
        assembly.AddPart(cube)
        assembly.AddPart(axes)

        self._orientation_widget = vtkOrientationMarkerWidget()
        self._orientation_widget.SetOrientationMarker(assembly)
        self._orientation_widget.SetInteractor(self._iren)
        self._orientation_widget.SetViewport(0.0, 0.0, 0.28, 0.28)
        self._orientation_widget.On()
        self._orientation_widget.InteractiveOff()

    def _install_observers(self) -> None:
        self._iren.AddObserver("KeyPressEvent", self._on_key_press)

    def _on_key_press(self, _obj, _event) -> None:
        key = self._iren.GetKeySym() or ""
        if key.lower() == "r":
            self.reset_view()

    # ================================================================== #
    # Public API
    # ================================================================== #
    def set_volume(self, volume) -> None:
        """Set the volume to render (``vtkImageData`` or object with
        ``image_data`` attribute)."""
        image = volume
        if not isinstance(volume, vtkImageData):
            image = getattr(volume, "image_data", None)
            if image is None:
                raise TypeError("volume must be vtkImageData or expose "
                                "'image_data'")
        self._image = image
        self._mapper.SetInputData(image)
        lo, hi = image.GetScalarRange()
        color, opacity = _build_preset("CT Bone", lo, hi)
        self.set_transfer_function(color, opacity)
        self.reset_view()
        self.render()

    def set_transfer_function(self, color_tf: vtkColorTransferFunction,
                              opacity_tf) -> None:
        """Apply colour and opacity transfer functions to the volume."""
        self._volume_property.SetColor(color_tf)
        self._volume_property.SetScalarOpacity(opacity_tf)
        # Keep python references alive (VTK holds raw pointers only).
        self._color_tf = color_tf
        self._opacity_tf = opacity_tf
        self._volume_property.Modified()
        self.render()

    def apply_preset(self, name: str, lo: Optional[float] = None,
                     hi: Optional[float] = None,
                     opacity_scale: float = 1.0) -> None:
        """Apply one of the built-in presets over the given scalar range."""
        if self._image is None:
            return
        if lo is None or hi is None:
            lo, hi = self._image.GetScalarRange()
        color, opacity = _build_preset(name, lo, hi, opacity_scale)
        self.set_transfer_function(color, opacity)

    def edit_transfer_function(self) -> None:
        """Open the transfer function editor dialog (blocking)."""
        rng = self._image.GetScalarRange() if self._image else (0.0, 1.0)
        dialog = TransferFunctionDialog(rng, self)
        if dialog.exec() == QDialog.Accepted:
            preset, lo, hi, scale = dialog.values()
            self.apply_preset(preset, lo, hi, scale)

    def add_mesh(self, polydata: vtkPolyData, name: Optional[str] = None,
                 color=(0.9, 0.85, 0.2), opacity: float = 1.0) -> vtkActor:
        """Overlay a surface mesh; returns the actor."""
        name = name or f"mesh_{len(self._meshes)}"
        mapper = vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetOpacity(opacity)
        self._renderer.AddActor(actor)
        self._meshes[name] = actor
        self.mesh_count_changed.emit(len(self._meshes))
        self.render()
        return actor

    def add_label_overlay(self, label, color=None, opacity: float = 0.85) -> vtkActor:
        """Overlay a label volume as a colored iso-surface (flying edges)."""
        import numpy as np
        from vtkmodules.util.numpy_support import numpy_to_vtk

        if color is None:
            color = getattr(label, "color", None)
        if isinstance(color, (tuple, list)):
            rgba = tuple(float(c) for c in color)
            color = (rgba[0], rgba[1], rgba[2]) if len(rgba) >= 3 else (1.0, 0.0, 0.0)
        else:
            color = (1.0, 0.0, 0.0)

        arr = np.asarray(label.array if hasattr(label, "array") else label)
        vtk_data = numpy_to_vtk((arr > 0).astype(np.uint8).ravel(), deep=True)
        vtk_data.SetNumberOfComponents(1)
        img = vtkImageData()
        img.SetDimensions(*reversed(arr.shape))
        img.GetPointData().SetScalars(vtk_data)

        fe = vtkFlyingEdges3D()
        fe.SetInputData(img)
        fe.SetValue(0, 0.5)
        fe.ComputeNormalsOn()
        fe.Update()

        mapper = vtkPolyDataMapper()
        mapper.SetInputData(fe.GetOutput())
        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetOpacity(opacity)
        self._renderer.AddActor(actor)
        self._meshes[f"label_{getattr(label, 'name', '')}"] = actor
        self.render()
        return actor

    def remove_mesh(self, name_or_actor) -> None:
        """Remove a mesh overlay by name or by actor reference."""
        actor = None
        if isinstance(name_or_actor, str):
            actor = self._meshes.pop(name_or_actor, None)
        else:
            for key, value in list(self._meshes.items()):
                if value is name_or_actor:
                    actor = self._meshes.pop(key)
                    break
        if actor is not None:
            self._renderer.RemoveActor(actor)
            self.mesh_count_changed.emit(len(self._meshes))
            self.render()

    def clear_meshes(self) -> None:
        for actor in self._meshes.values():
            self._renderer.RemoveActor(actor)
        self._meshes.clear()
        self.mesh_count_changed.emit(0)
        self.render()

    def reset_view(self) -> None:
        self._renderer.ResetCamera()
        self.render()

    def render(self) -> None:
        self._iren.GetRenderWindow().Render()

    def close(self) -> None:  # pragma: no cover - Qt lifecycle
        try:
            self._iren.close()
        except Exception:
            pass
        super().close()
