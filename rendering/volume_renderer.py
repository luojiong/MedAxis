"""VolumeRenderer — GPU ray-cast 3D rendering with transfer functions."""
from __future__ import annotations

import logging
from typing import Optional

from core.volume_data import VolumeData

try:
    import vtk

    HAS_VTK = True
except ImportError:  # pragma: no cover
    vtk = None  # type: ignore[assignment]
    HAS_VTK = False

from .helpers import numpy_to_vtk_image, volume_user_matrix
from .transfer_function import PRESETS, TransferFunction

logger = logging.getLogger(__name__)

__all__ = ["VolumeRenderer"]


def _require_vtk() -> None:
    if not HAS_VTK:
        raise RuntimeError("VTK is required for VolumeRenderer")


class VolumeRenderer:
    """Manage a ray-cast volume rendering for one ``vtkRenderer``.

    Supports transfer-function presets (CT-Default, CT-Bone, CT-Angio,
    CT-Lung, MR-Default), custom opacity/color points and shading tweaks.
    """

    def __init__(self) -> None:
        self.volume: Optional[VolumeData] = None
        self._renderer = None
        self._image_data = None
        self._mapper = None
        self._property = None
        self._actor = None
        self._visible = True

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def setup(self, renderer, volume: VolumeData,
              preset: str = "CT-Default") -> None:
        """Create the mapper/property/actor and add them to the renderer.

        Args:
            renderer: Target ``vtkRenderer``.
            volume: Volume to render.
            preset: Initial transfer-function preset name.
        """
        _require_vtk()
        self.volume = volume
        self._renderer = renderer

        self._image_data = numpy_to_vtk_image(
            volume.array, volume.spacing_mm, volume.origin_mm
        )

        self._mapper = vtk.vtkGPUVolumeRayCastMapper()
        self._mapper.SetInputData(self._image_data)
        self._mapper.SetImageSampleDistance(1.0)
        self._mapper.SetSampleDistance(0.8)
        self._mapper.SetAutoAdjustSampleDistances(True)

        if not self._mapper.IsRenderSupported(
                self._image_data.GetPointData().GetScalars().GetDataType(), 1):
            logger.warning("GPU ray cast unsupported; falling back to smart mapper")
            smart = vtk.vtkSmartVolumeMapper()
            smart.SetInputData(self._image_data)
            self._mapper = smart

        self._property = vtk.vtkVolumeProperty()
        self._property.SetInterpolationTypeToLinear()
        self._property.ShadeOn()
        self._property.SetAmbient(0.2)
        self._property.SetDiffuse(0.7)
        self._property.SetSpecular(0.2)
        self._property.SetSpecularPower(10.0)

        self._actor = vtk.vtkVolume()
        self._actor.SetMapper(self._mapper)
        self._actor.SetProperty(self._property)
        self._actor.SetUserMatrix(volume_user_matrix(volume))

        renderer.AddVolume(self._actor)
        self.apply_preset(preset)

    def teardown(self) -> None:
        """Remove the volume actor from the renderer."""
        if self._renderer is not None and self._actor is not None:
            self._renderer.RemoveVolume(self._actor)
        self._renderer = None
        self._actor = None

    # ------------------------------------------------------------------
    # Transfer functions
    # ------------------------------------------------------------------
    def set_transfer_function(
        self,
        opacity_points: list[tuple[float, float]],
        color_points: list[tuple[float, tuple[float, float, float]]],
    ) -> None:
        """Apply custom opacity and color points.

        Args:
            opacity_points: ``(value, opacity)`` pairs, opacity in [0, 1].
            color_points: ``(value, (r, g, b))`` pairs, channels in [0, 1].
        """
        if self._property is None:
            raise RuntimeError("VolumeRenderer.setup() not called")

        tf = TransferFunction(name="Custom",
                              opacity_points=opacity_points,
                              color_points=color_points)
        opacity_func, color_func = tf.to_vtk()
        self._property.SetScalarOpacity(opacity_func)
        self._property.SetColor(color_func)
        # Keep the VTK objects alive.
        self._tf = tf
        self._opacity_func = opacity_func
        self._color_func = color_func
        self._render()

    def apply_preset(self, name: str) -> None:
        """Apply one of the built-in presets by name."""
        if name not in PRESETS:
            raise KeyError(f"Unknown transfer-function preset: {name}. "
                           f"Available: {sorted(PRESETS)}")
        tf: TransferFunction = PRESETS[name]()
        self.set_transfer_function(tf.opacity_points, tf.color_points)

    @staticmethod
    def available_presets() -> list[str]:
        """Names of built-in transfer-function presets."""
        return sorted(PRESETS)

    # ------------------------------------------------------------------
    # Shading / visibility
    # ------------------------------------------------------------------
    def set_shading(self, ambient: float = 0.2, diffuse: float = 0.7,
                    specular: float = 0.2, specular_power: float = 10.0,
                    enabled: bool = True) -> None:
        """Configure the Phong shading parameters of the volume property."""
        if self._property is None:
            raise RuntimeError("VolumeRenderer.setup() not called")
        if enabled:
            self._property.ShadeOn()
        else:
            self._property.ShadeOff()
        self._property.SetAmbient(float(ambient))
        self._property.SetDiffuse(float(diffuse))
        self._property.SetSpecular(float(specular))
        self._property.SetSpecularPower(float(specular_power))
        self._render()

    def toggle_volume_rendering(self) -> bool:
        """Toggle actor visibility. Returns the new visibility state."""
        self._visible = not self._visible
        if self._actor is not None:
            self._actor.SetVisibility(self._visible)
            self._render()
        return self._visible

    def set_visible(self, visible: bool) -> None:
        """Explicitly set actor visibility."""
        self._visible = bool(visible)
        if self._actor is not None:
            self._actor.SetVisibility(self._visible)
            self._render()

    @property
    def visible(self) -> bool:
        return self._visible

    @property
    def actor(self):
        return self._actor

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _render(self) -> None:
        if self._renderer is not None:
            window = self._renderer.GetRenderWindow()
            if window is not None:
                window.Render()
