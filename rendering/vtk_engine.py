"""VTKEngine — singleton managing the VTK lifecycle and GPU resources."""
from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore import QObject, Signal

from core.volume_data import VolumeData

try:
    import vtk

    HAS_VTK = True
except ImportError:  # pragma: no cover
    vtk = None  # type: ignore[assignment]
    HAS_VTK = False

from .helpers import numpy_to_vtk_image, volume_user_matrix

logger = logging.getLogger(__name__)

__all__ = ["VTKEngine"]


class VTKEngine(QObject):
    """Singleton managing shared VTK resources.

    Owns a master (off-screen-capable) render window, creates renderers
    for each view, and builds the common actor/mapper/filter objects for
    a :class:`~core.volume_data.VolumeData`.

    Signals:
        render_completed() — emitted after :meth:`render_all` finishes.
    """

    render_completed = Signal()

    _instance: Optional["VTKEngine"] = None

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._initialized = False
        self._render_window = None
        self._renderers: list = []
        self._gpu_info: dict[str, object] = {}

    # ------------------------------------------------------------------
    # Singleton access
    # ------------------------------------------------------------------
    @classmethod
    def instance(cls, parent: Optional[QObject] = None) -> "VTKEngine":
        """Return the process-wide engine instance (created on first call)."""
        if cls._instance is None:
            cls._instance = cls(parent)
        return cls._instance

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def initialize(self) -> None:
        """Create the master render window and query GPU capabilities."""
        if self._initialized:
            return
        if not HAS_VTK:
            raise RuntimeError("VTK is not installed")

        self._render_window = vtk.vtkRenderWindow()
        self._render_window.SetOffScreenRendering(1)
        self._render_window.SetSize(640, 480)

        self._gpu_info = self._probe_gpu()
        logger.info(
            "VTK %s initialized | GPU: %s | OpenGL %s",
            vtk.vtkVersion.GetVTKVersion(),
            self._gpu_info.get("renderer", "unknown"),
            self._gpu_info.get("opengl_version", "unknown"),
        )
        self._initialized = True

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def render_window(self):
        return self._render_window

    def _probe_gpu(self) -> dict[str, object]:
        """Best-effort GPU capability query."""
        info: dict[str, object] = {}
        try:
            ren = vtk.vtkRenderer()
            self._render_window.AddRenderer(ren)
            self._render_window.Render()
            ogl = ren.GetRenderWindow().ReportCapabilities()
            self._render_window.RemoveRenderer(ren)
            for line in ogl.splitlines():
                if line.startswith("GL_RENDERER"):
                    info["renderer"] = line.split(":", 1)[-1].strip()
                elif line.startswith("GL_VERSION"):
                    info["opengl_version"] = line.split(":", 1)[-1].strip()
            info["gpu_accelerated"] = True
        except Exception as exc:  # noqa: BLE001
            logger.warning("GPU probe failed: %s", exc)
            info["gpu_accelerated"] = False
        return info

    def shutdown(self) -> None:
        """Release shared VTK resources."""
        self._renderers.clear()
        if self._render_window is not None:
            self._render_window.Finalize()
        self._render_window = None
        self._initialized = False

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------
    def create_renderer(self, background: tuple[float, float, float] = (0.0, 0.0, 0.0)):
        """Create a new ``vtkRenderer`` tracked by the engine."""
        self._ensure_initialized()
        renderer = vtk.vtkRenderer()
        renderer.SetBackground(*background)
        self._renderers.append(renderer)
        return renderer

    def create_image_data(self, volume: VolumeData):
        """Build a ``vtkImageData`` (+ user matrix) for a volume.

        Returns:
            ``(vtkImageData, vtkMatrix4x4)`` — apply the matrix on the
            actor to honor the volume direction cosines.
        """
        self._ensure_initialized()
        image = numpy_to_vtk_image(volume.array, volume.spacing_mm, volume.origin_mm)
        return image, volume_user_matrix(volume)

    def create_image_actor(self, volume: VolumeData):
        """Create a ``vtkImageActor`` displaying the full volume extent."""
        self._ensure_initialized()
        image, matrix = self.create_image_data(volume)
        actor = vtk.vtkImageActor()
        actor.GetMapper().SetInputData(image)
        actor.SetUserMatrix(matrix)
        actor.SetInterpolate(True)
        # Keep the image alive past this method's scope.
        actor._medaxis_image = image  # type: ignore[attr-defined]
        return actor

    def create_volume_mapper(self, volume: VolumeData):
        """Create a GPU ray-cast mapper configured for the volume.

        Falls back to ``vtkSmartVolumeMapper`` when the GPU mapper cannot
        handle the data.
        """
        self._ensure_initialized()
        image, _matrix = self.create_image_data(volume)

        mapper = vtk.vtkGPUVolumeRayCastMapper()
        mapper.SetInputData(image)
        mapper.SetImageSampleDistance(1.0)
        mapper.SetSampleDistance(0.8)
        mapper.SetAutoAdjustSampleDistances(True)

        if not mapper.IsRenderSupported(image.GetPointData().GetScalars().GetDataType(), 1):
            logger.warning("GPU ray casting unsupported; using smart volume mapper")
            smart = vtk.vtkSmartVolumeMapper()
            smart.SetInputData(image)
            smart._medaxis_image = image  # type: ignore[attr-defined]
            return smart

        mapper._medaxis_image = image  # type: ignore[attr-defined]
        return mapper

    def get_reslice_filter(self, volume: Optional[VolumeData] = None):
        """Create a ``vtkImageReslice`` filter (optionally pre-connected)."""
        self._ensure_initialized()
        reslice = vtk.vtkImageReslice()
        reslice.SetInterpolationModeToLinear()
        reslice.SetOutputDimensionality(2)
        reslice.SetAutoCropOutput(True)
        if volume is not None:
            image, _ = self.create_image_data(volume)
            reslice.SetInputData(image)
            reslice._medaxis_image = image  # type: ignore[attr-defined]
        return reslice

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render_all(self) -> None:
        """Render all registered renderers/windows."""
        for renderer in self._renderers:
            window = renderer.GetRenderWindow() if renderer is not None else None
            if window is not None:
                window.Render()
            elif renderer is not None:
                renderer.Render()
        self.render_completed.emit()

    def release_renderer(self, renderer) -> None:
        """Stop tracking a renderer (e.g. when a view is closed)."""
        if renderer in self._renderers:
            self._renderers.remove(renderer)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _ensure_initialized(self) -> None:
        if not self._initialized:
            self.initialize()
