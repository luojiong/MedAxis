"""VTK/NumPy interop and viewer decoration helpers."""
from __future__ import annotations

import logging

import numpy as np

try:
    import vtk
    from vtk.util import numpy_support

    HAS_VTK = True
except ImportError:  # pragma: no cover
    vtk = None  # type: ignore[assignment]
    numpy_support = None  # type: ignore[assignment]
    HAS_VTK = False

logger = logging.getLogger(__name__)

__all__ = [
    "numpy_to_vtk_image",
    "vtk_image_to_numpy",
    "volume_user_matrix",
    "create_annotation",
    "create_orientation_marker",
    "create_scale_bar",
]


def _require_vtk() -> None:
    if not HAS_VTK:
        raise RuntimeError("VTK is required for rendering helpers")


# ----------------------------------------------------------------------
# Data conversion
# ----------------------------------------------------------------------

def numpy_to_vtk_image(
    array: np.ndarray,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> "vtk.vtkImageData":
    """Convert a 3D numpy array (k, j, i) into a ``vtkImageData``.

    The direction cosine matrix is *not* representable on vtkImageData;
    apply it via :func:`volume_user_matrix` on the actor/mapper instead.

    Args:
        array: 3D array indexed ``[k, j, i]`` (z, y, x).
        spacing: Voxel size (sx, sy, sz) in mm.
        origin: World coordinate of voxel (0, 0, 0).

    Returns:
        A new ``vtkImageData``.
    """
    _require_vtk()
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D array, got {array.ndim}D")

    nk, nj, ni = array.shape
    image = vtk.vtkImageData()
    image.SetDimensions(ni, nj, nk)
    image.SetSpacing(float(spacing[0]), float(spacing[1]), float(spacing[2]))
    image.SetOrigin(float(origin[0]), float(origin[1]), float(origin[2]))

    flat = np.ascontiguousarray(array).ravel()  # i-fastest order matches VTK
    scalars = numpy_support.numpy_to_vtk(num_array=flat, deep=1,
                                         array_type=numpy_support.get_vtk_array_type(array.dtype))
    scalars.SetName("Scalars")
    image.GetPointData().SetScalars(scalars)
    return image


def vtk_image_to_numpy(image: "vtk.vtkImageData") -> np.ndarray:
    """Convert a ``vtkImageData`` into a numpy array shaped (k, j, i)."""
    _require_vtk()
    dims = image.GetDimensions()  # (ni, nj, nk)
    scalars = image.GetPointData().GetScalars()
    if scalars is None:
        raise ValueError("vtkImageData has no scalars")
    arr = numpy_support.vtk_to_numpy(scalars)
    n_comp = scalars.GetNumberOfComponents()
    if n_comp > 1:
        arr = arr.reshape(dims[2], dims[1], dims[0], n_comp)
    else:
        arr = arr.reshape(dims[2], dims[1], dims[0])
    return arr


def volume_user_matrix(volume) -> "vtk.vtkMatrix4x4":
    """Build a 4x4 VTK matrix carrying the volume's direction cosines.

    vtkImageData cannot encode direction; applying this matrix via
    ``actor.SetUserMatrix`` restores the full LPS pose.
    """
    _require_vtk()
    mat = vtk.vtkMatrix4x4()
    mat.Identity()
    direction = volume.direction_matrix
    for r in range(3):
        for c in range(3):
            mat.SetElement(r, c, float(direction[r, c]))
    return mat


# ----------------------------------------------------------------------
# Annotations / decorations
# ----------------------------------------------------------------------

# Corner indices for vtkCornerAnnotation
CORNER_LOWER_LEFT, CORNER_LOWER_RIGHT, CORNER_UPPER_LEFT, CORNER_UPPER_RIGHT = 0, 1, 2, 3


def create_annotation(text: str, corner: int = CORNER_UPPER_LEFT,
                      font_size: int = 14, color: tuple[float, float, float] = (1.0, 1.0, 1.0)):
    """Create a ``vtkCornerAnnotation`` actor with one line of text.

    Args:
        text: The annotation text.
        corner: One of ``CORNER_LOWER_LEFT/LOWER_RIGHT/UPPER_LEFT/UPPER_RIGHT``.
        font_size: Font size in points.
        color: Text color (r, g, b) in [0, 1].

    Returns:
        A ``vtkCornerAnnotation`` actor (add it to the renderer).
    """
    _require_vtk()
    annotation = vtk.vtkCornerAnnotation()
    annotation.SetText(corner, text)
    annotation.GetTextProperty().SetFontSize(font_size)
    annotation.GetTextProperty().SetColor(*color)
    annotation.GetTextProperty().SetShadow(True)
    return annotation


def create_orientation_marker(
    interactor=None,
    size: float = 0.12,
    viewport: tuple[float, float, float, float] = (0.86, 0.0, 1.0, 0.16),
):
    """Create an orientation axes marker.

    Args:
        interactor: Optional render-window interactor; when provided the
            returned widget is fully configured and interactive.
        size: Viewport fraction occupied by the marker.
        viewport: Marker viewport (xmin, ymin, xmax, ymax).

    Returns:
        ``(axes_actor, widget_or_None)`` — the widget is None when no
        interactor was given (caller must attach it later).
    """
    _require_vtk()
    axes = vtk.vtkAxesActor()
    axes.SetAxisLabels(1)
    axes.SetShaftTypeToCylinder()

    widget = None
    if interactor is not None:
        widget = vtk.vtkOrientationMarkerWidget()
        widget.SetOrientationMarker(axes)
        widget.SetInteractor(interactor)
        widget.SetViewport(*viewport)
        widget.SetEnabled(True)
        widget.InteractiveOff()
    return axes, widget


def create_scale_bar(
    renderer,
    spacing: float,
    units: str = "mm",
    target_fraction: float = 0.2,
    viewport_size: tuple[int, int] = (800, 600),
):
    """Create a simple scale-bar actor for a slice view.

    The bar shows a "nice" round physical length covering roughly
    ``target_fraction`` of the viewport width.

    Args:
        renderer: The ``vtkRenderer`` (used to query the camera later).
        spacing: In-plane pixel spacing in ``units``.
        units: Unit label, e.g. ``"mm"``.
        target_fraction: Fraction of the viewport the bar should cover.
        viewport_size: Approximate (width, height) of the view in pixels.

    Returns:
        ``(bar_actor, label_actor)`` — two 2D actors to add to the renderer.
    """
    _require_vtk()
    if spacing <= 0:
        spacing = 1.0

    # Nice round length close to the target physical size.
    target_units = target_fraction * viewport_size[0] * spacing
    exponent = np.floor(np.log10(max(target_units, 1e-9)))
    nice_length = float(10 ** exponent)
    for factor in (1.0, 2.0, 5.0, 10.0):
        if factor * 10 ** exponent <= target_units:
            nice_length = factor * 10 ** exponent
    pixel_length = nice_length / spacing

    # Horizontal bar near the lower-left corner (pixel coordinates).
    margin = 16.0
    x0, y0 = margin, margin
    x1 = x0 + pixel_length

    bar_source = vtk.vtkLineSource()
    bar_source.SetPoint1(x0, y0, 0.0)
    bar_source.SetPoint2(x1, y0, 0.0)

    bar_mapper = vtk.vtkPolyDataMapper2D()
    bar_mapper.SetInputConnection(bar_source.GetOutputPort())
    bar_actor = vtk.vtkActor2D()
    bar_actor.SetMapper(bar_mapper)
    bar_actor.GetProperty().SetColor(1.0, 1.0, 1.0)
    bar_actor.GetProperty().SetLineWidth(2.0)

    label = vtk.vtkTextActor()
    label.SetInput(f"{nice_length:g} {units}")
    label.GetTextProperty().SetFontSize(13)
    label.GetTextProperty().SetColor(1.0, 1.0, 1.0)
    label.GetTextProperty().SetShadow(True)
    label.SetDisplayCoord(int((x0 + x1) / 2.0), int(y0 + 6.0))
    label.GetTextProperty().SetJustificationToCentered()

    return bar_actor, label
