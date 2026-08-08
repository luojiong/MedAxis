"""Transfer functions for volume rendering."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

try:
    import vtk

    HAS_VTK = True
except ImportError:  # pragma: no cover
    vtk = None  # type: ignore[assignment]
    HAS_VTK = False

__all__ = ["TransferFunction", "PRESETS"]


@dataclass
class TransferFunction:
    """A piecewise opacity + color transfer function over scalar values.

    Attributes:
        name: Display name.
        opacity_points: List of ``(value, opacity)`` with opacity in [0, 1].
        color_points: List of ``(value, (r, g, b))`` with channels in [0, 1].
        value_range: Optional ``(min, max)`` the function is defined over.
    """

    name: str = "Custom"
    opacity_points: list[tuple[float, float]] = field(default_factory=list)
    color_points: list[tuple[float, tuple[float, float, float]]] = field(default_factory=list)
    value_range: Optional[tuple[float, float]] = None

    def __post_init__(self) -> None:
        self.opacity_points.sort(key=lambda p: p[0])
        self.color_points.sort(key=lambda p: p[0])

    # ------------------------------------------------------------------
    # VTK conversion
    # ------------------------------------------------------------------
    def to_vtk(self) -> tuple:
        """Build ``(vtkPiecewiseFunction, vtkColorTransferFunction)``.

        Raises:
            RuntimeError: If VTK is unavailable.
        """
        if not HAS_VTK:
            raise RuntimeError("VTK is required for to_vtk()")

        opacity = vtk.vtkPiecewiseFunction()
        for value, alpha in self.opacity_points:
            opacity.AddPoint(float(value), float(max(0.0, min(1.0, alpha))))

        color = vtk.vtkColorTransferFunction()
        for value, (r, g, b) in self.color_points:
            color.AddRGBPoint(float(value), float(r), float(g), float(b))

        return opacity, color

    # ------------------------------------------------------------------
    # JSON serialization
    # ------------------------------------------------------------------
    def to_json(self, path: Union[str, Path]) -> Path:
        """Serialize the transfer function to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "name": self.name,
            "opacity_points": [[float(v), float(a)] for v, a in self.opacity_points],
            "color_points": [[float(v), [float(r), float(g), float(b)]]
                             for v, (r, g, b) in self.color_points],
            "value_range": list(self.value_range) if self.value_range else None,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return path

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "TransferFunction":
        """Load a transfer function from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            name=data.get("name", "Custom"),
            opacity_points=[(float(v), float(a)) for v, a in data.get("opacity_points", [])],
            color_points=[(float(v), (float(c[0]), float(c[1]), float(c[2])))
                          for v, c in data.get("color_points", [])],
            value_range=tuple(data["value_range"]) if data.get("value_range") else None,
        )

    def to_dict(self) -> dict:
        """JSON-compatible dict (for embedding in scene.json)."""
        return {
            "name": self.name,
            "opacity_points": [[float(v), float(a)] for v, a in self.opacity_points],
            "color_points": [[float(v), list(c)] for v, c in self.color_points],
            "value_range": list(self.value_range) if self.value_range else None,
        }


# ----------------------------------------------------------------------
# Presets (typical CT / MR ranges in Hounsfield / normalized units)
# ----------------------------------------------------------------------

def ct_default() -> TransferFunction:
    """Soft-tissue CT preset."""
    return TransferFunction(
        name="CT-Default",
        opacity_points=[(-1024.0, 0.0), (-400.0, 0.0), (-100.0, 0.1),
                        (100.0, 0.35), (500.0, 0.65), (2000.0, 0.8)],
        color_points=[(-1024.0, (0.0, 0.0, 0.0)), (-100.0, (0.25, 0.15, 0.12)),
                      (100.0, (0.85, 0.55, 0.4)), (500.0, (0.95, 0.9, 0.8)),
                      (2000.0, (1.0, 1.0, 1.0))],
        value_range=(-1024.0, 3071.0),
    )


def ct_bone() -> TransferFunction:
    """Bone CT preset."""
    return TransferFunction(
        name="CT-Bone",
        opacity_points=[(150.0, 0.0), (300.0, 0.08), (800.0, 0.45),
                        (1500.0, 0.75), (3000.0, 0.9)],
        color_points=[(150.0, (0.3, 0.15, 0.1)), (500.0, (0.9, 0.75, 0.55)),
                      (1500.0, (1.0, 0.95, 0.85)), (3000.0, (1.0, 1.0, 1.0))],
        value_range=(-1024.0, 3071.0),
    )


def ct_angio() -> TransferFunction:
    """Contrast-enhanced vessels (CT angiography) preset."""
    return TransferFunction(
        name="CT-Angio",
        opacity_points=[(-1024.0, 0.0), (120.0, 0.0), (250.0, 0.25),
                        (500.0, 0.6), (1200.0, 0.85)],
        color_points=[(120.0, (0.35, 0.05, 0.05)), (300.0, (0.9, 0.2, 0.15)),
                      (700.0, (1.0, 0.8, 0.6)), (1200.0, (1.0, 1.0, 0.95))],
        value_range=(-1024.0, 3071.0),
    )


def ct_lung() -> TransferFunction:
    """Lung window CT preset (air-filled structures)."""
    return TransferFunction(
        name="CT-Lung",
        opacity_points=[(-1024.0, 0.0), (-900.0, 0.15), (-500.0, 0.35),
                        (-100.0, 0.05), (200.0, 0.4), (1000.0, 0.7)],
        color_points=[(-1024.0, (0.1, 0.1, 0.35)), (-600.0, (0.4, 0.6, 0.9)),
                      (-100.0, (0.9, 0.85, 0.8)), (1000.0, (1.0, 1.0, 1.0))],
        value_range=(-1024.0, 3071.0),
    )


def mr_default() -> TransferFunction:
    """Generic MR preset over normalized [0, 1000] intensities."""
    return TransferFunction(
        name="MR-Default",
        opacity_points=[(0.0, 0.0), (120.0, 0.0), (300.0, 0.25),
                        (600.0, 0.55), (1000.0, 0.8)],
        color_points=[(0.0, (0.0, 0.0, 0.0)), (200.0, (0.4, 0.3, 0.25)),
                      (500.0, (0.9, 0.8, 0.65)), (1000.0, (1.0, 1.0, 1.0))],
        value_range=(0.0, 1000.0),
    )


PRESETS: dict[str, callable] = {  # type: ignore[valid-type]
    "CT-Default": ct_default,
    "CT-Bone": ct_bone,
    "CT-Angio": ct_angio,
    "CT-Lung": ct_lung,
    "MR-Default": mr_default,
}


def get_preset(name: str) -> TransferFunction:
    """Return a preset by name (raises KeyError for unknown names)."""
    return PRESETS[name]()
