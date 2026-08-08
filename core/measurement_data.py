"""Measurement data models: distance, angle, and ROI statistics."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from .units import format_angle_deg, format_distance_mm

__all__ = ["DistanceMeasurement", "AngleMeasurement", "ROIStatistics"]


@dataclass
class DistanceMeasurement:
    """A straight-line distance measurement between two world points (mm)."""

    id: str = field(default_factory=lambda: uuid4().hex)
    point_a: tuple[float, float, float] = (0.0, 0.0, 0.0)
    point_b: tuple[float, float, float] = (0.0, 0.0, 0.0)
    label: str = ""
    volume_id: str = ""

    @property
    def length_mm(self) -> float:
        """Euclidean distance between the endpoints in mm."""
        a, b = self.point_a, self.point_b
        return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))

    def format_for_display(self) -> str:
        """Human-readable string, e.g. ``'D1: 12.3 mm'``."""
        name = self.label or "Distance"
        return f"{name}: {format_distance_mm(self.length_mm)}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "type": "distance",
            "id": self.id,
            "point_a": list(self.point_a),
            "point_b": list(self.point_b),
            "label": self.label,
            "volume_id": self.volume_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DistanceMeasurement":
        """Deserialize from :meth:`to_dict` output."""
        return cls(
            id=d["id"],
            point_a=tuple(d["point_a"]),  # type: ignore[arg-type]
            point_b=tuple(d["point_b"]),  # type: ignore[arg-type]
            label=d.get("label", ""),
            volume_id=d.get("volume_id", ""),
        )


@dataclass
class AngleMeasurement:
    """An angle measurement defined by three world points (mm).

    The angle is measured at ``vertex`` between rays to ``point_a`` and
    ``point_b``.
    """

    id: str = field(default_factory=lambda: uuid4().hex)
    point_a: tuple[float, float, float] = (0.0, 0.0, 0.0)
    vertex: tuple[float, float, float] = (0.0, 0.0, 0.0)
    point_b: tuple[float, float, float] = (0.0, 0.0, 0.0)
    label: str = ""
    volume_id: str = ""

    @property
    def angle_deg(self) -> float:
        """Interior angle at the vertex in degrees."""
        va = [ai - vi for ai, vi in zip(self.point_a, self.vertex)]
        vb = [bi - vi for bi, vi in zip(self.point_b, self.vertex)]
        dot = sum(x * y for x, y in zip(va, vb))
        na = math.sqrt(sum(x * x for x in va))
        nb = math.sqrt(sum(x * x for x in vb))
        if na == 0.0 or nb == 0.0:
            return 0.0
        cos_t = max(-1.0, min(1.0, dot / (na * nb)))
        return math.degrees(math.acos(cos_t))

    def format_for_display(self) -> str:
        """Human-readable string, e.g. ``'A1: 45.2°'``."""
        name = self.label or "Angle"
        return f"{name}: {format_angle_deg(self.angle_deg)}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "type": "angle",
            "id": self.id,
            "point_a": list(self.point_a),
            "vertex": list(self.vertex),
            "point_b": list(self.point_b),
            "label": self.label,
            "volume_id": self.volume_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AngleMeasurement":
        """Deserialize from :meth:`to_dict` output."""
        return cls(
            id=d["id"],
            point_a=tuple(d["point_a"]),  # type: ignore[arg-type]
            vertex=tuple(d["vertex"]),  # type: ignore[arg-type]
            point_b=tuple(d["point_b"]),  # type: ignore[arg-type]
            label=d.get("label", ""),
            volume_id=d.get("volume_id", ""),
        )


@dataclass
class ROIStatistics:
    """Statistics over a rectangular or freeform region of interest."""

    id: str = field(default_factory=lambda: uuid4().hex)
    label: str = ""
    volume_id: str = ""
    slice_index: int = 0
    axis: int = 0
    # ROI outline in slice pixel coordinates, (N, 2).
    points: list[tuple[float, float]] = field(default_factory=list)
    mean: float = 0.0
    std: float = 0.0
    min: float = 0.0
    max: float = 0.0
    area_mm2: float = 0.0
    pixel_count: int = 0

    def format_for_display(self) -> str:
        """Human-readable multi-line summary."""
        name = self.label or "ROI"
        return (
            f"{name}: mean={self.mean:.1f}, std={self.std:.1f}, "
            f"min={self.min:.1f}, max={self.max:.1f}, "
            f"area={self.area_mm2:.1f} mm\u00b2, n={self.pixel_count}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "type": "roi",
            "id": self.id,
            "label": self.label,
            "volume_id": self.volume_id,
            "slice_index": self.slice_index,
            "axis": self.axis,
            "points": [list(p) for p in self.points],
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "area_mm2": self.area_mm2,
            "pixel_count": self.pixel_count,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ROIStatistics":
        """Deserialize from :meth:`to_dict` output."""
        return cls(
            id=d["id"],
            label=d.get("label", ""),
            volume_id=d.get("volume_id", ""),
            slice_index=d.get("slice_index", 0),
            axis=d.get("axis", 0),
            points=[(float(p[0]), float(p[1])) for p in d.get("points", [])],
            mean=d.get("mean", 0.0),
            std=d.get("std", 0.0),
            min=d.get("min", 0.0),
            max=d.get("max", 0.0),
            area_mm2=d.get("area_mm2", 0.0),
            pixel_count=d.get("pixel_count", 0),
        )
