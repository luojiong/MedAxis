"""Project — top-level scene model persisted as scene.json."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

from .measurement_data import AngleMeasurement, DistanceMeasurement, ROIStatistics
from .patient_model import PatientModel

__all__ = ["Project"]

SCENE_VERSION = 1


@dataclass
class Project:
    """The MedAxis scene: everything that gets saved to ``scene.json``.

    Lists of labels / meshes / transfer functions store metadata dicts that
    reference on-disk binary files (e.g. ``labels/<id>.nii.gz``,
    ``meshes/<id>.stl``). :meth:`validate` checks those references.
    """

    version: int = SCENE_VERSION
    created: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    modified: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    patients: list[PatientModel] = field(default_factory=list)
    labels: list[dict[str, Any]] = field(default_factory=list)      # metadata + "file" ref
    meshes: list[dict[str, Any]] = field(default_factory=list)      # metadata + "file" ref
    measurements: list[Union[DistanceMeasurement, AngleMeasurement, ROIStatistics]] = field(
        default_factory=list
    )
    views: dict[str, Any] = field(default_factory=dict)             # camera/window state
    transfer_functions: list[dict[str, Any]] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def touch(self) -> None:
        """Update the modification timestamp."""
        self.modified = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Serialize the whole scene to a JSON-compatible dict."""
        self.touch()
        return {
            "version": self.version,
            "created": self.created,
            "modified": self.modified,
            "patients": [p.to_dict() for p in self.patients],
            "labels": self.labels,
            "meshes": self.meshes,
            "measurements": [m.to_dict() for m in self.measurements],
            "views": self.views,
            "transfer_functions": self.transfer_functions,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Project":
        """Reconstruct a :class:`Project` from :meth:`to_dict` output."""
        measurements: list[Any] = []
        for m in d.get("measurements", []):
            mtype = m.get("type")
            if mtype == "distance":
                measurements.append(DistanceMeasurement.from_dict(m))
            elif mtype == "angle":
                measurements.append(AngleMeasurement.from_dict(m))
            elif mtype == "roi":
                measurements.append(ROIStatistics.from_dict(m))

        return cls(
            version=d.get("version", SCENE_VERSION),
            created=d.get("created", ""),
            modified=d.get("modified", ""),
            patients=[PatientModel.from_dict(p) for p in d.get("patients", [])],
            labels=list(d.get("labels", [])),
            meshes=list(d.get("meshes", [])),
            measurements=measurements,
            views=dict(d.get("views", {})),
            transfer_functions=list(d.get("transfer_functions", [])),
        )

    def save(self, path: Union[str, Path]) -> None:
        """Write the scene to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Project":
        """Read a scene from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, base_dir: Optional[Union[str, Path]] = None) -> list[str]:
        """Check internal consistency and on-disk file references.

        Args:
            base_dir: Directory against which relative ``"file"`` paths are
                resolved (typically the project folder containing
                scene.json). Defaults to the current working directory.

        Returns:
            A list of human-readable issue strings; empty means valid.
        """
        issues: list[str] = []
        base = Path(base_dir) if base_dir is not None else Path.cwd()

        # Volume IDs referenced anywhere in the scene.
        volume_ids: set[str] = set()
        for patient in self.patients:
            for _study, series in patient.iter_series():
                if series.volume_id:
                    volume_ids.add(series.volume_id)

        def resolve(ref: Optional[str]) -> Optional[Path]:
            if not ref:
                return None
            p = Path(ref)
            return p if p.is_absolute() else base / p

        # Label entries: file exists, parent volume referenced.
        label_ids = set()
        for entry in self.labels:
            lid = entry.get("id")
            label_ids.add(lid)
            f = resolve(entry.get("file"))
            if f is None:
                issues.append(f"Label '{lid}' has no file reference")
            elif not f.exists():
                issues.append(f"Label '{lid}' file missing: {f}")
            parent = entry.get("parent_volume_id")
            if parent and volume_ids and parent not in volume_ids:
                issues.append(f"Label '{lid}' references unknown volume '{parent}'")

        if len(label_ids) != len(self.labels):
            issues.append("Duplicate label ids present")

        # Mesh entries: file exists.
        mesh_ids = set()
        for entry in self.meshes:
            mid = entry.get("id")
            mesh_ids.add(mid)
            f = resolve(entry.get("file"))
            if f is None:
                issues.append(f"Mesh '{mid}' has no file reference")
            elif not f.exists():
                issues.append(f"Mesh '{mid}' file missing: {f}")
            parent = entry.get("parent_label_id")
            if parent and parent not in label_ids:
                issues.append(f"Mesh '{mid}' references unknown label '{parent}'")

        if len(mesh_ids) != len(self.meshes):
            issues.append("Duplicate mesh ids present")

        # Measurements: volume references.
        for m in self.measurements:
            vid = getattr(m, "volume_id", "")
            if vid and volume_ids and vid not in volume_ids:
                issues.append(f"Measurement '{m.id}' references unknown volume '{vid}'")

        return issues
