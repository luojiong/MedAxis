"""Project I/O — save/load .medaxis project archives with autosave."""
from __future__ import annotations

import json
import logging
import shutil
import tempfile
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from core.project import Project, SCENE_VERSION
from core.volume_data import VolumeData

logger = logging.getLogger(__name__)

__all__ = ["ProjectIO", "PROJECT_EXTENSION"]

PROJECT_EXTENSION = ".medaxis"
_SCENE_FILENAME = "scene.json"
_VOLUMES_DIR = "volumes"
_LABELS_DIR = "labels"
_MESHES_DIR = "meshes"

AUTOSAVE_DIR = Path.home() / ".medaxis" / "autosave"


class ProjectIO:
    """Save/load MedAxis project archives (``.medaxis`` ZIP files).

    Archive layout::

        scene.json                  # Project.to_dict()
        volumes/<volume_id>.nii.gz  # volume files
        labels/<label_id>.nii.gz    # label files referenced by scene.json
        meshes/<mesh_id>.stl        # mesh files referenced by scene.json

    Labels/meshes entries in ``scene.json`` carry a relative ``"file"``
    path; existing files are copied into the archive. Volumes are not
    part of the scene model, so they are passed in via the ``volumes``
    mapping and recorded under a top-level ``"volumes"`` key.
    """

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    def save_project(
        self,
        project: Project,
        path: Union[str, Path],
        volumes: Optional[dict[str, VolumeData]] = None,
        labels: Optional[dict[str, Any]] = None,
        meshes: Optional[dict[str, Any]] = None,
    ) -> Path:
        """Save a project to a ``.medaxis`` archive.

        Args:
            project: The scene model.
            path: Destination file path.
            volumes: Optional mapping ``volume_id -> VolumeData`` to embed.
                Also honors a ``project.volumes`` attribute when present.

        Returns:
            The written archive path.
        """
        path = Path(path)
        if path.suffix != PROJECT_EXTENSION:
            path = path.with_suffix(PROJECT_EXTENSION)
        path.parent.mkdir(parents=True, exist_ok=True)

        merged_volumes: dict[str, VolumeData] = {}
        attr_volumes = getattr(project, "volumes", None)
        if isinstance(attr_volumes, dict):
            merged_volumes.update(attr_volumes)
        if volumes:
            merged_volumes.update(volumes)

        with tempfile.TemporaryDirectory(prefix="medaxis_save_") as tmp_name:
            tmp = Path(tmp_name)

            # Volume files + scene entries.
            volume_entries: list[dict[str, Any]] = []
            for vid, volume in merged_volumes.items():
                rel = f"{_VOLUMES_DIR}/{vid}.nii.gz"
                written = self._write_volume_file(volume, tmp / rel)
                volume_entries.append({
                    "id": vid,
                    "name": volume.name,
                    "modality": volume.modality,
                    "file": written.relative_to(tmp).as_posix(),
                })

            # Copy referenced label/mesh files into the archive.
            self._copy_referenced_files(project.labels, tmp, _LABELS_DIR)
            self._copy_referenced_files(project.meshes, tmp, _MESHES_DIR)

            # scene.json
            scene = project.to_dict()
            if labels:
                scene["labels"] = []
                for label_id, label in labels.items():
                    rel = f"{_LABELS_DIR}/{label_id}.npz"
                    label_path = tmp / rel
                    label_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(label_path, array=label.array)
                    entry = label.to_dict()
                    entry["file"] = rel
                    scene["labels"].append(entry)
            if meshes:
                scene["meshes"] = []
                for mesh_id, mesh in meshes.items():
                    rel = f"{_MESHES_DIR}/{mesh_id}.ply"
                    mesh_path = tmp / rel
                    mesh_path.parent.mkdir(parents=True, exist_ok=True)
                    mesh.save_ply(mesh_path)
                    entry = mesh.to_dict()
                    entry["file"] = rel
                    scene["meshes"].append(entry)
            scene["volumes"] = volume_entries
            scene["app"] = "MedAxis"
            scene["saved_at"] = datetime.now().isoformat()
            with open(tmp / _SCENE_FILENAME, "w", encoding="utf-8") as f:
                json.dump(scene, f, indent=2, ensure_ascii=False)

            self._zip_directory(tmp, path)

        project.path = str(path)  # type: ignore[attr-defined]
        logger.info("Saved project to %s", path)
        return path

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    def load_project(
        self,
        path: Union[str, Path],
        extract_to: Optional[Union[str, Path]] = None,
    ) -> Project:
        """Load a ``.medaxis`` archive into a :class:`Project`.

        The archive is extracted next to the file (or to ``extract_to``)
        so that relative ``"file"`` references resolve. Validation issues
        are logged but non-fatal.

        Returns:
            The reconstructed project with ``.path`` and ``.base_dir``
            attributes set.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Project file not found: {path}")

        if extract_to is None:
            extract_to = path.parent / (path.stem + "_extracted")
        extract_to = Path(extract_to)
        extract_to.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(path, "r") as zf:
            names = zf.namelist()
            if _SCENE_FILENAME not in names:
                raise ValueError(f"Invalid project archive (missing {_SCENE_FILENAME}): {path}")
            extract_root = extract_to.resolve()
            for member in zf.infolist():
                target = (extract_to / member.filename).resolve()
                if extract_root not in target.parents and target != extract_root:
                    raise ValueError(f"Unsafe project archive member: {member.filename}")
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member, "r") as source, open(target, "wb") as destination:
                    shutil.copyfileobj(source, destination)

        with open(extract_to / _SCENE_FILENAME, "r", encoding="utf-8") as f:
            scene = json.load(f)

        if scene.get("version", SCENE_VERSION) > SCENE_VERSION:
            logger.warning(
                "Project version %s is newer than supported %s",
                scene.get("version"), SCENE_VERSION,
            )

        project = Project.from_dict(scene)
        project.path = str(path)  # type: ignore[attr-defined]
        project.base_dir = str(extract_to)  # type: ignore[attr-defined]
        project.volume_entries = scene.get("volumes", [])  # type: ignore[attr-defined]

        issues = project.validate(base_dir=extract_to)
        for issue in issues:
            logger.warning("Project validation: %s", issue)
        logger.info("Loaded project from %s (%d issues)", path, len(issues))
        return project

    def load_volumes(self, project: Project) -> dict[str, VolumeData]:
        """Materialize the volume entries stored in a loaded project archive."""
        base_dir = Path(getattr(project, "base_dir", Path.cwd()))
        result: dict[str, VolumeData] = {}
        for entry in getattr(project, "volume_entries", []):
            file_ref = entry.get("file")
            if not file_ref:
                continue
            source = base_dir / file_ref
            try:
                if source.suffix.lower() == ".npz":
                    with np.load(source) as data:
                        volume = VolumeData(
                            array=data["array"],
                            spacing=tuple(float(v) for v in data["spacing"]),
                            origin=tuple(float(v) for v in data["origin"]),
                            direction=tuple(float(v) for v in data["direction"]),
                            name=entry.get("name", source.stem),
                            modality=entry.get("modality", ""),
                        )
                else:
                    from .nifti_reader import NIfTIReader
                    volume = NIfTIReader().read(source)
                result[str(entry.get("id", id(volume)))] = volume
            except Exception:  # noqa: BLE001
                logger.exception("Unable to restore volume from %s", source)
        return result

    # ------------------------------------------------------------------
    # Autosave / crash recovery
    # ------------------------------------------------------------------
    def autosave(self, project: Project,
                 volumes: Optional[dict[str, VolumeData]] = None,
                 labels: Optional[dict[str, Any]] = None,
                 meshes: Optional[dict[str, Any]] = None) -> Path:
        """Write a timestamped autosave archive under ~/.medaxis/autosave/."""
        AUTOSAVE_DIR.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        name = getattr(project, "name", None) or "project"
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(name))
        target = AUTOSAVE_DIR / f"autosave_{safe_name}_{stamp}{PROJECT_EXTENSION}"
        try:
            self.save_project(project, target, volumes=volumes, labels=labels, meshes=meshes)
            self._prune_autosaves(keep=5, prefix=f"autosave_{safe_name}_")
            logger.info("Autosaved project to %s", target)
        except Exception:  # noqa: BLE001
            logger.exception("Autosave failed")
            raise
        return target

    def detect_crash_recovery(self) -> list[dict[str, Any]]:
        """List autosave archives available for crash recovery.

        Returns:
            List of dicts with keys ``path``, ``modified`` (epoch seconds),
            ``size_bytes`` and ``name`` — newest first.
        """
        if not AUTOSAVE_DIR.is_dir():
            return []
        results: list[dict[str, Any]] = []
        for entry in AUTOSAVE_DIR.glob(f"autosave_*{PROJECT_EXTENSION}"):
            try:
                stat = entry.stat()
                results.append({
                    "path": str(entry),
                    "modified": stat.st_mtime,
                    "size_bytes": stat.st_size,
                    "name": entry.stem,
                })
            except OSError:
                continue
        results.sort(key=lambda d: d["modified"], reverse=True)
        return results

    def clear_autosaves(self) -> int:
        """Delete all autosave archives. Returns the number removed."""
        if not AUTOSAVE_DIR.is_dir():
            return 0
        count = 0
        for entry in AUTOSAVE_DIR.glob(f"autosave_*{PROJECT_EXTENSION}"):
            try:
                entry.unlink()
                count += 1
            except OSError:
                continue
        return count

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _write_volume_file(volume: VolumeData, target: Path) -> Path:
        """Write one volume as .nii.gz (falls back to .npz if no nibabel)."""
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            from .nifti_writer import NIfTIWriter

            NIfTIWriter().write_volume(volume, target)
            return target
        except Exception as exc:  # noqa: BLE001
            logger.warning("nibabel unavailable (%s); storing volume as .npz", exc)
            target = target.with_suffix(".npz")
            np.savez_compressed(
                target,
                array=volume.array,
                spacing=np.asarray(volume.spacing_mm),
                origin=np.asarray(volume.origin_mm),
                direction=np.asarray(volume.direction),
            )
            return target

    @staticmethod
    def _copy_referenced_files(entries: list[dict[str, Any]], tmp_root: Path,
                               default_dir: str) -> None:
        """Copy existing referenced files into the staging directory."""
        for entry in entries:
            ref = entry.get("file")
            if not ref:
                continue
            src = Path(ref)
            if not src.is_file():
                # Try resolving relative to common bases.
                for base in (Path.cwd(),):
                    candidate = base / ref
                    if candidate.is_file():
                        src = candidate
                        break
            if not src.is_file():
                logger.warning("Referenced file missing, skipping: %s", ref)
                continue
            dst = tmp_root / ref
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    @staticmethod
    def _zip_directory(root: Path, archive_path: Path) -> None:
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=6) as zf:
            for file_path in sorted(root.rglob("*")):
                if file_path.is_file():
                    zf.write(file_path, file_path.relative_to(root).as_posix())

    @staticmethod
    def _prune_autosaves(keep: int, prefix: str) -> None:
        archives = sorted(
            AUTOSAVE_DIR.glob(f"{prefix}*{PROJECT_EXTENSION}"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for old in archives[keep:]:
            try:
                old.unlink()
            except OSError:
                continue
