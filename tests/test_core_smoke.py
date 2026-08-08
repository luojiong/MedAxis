"""Small dependency-light regression tests for the application foundation."""

from pathlib import Path

import numpy as np

from core.volume_data import VolumeData
from core.label_data import LabelData
from core.mesh_data import MeshData
from core.native_extensions import NativeExtensionRegistry
from medaxis_io.file_manager import FileFormat, FileManager
from medaxis_io.project_io import ProjectIO


def test_file_format_detection_by_extension(tmp_path: Path) -> None:
    assert FileManager.detect_format(tmp_path / "scan.nii.gz") == FileFormat.NIFTI
    assert FileManager.detect_format(tmp_path / "scan.nrrd") == FileFormat.NRRD
    assert FileManager.detect_format(tmp_path / "session.medaxis") == FileFormat.PROJECT


def test_project_archive_records_fallback_volume_path(tmp_path: Path) -> None:
    volume = VolumeData(np.zeros((2, 3, 4), dtype=np.int16), name="demo")
    from core.project import Project

    path = ProjectIO().save_project(Project(), tmp_path / "demo.medaxis", {"v0": volume})
    loaded = ProjectIO().load_project(path, tmp_path / "extracted")

    assert loaded.volume_entries[0]["file"].startswith("volumes/")
    assert (tmp_path / "extracted" / loaded.volume_entries[0]["file"]).is_file()


def test_project_archive_round_trips_embedded_workspace_data(tmp_path: Path) -> None:
    from core.project import Project

    volume = VolumeData(np.zeros((2, 3, 4), dtype=np.int16), name="demo")
    label = LabelData("label-1", "Tumor", "v0", np.ones((2, 3, 4), dtype=np.int16))
    mesh = MeshData(
        vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
        id="mesh-1",
    )

    project = Project(labels=[label.to_dict()], meshes=[mesh.to_dict()])
    archive = ProjectIO().save_project(
        project,
        tmp_path / "workspace.medaxis",
        {"v0": volume},
        labels={label.id: label},
        meshes={mesh.id: mesh},
    )
    loaded = ProjectIO().load_project(archive, tmp_path / "workspace")
    restored = ProjectIO().load_volumes(loaded)

    assert np.array_equal(restored["v0"].array, volume.array)
    assert (tmp_path / "workspace" / loaded.labels[0]["file"]).is_file()
    assert (tmp_path / "workspace" / loaded.meshes[0]["file"]).is_file()


def test_native_extension_registry_reports_missing_extension() -> None:
    registry = NativeExtensionRegistry()
    status = registry.load("medaxis_missing_for_test")
    assert not status.available
    assert status.module is None


def test_native_extension_status_includes_every_compiled_module() -> None:
    statuses = NativeExtensionRegistry().status()

    assert {"medaxis_bridge", "medaxis_itk", "medaxis_cgal", "medaxis_occ", "medaxis_radiomics"} <= statuses.keys()
