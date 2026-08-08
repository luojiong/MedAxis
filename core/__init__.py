"""MedAxis core data model package.

Public data models for the MedAxis medical imaging platform:

* :class:`ImageData` — single 2D DICOM slice
* :class:`VolumeData` — 3D image volume
* :class:`LabelData` / :class:`LabelStats` — segmentation labels
* :class:`MeshData` — surface meshes
* :class:`PatientModel` / :class:`StudyModel` / :class:`SeriesModel` — DICOM hierarchy
* :class:`DistanceMeasurement`, :class:`AngleMeasurement`, :class:`ROIStatistics`
* :class:`Project` — scene.json model
* :class:`UndoManager` and undo commands
* Coordinate transforms and unit utilities
"""

from .image_data import ImageData
from .volume_data import VolumeData
from .label_data import LabelData, LabelStats
from .mesh_data import MeshData
from .patient_model import PatientModel, StudyModel, SeriesModel
from .measurement_data import DistanceMeasurement, AngleMeasurement, ROIStatistics
from .project import Project
from .undo_manager import (
    UndoManager,
    EditLabelCommand,
    DeleteLabelCommand,
    AddMeasurementCommand,
    ChangeViewCommand,
)
from .transform import (
    build_affine_matrix,
    world_to_voxel_coords,
    voxel_to_world_coords,
    LPS_to_RAS,
    RAS_to_LPS,
)
from .units import (
    mm_to_cm,
    cm_to_mm,
    format_distance_mm,
    format_area_mm2,
    format_volume_ml,
    format_angle_deg,
)
from .label_manager import LabelManager

__all__ = [
    # Data models
    "ImageData",
    "VolumeData",
    "LabelData",
    "LabelStats",
    "MeshData",
    "PatientModel",
    "StudyModel",
    "SeriesModel",
    "DistanceMeasurement",
    "AngleMeasurement",
    "ROIStatistics",
    "Project",
    # Undo
    "UndoManager",
    "EditLabelCommand",
    "DeleteLabelCommand",
    "AddMeasurementCommand",
    "ChangeViewCommand",
    # Transforms
    "build_affine_matrix",
    "world_to_voxel_coords",
    "voxel_to_world_coords",
    "LPS_to_RAS",
    "RAS_to_LPS",
    # Units
    "mm_to_cm",
    "cm_to_mm",
    "format_distance_mm",
    "format_area_mm2",
    "format_volume_ml",
    "format_angle_deg",
    "LabelManager",
]
