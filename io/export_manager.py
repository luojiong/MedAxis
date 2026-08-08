"""ExportManager — unified export entry point for all data types."""
from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from PySide6.QtCore import QObject, Signal

from core.volume_data import VolumeData
from core.label_data import LabelData
from core.mesh_data import MeshData
from .dicom_writer import DICOMWriter
from .nifti_writer import NIfTIWriter
from .formats.stl_writer import STLWriter
from .formats.obj_writer import OBJWriter

logger = logging.getLogger(__name__)

__all__ = ["ExportManager", "ExportFormat"]


class ExportFormat(str, Enum):
    """Supported export formats."""

    # Volumes / labels
    DICOM = "dicom"
    NIFTI = "nifti"
    NRRD = "nrrd"
    DICOM_SEG = "dicom_seg"
    # Meshes
    STL = "stl"
    OBJ = "obj"
    PLY = "ply"
    # Screenshots
    PNG = "png"
    JPEG = "jpeg"
    # Reports
    PDF = "pdf"


_EXTENSION_MAP = {
    ExportFormat.DICOM: ".dcm",
    ExportFormat.NIFTI: ".nii.gz",
    ExportFormat.NRRD: ".nrrd",
    ExportFormat.DICOM_SEG: ".dcm",
    ExportFormat.STL: ".stl",
    ExportFormat.OBJ: ".obj",
    ExportFormat.PLY: ".ply",
    ExportFormat.PNG: ".png",
    ExportFormat.JPEG: ".jpg",
    ExportFormat.PDF: ".pdf",
}


class ExportManager(QObject):
    """Central export dispatcher for volumes, labels, meshes and reports.

    Signals:
        export_progress(float) — fraction in [0, 1]
        export_complete(str)   — output path on success
        export_error(str)      — human-readable error message
    """

    export_progress = Signal(float)
    export_complete = Signal(str)
    export_error = Signal(str)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self.dicom_writer = DICOMWriter()
        self.nifti_writer = NIfTIWriter()
        self.stl_writer = STLWriter()
        self.obj_writer = OBJWriter()

    # ------------------------------------------------------------------
    # Volume / label / mesh exports
    # ------------------------------------------------------------------
    def export_volume(self, volume: VolumeData, format: Union[ExportFormat, str],
                      path: Union[str, Path]) -> Optional[Path]:
        """Export a volume in the requested format."""
        format = ExportFormat(format)
        try:
            path = Path(path)
            self.export_progress.emit(0.1)
            if format == ExportFormat.NIFTI:
                out = self.nifti_writer.write_volume(volume, path)
            elif format == ExportFormat.DICOM:
                # Export the middle slice as a secondary capture stand-in
                # (full multi-frame DICOM export lives in plugins).
                mid = volume.array.shape[0] // 2
                slice2d = volume.array[mid].astype(np.float32)
                lo, hi = np.percentile(slice2d, [1, 99])
                norm = np.clip((slice2d - lo) / max(hi - lo, 1e-6), 0, 1)
                rgb = np.stack([norm] * 3, axis=-1) * 255.0
                out = self.dicom_writer.write_secondary_capture(volume, rgb.astype(np.uint8), path)
            elif format == ExportFormat.NRRD:  # type: ignore[comparison-overlap]
                out = self._export_nrrd(volume, path)
            else:
                raise ValueError(f"Unsupported volume export format: {format}")
            self.export_progress.emit(1.0)
            self.export_complete.emit(str(out))
            return out
        except Exception as exc:  # noqa: BLE001
            logger.exception("Volume export failed")
            self.export_error.emit(f"Volume export failed: {exc}")
            return None

    def export_label(self, label: LabelData, format: Union[ExportFormat, str],
                     path: Union[str, Path],
                     volume: Optional[VolumeData] = None) -> Optional[Path]:
        """Export a label map as NIfTI or DICOM SEG."""
        format = ExportFormat(format)
        try:
            if format == ExportFormat.NIFTI:
                out = self.nifti_writer.write_label(label, path, volume=volume)
            elif format == ExportFormat.DICOM_SEG:
                if volume is None:
                    raise ValueError("DICOM SEG export requires the parent volume")
                out = self.dicom_writer.write_segmentation(label, volume, path)
            else:
                raise ValueError(f"Unsupported label export format: {format}")
            self.export_progress.emit(1.0)
            self.export_complete.emit(str(out))
            return out
        except Exception as exc:  # noqa: BLE001
            logger.exception("Label export failed")
            self.export_error.emit(f"Label export failed: {exc}")
            return None

    def export_mesh(self, mesh: MeshData, format: Union[ExportFormat, str],
                    path: Union[str, Path]) -> Optional[Path]:
        """Export a surface mesh as STL, OBJ (+MTL) or PLY."""
        format = ExportFormat(format)
        try:
            path = Path(path)
            self.export_progress.emit(0.2)
            if format == ExportFormat.STL:
                out = self.stl_writer.write(mesh, path)
            elif format == ExportFormat.OBJ:
                out = self.obj_writer.write(mesh, path)
            elif format == ExportFormat.PLY:
                mesh.save_ply(path)
                out = path
            else:
                raise ValueError(f"Unsupported mesh export format: {format}")
            self.export_progress.emit(1.0)
            self.export_complete.emit(str(out))
            return out
        except Exception as exc:  # noqa: BLE001
            logger.exception("Mesh export failed")
            self.export_error.emit(f"Mesh export failed: {exc}")
            return None

    # ------------------------------------------------------------------
    # Screenshots
    # ------------------------------------------------------------------
    def export_screenshot(self, view: Any, format: Union[ExportFormat, str],
                          path: Union[str, Path]) -> Optional[Path]:
        """Export a viewer screenshot as PNG/JPEG.

        Args:
            view: Either a numpy RGB(A) array, or a VTK render window /
                anything exposing ``capture_screenshot() -> np.ndarray``.
        """
        format = ExportFormat(format)
        if format not in (ExportFormat.PNG, ExportFormat.JPEG):
            self.export_error.emit(f"Unsupported screenshot format: {format}")
            return None
        try:
            image = self._capture(view)
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.suffix:
                path = path.with_suffix(_EXTENSION_MAP[format])

            if self._save_with_pil(image, path, format):
                out = path
            else:
                out = self._save_with_vtk(image, path)
            self.export_progress.emit(1.0)
            self.export_complete.emit(str(out))
            return out
        except Exception as exc:  # noqa: BLE001
            logger.exception("Screenshot export failed")
            self.export_error.emit(f"Screenshot export failed: {exc}")
            return None

    @staticmethod
    def _capture(view: Any) -> np.ndarray:
        """Normalize the view argument to an RGB uint8 array."""
        if isinstance(view, np.ndarray):
            return view[:, :, :3].astype(np.uint8) if view.ndim == 3 else view
        if hasattr(view, "capture_screenshot"):
            return np.asarray(view.capture_screenshot())[:, :, :3].astype(np.uint8)
        # VTK render window fallback.
        try:
            import vtk
            from vtk.util import numpy_support

            win = view
            w2i = vtk.vtkWindowToImageFilter()
            w2i.SetInput(win)
            w2i.ReadFrontBufferOff()
            w2i.Update()
            img = w2i.GetOutput()
            w, h, _ = img.GetDimensions()
            arr = numpy_support.vtk_to_numpy(img.GetPointData().GetScalars())
            return np.flipud(arr.reshape(h, w, -1))[:, :, :3].astype(np.uint8)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Cannot capture screenshot from {type(view)!r}") from exc

    @staticmethod
    def _save_with_pil(image: np.ndarray, path: Path, format: ExportFormat) -> bool:
        try:
            from PIL import Image  # type: ignore

            Image.fromarray(image).save(path, quality=92 if format == ExportFormat.JPEG else None)
            return True
        except ImportError:
            return False

    @staticmethod
    def _save_with_vtk(image: np.ndarray, path: Path) -> Path:
        import vtk
        from vtk.util import numpy_support

        h, w, c = image.shape
        img = vtk.vtkImageData()
        img.SetDimensions(w, h, 1)
        flat = np.ascontiguousarray(np.flipud(image)).reshape(-1, c)
        img.GetPointData().SetScalars(numpy_support.numpy_to_vtk(flat, deep=1))

        writer = vtk.vtkPNGWriter() if path.suffix.lower() == ".png" else vtk.vtkJPEGWriter()
        writer.SetFileName(str(path))
        writer.SetInputData(img)
        writer.Write()
        return path

    # ------------------------------------------------------------------
    # Reports
    # ------------------------------------------------------------------
    def export_report(self, project: Any, format: Union[ExportFormat, str],
                      path: Union[str, Path]) -> Optional[Path]:
        """Export a project summary report as PDF (reportlab)."""
        format = ExportFormat(format)
        if format != ExportFormat.PDF:
            self.export_error.emit(f"Unsupported report format: {format}")
            return None
        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix.lower() != ".pdf":
                path = path.with_suffix(".pdf")
            self._write_pdf_report(project, path)
            self.export_progress.emit(1.0)
            self.export_complete.emit(str(path))
            return path
        except Exception as exc:  # noqa: BLE001
            logger.exception("Report export failed")
            self.export_error.emit(f"Report export failed: {exc}")
            return None

    @staticmethod
    def _write_pdf_report(project: Any, path: Path) -> None:
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.lib.units import mm
            from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
            from reportlab.lib import colors
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("reportlab is required for PDF reports") from exc

        doc = SimpleDocTemplate(str(path), pagesize=A4,
                                title="MedAxis Report", author="MedAxis")
        styles = getSampleStyleSheet()
        story: list = [
            Paragraph("MedAxis Report", styles["Title"]),
            Paragraph(f"Generated: {__import__('datetime').datetime.now().isoformat(timespec='seconds')}",
                      styles["Normal"]),
            Spacer(1, 6 * mm),
        ]

        # Patients
        patients = getattr(project, "patients", [])
        if patients:
            story.append(Paragraph("Patients", styles["Heading2"]))
            rows = [["Patient ID", "Name", "Studies"]]
            for p in patients:
                rows.append([getattr(p, "patient_id", ""),
                             getattr(p, "patient_name", ""),
                             str(len(getattr(p, "studies", [])))])
            table = Table(rows)
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2b3a55")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ]))
            story.append(table)
            story.append(Spacer(1, 4 * mm))

        # Labels
        labels = getattr(project, "labels", [])
        if labels:
            story.append(Paragraph("Segmentations", styles["Heading2"]))
            for entry in labels:
                story.append(Paragraph(
                    f"- {entry.get('name', entry.get('id', 'label'))} "
                    f"(source: {entry.get('source', 'n/a')})", styles["Normal"]))
            story.append(Spacer(1, 4 * mm))

        # Measurements
        measurements = getattr(project, "measurements", [])
        if measurements:
            story.append(Paragraph("Measurements", styles["Heading2"]))
            for m in measurements:
                if hasattr(m, "format_for_display"):
                    story.append(Paragraph(f"- {m.format_for_display()}", styles["Normal"]))
                else:
                    story.append(Paragraph(f"- {m}", styles["Normal"]))

        if len(story) == 3:
            story.append(Paragraph("Project is empty.", styles["Normal"]))

        doc.build(story)

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _export_nrrd(volume: VolumeData, path: Path) -> Path:
        try:
            import nrrd  # type: ignore

            header = {
                "space": "left-posterior-superior",
                "space origin": list(volume.origin_mm),
                "space directions": (volume.direction_matrix
                                     * np.asarray(volume.spacing_mm)[np.newaxis, :]).tolist(),
                "kinds": ["domain", "domain", "domain"],
            }
            path = Path(path)
            if path.suffix.lower() not in (".nrrd", ".nhdr"):
                path = path.with_suffix(".nrrd")
            path.parent.mkdir(parents=True, exist_ok=True)
            nrrd.write(str(path), volume.array, header)
            return path
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("pynrrd is required for NRRD export") from exc

    @staticmethod
    def default_extension(format: Union[ExportFormat, str]) -> str:
        """Return the canonical file extension for a format."""
        return _EXTENSION_MAP[ExportFormat(format)]
