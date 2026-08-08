"""DICOM reading — single files, folders, and async loading.

Uses pydicom for parsing and GDCM (via pydicom's gdcm pixel-data handler)
for compressed transfer syntaxes (JPEG, JPEG-LS, JPEG2000, RLE).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np

from core.image_data import ImageData

try:
    import pydicom
    from pydicom.dataset import Dataset, FileDataset

    HAS_PYDICOM = True
except ImportError:  # pragma: no cover
    pydicom = None  # type: ignore[assignment]
    Dataset = None  # type: ignore[assignment, misc]
    FileDataset = None  # type: ignore[assignment, misc]
    HAS_PYDICOM = False

try:
    from pydicom.pixel_data_handlers import gdcm_handler

    HAS_GDCM = gdcm_handler.is_available()
except ImportError:  # pragma: no cover
    gdcm_handler = None  # type: ignore[assignment]
    HAS_GDCM = False

try:
    from PySide6.QtCore import QObject, QThread, Signal
except ImportError:  # pragma: no cover
    QObject = object  # type: ignore[assignment, misc]
    QThread = None  # type: ignore[assignment, misc]

    def Signal(*args, **kwargs):  # type: ignore[no-redef]
        return None

logger = logging.getLogger(__name__)

__all__ = ["DICOMReader", "DICOMLoadWorker", "HAS_PYDICOM", "HAS_GDCM"]

# Transfer syntaxes that require a compressed-syntax decoder (GDCM).
_COMPRESSED_SYNTAXES = {
    "1.2.840.10008.1.2.4.50",   # JPEG Baseline
    "1.2.840.10008.1.2.4.51",   # JPEG Extended
    "1.2.840.10008.1.2.4.57",   # JPEG Lossless NH
    "1.2.840.10008.1.2.4.70",   # JPEG Lossless
    "1.2.840.10008.1.2.4.80",   # JPEG-LS Lossless
    "1.2.840.10008.1.2.4.81",   # JPEG-LS Near-Lossless
    "1.2.840.10008.1.2.4.90",   # JPEG 2000 Lossless
    "1.2.840.10008.1.2.4.91",   # JPEG 2000
    "1.2.840.10008.1.2.5",      # RLE Lossless
}

_DEFAULT_ORIENTATION = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


def _is_dicom_filename(name: str) -> bool:
    """Return True for *.dcm, *.DCM and extension-less files."""
    lower = name.lower()
    if lower.endswith(".dcm"):
        return True
    return "." not in name


class DICOMReader:
    """Read DICOM files and folders into :class:`~core.image_data.ImageData`.

    Attributes:
        progress_callback: Optional ``fn(fraction: float, message: str)``
            invoked during folder reads with ``fraction`` in [0, 1].
    """

    def __init__(self, progress_callback: Optional[Callable[[float, str], None]] = None) -> None:
        self.progress_callback = progress_callback
        self.last_error: str = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def read_file(self, path: Union[str, Path]) -> ImageData:
        """Read a single DICOM file into an :class:`ImageData`.

        Raises:
            RuntimeError: If pydicom is unavailable or the file is unreadable.
        """
        if not HAS_PYDICOM:
            raise RuntimeError("pydicom is required to read DICOM files")
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"DICOM file not found: {path}")

        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=False, force=True)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to parse DICOM file {path}: {exc}") from exc

        return self._dataset_to_image_data(ds, path)

    def read_folder(
        self,
        path: Union[str, Path],
        recursive: bool = True,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> list[ImageData]:
        """Recursively read a DICOM folder into a list of slices.

        Files are matched by ``.dcm``/``.DCM`` extension or by having no
        extension at all. Non-DICOM files are skipped with a warning.

        Args:
            path: Folder to scan.
            recursive: Descend into sub-directories.
            progress_callback: Overrides :attr:`progress_callback` if given.

        Returns:
            List of :class:`ImageData` instances (unsorted).
        """
        folder = Path(path)
        if not folder.is_dir():
            raise NotADirectoryError(f"Not a DICOM folder: {path}")

        callback = progress_callback or self.progress_callback
        candidates = self._collect_files(folder, recursive)
        total = len(candidates)
        if total == 0:
            logger.warning("No DICOM candidate files found in %s", folder)
            return []

        images: list[ImageData] = []
        for index, file_path in enumerate(candidates):
            try:
                images.append(self.read_file(file_path))
            except Exception as exc:  # noqa: BLE001
                logger.debug("Skipping non-DICOM file %s: %s", file_path, exc)
            if callback is not None:
                callback((index + 1) / total, f"Reading {file_path.name}")
        logger.info("Read %d DICOM instances from %s", len(images), folder)
        return images

    def load_folder_async(
        self,
        path: Union[str, Path],
        on_finished: Optional[Callable[[list[ImageData]], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
        on_progress: Optional[Callable[[float], None]] = None,
        parent: Optional[object] = None,
    ) -> "DICOMLoadWorker":
        """Start an asynchronous folder load on a background QThread.

        Returns:
            The started :class:`DICOMLoadWorker` (keep a reference to it).
        """
        worker = DICOMLoadWorker(self, Path(path), parent)
        if on_finished is not None:
            worker.finished_list.connect(on_finished)  # type: ignore[union-attr]
        if on_error is not None:
            worker.error.connect(on_error)  # type: ignore[union-attr]
        if on_progress is not None:
            worker.progress.connect(on_progress)  # type: ignore[union-attr]
        worker.start()  # type: ignore[union-attr]
        return worker

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _collect_files(folder: Path, recursive: bool) -> list[Path]:
        walker = folder.rglob("*") if recursive else folder.glob("*")
        files = [p for p in walker if p.is_file() and _is_dicom_filename(p.name)]
        files.sort()
        return files

    def _dataset_to_image_data(self, ds: "Dataset", path: Path) -> ImageData:
        """Convert a pydicom dataset into an :class:`ImageData`."""
        pixel_array = self._get_pixel_array(ds)

        # Geometry
        origin = tuple(float(v) for v in getattr(ds, "ImagePositionPatient", (0.0, 0.0, 0.0)))
        orientation = tuple(float(v) for v in getattr(ds, "ImageOrientationPatient", _DEFAULT_ORIENTATION))
        pixel_spacing = getattr(ds, "PixelSpacing", (1.0, 1.0))
        spacing = (float(pixel_spacing[0]), float(pixel_spacing[1]))
        slice_location = getattr(ds, "SliceLocation", None)
        if slice_location is not None:
            slice_location = float(slice_location)

        # Modality LUT rescale
        rescale_slope = float(getattr(ds, "RescaleSlope", 1.0))
        rescale_intercept = float(getattr(ds, "RescaleIntercept", 0.0))

        # Window center/width may be multi-valued; take the first.
        window_center = self._first_value(getattr(ds, "WindowCenter", None))
        window_width = self._first_value(getattr(ds, "WindowWidth", None))

        num_frames = int(getattr(ds, "NumberOfFrames", 1))

        image = ImageData(
            array=pixel_array,
            spacing=spacing,
            origin=origin,
            orientation=orientation,
            instance_uid=str(getattr(ds, "SOPInstanceUID", pydicom.uid.generate_uid())),
            series_uid=str(getattr(ds, "SeriesInstanceUID", "")),
            sop_class_uid=str(getattr(ds, "SOPClassUID", "")),
            slice_location=slice_location,
            window_center=window_center,
            window_width=window_width,
            rescale_slope=rescale_slope,
            rescale_intercept=rescale_intercept,
        )

        # Extra (non-core) attributes used downstream.
        image.num_frames = num_frames
        image.instance_number = int(getattr(ds, "InstanceNumber", 0) or 0)
        image.slice_thickness = float(getattr(ds, "SliceThickness", 1.0) or 1.0)
        image.patient_name = str(getattr(ds, "PatientName", ""))
        image.patient_id = str(getattr(ds, "PatientID", ""))
        image.study_instance_uid = str(getattr(ds, "StudyInstanceUID", ""))
        image.modality = str(getattr(ds, "Modality", ""))
        image.series_description = str(getattr(ds, "SeriesDescription", ""))
        image.source_path = str(path)

        # Apply rescale eagerly for quantitative work (keeps raw in metadata).
        if rescale_slope != 1.0 or rescale_intercept != 0.0:
            image.array = pixel_array.astype(np.float32) * rescale_slope + rescale_intercept
        image.metadata = {
            "TransferSyntaxUID": str(getattr(ds.file_meta, "TransferSyntaxUID", "")),
            "Manufacturer": str(getattr(ds, "Manufacturer", "")),
            "StudyDescription": str(getattr(ds, "StudyDescription", "")),
            "SeriesDescription": image.series_description,
            "SeriesNumber": int(getattr(ds, "SeriesNumber", 0) or 0),
            "AcquisitionNumber": int(getattr(ds, "AcquisitionNumber", 0) or 0),
            "Rows": int(getattr(ds, "Rows", 0)),
            "Columns": int(getattr(ds, "Columns", 0)),
            "BitsAllocated": int(getattr(ds, "BitsAllocated", 0)),
            "PhotometricInterpretation": str(getattr(ds, "PhotometricInterpretation", "")),
            "NumberOfFrames": num_frames,
        }
        return image

    @staticmethod
    def _first_value(value) -> Optional[float]:
        if value is None:
            return None
        try:  # MultiValue
            return float(value[0])
        except (TypeError, IndexError):
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

    @staticmethod
    def _get_pixel_array(ds: "Dataset") -> np.ndarray:
        """Decode pixel data, falling back to GDCM for compressed syntaxes."""
        try:
            arr = ds.pixel_array
            return np.asarray(arr)
        except Exception as exc:  # noqa: BLE001
            syntax = str(getattr(ds.file_meta, "TransferSyntaxUID", ""))
            if HAS_GDCM and (syntax in _COMPRESSED_SYNTAXES or True):
                try:
                    logger.debug("Falling back to GDCM decoder for %s", syntax)
                    return np.asarray(gdcm_handler.get_pixel_data(ds))
                except Exception as gdcm_exc:  # noqa: BLE001
                    raise RuntimeError(
                        f"Cannot decode pixel data (transfer syntax {syntax}): "
                        f"{exc}; GDCM fallback also failed: {gdcm_exc}"
                    ) from gdcm_exc
            raise RuntimeError(f"Cannot decode pixel data: {exc}") from exc


class DICOMLoadWorker(QThread):  # type: ignore[misc, valid-type]
    """Background QThread that reads a DICOM folder.

    Signals:
        slice_ready(ImageData)       — emitted for each parsed instance
        progress(float)              — fraction in [0, 1]
        finished_list(list)          — all slices on completion
        error(str)                   — fatal error message
    """

    slice_ready = Signal(object)
    progress = Signal(float)
    finished_list = Signal(list)
    error = Signal(str)

    def __init__(self, reader: DICOMReader, folder: Path, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._reader = reader
        self._folder = folder
        self._cancel_requested = False

    def cancel(self) -> None:
        """Request cooperative cancellation."""
        self._cancel_requested = True

    def run(self) -> None:  # noqa: D401
        images: list[ImageData] = []
        try:
            candidates = self._reader._collect_files(self._folder, recursive=True)
            total = len(candidates)
            if total == 0:
                self.error.emit(f"No DICOM files found in {self._folder}")
                return
            for index, file_path in enumerate(candidates):
                if self._cancel_requested:
                    logger.info("DICOM load cancelled at %d/%d", index, total)
                    break
                try:
                    image = self._reader.read_file(file_path)
                    images.append(image)
                    self.slice_ready.emit(image)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Skipping %s: %s", file_path, exc)
                self.progress.emit((index + 1) / total)
            self.finished_list.emit(images)
        except Exception as exc:  # noqa: BLE001
            logger.exception("DICOM folder load failed")
            self.error.emit(str(exc))
