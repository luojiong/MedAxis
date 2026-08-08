"""DICOM writing — secondary capture, segmentation, structured reports."""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import numpy as np

from core.volume_data import VolumeData
from core.label_data import LabelData

try:
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import (
        ExplicitVRLittleEndian,
        generate_uid,
    )

    HAS_PYDICOM = True
except ImportError:  # pragma: no cover
    pydicom = None  # type: ignore[assignment]
    HAS_PYDICOM = False

try:
    import highdicom as hd

    HAS_HIGHDICOM = True
except ImportError:  # pragma: no cover
    hd = None  # type: ignore[assignment]
    HAS_HIGHDICOM = False

logger = logging.getLogger(__name__)

__all__ = ["DICOMWriter"]

# SOP class UIDs
_SECONDARY_CAPTURE = "1.2.840.10008.5.1.4.1.1.7"      # SC Image Storage
_SEGMENTATION = "1.2.840.10008.5.1.4.1.1.66.4"       # Segmentation Storage
_BASIC_TEXT_SR = "1.2.840.10008.5.1.4.1.1.88.11"     # Basic Text SR Storage

_APP_NAME = "MEDAXIS"


class DICOMWriter:
    """Write DICOM objects derived from MedAxis data.

    Uses ``highdicom`` when available for segmentations; otherwise falls
    back to a minimal pydicom implementation.
    """

    def __init__(self, implementation_name: str = _APP_NAME) -> None:
        self.implementation_name = implementation_name

    # ------------------------------------------------------------------
    # Secondary Capture (view screenshots)
    # ------------------------------------------------------------------
    def write_secondary_capture(
        self,
        volume: Optional[VolumeData],
        view_screenshot: np.ndarray,
        path: Union[str, Path],
    ) -> Path:
        """Write a viewer screenshot as a DICOM Secondary Capture.

        Args:
            volume: Source volume used to copy patient/study context
                (may be None for anonymous captures).
            view_screenshot: RGB(A) uint8 array, shape (H, W, 3|4).
            path: Output ``.dcm`` path.

        Returns:
            The written path.
        """
        self._require_pydicom()
        img = np.asarray(view_screenshot)
        if img.ndim != 3 or img.shape[2] not in (3, 4):
            raise ValueError("Screenshot must be HxWx3 or HxWx4 uint8")
        img = img[:, :, :3].astype(np.uint8)

        ds = self._base_dataset(_SECONDARY_CAPTURE)
        src = self._source_context(volume)
        ds.PatientName = src.get("PatientName", "Anonymous")
        ds.PatientID = src.get("PatientID", "ANON")
        ds.StudyInstanceUID = src.get("StudyInstanceUID", generate_uid())
        ds.StudyID = src.get("StudyID", "1")
        ds.AccessionNumber = src.get("AccessionNumber", "")
        ds.ReferringPhysicianName = src.get("ReferringPhysicianName", "")
        ds.StudyDate = src.get("StudyDate", ds.ContentDate)
        ds.StudyTime = src.get("StudyTime", ds.ContentTime)

        ds.Modality = "OT"
        ds.SeriesDescription = "MedAxis View Screenshot"
        ds.ImageType = ["DERIVED", "SECONDARY", "SCREEN SAVE"]
        ds.InstanceNumber = 1
        ds.Rows = img.shape[0]
        ds.Columns = img.shape[1]
        ds.SamplesPerPixel = 3
        ds.PhotometricInterpretation = "RGB"
        ds.PlanarConfiguration = 0
        ds.BitsAllocated = 8
        ds.BitsStored = 8
        ds.HighBit = 7
        ds.PixelRepresentation = 0
        ds.LossyImageCompression = "00"
        ds.PixelData = img.tobytes()

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        ds.save_as(str(path), write_like_original=False)
        logger.info("Wrote DICOM SC: %s", path)
        return path

    # ------------------------------------------------------------------
    # Segmentation (SEG)
    # ------------------------------------------------------------------
    def write_segmentation(
        self,
        label_data: LabelData,
        volume: VolumeData,
        path: Union[str, Path],
    ) -> Path:
        """Write a label map as a DICOM Segmentation object.

        Prefers ``highdicom``; falls back to a minimal pydicom SEG.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if HAS_HIGHDICOM:
            self._write_seg_highdicom(label_data, volume, path)
        else:
            logger.warning("highdicom unavailable; writing minimal pydicom SEG")
            self._write_seg_pydicom(label_data, volume, path)
        logger.info("Wrote DICOM SEG: %s", path)
        return path

    def _write_seg_highdicom(self, label: LabelData, volume: VolumeData, path: Path) -> None:
        from highdicom.seg import Segmentation, SegmentDescription, SegmentationTypeValues
        from highdicom.sr.coding import CodedConcept

        array = label.array.astype(np.uint8)
        segments = []
        for value in sorted(int(v) for v in np.unique(array) if v != 0):
            name = label.class_names.get(value, f"Label {value}")
            segments.append(
                SegmentDescription(
                    segment_number=value,
                    segment_label=name,
                    segmented_property_category=CodedConcept("M-01000", "SRT", "Morphologically Abnormal Structure"),
                    segmented_property_type=CodedConcept("M-8000/0", "SRT", "Neoplasm"),
                    algorithm_type="MANUAL",
                )
            )
        if not segments:
            raise ValueError("Label map has no non-zero segments")

        # Stack of binary frames: one 2D frame per (label value, slice).
        frames: list[np.ndarray] = []
        segment_numbers: list[int] = []
        for seg in segments:
            mask = (array == seg.segment_number)
            for k in range(mask.shape[0]):
                if mask[k].any():
                    frames.append(mask[k].astype(np.uint8))
                    segment_numbers.append(seg.segment_number)

        seg = Segmentation(
            source_images=[],  # reference-less; highdicom accepts empty with geometry kwargs
            pixel_array=np.stack(frames, axis=0),
            segmentation_type=SegmentationTypeValues.BINARY,
            segment_descriptions=segments,
            series_instance_uid=generate_uid(),
            series_number=100,
            sop_instance_uid=generate_uid(),
            instance_number=1,
            manufacturer=self.implementation_name,
            manufacturer_model_name="MedAxis Segmentation",
            software_versions="1.0",
            device_serial_number="0",
            transfer_syntax_uid=ExplicitVRLittleEndian,
        )
        seg.save_as(str(path))

    def _write_seg_pydicom(self, label: LabelData, volume: VolumeData, path: Path) -> None:
        ds = self._base_dataset(_SEGMENTATION)
        src = self._source_context(volume)
        ds.PatientName = src.get("PatientName", "Anonymous")
        ds.PatientID = src.get("PatientID", "ANON")
        ds.StudyInstanceUID = src.get("StudyInstanceUID", generate_uid())

        ds.Modality = "SEG"
        ds.SeriesDescription = label.name or "MedAxis Segmentation"
        ds.InstanceNumber = 1
        ds.SegmentNumber = 1
        ds.ContentLabel = (label.name or "SEGMENTATION")[:16].upper()
        ds.ContentDescription = "Created by MedAxis"
        ds.ContentCreatorName = self.implementation_name

        array = (label.array != 0).astype(np.uint8)
        nk, nj, ni = array.shape
        ds.Rows = nj
        ds.Columns = ni
        ds.NumberOfFrames = nk
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 1
        ds.BitsStored = 1
        ds.HighBit = 0
        ds.PixelRepresentation = 0

        # Pack bits (LSB first), pad each frame to a byte boundary.
        packed_frames = []
        frame_bytes = (ni * nj + 7) // 8
        for k in range(nk):
            bits = np.packbits(array[k].ravel(), bitorder="little")
            padded = np.zeros(frame_bytes, dtype=np.uint8)
            padded[: len(bits)] = bits
            packed_frames.append(padded.tobytes())
        ds.PixelData = b"".join(packed_frames)

        # Geometry from the volume.
        ds.ImagePositionPatient = [float(v) for v in volume.origin_mm]
        dir_mat = volume.direction_matrix
        ds.ImageOrientationPatient = [
            float(dir_mat[0, 0]), float(dir_mat[1, 0]), float(dir_mat[2, 0]),
            float(dir_mat[0, 1]), float(dir_mat[1, 1]), float(dir_mat[2, 1]),
        ]
        ds.PixelSpacing = [float(volume.spacing[1]), float(volume.spacing[0])]
        ds.SpacingBetweenSlices = float(volume.spacing[2])

        ds.save_as(str(path), write_like_original=False)

    # ------------------------------------------------------------------
    # Structured Report (SR)
    # ------------------------------------------------------------------
    def write_structured_report(
        self,
        measurements: list[dict],
        patient_info: dict,
        path: Union[str, Path],
    ) -> Path:
        """Write measurements as a DICOM Basic Text SR.

        Args:
            measurements: List of dicts; each must contain ``"text"`` and
                may contain ``"type"``, ``"value"``, ``"units"``.
            patient_info: Dict with optional keys PatientName, PatientID,
                StudyInstanceUID, StudyDescription.
            path: Output ``.dcm`` path.
        """
        self._require_pydicom()
        ds = self._base_dataset(_BASIC_TEXT_SR)
        ds.PatientName = patient_info.get("PatientName", "Anonymous")
        ds.PatientID = patient_info.get("PatientID", "ANON")
        ds.StudyInstanceUID = patient_info.get("StudyInstanceUID", generate_uid())
        ds.StudyDescription = patient_info.get("StudyDescription", "")
        ds.Modality = "SR"
        ds.SeriesDescription = "MedAxis Measurement Report"
        ds.InstanceNumber = 1
        ds.CompletionFlag = "COMPLETE"
        ds.VerificationFlag = "UNVERIFIED"
        ds.ValueType = "CONTAINER"
        ds.ConceptNameCodeSequence = self._code_sequence(
            "126000", "DCM", "Imaging Measurement Report"
        )

        content: list[Dataset] = []
        for index, m in enumerate(measurements, start=1):
            item = Dataset()
            item.RelationshipType = "CONTAINS"
            item.ValueType = "TEXT"
            item.ConceptNameCodeSequence = self._code_sequence(
                "125007", "DCM", m.get("type", f"Measurement {index}")
            )
            item.TextValue = str(m.get("text", ""))
            content.append(item)

            if "value" in m:
                num_item = Dataset()
                num_item.RelationshipType = "HAS OBS CONTEXT"
                num_item.ValueType = "NUM"
                num_item.ConceptNameCodeSequence = self._code_sequence(
                    "125007", "DCM", "Value"
                )
                measured = Dataset()
                measured.NumericValue = str(m["value"])
                measured.MeasurementUnitsCodeSequence = self._code_sequence(
                    "mm", "UCUM", m.get("units", "mm")
                )
                num_item.MeasuredValueSequence = [measured]
                content.append(num_item)

        ds.ContentSequence = content
        ds.ContentDate = ds.ContentDate
        ds.ContentTime = ds.ContentTime

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        ds.save_as(str(path), write_like_original=False)
        logger.info("Wrote DICOM SR: %s (%d measurements)", path, len(measurements))
        return path

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _require_pydicom() -> None:
        if not HAS_PYDICOM:
            raise RuntimeError("pydicom is required for DICOM writing")

    def _base_dataset(self, sop_class_uid: str) -> "Dataset":
        """Create a minimal dataset with file meta and common type-1 tags."""
        now = datetime.now()
        ds = Dataset()
        ds.file_meta = FileMetaDataset()
        ds.file_meta.MediaStorageSOPClassUID = sop_class_uid
        ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.file_meta.ImplementationClassUID = generate_uid()
        ds.file_meta.ImplementationVersionName = self.implementation_name

        ds.is_little_endian = True
        ds.is_implicit_VR = False

        ds.SOPClassUID = sop_class_uid
        ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
        ds.SeriesInstanceUID = generate_uid()
        ds.FrameOfReferenceUID = generate_uid()
        ds.ContentDate = now.strftime("%Y%m%d")
        ds.ContentTime = now.strftime("%H%M%S")
        ds.InstanceCreationDate = ds.ContentDate
        ds.InstanceCreationTime = ds.ContentTime
        ds.Manufacturer = self.implementation_name
        return ds

    @staticmethod
    def _source_context(volume: Optional[VolumeData]) -> dict:
        """Extract DICOM patient/study context from a volume, if present."""
        if volume is None:
            return {}
        return {
            "PatientName": getattr(volume, "patient_name", "Anonymous"),
            "PatientID": getattr(volume, "patient_id", "ANON"),
            "StudyInstanceUID": getattr(volume, "study_instance_uid", "") or None,
            "StudyID": "1",
            "AccessionNumber": "",
            "ReferringPhysicianName": "",
            "StudyDate": datetime.now().strftime("%Y%m%d"),
            "StudyTime": datetime.now().strftime("%H%M%S"),
        }

    @staticmethod
    def _code_sequence(code_value: str, scheme: str, meaning: str) -> list["Dataset"]:
        item = Dataset()
        item.CodeValue = code_value
        item.CodingSchemeDesignator = scheme
        item.CodeMeaning = meaning
        return [item]
