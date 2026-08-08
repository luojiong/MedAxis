"""Patient / Study / Series hierarchical models for DICOM organization.

Structure: Patient -> Studies -> Series -> (volumes loaded from series).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

__all__ = ["PatientModel", "StudyModel", "SeriesModel"]


@dataclass
class SeriesModel:
    """A DICOM series (leaf of the hierarchy, maps to one or more volumes)."""

    series_uid: str
    series_number: int = 0
    series_description: str = ""
    modality: str = ""
    num_instances: int = 0
    volume_id: Optional[str] = None  # set once the series is loaded as a volume

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "series_uid": self.series_uid,
            "series_number": self.series_number,
            "series_description": self.series_description,
            "modality": self.modality,
            "num_instances": self.num_instances,
            "volume_id": self.volume_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SeriesModel":
        """Deserialize from :meth:`to_dict` output."""
        return cls(
            series_uid=d["series_uid"],
            series_number=d.get("series_number", 0),
            series_description=d.get("series_description", ""),
            modality=d.get("modality", ""),
            num_instances=d.get("num_instances", 0),
            volume_id=d.get("volume_id"),
        )


@dataclass
class StudyModel:
    """A DICOM study containing one or more series."""

    study_uid: str
    study_date: str = ""          # YYYYMMDD
    study_time: str = ""          # HHMMSS
    study_description: str = ""
    modality: str = ""
    series: list[SeriesModel] = field(default_factory=list)

    def add_series(self, series: SeriesModel) -> None:
        """Add a series (replaces an existing one with the same UID)."""
        self.series = [s for s in self.series if s.series_uid != series.series_uid]
        self.series.append(series)

    def get_series(self, series_uid: str) -> Optional[SeriesModel]:
        """Find a series by UID."""
        for s in self.series:
            if s.series_uid == series_uid:
                return s
        return None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "study_uid": self.study_uid,
            "study_date": self.study_date,
            "study_time": self.study_time,
            "study_description": self.study_description,
            "modality": self.modality,
            "series": [s.to_dict() for s in self.series],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StudyModel":
        """Deserialize from :meth:`to_dict` output."""
        return cls(
            study_uid=d["study_uid"],
            study_date=d.get("study_date", ""),
            study_time=d.get("study_time", ""),
            study_description=d.get("study_description", ""),
            modality=d.get("modality", ""),
            series=[SeriesModel.from_dict(s) for s in d.get("series", [])],
        )


@dataclass
class PatientModel:
    """A DICOM patient containing one or more studies."""

    patient_id: str
    patient_name: str = ""
    birth_date: str = ""  # YYYYMMDD
    sex: str = ""         # "M" / "F" / "O"
    studies: list[StudyModel] = field(default_factory=list)

    def add_study(self, study: StudyModel) -> None:
        """Add a study (replaces an existing one with the same UID)."""
        self.studies = [s for s in self.studies if s.study_uid != study.study_uid]
        self.studies.append(study)

    def get_study(self, study_uid: str) -> Optional[StudyModel]:
        """Find a study by UID."""
        for s in self.studies:
            if s.study_uid == study_uid:
                return s
        return None

    def iter_series(self):
        """Iterate over (study, series) pairs for the whole patient."""
        for study in self.studies:
            for series in study.series:
                yield study, series

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "patient_id": self.patient_id,
            "patient_name": self.patient_name,
            "birth_date": self.birth_date,
            "sex": self.sex,
            "studies": [s.to_dict() for s in self.studies],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PatientModel":
        """Deserialize from :meth:`to_dict` output."""
        return cls(
            patient_id=d["patient_id"],
            patient_name=d.get("patient_name", ""),
            birth_date=d.get("birth_date", ""),
            sex=d.get("sex", ""),
            studies=[StudyModel.from_dict(s) for s in d.get("studies", [])],
        )

    # ------------------------------------------------------------------
    # Qt integration
    # ------------------------------------------------------------------

    def get_dicom_tree_model(self):
        """Build a ``QStandardItemModel`` for displaying the hierarchy in a
        ``QTreeView``.

        Requires PyQt6. Raises ``ImportError`` if Qt is unavailable.
        """
        try:
            from PyQt6.QtGui import QStandardItem, QStandardItemModel
        except ImportError as exc:  # pragma: no cover
            raise ImportError("PyQt6 is required for get_dicom_tree_model()") from exc

        model = QStandardItemModel()
        model.setHorizontalHeaderLabels(["Patient / Study / Series"])

        patient_item = QStandardItem(f"{self.patient_name or self.patient_id}")
        patient_item.setToolTip(
            f"ID: {self.patient_id}\nDOB: {self.birth_date}\nSex: {self.sex}"
        )
        patient_item.setData(self.patient_id, 0x0100)  # Qt.ItemDataRole.UserRole

        for study in self.studies:
            desc = study.study_description or study.study_date or "Study"
            study_item = QStandardItem(f"{desc} [{study.modality or '—'}]")
            study_item.setToolTip(f"UID: {study.study_uid}\nDate: {study.study_date}")
            study_item.setData(study.study_uid, 0x0100)

            for series in study.series:
                sdesc = series.series_description or f"Series {series.series_number}"
                series_item = QStandardItem(
                    f"#{series.series_number} {sdesc} "
                    f"({series.modality}, {series.num_instances} img)"
                )
                series_item.setToolTip(f"UID: {series.series_uid}")
                series_item.setData(series.series_uid, 0x0100)
                study_item.appendRow(series_item)

            patient_item.appendRow(study_item)

        model.appendRow(patient_item)
        return model
