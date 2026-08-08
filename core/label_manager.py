"""Project-scoped label map manager."""

from __future__ import annotations

from typing import Iterable, Optional
from uuid import uuid4

import numpy as np

from .label_data import LabelData


class LabelManager:
    """Owns label-map identity, ordering, visibility, and metadata."""

    def __init__(self) -> None:
        self._labels: dict[str, LabelData] = {}
        self._order: list[str] = []

    def add(self, label: LabelData) -> LabelData:
        if label.id in self._labels:
            raise ValueError(f"Label already exists: {label.id}")
        self._labels[label.id] = label
        self._order.append(label.id)
        return label

    def create(
        self,
        array: np.ndarray,
        parent_volume_id: str,
        name: str = "Label",
        source: str = "manual",
        label_id: Optional[str] = None,
    ) -> LabelData:
        return self.add(LabelData(
            id=label_id or str(uuid4()),
            name=name,
            parent_volume_id=str(parent_volume_id),
            array=np.asarray(array, dtype=np.int16),
            source=source,
        ))

    def get(self, label_id: str) -> Optional[LabelData]:
        return self._labels.get(str(label_id))

    def remove(self, label_id: str) -> Optional[LabelData]:
        label_id = str(label_id)
        label = self._labels.pop(label_id, None)
        if label is not None:
            self._order.remove(label_id)
        return label

    def clear(self) -> None:
        self._labels.clear()
        self._order.clear()

    def list(self) -> list[LabelData]:
        return [self._labels[label_id] for label_id in self._order]

    def for_volume(self, volume_id: str) -> list[LabelData]:
        return [label for label in self.list() if label.parent_volume_id == str(volume_id)]

    def replace(self, labels: Iterable[LabelData]) -> None:
        self.clear()
        for label in labels:
            self.add(label)

    def metadata(self) -> list[dict]:
        return [label.to_dict() for label in self.list()]
