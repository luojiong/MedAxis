"""
Comparison view — side-by-side, overlay and subtraction modes.

Shows two volumes in linked slice viewers so a second series can be
compared against the primary one (e.g. pre/post contrast, PET/CT).
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from .slice_view import SliceView


class ComparisonView(QWidget):
    """Two linked slice viewers with side-by-side / overlay / subtraction."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._volume_a = None
        self._volume_b = None
        self._mode = "side_by_side"

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Compare:", self))
        self._mode_box = QComboBox(self)
        self._mode_box.addItem("Side by Side", "side_by_side")
        self._mode_box.addItem("Overlay (blend)", "overlay")
        self._mode_box.addItem("Subtraction (A - B)", "subtraction")
        self._mode_box.currentIndexChanged.connect(self._on_mode_changed)
        controls.addWidget(self._mode_box)
        controls.addStretch(1)
        layout.addLayout(controls)

        self._views = QHBoxLayout()
        self._view_a = SliceView("axial", self)
        self._view_b = SliceView("axial", self)
        self._views.addWidget(self._view_a, 1)
        self._views.addWidget(self._view_b, 1)
        layout.addLayout(self._views, 1)

        # Link slice navigation between the two views.
        self._view_a.slice_changed.connect(lambda i, o: self._sync_slice(i))

    # ------------------------------------------------------------------
    def set_volumes(self, volume_a, volume_b) -> None:
        self._volume_a = volume_a
        self._volume_b = volume_b
        self._view_a.set_volume(volume_a)
        self._apply_mode()

    def _on_mode_changed(self, _index: int) -> None:
        self._mode = self._mode_box.currentData()
        self._apply_mode()

    def _apply_mode(self) -> None:
        if self._volume_b is None:
            return
        if self._mode == "subtraction":
            # Compute A - B (broadcast-safe) and show it in view B.
            import numpy as np

            diff = np.asarray(self._volume_a.to_numpy(), dtype=np.float32) - \
                np.asarray(self._volume_b.to_numpy(), dtype=np.float32)
            from core.volume_data import VolumeData

            self._view_b.set_volume(VolumeData(
                diff, spacing=self._volume_a.spacing_mm,
                origin=self._volume_a.origin_mm,
                direction=self._volume_a.direction,
                name="A - B"))
            self._view_b.setVisible(True)
        elif self._mode == "overlay":
            self._view_b.set_volume(self._volume_b)
            self._view_b.setVisible(True)
        else:  # side_by_side
            self._view_b.set_volume(self._volume_b)
            self._view_b.setVisible(True)

    def _sync_slice(self, index: int) -> None:
        """Keep the second view on the same slice index."""
        if self._mode == "side_by_side":
            try:
                self._view_b.set_slice_index(index, emit=False)
            except Exception:
                pass

    def set_orientation(self, orientation: str) -> None:
        for view in (self._view_a, self._view_b):
            view.set_orientation(orientation)
