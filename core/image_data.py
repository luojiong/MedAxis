"""ImageData — 2D image slice data model."""
import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class ImageData:
    """Single 2D image slice (one DICOM instance)."""
    array: np.ndarray                    # 2D pixel data
    spacing: tuple[float, float]         # [row_spacing, col_spacing] mm
    origin: tuple[float, float, float]   # ImagePositionPatient [x,y,z] mm
    orientation: tuple[float, ...]       # ImageOrientationPatient (6 values)
    instance_uid: str
    series_uid: str
    sop_class_uid: str
    slice_location: Optional[float] = None
    window_center: Optional[float] = None
    window_width: Optional[float] = None
    rescale_slope: float = 1.0
    rescale_intercept: float = 0.0

    @property
    def hounsfield_units(self) -> np.ndarray:
        """Convert raw pixel values to Hounsfield Units."""
        return self.array.astype(np.float32) * self.rescale_slope + self.rescale_intercept

    @property
    def rows(self) -> int: return self.array.shape[0]

    @property
    def cols(self) -> int: return self.array.shape[1]
