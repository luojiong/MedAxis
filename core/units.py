"""Unit conversion and human-readable formatting utilities."""
from __future__ import annotations

__all__ = [
    "mm_to_cm",
    "cm_to_mm",
    "mm_to_m",
    "m_to_mm",
    "mm_to_in",
    "in_to_mm",
    "mm3_to_ml",
    "ml_to_mm3",
    "deg_to_rad",
    "rad_to_deg",
    "format_distance_mm",
    "format_area_mm2",
    "format_volume_ml",
    "format_angle_deg",
]


# ----------------------------------------------------------------------
# Conversions
# ----------------------------------------------------------------------

def mm_to_cm(value: float) -> float:
    """Millimetres to centimetres."""
    return value / 10.0


def cm_to_mm(value: float) -> float:
    """Centimetres to millimetres."""
    return value * 10.0


def mm_to_m(value: float) -> float:
    """Millimetres to metres."""
    return value / 1000.0


def m_to_mm(value: float) -> float:
    """Metres to millimetres."""
    return value * 1000.0


def mm_to_in(value: float) -> float:
    """Millimetres to inches."""
    return value / 25.4


def in_to_mm(value: float) -> float:
    """Inches to millimetres."""
    return value * 25.4


def mm3_to_ml(value: float) -> float:
    """Cubic millimetres to millilitres (1 mL = 1000 mm^3)."""
    return value / 1000.0


def ml_to_mm3(value: float) -> float:
    """Millilitres to cubic millimetres."""
    return value * 1000.0


def deg_to_rad(value: float) -> float:
    """Degrees to radians."""
    import math
    return math.radians(value)


def rad_to_deg(value: float) -> float:
    """Radians to degrees."""
    import math
    return math.degrees(value)


# ----------------------------------------------------------------------
# Formatting
# ----------------------------------------------------------------------

def format_distance_mm(value: float) -> str:
    """Format a distance in mm as a human-readable string.

    Examples:
        >>> format_distance_mm(12.345)
        '12.3 mm'
        >>> format_distance_mm(123.4)
        '12.34 cm'
    """
    if abs(value) >= 100.0:
        return f"{mm_to_cm(value):.2f} cm"
    return f"{value:.1f} mm"


def format_area_mm2(value: float) -> str:
    """Format an area in mm^2 as a human-readable string."""
    if abs(value) >= 100.0:
        return f"{value / 100.0:.2f} cm\u00b2"
    return f"{value:.1f} mm\u00b2"


def format_volume_ml(value: float) -> str:
    """Format a volume in mL as a human-readable string.

    Examples:
        >>> format_volume_ml(1234.5)
        '1234.5 mL'
        >>> format_volume_ml(0.42)
        '0.42 mL'
    """
    if abs(value) >= 1000.0:
        return f"{value / 1000.0:.2f} L"
    return f"{value:.1f} mL"


def format_angle_deg(value: float) -> str:
    """Format an angle in degrees, e.g. ``'45.2°'``."""
    return f"{value:.1f}\u00b0"
