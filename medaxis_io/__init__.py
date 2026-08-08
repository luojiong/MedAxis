"""Import-safe alias for MedAxis' file I/O package.

The source directory is named ``io`` for historical reasons, but Python's
standard-library ``io`` module is a built-in and always wins an absolute
import.  Point this package at the existing source directory so imports such
as ``medaxis_io.file_manager`` keep one canonical implementation.
"""

from pathlib import Path

__path__ = [str(Path(__file__).resolve().parent.parent / "io")]

