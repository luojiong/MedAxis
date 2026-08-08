"""Base classes for format readers/writers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Union


class BaseReader(ABC):
    """Abstract base class for file readers."""

    #: File extensions handled by this reader, e.g. ``(".nrrd", ".nhdr")``.
    supported_extensions: tuple[str, ...] = ()

    @abstractmethod
    def read(self, path: Union[str, Path], **kwargs: Any) -> Any:
        """Read a file and return the loaded data object.

        Args:
            path: Path to the file.
            **kwargs: Format-specific options.

        Returns:
            The loaded data (e.g. ``VolumeData``).
        """

    def can_read(self, path: Union[str, Path]) -> bool:
        """Return True when the path's extension is supported."""
        name = Path(path).name.lower()
        return any(name.endswith(ext) for ext in self.supported_extensions)


class BaseWriter(ABC):
    """Abstract base class for file writers."""

    #: File extensions produced by this writer, e.g. ``(".stl",)``.
    supported_extensions: tuple[str, ...] = ()

    @abstractmethod
    def write(self, data: Any, path: Union[str, Path], **kwargs: Any) -> Path:
        """Write ``data`` to ``path`` and return the written path.

        Args:
            data: The object to serialize (mesh, volume, label...).
            path: Destination path.
            **kwargs: Format-specific options.
        """
