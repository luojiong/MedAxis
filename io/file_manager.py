"""FileManager — unified file I/O entry point with async loading queue."""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional, Union

from PySide6.QtCore import QObject, QSettings, Signal, Slot

from core.volume_data import VolumeData
from utils.async_helpers import AsyncTaskRunner

# NOTE: relative imports — Python's built-in ``io`` module shadows an
# absolute ``import io`` of this package.
from .dicom_reader import DICOMReader
from .image_series import SeriesBuilder
from .nifti_reader import NIfTIReader

logger = logging.getLogger(__name__)

try:
    from .formats.nrrd_reader import NRRDReader
    from .formats.raw_reader import RAWReader
except ImportError:  # pragma: no cover
    NRRDReader = None  # type: ignore[assignment, misc]
    RAWReader = None  # type: ignore[assignment, misc]

__all__ = ["FileManager", "FileFormat"]

MAX_RECENT_FILES = 10


class FileFormat:
    """Known file formats."""
    DICOM = "dicom"
    NIFTI = "nifti"
    NRRD = "nrrd"
    RAW = "raw"
    PROJECT = "project"
    UNKNOWN = "unknown"


class FileManager(QObject):
    """Unified entry point for opening medical image volumes.

    Auto-detects the format (DICOM file/folder, NIfTI, NRRD, RAW) and
    dispatches to the matching reader. Supports synchronous loads,
    coroutine loads (``open_async``) and a queued QThread load path with
    progress signals.

    Signals:
        volume_loaded(VolumeData)  — a volume finished loading
        load_progress(float)       — fraction in [0, 1]
        load_error(str)            — human-readable error message
    """

    volume_loaded = Signal(object)
    load_progress = Signal(float)
    load_error = Signal(str)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._settings = QSettings("MedAxis", "MedAxis")
        self.dicom_reader = DICOMReader()
        self.nifti_reader = NIfTIReader()
        self.nrrd_reader = NRRDReader() if NRRDReader is not None else None
        self.raw_reader = RAWReader() if RAWReader is not None else None
        self.series_builder = SeriesBuilder()

        self.recent_files: list[str] = self._load_recent()
        self._queue: list[tuple[str, dict]] = []
        self._active_runner: Optional[AsyncTaskRunner] = None

    # ------------------------------------------------------------------
    # Format detection
    # ------------------------------------------------------------------
    @staticmethod
    def detect_format(path: Union[str, Path]) -> str:
        """Detect the file format by extension, then content sniffing."""
        path = Path(path)
        name = path.name.lower()

        if path.is_dir():
            return FileFormat.DICOM
        if name.endswith((".nii", ".nii.gz")):
            return FileFormat.NIFTI
        if name.endswith((".nrrd", ".nhdr")):
            return FileFormat.NRRD
        if name.endswith((".medaxis",)):
            return FileFormat.PROJECT
        if name.endswith((".raw", ".r32", ".r16", ".f32", ".u8", ".u16")):
            return FileFormat.RAW
        if name.endswith(".dcm"):
            return FileFormat.DICOM

        # Content sniffing for extension-less / unusual names.
        return FileManager._sniff_content(path)

    @staticmethod
    def _sniff_content(path: Path) -> str:
        """Identify format from the first bytes of the file."""
        try:
            with open(path, "rb") as f:
                head = f.read(352)
        except OSError:
            return FileFormat.UNKNOWN
        if len(head) >= 132 and head[128:132] == b"DICM":
            return FileFormat.DICOM
        if len(head) >= 4 and head[0:4] in (b"\x5c\x01\x00\x00", b"\xe8\x01\x00\x00"):
            return FileFormat.NIFTI  # sizeof(hdr)=348 / 540 (NIfTI-2)
        if head.startswith(b"NRRD") or head.startswith(b"# NRRD"):
            return FileFormat.NRRD
        if head[:2] == b"PK":
            return FileFormat.PROJECT
        if "." not in path.name:
            return FileFormat.DICOM  # extension-less: try DICOM
        return FileFormat.UNKNOWN

    # ------------------------------------------------------------------
    # Synchronous loading
    # ------------------------------------------------------------------
    def open(self, path: Union[str, Path], **kwargs) -> VolumeData:
        """Open a file/folder synchronously and return the volume.

        Keyword args are forwarded to the reader (e.g. ``dims``,
        ``spacing``, ``dtype`` for RAW files).

        Raises:
            ValueError: For unrecognized formats.
        """
        path = Path(path)
        fmt = self.detect_format(path)
        volume = self._load_by_format(path, fmt, kwargs)
        self._add_recent(str(path.resolve()))
        return volume

    def open_multiple(self, paths: list[Union[str, Path]], **kwargs) -> list[VolumeData]:
        """Open several paths, returning all successfully loaded volumes."""
        volumes: list[VolumeData] = []
        for p in paths:
            try:
                volumes.append(self.open(p, **kwargs))
            except Exception as exc:  # noqa: BLE001
                logger.exception("Failed to open %s", p)
                self.load_error.emit(f"Failed to open {p}: {exc}")
        return volumes

    def _load_by_format(self, path: Path, fmt: str, kwargs: dict) -> VolumeData:
        if fmt == FileFormat.NIFTI:
            return self.nifti_reader.read(path)
        if fmt == FileFormat.NRRD:
            if self.nrrd_reader is None:
                raise RuntimeError("NRRD support unavailable (install pynrrd or itk)")
            return self.nrrd_reader.read(path)
        if fmt == FileFormat.RAW:
            if self.raw_reader is None:
                raise RuntimeError("RAW reader unavailable")
            return self.raw_reader.read(path, **kwargs)
        if fmt == FileFormat.DICOM:
            return self._load_dicom(path)
        raise ValueError(f"Unrecognized file format: {path}")

    def _load_dicom(self, path: Path) -> VolumeData:
        """Load a DICOM file or folder, choosing the largest series."""
        if path.is_dir():
            images = self.dicom_reader.read_folder(
                path,
                progress_callback=lambda frac, msg: self.load_progress.emit(frac),
            )
        else:
            images = [self.dicom_reader.read_file(path)]

        if not images:
            raise ValueError(f"No DICOM instances found in {path}")

        series_map = self.series_builder.build_series(images)
        # Prefer the series with the most instances.
        best_uid = max(series_map, key=lambda uid: len(series_map[uid]))
        slices = series_map[best_uid]
        if len(series_map) > 1:
            logger.info("Multiple series present; selected %s (%d instances)",
                        best_uid, len(slices))
        volume = self.series_builder.build_volume(slices, name=path.name)
        self.load_progress.emit(1.0)
        return volume

    # ------------------------------------------------------------------
    # Async loading (coroutine, used by MCP tools)
    # ------------------------------------------------------------------
    async def open_async(self, path: Union[str, Path], **kwargs) -> VolumeData:
        """Open a path on the default thread-pool executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.open(path, **kwargs))

    # ------------------------------------------------------------------
    # Queued QThread loading
    # ------------------------------------------------------------------
    @Slot(str)
    def open_queued(self, path: Union[str, Path], **kwargs) -> None:
        """Queue a background load; emits volume_loaded/load_error when done."""
        self._queue.append((str(path), kwargs))
        if self._active_runner is None:
            self._start_next()

    def _start_next(self) -> None:
        if not self._queue:
            self._active_runner = None
            return
        path, kwargs = self._queue.pop(0)
        runner = AsyncTaskRunner(self.open, path, description=f"load {path}",
                                 cancellable=False, parent=self, **kwargs)
        runner.finished_ok.connect(self._on_loaded)
        runner.failed.connect(self._on_failed)
        self._active_runner = runner
        runner.start()

    def _on_loaded(self, volume: VolumeData) -> None:
        self.volume_loaded.emit(volume)
        self._start_next()

    def _on_failed(self, message: str) -> None:
        self.load_error.emit(message)
        self._start_next()

    # ------------------------------------------------------------------
    # Recent files
    # ------------------------------------------------------------------
    def _load_recent(self) -> list[str]:
        value = self._settings.value("filemanager/recent_files", [])
        if isinstance(value, str):
            value = [value]
        return [str(p) for p in value] if value else []

    def _add_recent(self, path: str) -> None:
        if path in self.recent_files:
            self.recent_files.remove(path)
        self.recent_files.insert(0, path)
        self.recent_files = self.recent_files[:MAX_RECENT_FILES]
        self._settings.setValue("filemanager/recent_files", self.recent_files)

    def clear_recent(self) -> None:
        self.recent_files = []
        self._settings.setValue("filemanager/recent_files", [])
