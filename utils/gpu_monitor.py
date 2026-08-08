"""MedAxis — GPU Monitor.

Query GPU information (name, driver version, memory) using nvidia-smi
when available, falling back to WMI on Windows. All queries are cached
briefly to avoid repeated subprocess spawns.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

_CACHE_TTL_SECONDS = 5.0


@dataclass
class GPUInfo:
    """Snapshot of a single GPU's state."""
    index: int = 0
    gpu_name: str = "Unknown"
    driver_version: str = "Unknown"
    total_memory_mb: float = 0.0
    used_memory_mb: float = 0.0
    free_memory_mb: float = 0.0

    @property
    def utilization_percent(self) -> float:
        if self.total_memory_mb <= 0:
            return 0.0
        return round(100.0 * self.used_memory_mb / self.total_memory_mb, 1)

    def summary(self) -> str:
        return (f"{self.gpu_name} | driver {self.driver_version} | "
                f"{self.used_memory_mb:.0f}/{self.total_memory_mb:.0f} MB used "
                f"({self.utilization_percent}%)")


@dataclass
class GPUReport:
    """Aggregate report for all detected GPUs."""
    available: bool = False
    gpus: List[GPUInfo] = field(default_factory=list)
    source: str = "none"   # nvidia-smi | wmi | none

    @property
    def primary(self) -> Optional[GPUInfo]:
        return self.gpus[0] if self.gpus else None

    @property
    def total_memory_mb(self) -> float:
        return sum(g.total_memory_mb for g in self.gpus)

    @property
    def used_memory_mb(self) -> float:
        return sum(g.used_memory_mb for g in self.gpus)

    @property
    def free_memory_mb(self) -> float:
        return sum(g.free_memory_mb for g in self.gpus)

    @property
    def gpu_name(self) -> str:
        return self.primary.gpu_name if self.primary else "No GPU detected"

    @property
    def driver_version(self) -> str:
        return self.primary.driver_version if self.primary else "N/A"


class GPUMonitor:
    """Queries GPU state with a small TTL cache."""

    def __init__(self, cache_ttl: float = _CACHE_TTL_SECONDS) -> None:
        self._cache_ttl = cache_ttl
        self._last_query: float = 0.0
        self._report: Optional[GPUReport] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def is_available(self) -> bool:
        return self.query().available

    def query(self, force: bool = False) -> GPUReport:
        """Return a (possibly cached) GPU report."""
        now = time.monotonic()
        if (not force and self._report is not None
                and now - self._last_query < self._cache_ttl):
            return self._report

        report = self._query_nvidia_smi()
        if not report.available:
            report = self._query_wmi()

        self._report = report
        self._last_query = now
        return report

    # ------------------------------------------------------------------
    # nvidia-smi backend
    # ------------------------------------------------------------------
    @staticmethod
    def _query_nvidia_smi() -> GPUReport:
        exe = shutil.which("nvidia-smi")
        if exe is None:
            return GPUReport(available=False)

        try:
            out = subprocess.run(
                [exe,
                 "--query-gpu=index,name,driver_version,"
                 "memory.total,memory.used,memory.free",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5, check=False,
            )
        except (subprocess.SubprocessError, OSError):
            logger.debug("nvidia-smi execution failed.", exc_info=True)
            return GPUReport(available=False)

        if out.returncode != 0:
            logger.debug("nvidia-smi returned %d: %s",
                         out.returncode, out.stderr.strip())
            return GPUReport(available=False)

        gpus: List[GPUInfo] = []
        for line in out.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            try:
                gpus.append(GPUInfo(
                    index=int(parts[0]),
                    gpu_name=parts[1],
                    driver_version=parts[2],
                    total_memory_mb=float(parts[3]),
                    used_memory_mb=float(parts[4]),
                    free_memory_mb=float(parts[5]),
                ))
            except ValueError:
                logger.debug("Unparsable nvidia-smi line: %r", line)

        return GPUReport(available=bool(gpus), gpus=gpus, source="nvidia-smi")

    # ------------------------------------------------------------------
    # WMI backend (Windows fallback — memory info limited)
    # ------------------------------------------------------------------
    @staticmethod
    def _query_wmi() -> GPUReport:
        import sys
        if sys.platform != "win32":
            return GPUReport(available=False)

        try:
            # Prefer the `wmi` package when installed.
            import wmi as wmi_module  # type: ignore
            conn = wmi_module.WMI()
            gpus: List[GPUInfo] = []
            for i, adapter in enumerate(conn.Win32_VideoController()):
                total_mb = float(getattr(adapter, "AdapterRAM", 0) or 0) / (1024 * 1024)
                gpus.append(GPUInfo(
                    index=i,
                    gpu_name=str(getattr(adapter, "Name", "Unknown")),
                    driver_version=str(getattr(adapter, "DriverVersion", "Unknown")),
                    total_memory_mb=total_mb,
                    used_memory_mb=0.0,
                    free_memory_mb=total_mb,
                ))
            return GPUReport(available=bool(gpus), gpus=gpus, source="wmi")
        except ImportError:
            pass
        except Exception:  # noqa: BLE001
            logger.debug("WMI query via wmi package failed.", exc_info=True)

        # Last resort: PowerShell CIM query (no extra dependencies).
        try:
            ps = shutil.which("powershell")
            if ps is None:
                return GPUReport(available=False)
            out = subprocess.run(
                [ps, "-NoProfile", "-Command",
                 "Get-CimInstance Win32_VideoController | "
                 "Select-Object Name,DriverVersion,AdapterRAM | "
                 "ForEach-Object { \"$($_.Name)|$($_.DriverVersion)|$($_.AdapterRAM)\" }"],
                capture_output=True, text=True, timeout=10, check=False,
            )
            if out.returncode != 0:
                return GPUReport(available=False)
            gpus = []
            for i, line in enumerate(l.strip() for l in out.stdout.splitlines()):
                if not line:
                    continue
                parts = line.split("|")
                try:
                    total_mb = float(parts[2] or 0) / (1024 * 1024) if len(parts) > 2 else 0.0
                except ValueError:
                    total_mb = 0.0
                gpus.append(GPUInfo(
                    index=i,
                    gpu_name=parts[0] if parts else "Unknown",
                    driver_version=parts[1] if len(parts) > 1 else "Unknown",
                    total_memory_mb=total_mb,
                    used_memory_mb=0.0,
                    free_memory_mb=total_mb,
                ))
            return GPUReport(available=bool(gpus), gpus=gpus, source="wmi")
        except (subprocess.SubprocessError, OSError):
            logger.debug("PowerShell GPU query failed.", exc_info=True)
            return GPUReport(available=False)


# Module-level convenience singleton
_monitor = GPUMonitor()


def get_gpu_report(force: bool = False) -> GPUReport:
    return _monitor.query(force=force)


def is_gpu_available() -> bool:
    return _monitor.is_available


if __name__ == "__main__":
    report = get_gpu_report(force=True)
    print(f"GPU available: {report.available} (source: {report.source})")
    for gpu in report.gpus:
        print(gpu.summary())
