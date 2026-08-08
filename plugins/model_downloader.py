"""
Model downloader — fetch AI model archives from HuggingFace / Zenodo /
custom URLs into the local model cache.
"""
from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path
from typing import Callable, Optional

MODEL_CACHE = Path.home() / ".medaxis" / "models"


class ModelDownloader:
    """Downloads model packages with progress reporting."""

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        self.cache_dir = Path(cache_dir or MODEL_CACHE)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def download(self, url: str, name: str,
                 progress: Optional[Callable[[float], None]] = None,
                 headers: Optional[dict] = None) -> Path:
        """Download ``url`` into the cache as ``name``; returns the path."""
        target = self.cache_dir / name
        if target.exists() and target.stat().st_size > 0:
            return target

        tmp = target.with_suffix(target.suffix + ".part")
        request = urllib.request.Request(url, headers=headers or {})
        with urllib.request.urlopen(request) as resp, open(tmp, "wb") as out:
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            while True:
                chunk = resp.read(1024 * 256)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if progress is not None and total:
                    progress(min(done / total, 1.0))
        shutil.move(str(tmp), str(target))
        return target

    def extract_archive(self, archive: Path, name: Optional[str] = None) -> Path:
        """Extract a .zip/.tar.gz model archive; returns the output dir."""
        import tarfile
        import zipfile

        out_dir = self.cache_dir / (name or archive.stem)
        out_dir.mkdir(parents=True, exist_ok=True)
        if archive.suffix == ".zip" or str(archive).endswith(".whl"):
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(out_dir)
        else:
            with tarfile.open(archive) as tf:
                tf.extractall(out_dir)
        return out_dir

    def list_models(self) -> list[Path]:
        """Model files/archives present in the cache."""
        entries = []
        for entry in sorted(self.cache_dir.iterdir()):
            if entry.is_file():
                entries.append(entry)
            elif entry.is_dir() and (entry / "model.onnx").exists():
                entries.append(entry / "model.onnx")
        return entries


# Built-in model catalog (external AI services, blueprint §6.4).
MODEL_CATALOG = [
    {
        "id": "totalsegmentator-v2",
        "name": "TotalSegmentator v2",
        "modality": "CT",
        "num_classes": 104,
        "source": "https://github.com/wasserth/TotalSegmentator",
        "description": "Whole-body CT multi-organ segmentation (external service).",
    },
    {
        "id": "medsam",
        "name": "MedSAM",
        "modality": "Any",
        "num_classes": 1,
        "source": "https://github.com/bowang-lab/MedSAM",
        "description": "Promptable segmentation foundation model (external service).",
    },
    {
        "id": "nnunet-2d",
        "name": "nnU-Net v2 (user-trained)",
        "modality": "Any",
        "num_classes": "task-dependent",
        "source": "https://github.com/MIC-DKFZ/nnUNet",
        "description": "User-trained nnU-Net v2 inference server (external service).",
    },
    {
        "id": "monai-deploy",
        "name": "MONAI Deploy",
        "modality": "Any",
        "num_classes": "task-dependent",
        "source": "https://github.com/Project-MONAI/monai-deploy",
        "description": "Standard MONAI informatics gateway (external service).",
    },
]
