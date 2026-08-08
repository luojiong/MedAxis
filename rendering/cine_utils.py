"""Window/level presets loader + cine recording (MP4 via ffmpeg)."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import List, Optional

_PRESETS_PATH = Path(__file__).resolve().parent.parent / "resources" / "presets" / "window_level_presets.json"


def load_presets() -> List[dict]:
    """Load the window/level preset table from resources."""
    if not _PRESETS_PATH.exists():
        return []
    with open(_PRESETS_PATH, "r", encoding="utf-8") as fh:
        return list(json.load(fh).get("presets", []))


def apply_preset(view, name: str) -> Optional[dict]:
    """Apply a named preset to a slice view; returns the preset dict."""
    for preset in load_presets():
        if preset["name"] == name:
            view.set_window_level(float(preset["window"]), float(preset["level"]))
            return preset
    return None


def find_ffmpeg() -> Optional[str]:
    """Locate the ffmpeg binary (conda env, system PATH)."""
    candidates = [
        os.path.join(os.environ.get("MEDAXIS_CPP_PREFIX", "").split(";")[0], "Library", "bin", "ffmpeg.exe")
    ]
    for cand in candidates:
        if cand and os.path.exists(cand):
            return cand
    return shutil_which("ffmpeg")


def shutil_which(name: str) -> Optional[str]:
    import shutil

    return shutil.which(name)


class CineRecorder:
    """Records a view's cine playback to MP4 (H.264) via ffmpeg.

    Frames are captured from the widget with QWidget.grab() and piped to
    ffmpeg's stdin as raw RGB.
    """

    def __init__(self, view) -> None:
        self._view = view
        self._process: Optional[subprocess.Popen] = None
        self._frame_count = 0

    def start(self, path: str, fps: float = 10.0) -> bool:
        ffmpeg = find_ffmpeg()
        if ffmpeg is None:
            raise RuntimeError("ffmpeg not found (install it or set MEDAXIS_CPP_PREFIX)")
        self._path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{self._view.width()}x{self._view.height()}",
            "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "veryfast", path,
        ]
        self._process = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self._frame_count = 0
        return True

    def capture_frame(self) -> int:
        """Grab one frame and feed it to ffmpeg; returns the frame count."""
        if self._process is None:
            return 0
        import numpy as np
        from PySide6.QtGui import QImage

        pixmap = self._view.grab()
        image = pixmap.toImage().convertToFormat(QImage.Format.Format_RGB888)
        width, height = image.width(), image.height()
        # PySide6: bits() returns a memoryview sized to the image data.
        arr = np.frombuffer(image.constBits(), dtype=np.uint8)
        expected = height * image.bytesPerLine()
        if arr.size != expected:
            arr = np.frombuffer(bytes(image.constBits()), dtype=np.uint8)
        arr = arr.reshape(height, image.bytesPerLine())
        arr = arr[:, : width * 3]  # strip padding
        self._process.stdin.write(arr.tobytes())
        self._frame_count += 1
        return self._frame_count

    def stop(self) -> str:
        """Finalize the recording; returns the output path."""
        if self._process is not None:
            try:
                self._process.stdin.close()
            except Exception:
                pass
            self._process.wait(timeout=60)
            self._process = None
        return getattr(self, "_path", "")

    @property
    def recording(self) -> bool:
        return self._process is not None

    @property
    def frame_count(self) -> int:
        return self._frame_count
