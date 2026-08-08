"""
Script manager — script library with built-in templates, history and
import/export.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

#: Built-in script templates (blueprint Phase 5b).
_TEMPLATES: dict[str, str] = {
    "DICOM 批量导入": """\
# 批量导入 DICOM 文件夹中的所有序列
import os
from pathlib import Path

folder = Path(medaxis.ui.dialog_input("输入 DICOM 文件夹路径:", default=""))
if folder.is_dir():
    for entry in sorted(folder.iterdir()):
        if entry.is_dir():
            try:
                vol = medaxis.io.open_dicom(str(entry))
                print(f"loaded {vol.name} {vol.dimensions}")
            except Exception as exc:
                print(f"skip {entry.name}: {exc}")
""",
    "肺部分割": """\
# 阈值 + 最大连通域提取肺部
vol = medaxis.data.volumes()[0]
label = medaxis.algorithms.threshold(vol, lower=-1024, upper=-400)
label = medaxis.algorithms.morphology(label, operation="opening", radius=2)
print("lung voxels:", label)
""",
    "骨骼 STL 导出": """\
# 骨窗阈值 → 表面重建 → STL 导出
vol = medaxis.data.volumes()[0]
bone = medaxis.algorithms.threshold(vol, lower=300, upper=3000)
mesh = medaxis.algorithms.marching_cubes(bone)
medaxis.io.export_mesh(mesh, "bone.stl")
print("exported bone.stl")
""",
    "测量报告": """\
# 对当前 label 输出体积/统计报告
labels = medaxis.data.labels()
for label in labels:
    stats = medaxis.algorithms.label_statistics(label)
    print(label.name, stats)
""",
    "影像组学特征": """\
# 提取 full radiomics suite (一阶+纹理+形状)
vol = medaxis.data.volumes()[0]
label = medaxis.data.labels()[0]
features = medaxis.algorithms.radiomics(vol, label)
for name, value in sorted(features.items()):
    print(f"{name}: {value:.4f}")
""",
}


class ScriptManager:
    """Loads/saves scripts and provides the built-in template library."""

    def __init__(self, library_dir: Optional[str] = None) -> None:
        self._library_dir = Path(library_dir or
                                 os.path.join(Path.home(), ".medaxis", "scripts"))
        self._library_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def templates(self) -> dict[str, str]:
        return dict(_TEMPLATES)

    def list_scripts(self) -> List[Path]:
        return sorted(self._library_dir.glob("*.py"))

    def save_script(self, name: str, code: str) -> Path:
        path = self._library_dir / (name if name.endswith(".py") else name + ".py")
        path.write_text(code, encoding="utf-8")
        return path

    def load_script(self, name: str) -> str:
        path = self._library_dir / (name if name.endswith(".py") else name + ".py")
        return path.read_text(encoding="utf-8")

    def delete_script(self, name: str) -> None:
        path = self._library_dir / (name if name.endswith(".py") else name + ".py")
        if path.exists():
            path.unlink()

    def recent(self, limit: int = 10) -> List[Path]:
        files = sorted(self._library_dir.glob("*.py"), key=lambda p: p.stat().st_mtime,
                       reverse=True)
        return files[:limit]
