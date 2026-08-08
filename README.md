# MedAxis — Medical Imaging Rendering & Segmentation Platform

A high-performance medical imaging workstation built with Python Qt, VTK C++, ITK C++, OpenCASCADE, and CGAL.

## Quick Start (uv)

```bash
# 安装 uv (https://docs.astral.sh/uv/)
winget install astral-sh.uv

# 创建环境并安装依赖 (Python 3.12 + 全部运行时依赖 + dev 工具)
uv sync

# 运行
uv run medaxis

# 测试 / 静态检查
uv run pytest -q
uv run ruff check app core geometry io medaxis_io mcp plugins processing rendering scripts ui tests
```

> `.python-version` 固定 Python 3.12;`uv.lock` 锁定全部依赖。
> 原生扩展的 DLL 搜索路径写在 `~/.medaxis/config.yaml` 的 `native.cpp_prefix`
> (或环境变量 `MEDAXIS_CPP_PREFIX`,分号分隔),见下方 Native Build。

## Native Build (C++ 扩展)

原生扩展产出 5 个 pybind11 模块,部署到 `medaxis/_native/`:

| 模块 | 来源 |
| --- | --- |
| `medaxis_bridge` | `native/bridge` — numpy ↔ vtkImageData ↔ itkImage 零拷贝桥 |
| `medaxis_itk` | `native/itk_ext` — 阈值/形态学/区域生长/LevelSet 等 ITK Filter |
| `medaxis_cgal` | `native/cgal_ext` — 网格重建/表面处理/曲率 (CGAL) |
| `medaxis_occ` | `native/occ_ext` — 曲线采样/Frenet 标架 (OpenCASCADE) |
| `medaxis_radiomics` | `native/radiomics_ext` — 一阶影像组学特征 |

### 依赖准备 (Windows)

C++ 依赖通过 conda-forge (micromamba) 获取预编译库,ITK 需要源码构建:

```bash
# 1. micromamba + C++ 依赖 (vtk 9.6.2 / cgal 6.0.1 / occt 7.8.1 / eigen / boost)
#    vtk 需 --no-deps 安装: conda 的 qt6-main 包含 >260 字符路径, micromamba 解压失败;
#    随后补装 vtk 的第三方依赖 (expat/lzma/tbb 开发包等)
micromamba create -n medaxis-cpp -c conda-forge cgal eigen boost-cpp gmp mpfr occt=7.8 -y
micromamba install -n medaxis-cpp --no-deps -y vtk=9.6.2 vtk-base=9.6.2
micromamba install -n medaxis-cpp -c conda-forge -y expat liblzma-devel tbb-devel ffmpeg
#    注意: 若 micromamba 报 "remove_all: directory is not empty",
#    用 build-deps/extract_longpath_pkgs.py 手动解压到包缓存

# 2. ITK 5.4.7 源码构建 (conda-forge 的 itk 不含 C++ 头文件)
git clone --depth 1 --branch v5.4.7 https://github.com/InsightSoftwareConsortium/ITK.git build-deps/ITK-5.4.7
cmake -S build-deps/ITK-5.4.7 -B build-deps/itk-build -G "Visual Studio 18 2026" -A x64 ^
  -DCMAKE_CONFIGURATION_TYPES=Release -DBUILD_SHARED_LIBS=ON -DBUILD_TESTING=OFF ^
  -DITK_BUILD_DEFAULT_MODULES=OFF ^
  -DModule_ITKCommon=ON -DModule_ITKImageFilterBase=ON -DModule_ITKImageIntensity=ON ^
  -DModule_ITKImageGradient=ON -DModule_ITKImageGrid=ON -DModule_ITKThresholding=ON ^
  -DModule_ITKStatistics=ON -DModule_ITKRegionGrowing=ON -DModule_ITKWatersheds=ON ^
  -DModule_ITKLevelSets=ON -DModule_ITKMathematicalMorphology=ON ^
  -DModule_ITKBinaryMathematicalMorphology=ON -DModule_ITKConnectedComponents=ON ^
  -DModule_ITKDistanceMap=ON -DModule_ITKRegistrationCommon=ON -DModule_ITKSmoothing=ON ^
  -DModule_ITKAnisotropicSmoothing=ON ^
  -DCMAKE_INSTALL_PREFIX=D:/1AProject/MedAxis/build-deps/itk-install
cmake --build build-deps/itk-build --config Release --parallel 8
cmake --install build-deps/itk-build --config Release

# 3. 记录 C++ 前缀到 ~/.medaxis/config.yaml:
#    uv run python -c "from utils.config import AppConfig; c=AppConfig.instance(); \
#      c.set('native','cpp_prefix', r'<mamba env>;<itk-install>'); c.save()"
#    (或导出环境变量 MEDAXIS_CPP_PREFIX, 分号分隔)
```

### 编译 MedAxis 原生模块

```bash
uv pip install pybind11   # 提供 CMake 配置 (已加入 dev 依赖组)
cmake -S . -B build-native -G "Visual Studio 18 2026" -A x64 ^
  -DMEDAXIS_BUILD_NATIVE=ON ^
  "-DCMAKE_PREFIX_PATH=<micromamba env>;<micromamba env>/Library;<itk-install>" ^
  "-DPython_EXECUTABLE=<repo>/.venv/Scripts/python.exe" ^
  -DPYBIND11_FINDPYTHON=ON ^
  "-Dpybind11_DIR=<repo>/.venv/Lib/site-packages/pybind11/share/cmake/pybind11"
cmake --build build-native --config Release --parallel 8

# 部署 .pyd 到 medaxis/_native/ (core/native_extensions.py 自动加入搜索路径)
copy build-native\native\bridge\Release\medaxis_bridge.pyd medaxis\_native\
copy build-native\native\itk_ext\Release\medaxis_itk.pyd     medaxis\_native\
copy build-native\native\cgal_ext\Release\medaxis_cgal.pyd    medaxis\_native\
copy build-native\native\occ_ext\Release\medaxis_occ.pyd     medaxis\_native\
copy build-native\native\radiomics_ext\Release\medaxis_radiomics.pyd medaxis\_native\

# 验证 (5 个模块应全部 available=True)
uv run python -c "from core.native_extensions import NativeExtensionRegistry; print(NativeExtensionRegistry().status())"
```

> 版本对齐要求:Python 侧 `vtk==9.6.2` / `itk==5.4.7` 必须与 C++ 侧一致
> (pyproject.toml 已 pin),否则同一进程内会出现两套 VTK/ITK ABI。
> Windows 长路径注意:conda 部分包含 >260 字符路径,micromamba 解压失败时
> 可用 `build-deps/extract_longpath_pkgs.py` 手动解压到包缓存。

## Configuration

- Config: `~/.medaxis/config.yaml`
- Logs: `~/.medaxis/logs/`
- Autosave: `~/.medaxis/autosave/`
