# 🩻 MedAxis — Medical Imaging Rendering & Segmentation Platform

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org)
[![PySide6](https://img.shields.io/badge/UI-PySide6-41CD52?logo=qt&logoColor=white)](https://doc.qt.io/qtforpython/)
[![VTK](https://img.shields.io/badge/VTK-9.6-006F8E)](https://vtk.org)
[![ITK](https://img.shields.io/badge/ITK-5.4-0071BC)](https://itk.org)
[![CGAL](https://img.shields.io/badge/CGAL-6.0-4B0082)](https://www.cgal.org)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue)](LICENSE)

A high-performance medical imaging workstation built with **Python Qt, VTK C++, ITK C++, OpenCASCADE and CGAL** — 40% Python orchestration + 60% C++ compute, exactly as the design blueprint specifies.

> 🧠 **Core principle:** Python is the *commander*, C++ is the *soldier*. Every performance-sensitive algorithm executes in the compiled native layer; Python only handles UI, orchestration, MCP protocol and plugins.

---

## ✨ Highlights

| | |
|---|---|
| 🗂️ **I/O** | DICOM (folder/batch), NIfTI, NRRD, MetaImage, RAW, `.medaxis` project archives |
| 🧭 **Rendering** | Axial / coronal / sagittal slices, MPR (oblique reslice), CPR (curved reformation), 3D volume rendering, label overlays, scale bar & ROI annotations |
| 🔬 **Algorithms** | **95 registered algorithms** — filtering, morphology, thresholding, segmentation, reconstruction, registration, arithmetic, enhancement, radiomics |
| ⚡ **Native compute** | **53 algorithms run in C++/ITK** (zero-copy numpy ↔ ITK bridge); e.g. Gaussian on 64³: **2.4 ms native vs 9.9 s** via the Python binding |
| 🧮 **Radiomics** | **114 features** — first-order, GLCM(30), GLRLM(15), GLSZM(15), GLDM(14), NGTDM(5), shape 2D/3D — computed in C++ |
| 🩸 **Vascular tools** | Vessel centerline extraction (3D thinning + Dijkstra) and **stent view** (circular unwrap with MIP) driving the CPR view |
| 🤖 **MCP & AI** | Built-in MCP server (stdio/SSE, JSON-RPC, permissions, audit) with 12 tools incl. `run_algorithm` and `run_script`; external AI service clients (REST + TotalSegmentator adapter) |
| 📜 **Scripting** | Mimics-style console, script editor, action recorder, 3-level sandbox, hook system, template library |
| 🧩 **Plugins** | `.py` / `.pyd` plugin discovery with automatic MCP tool exposure; AI Model Zoo + downloader |
| 💾 **Workflow** | Label editing (8 tools + undo/redo + boolean ops), exports: NIfTI / NRRD / DICOM / DICOM-SEG / **RTSTRUCT** / STL / OBJ / PLY / **3MF** / PNG / PDF, comparison view, cine playback + MP4 recording, autosave |

---

## 🚀 Quick Start

```bash
# 1. Install uv (https://docs.astral.sh/uv/)
winget install astral-sh.uv

# 2. Create the environment (Python 3.12 + all dependencies + dev tools)
uv sync

# 3. Run
uv run medaxis
```

> 📌 `.python-version` pins Python 3.12; `uv.lock` locks every dependency.
> Native extension DLL search paths are read from `native.cpp_prefix` in
> `~/.medaxis/config.yaml` (or the `MEDAXIS_CPP_PREFIX` env var) — see
> [Native Build](#-native-build-windows).

### ✅ Verify the installation

```bash
uv run pytest -q                      # 10 tests
uv run ruff check app core geometry io medaxis_io mcp plugins processing rendering scripts ui tests
uv run python -c "from core.native_extensions import NativeExtensionRegistry; print(NativeExtensionRegistry().status())"
# → all five medaxis_* modules should report available=True
```

---

## 🔧 Native Build (Windows)

The five pybind11 extension modules are deployed to `medaxis/_native/`:

| Module | Source | Purpose |
|---|---|---|
| `medaxis_bridge` | `native/bridge` | Zero-copy numpy ↔ vtkImageData ↔ itkImage bridge |
| `medaxis_itk` | `native/itk_ext` | ITK filters: filtering, morphology, segmentation, registration (7 dispatch entry points) |
| `medaxis_cgal` | `native/cgal_ext` | Mesh reconstruction / processing / curvature (CGAL 6) |
| `medaxis_occ` | `native/occ_ext` | Curve sampling / Frenet frames / arc length (OpenCASCADE 7.8) |
| `medaxis_radiomics` | `native/radiomics_ext` | First-order + GLCM/GLRLM/GLSZM/GLDM/NGTDM texture features (C++) |

### 📦 Dependencies (conda-forge + one source build)

C++ dependencies come from a micromamba/conda-forge environment; **ITK must be built from source** (conda-forge's `itk` ships no C++ dev files):

```bash
# 1. micromamba + C++ deps (vtk 9.6.2 / cgal 6.0.1 / occt 7.8.1 / eigen / boost)
#    ⚠️ vtk needs --no-deps: conda's qt6-main contains >260-char paths that
#       micromamba cannot extract (see build-deps/extract_longpath_pkgs.py)
micromamba create -n medaxis-cpp -c conda-forge cgal eigen boost-cpp gmp mpfr occt=7.8 -y
micromamba install -n medaxis-cpp --no-deps -y vtk=9.6.2 vtk-base=9.6.2
micromamba install -n medaxis-cpp -c conda-forge -y expat liblzma-devel tbb-devel ffmpeg

# 2. ITK 5.4.7 source build
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
  -DModule_ITKDenoising=ON -DModule_ITKClassifiers=ON -DModule_ITKDisplacementField=ON ^
  -DModule_ITKPDEDeformableRegistration=ON -DModule_ITKRegistrationMethodsv4=ON ^
  -DModule_ITKOptimizers=ON -DModule_ITKOptimizersv4=ON ^
  -DCMAKE_INSTALL_PREFIX=D:/1AProject/MedAxis/build-deps/itk-install
cmake --build build-deps/itk-build --config Release --parallel 8
cmake --install build-deps/itk-build --config Release

# 3. Record the C++ prefixes (uv run python):
#    from utils.config import AppConfig; c = AppConfig.instance()
#    c.set('native', 'cpp_prefix', r'<mamba env>;<itk-install>'); c.save()
```

### 🛠️ Compile the native modules

```bash
uv pip install pybind11   # CMake configs (already in the dev group)
cmake -S . -B build-native -G "Visual Studio 18 2026" -A x64 ^
  -DMEDAXIS_BUILD_NATIVE=ON ^
  "-DCMAKE_PREFIX_PATH=<micromamba env>;<micromamba env>/Library;<itk-install>" ^
  "-DPython_EXECUTABLE=<repo>/.venv/Scripts/python.exe" ^
  -DPYBIND11_FINDPYTHON=ON ^
  "-Dpybind11_DIR=<repo>/.venv/Lib/site-packages/pybind11/share/cmake/pybind11"
cmake --build build-native --config Release --parallel 8

# Deploy the .pyd files to medaxis/_native/ (auto-added to sys.path)
copy build-native\native\bridge\Release\medaxis_bridge.pyd        medaxis\_native\
copy build-native\native\itk_ext\Release\medaxis_itk.pyd          medaxis\_native\
copy build-native\native\cgal_ext\Release\medaxis_cgal.pyd        medaxis\_native\
copy build-native\native\occ_ext\Release\medaxis_occ.pyd          medaxis\_native\
copy build-native\native\radiomics_ext\Release\medaxis_radiomics.pyd medaxis\_native\
```

> ⚠️ **Version alignment:** the Python side is pinned to `vtk==9.6.2` and
> `itk==5.4.7` (in `pyproject.toml`) to match the C++ layer — mixing versions
> would load two VTK/ITK ABIs into one process.
> 💡 **Long paths:** some conda packages contain >260-char paths that break
> micromamba's extractor; use `build-deps/extract_longpath_pkgs.py` to unpack
> them manually into the package cache.

---

## 🏗️ Architecture

```
┌────────────────────────── Python (orchestration) ──────────────────────────┐
│  app/        application controller, workspace, autosave                    │
│  ui/         Qt widgets: slice/mpr/cpr/volume views, panels, label editor   │
│  io/         DICOM / NIfTI / NRRD / RAW readers, DICOM writers, exports     │
│  core/       VolumeData / LabelData / MeshData models, project archives     │
│  processing/ AlgorithmRegistry (95 algorithms) + native dispatch            │
│  mcp/        MCP server: JSON-RPC, stdio/SSE, 12 tools, audit               │
│  scripts/    console, editor, recorder, sandbox, hooks, templates           │
│  plugins/    plugin manager, model zoo, AI service clients                  │
│  geometry/   centerline, stent view, CGAL/OCC wrappers                      │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │ pybind11 (zero-copy numpy)
┌────────────────────────── C++ (compute) ────────────────────────────────────┐
│  native/bridge        numpy ↔ vtkImageData ↔ itkImage                       │
│  native/itk_ext       filtering · morphology · segmentation · registration  │
│  native/cgal_ext      mesh reconstruction · processing · curvature          │
│  native/occ_ext       curve sampling · Frenet frames                        │
│  native/radiomics_ext first-order + 5 texture families                      │
│  native/vtk_ext       custom VTK filters (reslice / CPR)                    │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 📦 Algorithm execution model

Every algorithm registered in `AlgorithmRegistry` is **dispatched to the
compiled backend first** (`processing/native_backend.py`); the Python
implementation only runs as a fallback for algorithms without a native twin.

```
medaxis.algorithms.threshold(vol, lower=50, upper=300)
        │
        ▼
AlgorithmRegistry.run ──► medaxis_itk.threshold_3d (C++/ITK) ◄── 53 algorithms
        │                        ▲
        └──► Python fallback ────┘   (thinning, NL-means, …)
```

---

## 🧪 Testing

```bash
uv run pytest -q                          # unit + regression tests
uv run python build-deps/native_ops_smoke.py   # 47-op native smoke test
uv run python build-deps/e2e_test.py           # load → segment → filter → render
uv run python build-deps/views_e2e.py          # 4-view + overlay rendering
uv run python build-deps/mpr_cpr_e2e.py        # MPR/CPR + centerline
uv run python build-deps/scripts_e2e.py        # sandbox/console/recorder/hooks
```

---

## 🗺️ Roadmap status

All blueprint phases (0–8) are implemented at the core level:

| Phase | Status |
|---|---|
| 0 · Native build & bindings | ✅ Complete (Windows, MSVC + conda-forge + ITK source) |
| 1 · Application foundation | ✅ Complete (uv toolchain, Qt shell, CI) |
| 2 · I/O & data model | ✅ Complete |
| 3 · Rendering engine | ✅ Complete (slice/MPR/CPR/volume, annotations, desktop-verified) |
| 4 · Processing catalog | ✅ Complete (95 algorithms, 53 native) |
| 5 · MCP & AI | ✅ Complete (12 tools, AI clients, verified E2E) |
| 5b · Scripting | ✅ Complete (console/editor/recorder/sandbox/hooks) |
| 6 · Geometry & advanced viz | ✅ Complete (centerline, stent view, CPR) |
| 7 · Plugins & ecosystem | ✅ Complete (py/pyd plugins, MCP auto-exposure, model zoo) |
| 8 · Persistence & workflow | ✅ Complete (archives, exports, RTSTRUCT/3MF, comparison, cine) |

> 📋 Full details: [`docs/implementation_status.md`](docs/implementation_status.md)
> and the design blueprint [`docs/design_plan.html`](docs/design_plan.html).

---

## ⚙️ Configuration

| Item | Location |
|---|---|
| 🎛️ Config | `~/.medaxis/config.yaml` |
| 📜 Logs | `~/.medaxis/logs/` |
| 💾 Autosave | `~/.medaxis/autosave/` |
| 📦 Script library | `~/.medaxis/scripts/` |
| 🤖 Model cache | `~/.medaxis/models/` |
| 🧩 Plugins | `~/.medaxis/plugins/` |

---

## 📄 License

Apache-2.0 — see [LICENSE](LICENSE).
