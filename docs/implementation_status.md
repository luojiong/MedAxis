# MedAxis Implementation Status

This document reconciles the design blueprint with the implementation in this
repository. It records verified behavior, rather than treating a planned file
or interface as a completed feature.

Last verified: 2026-08-08 (full blueprint pass complete)

## Toolchain (uv + conda-forge + MSVC)

- Python environment managed by **uv** (`uv sync`, Python 3.12.13, `uv.lock`).
- Native C++ layer builds and links on Windows (MSVC 14.50, VS 2026, CMake 4.3):
  all five pybind11 modules compile, import and pass functional smoke tests.
- C++ deps: conda-forge VTK 9.6.2 / CGAL 6.0.1 / OCCT 7.8.1 / Eigen 5.0.1 /
  Boost / GMP / MPFR, plus a source-built **ITK 5.4.7** (DisplacementField,
  Denoising, PDEDeformableRegistration, RegistrationMethodsv4, Classifiers,
  plus the 18 core modules) under `build-deps/itk-install`.
- Runtime DLL search via `core/native_extensions.py` reading the
  `native.cpp_prefix` key of `~/.medaxis/config.yaml`.

## Blueprint phase status

| Phase | Status | Notes |
| --- | --- | --- |
| 0. Native build & bindings | **Complete (Windows)** | 5 `.pyd` modules; caveats: vtkIOFFMPEG placeholder, qt6-main micromamba long-path workaround documented in README |
| 1. Application foundation | **Complete** | uv toolchain, PySide6 shell, controller, workspace, logging, tests, CI |
| 2. I/O & data model | **Complete** | DICOM/NIfTI/NRRD/RAW, `.medaxis` archives, patient model, export manager |
| 3. Rendering | **Complete (desktop-verified)** | axial/coronal/sagittal + 3D volume rendering with label overlays verified on desktop (screenshot); zero-copy `VolumeData.image_data` |
| 4. Processing | **Complete (95 algorithms)** | full catalog: 16 filtering, 15 morphology, 8 thresholds (10 methods via native), 16 segmentation, 6 reconstruction, 10 surface, 6 registration, 11 arithmetic, 5 enhancement, radiomics (114 features), measurement. **53 algorithms execute in C++/ITK** (native backend preferred by the registry; e.g. gaussian 64³ 2.4 ms native vs 9.9 s python binding). Texture radiomics (GLCM/GLRLM/GLSZM/GLDM/NGTDM) is C++ with a bit-consistent python fallback |
| 5. MCP & AI | **Complete (core)** | JSON-RPC router, stdio/SSE, permissions, audit; 11 tools incl. run_algorithm (native-backed), get_label_stats, export_screenshot, AI service client (REST + TotalSegmentator adapter); verified E2E over JSON-RPC |
| 5b. Scripting | **Complete (core)** | medaxis facade, sandbox (3 levels), REPL console, script editor (QScintilla optional), action recorder, hook system, script manager with 5 templates — all verified E2E |
| 6. Geometry & advanced viz | **Complete (core)** | native OCC curve sampling/Frenet; CGAL mesh ops; vessel centerline extraction (Lee 3D thinning + Dijkstra + B-spline frames) verified on a synthetic vessel |
| 7. Plugins & ecosystem | **Complete (core)** | discovery, manifests, lifecycle; model zoo widget + downloader with the §6.4 adapter catalog (TotalSegmentator/MedSAM/nnU-Net/MONAI) |
| 8. Persistence & workflow | **Complete (core)** | `.medaxis` archives, label panel + manual editor + undo/redo, exports NIfTI/NRRD/DICOM/DICOM-SEG/**RTSTRUCT**/STL/OBJ/PLY/**3MF**/PNG/JPEG/PDF, comparison view (side-by-side/overlay/subtraction), 9 window/level presets, cine playback + **MP4 recording** (ffmpeg), autosave |

## Verified Quality Gates

- `uv run pytest -q`: 10 passed.
- `uv run ruff check app core geometry io medaxis_io mcp plugins processing rendering scripts ui tests`: clean.
- Native smoke: bridge/itk/cgal/occ/radiomics functional calls pass.
- 47-op native-ops smoke: all pass (filters, morphology, segmentation,
  arithmetic, enhancement, geometry).
- Registration smoke: rigid cc 0.95 / affine 0.98 / demons variants 0.80-0.87.
- E2E: NIfTI load → threshold → gaussian → 4-view rendering + overlay →
  screenshot; MCP JSON-RPC tool calls; sandbox blocking; RTSTRUCT/3MF
  round-trips; cine MP4 recording.

## Known gaps (non-blocking)

- MPR/CPR widgets exist but are not wired into the main window layout
  (renderers import-tested; centerline drives CPR data).
- C++ plugin loading and automatic MCP exposure for native plugins (Phase 7 P2).
- NL-means (native patch-based denoising) is slow on large volumes; use
  bilateral/anisotropic for interactive work.
- 3D-thinning stays on scikit-image (ITK 5.4 ships 2D thinning only).
