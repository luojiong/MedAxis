# MedAxis Implementation Status

This document reconciles the design blueprint with the implementation in this
repository. It records verified behavior, rather than treating a planned file
or interface as a completed feature.

Last verified: 2026-08-08 (native build completed)

## Toolchain (uv + conda-forge + MSVC)

- Python environment managed by **uv** (`uv sync`, Python 3.12.13 pinned in
  `.python-version`, deps locked in `uv.lock`).
- Native C++ layer **builds and links on Windows** (MSVC 14.50, VS 2026
  generator, CMake 4.3.1). All five pybind11 modules compile, import, and
  pass functional smoke tests:
  - `medaxis_bridge` — numpy ↔ vtkImageData ↔ itkImage zero-copy volume
    management (numpy round-trip verified).
  - `medaxis_itk` — ITK threshold pipeline (Otsu etc.) on numpy buffers
    (verified).
  - `medaxis_cgal` — mesh reconstruction/processing/curvature via CGAL 6.0.1
    (delaunay, smooth, decimate, subdivide, mean/Gaussian curvature verified).
  - `medaxis_occ` — OpenCASCADE 7.8.1 curve sampling / Frenet frames /
    arc length (verified).
  - `medaxis_radiomics` — first-order radiomics features (verified).
- C++ dependencies: conda-forge (micromamba) `medaxis-cpp` env for
  VTK 9.6.2 / CGAL 6.0.1 / OCCT 7.8.1 / Eigen 5.0.1 / Boost 1.90 / GMP / MPFR,
  plus a source build of **ITK 5.4.7** (conda-forge's itk ships no C++ dev
  files) installed under `build-deps/itk-install`.
- Runtime DLL search is wired through `core/native_extensions.py`, which
  reads the `native.cpp_prefix` key of `~/.medaxis/config.yaml`
  (fallback: `MEDAXIS_CPP_PREFIX` env var, `;`-separated).

## Blueprint phase status

| Blueprint phase | Status | Evidence and remaining scope |
| --- | --- | --- |
| 0. Native build and bindings | Complete on Windows | All native targets compile, link, import, and pass smoke tests. Known caveats: vtk 9.6.2 conda package is missing the vtkIOFFMPEG module (placeholder files added); conda qt6-main cannot be extracted by micromamba due to >260-char paths (vtk installed with `--no-deps`; Qt-dependent VTK modules unused); CGAL 6.0.1 lacks `<CGAL/marching_cubes.h>` so `marching_cubes_surface` was removed from the C++ module (Python falls back to VTK). |
| 1. Application foundation | Complete | PySide6 application controller, workspace, docks, configuration, logging, linting, tests, pre-commit, and cross-platform CI are present. |
| 2. I/O and data model | Complete for implemented formats | DICOM, NIfTI, NRRD, RAW, project archive handling, volume/label/mesh models, and patient hierarchy are present. |
| 3. Rendering | Partial | Slice, MPR, CPR, volume, label overlay, crosshair, window/level, and view-state code are present and import-tested. Desktop OpenGL rendering has not been exercised in this headless environment. |
| 4. Processing | Partial | Registry, typed parameter validation, pipelines, built-in filtering, morphology, thresholding, region growing, watershed, flood fill, reconstruction, surface operations, and first-order radiomics are available. The blueprint's full 3D Slicer-sized algorithm catalog and 119-feature radiomics suite are not implemented. |
| 5. MCP and AI | Partial | JSON-RPC router, stdio/SSE transport, tools, resources, ordered permissions, bounded audit events, and external AI client integration are present. Natural-language parameter suggestion and a complete MCP client bridge are not implemented. |
| 5b. Scripting | Partial | The stable `medaxis` facade is available. Embedded IPython console, editor, action recorder, process-isolated sandbox, script library, and remote debugger are not implemented. |
| 6. Geometry and advanced visualization | Partial | Python fallbacks and working C++ modules for curve sampling, mesh operations, and radiomics are present. Advanced centerline/stent workflows are outstanding. |
| 7. Plugins | Partial | Python plugin discovery, manifests, lifecycle management, and built-in registration are present. Native plugin loading and automatic MCP exposure for native plugins are not implemented. |
| 8. Persistence and workflow | Partial | `.medaxis` archives, labels, manual editor, undo/redo, autosave, view persistence, and core image/label/mesh exports are present. DICOM RTSTRUCT, all planned export formats, comparison workflow, and recorded cine video remain outstanding. |

## Verified Quality Gates

- `uv run pytest -q`: **10 tests passed** (PySide6 + pytest-qt).
- `uv run ruff check app core geometry io medaxis_io mcp plugins processing rendering scripts ui tests`: passed.
- `uv run python -m compileall`: passed.
- Native smoke test (bridge/itk/cgal/occ/radiomics functional calls): passed.
- `cmake -S . -B build-native` (VS 2026, native ON): configures without
  skipped targets; `cmake --build --config Release`: all 5 `.pyd` built.
