"""
CGAL wrapper — mesh reconstruction, surface processing, curvature estimation.
Falls back to VTK when CGAL pybind11 module is unavailable.
"""
import numpy as np

from core.native_extensions import get_native_module


def _native_mesh(native, vertices: np.ndarray, faces: np.ndarray):
    mesh = native.MeshData()
    mesh.vertices = np.asarray(vertices, dtype=float).reshape(-1).tolist()
    mesh.faces = np.asarray(faces, dtype=np.int64).reshape(-1).tolist()
    return mesh


def _mesh_arrays(mesh):
    vertices = np.asarray(mesh.vertices, dtype=float).reshape(-1, 3)
    faces = np.asarray(mesh.faces, dtype=np.int64).reshape(-1, 3)
    return vertices, faces


def marching_cubes(label_array: np.ndarray, spacing: tuple = (1, 1, 1), iso_value: float = 0.5) -> dict:
    """Extract a surface mesh, preferring the compiled CGAL backend."""
    native = get_native_module("medaxis_cgal")
    if native is not None:
        try:
            mesh = native.marching_cubes_surface(
                np.asarray(label_array, dtype=np.float32),
                tuple(int(value) for value in label_array.shape),
                tuple(float(value) for value in spacing),
                float(iso_value),
            )
            vertices, faces = _mesh_arrays(mesh)
            return {"vertices": vertices, "faces": faces, "normals": np.zeros((len(vertices), 3))}
        except (RuntimeError, ValueError, TypeError):
            pass

    try:
        import vtk
        from vtkmodules.util.numpy_support import numpy_to_vtk

        vtk_data = numpy_to_vtk(label_array.ravel(), deep=False, array_type=vtk.VTK_FLOAT)
        vtk_data.SetNumberOfComponents(1)

        img = vtk.vtkImageData()
        img.SetDimensions(label_array.shape)
        img.SetSpacing(spacing)
        img.GetPointData().SetScalars(vtk_data)

        mc = vtk.vtkMarchingCubes()
        mc.SetInputData(img)
        mc.SetValue(0, iso_value)
        mc.ComputeNormalsOn()
        mc.Update()

        poly = mc.GetOutput()
        verts = np.array([poly.GetPoint(i) for i in range(poly.GetNumberOfPoints())])
        faces = np.array([[poly.GetCell(i).GetPointId(j) for j in range(3)] for i in range(poly.GetNumberOfCells())])

        return {"vertices": verts, "faces": faces, "normals": np.zeros((len(verts), 3))}
    except ImportError:
        return {"vertices": np.zeros((0, 3)), "faces": np.zeros((0, 3), dtype=int), "normals": np.zeros((0, 3))}


def mesh_smooth(vertices: np.ndarray, faces: np.ndarray, iterations: int = 20, method: str = "laplacian") -> np.ndarray:
    """Smooth mesh vertices."""
    native = get_native_module("medaxis_cgal")
    if native is not None:
        try:
            mode = native.SmoothMode.TAUBIN if method == "taubin" else native.SmoothMode.LAPLACIAN
            output = native.mesh_smooth(_native_mesh(native, vertices, faces), mode, int(iterations))
            return _mesh_arrays(output)[0]
        except (RuntimeError, ValueError, TypeError):
            pass

    import vtk
    poly = _build_polydata(vertices, faces)
    if method == "taubin":
        smoother = vtk.vtkWindowedSincPolyDataFilter()
        smoother.SetNumberOfIterations(iterations)
    else:
        smoother = vtk.vtkSmoothPolyDataFilter()
        smoother.SetNumberOfIterations(iterations)
        smoother.SetRelaxationFactor(0.1)
    smoother.SetInputData(poly)
    smoother.Update()
    out = smoother.GetOutput()
    return np.array([out.GetPoint(i) for i in range(out.GetNumberOfPoints())])


def mesh_decimate(vertices: np.ndarray, faces: np.ndarray, target_reduction: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Decimate mesh to reduce triangle count."""
    native = get_native_module("medaxis_cgal")
    if native is not None:
        try:
            target_faces = max(1, int(len(faces) * (1.0 - target_reduction)))
            output = native.mesh_decimate(_native_mesh(native, vertices, faces), target_faces)
            return _mesh_arrays(output)
        except (RuntimeError, ValueError, TypeError):
            pass

    import vtk
    poly = _build_polydata(vertices, faces)
    decimator = vtk.vtkQuadricDecimation()
    decimator.SetInputData(poly)
    decimator.SetTargetReduction(target_reduction)
    decimator.Update()
    out = decimator.GetOutput()
    out_verts = np.array([out.GetPoint(i) for i in range(out.GetNumberOfPoints())])
    out_faces = np.array([[out.GetCell(i).GetPointId(j) for j in range(3)] for i in range(out.GetNumberOfCells())])
    return out_verts, out_faces


def mesh_curvature(vertices: np.ndarray, faces: np.ndarray) -> dict:
    """Compute mean and Gaussian curvature per vertex."""
    native = get_native_module("medaxis_cgal")
    if native is not None:
        try:
            mesh = _native_mesh(native, vertices, faces)
            radius = max(1.0e-6, float(np.linalg.norm(np.ptp(vertices, axis=0))) * 0.02)
            return {
                "gaussian": np.asarray(native.compute_gaussian_curvature(mesh, radius), dtype=float),
                "mean": np.asarray(native.compute_mean_curvature(mesh, radius), dtype=float),
            }
        except (RuntimeError, ValueError, TypeError):
            pass

    import vtk
    poly = _build_polydata(vertices, faces)
    curv = vtk.vtkCurvatures()
    curv.SetInputData(poly)
    curv.SetCurvatureTypeToGaussian()
    curv.Update()
    gaussian = np.array([curv.GetOutput().GetPointData().GetScalars().GetValue(i) for i in range(len(vertices))])
    curv.SetCurvatureTypeToMean()
    curv.Update()
    mean = np.array([curv.GetOutput().GetPointData().GetScalars().GetValue(i) for i in range(len(vertices))])
    return {"gaussian": gaussian, "mean": mean}


def _build_polydata(vertices: np.ndarray, faces: np.ndarray):
    """Build vtkPolyData from numpy arrays."""
    import vtk
    from vtkmodules.util.numpy_support import numpy_to_vtk
    poly = vtk.vtkPolyData()
    pts = vtk.vtkPoints()
    pts.SetData(numpy_to_vtk(vertices, deep=True))
    poly.SetPoints(pts)
    cells = vtk.vtkCellArray()
    for face in faces:
        tri = vtk.vtkTriangle()
        for j, pid in enumerate(face[:3]):
            tri.GetPointIds().SetId(j, int(pid))
        cells.InsertNextCell(tri)
    poly.SetPolys(cells)
    return poly


def poisson_reconstruct(points: np.ndarray, normals: np.ndarray,
                        smoothing_angle: float = 30.0) -> dict:
    """Poisson surface reconstruction (native CGAL backend)."""
    native = get_native_module("medaxis_cgal")
    if native is not None:
        try:
            mesh = native.poisson_reconstruct(
                np.asarray(points, dtype=float).reshape(-1).tolist(),
                np.asarray(normals, dtype=float).reshape(-1).tolist(),
                float(smoothing_angle),
            )
            vertices, faces = _mesh_arrays(mesh)
            return {"vertices": vertices, "faces": faces}
        except (RuntimeError, ValueError, TypeError):
            pass
    raise RuntimeError("poisson_reconstruct requires the medaxis_cgal native module")


def advancing_front_reconstruct(points: np.ndarray) -> dict:
    """Advancing-front reconstruction from an unoriented point cloud."""
    native = get_native_module("medaxis_cgal")
    if native is not None:
        try:
            mesh = native.advancing_front_reconstruct(
                np.asarray(points, dtype=float).reshape(-1).tolist())
            vertices, faces = _mesh_arrays(mesh)
            return {"vertices": vertices, "faces": faces}
        except (RuntimeError, ValueError, TypeError):
            pass
    raise RuntimeError("advancing_front_reconstruct requires the medaxis_cgal native module")


def delaunay_tetrahedralize(points: np.ndarray) -> dict:
    """3D Delaunay tetrahedralization (native CGAL backend)."""
    native = get_native_module("medaxis_cgal")
    if native is not None:
        try:
            mesh = native.delaunay_triangulate(
                np.asarray(points, dtype=float).reshape(-1).tolist())
            return {
                "vertices": np.asarray(mesh.vertices, dtype=float).reshape(-1, 3),
                "tets": np.asarray(mesh.tets, dtype=np.int64).reshape(-1, 4),
            }
        except (RuntimeError, ValueError, TypeError):
            pass
    raise RuntimeError("delaunay_tetrahedralize requires the medaxis_cgal native module")


def mesh_fill_holes(vertices: np.ndarray, faces: np.ndarray,
                    max_hole_edges: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Fill boundary holes of a mesh (native CGAL backend)."""
    native = get_native_module("medaxis_cgal")
    if native is not None:
        try:
            output = native.mesh_fill_holes(
                _native_mesh(native, vertices, faces), int(max_hole_edges))
            return _mesh_arrays(output)
        except (RuntimeError, ValueError, TypeError):
            pass
    raise RuntimeError("mesh_fill_holes requires the medaxis_cgal native module")


def mesh_boolean(a_vertices, a_faces, b_vertices, b_faces,
                 operation: str = "union") -> tuple[np.ndarray, np.ndarray]:
    """Mesh boolean operation: union / intersect / difference (native CGAL)."""
    native = get_native_module("medaxis_cgal")
    if native is not None:
        try:
            ma = _native_mesh(native, a_vertices, a_faces)
            mb = _native_mesh(native, b_vertices, b_faces)
            if operation == "union":
                output = native.mesh_boolean_union(ma, mb)
            elif operation == "intersect":
                output = native.mesh_boolean_intersect(ma, mb)
            else:
                output = native.mesh_boolean_diff(ma, mb)
            return _mesh_arrays(output)
        except (RuntimeError, ValueError, TypeError):
            pass
    raise RuntimeError("mesh_boolean requires the medaxis_cgal native module")


def mesh_subdivide(vertices: np.ndarray, faces: np.ndarray,
                   scheme: str = "loop", iterations: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Subdivide a mesh (Loop / Catmull-Clark, native CGAL backend)."""
    native = get_native_module("medaxis_cgal")
    if native is not None:
        try:
            mode = (native.SubdivideScheme.CATMULL_CLARK if scheme == "catmull_clark"
                    else native.SubdivideScheme.LOOP)
            output = native.mesh_subdivide(
                _native_mesh(native, vertices, faces), mode, int(iterations))
            return _mesh_arrays(output)
        except (RuntimeError, ValueError, TypeError):
            pass
    raise RuntimeError("mesh_subdivide requires the medaxis_cgal native module")


def mesh_surface_stats(vertices: np.ndarray, faces: np.ndarray) -> dict:
    """Surface area and enclosed volume via VTK mass properties."""
    import vtk
    poly = _build_polydata(vertices, faces)
    mass = vtk.vtkMassProperties()
    mass.SetInputData(poly)
    mass.Update()
    return {
        "surface_area_mm2": mass.GetSurfaceArea(),
        "volume_mm3": mass.GetVolume(),
    }
