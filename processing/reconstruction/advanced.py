"""
Reconstruction — surface nets (skimage), Poisson / advancing-front / Delaunay
(native CGAL backend).
"""
import numpy as np
from ..algorithm_registry import (
    AlgorithmRegistry,
    AlgorithmDefinition,
    AlgorithmCategory,
    AlgorithmResult,
    AlgorithmParameter,
)

_registry = AlgorithmRegistry()


def _surface_nets(label_data, params, progress_callback=None):
    """Surface Nets (smooth dual contouring) via scikit-image marching cubes."""
    import numpy as np
    from skimage.measure import marching_cubes
    from core.mesh_data import MeshData

    arr = np.asarray(label_data.array)
    spacing = params.get("spacing", getattr(label_data, "spacing_mm", (1.0, 1.0, 1.0)))
    iso = params.get("iso_value", 0.5)
    verts, faces, _normals, _values = marching_cubes(
        arr.astype(np.float32), level=iso, spacing=spacing, method="lewiner")
    mesh = MeshData(vertices=verts, faces=faces.astype(np.int64),
                    name=f"SN_{getattr(label_data, 'name', '')}")
    return AlgorithmResult(algorithm_id="surface_nets", mesh_data=mesh)


def _poisson_reconstruct(mesh_data, params, progress_callback=None):
    """Poisson surface reconstruction of a point cloud (native CGAL)."""
    from geometry.cgal_wrapper import poisson_reconstruct
    from core.mesh_data import MeshData
    points = params.get("points", getattr(mesh_data, "vertices", None))
    normals = params.get("normals", None)
    if points is None:
        raise ValueError("poisson_reconstruct needs a point cloud (use points param)")
    if normals is None:
        import numpy as np
        normals = np.zeros_like(np.asarray(points, dtype=float))
    result = poisson_reconstruct(points, normals, float(params.get("smoothing_angle", 30.0)))
    mesh = MeshData(vertices=result["vertices"], faces=result["faces"],
                    name=f"poisson_{getattr(mesh_data, 'name', '')}")
    return AlgorithmResult(algorithm_id="poisson_reconstruct", mesh_data=mesh)


def _advancing_front(mesh_data, params, progress_callback=None):
    """Advancing-front reconstruction of a point cloud (native CGAL)."""
    from geometry.cgal_wrapper import advancing_front_reconstruct
    from core.mesh_data import MeshData
    points = params.get("points", getattr(mesh_data, "vertices", None))
    if points is None:
        raise ValueError("advancing_front needs a point cloud (use points param)")
    result = advancing_front_reconstruct(points)
    mesh = MeshData(vertices=result["vertices"], faces=result["faces"],
                    name=f"af_{getattr(mesh_data, 'name', '')}")
    return AlgorithmResult(algorithm_id="advancing_front", mesh_data=mesh)


def _delaunay(mesh_data, params, progress_callback=None):
    """3D Delaunay tetrahedralization (native CGAL)."""
    from geometry.cgal_wrapper import delaunay_tetrahedralize
    from core.mesh_data import MeshData
    points = params.get("points", getattr(mesh_data, "vertices", None))
    if points is None:
        raise ValueError("delaunay needs a point cloud (use points param)")
    result = delaunay_tetrahedralize(points)
    mesh = MeshData(vertices=result["vertices"], faces=np.zeros((0, 3), dtype=np.int64),
                    tets=result["tets"], name=f"dt_{getattr(mesh_data, 'name', '')}")
    return AlgorithmResult(algorithm_id="delaunay", mesh_data=mesh)


def register_reconstruction_extra():
    _registry.register(AlgorithmDefinition(
        id="surface_nets", name="Surface Nets", category=AlgorithmCategory.RECONSTRUCTION,
        description="Smooth dual-contouring surface extraction (scikit-image).",
        parameters=[AlgorithmParameter("iso_value", "float", "Iso Value", default=0.5, min_val=0.0, max_val=1.0)],
        run_func=_surface_nets, estimated_time_sec=5.0, input_parameter="label_data",
    ))
    _registry.register(AlgorithmDefinition(
        id="poisson_reconstruct", name="Poisson Reconstruction", category=AlgorithmCategory.RECONSTRUCTION,
        description="Poisson surface reconstruction of an oriented point cloud (CGAL).",
        parameters=[AlgorithmParameter("smoothing_angle", "float", "Smoothing Angle (deg)", default=30.0, min_val=0.0, max_val=180.0)],
        run_func=_poisson_reconstruct, estimated_time_sec=30.0, input_parameter="mesh_data",
    ))
    _registry.register(AlgorithmDefinition(
        id="advancing_front", name="Advancing Front", category=AlgorithmCategory.RECONSTRUCTION,
        description="Advancing-front reconstruction of an unoriented point cloud (CGAL).",
        parameters=[], run_func=_advancing_front, estimated_time_sec=15.0, input_parameter="mesh_data",
    ))
    _registry.register(AlgorithmDefinition(
        id="delaunay", name="Delaunay Tetrahedralization", category=AlgorithmCategory.RECONSTRUCTION,
        description="3D Delaunay tetrahedralization of a point cloud (CGAL).",
        parameters=[], run_func=_delaunay, estimated_time_sec=10.0, input_parameter="mesh_data",
    ))
