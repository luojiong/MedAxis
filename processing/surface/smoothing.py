"""
Surface processing algorithms — smooth, decimate, subdivide, curvature.
"""
from ..algorithm_registry import AlgorithmRegistry, AlgorithmDefinition, AlgorithmCategory, AlgorithmResult, AlgorithmParameter

_registry = AlgorithmRegistry()


def _mesh_smooth(mesh_data, params, progress_callback=None):
    from geometry.cgal_wrapper import mesh_smooth
    iterations = params.get("iterations", 20)
    method = params.get("method", "laplacian")
    new_verts = mesh_smooth(mesh_data.vertices, mesh_data.faces, iterations=iterations, method=method)
    from core.mesh_data import MeshData
    result = MeshData(vertices=new_verts, faces=mesh_data.faces.copy(), name=f"{mesh_data.name}_smoothed")
    return AlgorithmResult(algorithm_id="mesh_smooth", mesh_data=result)


def _mesh_decimate(mesh_data, params, progress_callback=None):
    from geometry.cgal_wrapper import mesh_decimate
    reduction = params.get("reduction", 0.5)
    new_verts, new_faces = mesh_decimate(mesh_data.vertices, mesh_data.faces, target_reduction=reduction)
    from core.mesh_data import MeshData
    result = MeshData(vertices=new_verts, faces=new_faces, name=f"{mesh_data.name}_decimated")
    return AlgorithmResult(algorithm_id="mesh_decimate", mesh_data=result)


def _mesh_curvature(mesh_data, params, progress_callback=None):
    from geometry.cgal_wrapper import mesh_curvature
    result = mesh_curvature(mesh_data.vertices, mesh_data.faces)
    return AlgorithmResult(algorithm_id="mesh_curvature", statistics=result)


def register_surface_algorithms():
    _registry.register(AlgorithmDefinition(
        id="mesh_smooth", name="Mesh Smooth", category=AlgorithmCategory.SURFACE,
        parameters=[AlgorithmParameter("iterations", "int", "Iterations", default=20, min_val=1, max_val=200),
                     AlgorithmParameter("method", "choice", "Method", choices=["laplacian", "taubin"], default="laplacian")],
        run_func=_mesh_smooth,
        input_parameter="mesh_data",
    ))
    _registry.register(AlgorithmDefinition(
        id="mesh_decimate", name="Mesh Decimate", category=AlgorithmCategory.SURFACE,
        parameters=[AlgorithmParameter("reduction", "float", "Target Reduction", default=0.5, min_val=0.1, max_val=0.95)],
        run_func=_mesh_decimate,
        input_parameter="mesh_data",
    ))
    _registry.register(AlgorithmDefinition(
        id="mesh_curvature", name="Mesh Curvature", category=AlgorithmCategory.SURFACE,
        run_func=_mesh_curvature,
        input_parameter="mesh_data",
    ))
