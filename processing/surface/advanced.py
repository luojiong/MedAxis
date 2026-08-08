"""
Surface processing extras — fill holes, boolean ops, subdivision, stats.
"""
from ..algorithm_registry import (
    AlgorithmRegistry,
    AlgorithmDefinition,
    AlgorithmCategory,
    AlgorithmResult,
    AlgorithmParameter,
)

_registry = AlgorithmRegistry()


def _mesh_fill_holes(mesh_data, params, progress_callback=None):
    from geometry.cgal_wrapper import mesh_fill_holes
    from core.mesh_data import MeshData
    new_verts, new_faces = mesh_fill_holes(
        mesh_data.vertices, mesh_data.faces, int(params.get("max_hole_edges", 0)))
    mesh = MeshData(vertices=new_verts, faces=new_faces, name=f"{mesh_data.name}_filled")
    return AlgorithmResult(algorithm_id="mesh_fill_holes", mesh_data=mesh)


def _mesh_boolean(op: str, label: str):
    def run(mesh_data, params, progress_callback=None):
        from geometry.cgal_wrapper import mesh_boolean
        from core.mesh_data import MeshData
        b = params.get("other_mesh", None)
        if b is None:
            raise ValueError(f"{op}: 'other_mesh' (MeshData) parameter is required")
        new_verts, new_faces = mesh_boolean(
            mesh_data.vertices, mesh_data.faces, b.vertices, b.faces, operation=op)
        mesh = MeshData(vertices=new_verts, faces=new_faces,
                        name=f"{mesh_data.name}_{op}")
        return AlgorithmResult(algorithm_id=f"mesh_boolean_{op}", mesh_data=mesh)
    run.__name__ = f"_mesh_boolean_{op}"
    return run


def _mesh_subdivide(mesh_data, params, progress_callback=None):
    from geometry.cgal_wrapper import mesh_subdivide
    from core.mesh_data import MeshData
    new_verts, new_faces = mesh_subdivide(
        mesh_data.vertices, mesh_data.faces,
        scheme=params.get("scheme", "loop"), iterations=int(params.get("iterations", 1)))
    mesh = MeshData(vertices=new_verts, faces=new_faces, name=f"{mesh_data.name}_subdivided")
    return AlgorithmResult(algorithm_id="mesh_subdivide", mesh_data=mesh)


def _mesh_stats(mesh_data, params, progress_callback=None):
    from geometry.cgal_wrapper import mesh_surface_stats
    stats = mesh_surface_stats(mesh_data.vertices, mesh_data.faces)
    stats["vertices"] = len(mesh_data.vertices)
    stats["faces"] = len(mesh_data.faces)
    return AlgorithmResult(algorithm_id="mesh_stats", statistics=stats)


def _mesh_to_label(mesh_data, params, progress_callback=None):
    """Rasterize a mesh into a label volume (voxelization)."""
    import vtk
    from vtkmodules.util.numpy_support import vtk_to_numpy
    from geometry.cgal_wrapper import _build_polydata

    dims = tuple(int(v) for v in params.get("dims", [64, 64, 64]))
    spacing = tuple(float(v) for v in params.get("spacing", [1.0, 1.0, 1.0]))
    poly = _build_polydata(mesh_data.vertices, mesh_data.faces)

    # Position the volume so the mesh fits inside.
    bounds = poly.GetBounds()  # (xmin, xmax, ymin, ymax, zmin, zmax)
    origin = (bounds[0], bounds[2], bounds[4])

    img = vtk.vtkImageData()
    img.SetDimensions(*dims)
    img.SetSpacing(*spacing)
    img.SetOrigin(*origin)
    img.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

    # stencil: voxels inside the mesh are set to 1.
    stencil = vtk.vtkPolyDataToImageStencil()
    stencil.SetInputData(poly)
    stencil.SetOutputOrigin(origin)
    stencil.SetOutputSpacing(spacing)
    stencil.SetOutputWholeExtent(0, dims[0] - 1, 0, dims[1] - 1, 0, dims[2] - 1)
    stencil.Update()

    img_stencil = vtk.vtkImageStencil()
    img_stencil.SetInputData(img)
    img_stencil.SetStencilConnection(stencil.GetOutputPort())
    img_stencil.ReverseStencilOff()
    img_stencil.SetBackgroundValue(1)
    img_stencil.Update()

    arr = vtk_to_numpy(img_stencil.GetOutput().GetPointData().GetScalars()).reshape(dims)
    return AlgorithmResult(algorithm_id="mesh_to_label", label_data=arr)


def register_surface_extra():
    _registry.register(AlgorithmDefinition(
        id="mesh_fill_holes", name="Mesh Fill Holes", category=AlgorithmCategory.SURFACE,
        description="Fill boundary holes of a mesh (CGAL).",
        parameters=[AlgorithmParameter("max_hole_edges", "int", "Max Hole Edges (0=all)", default=0, min_val=0, max_val=1000)],
        run_func=_mesh_fill_holes, input_parameter="mesh_data",
    ))
    for op, label in [("union", "Union"), ("intersect", "Intersect"), ("diff", "Difference")]:
        _registry.register(AlgorithmDefinition(
            id=f"mesh_boolean_{op}", name=f"Mesh Boolean {label}", category=AlgorithmCategory.SURFACE,
            description=f"Mesh boolean {op} (CGAL corefinement). Requires 'other_mesh'.",
            parameters=[AlgorithmParameter("other_mesh", "mesh", "Other Mesh")],
            run_func=_mesh_boolean(op, label), input_parameter="mesh_data",
        ))
    _registry.register(AlgorithmDefinition(
        id="mesh_subdivide", name="Mesh Subdivide", category=AlgorithmCategory.SURFACE,
        description="Loop / Catmull-Clark subdivision (CGAL).",
        parameters=[
            AlgorithmParameter("scheme", "choice", "Scheme", choices=["loop", "catmull_clark"], default="loop"),
            AlgorithmParameter("iterations", "int", "Iterations", default=1, min_val=1, max_val=4),
        ],
        run_func=_mesh_subdivide, input_parameter="mesh_data",
    ))
    _registry.register(AlgorithmDefinition(
        id="mesh_stats", name="Mesh Surface Stats", category=AlgorithmCategory.SURFACE,
        description="Surface area and enclosed volume (VTK mass properties).",
        parameters=[], run_func=_mesh_stats, input_parameter="mesh_data",
    ))
    _registry.register(AlgorithmDefinition(
        id="mesh_to_label", name="Mesh to Label", category=AlgorithmCategory.SURFACE,
        description="Rasterize a mesh into a label volume.",
        parameters=[
            AlgorithmParameter("dims", "int3", "Output Dims", default=[64, 64, 64]),
            AlgorithmParameter("spacing", "float3", "Output Spacing", default=[1.0, 1.0, 1.0]),
        ],
        run_func=_mesh_to_label, input_parameter="mesh_data",
    ))
