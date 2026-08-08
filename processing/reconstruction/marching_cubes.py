"""
3D Reconstruction algorithms — Marching Cubes, Flying Edges, Poisson.
"""
from ..algorithm_registry import AlgorithmRegistry, AlgorithmDefinition, AlgorithmCategory, AlgorithmResult, AlgorithmParameter

_registry = AlgorithmRegistry()


def _marching_cubes(label_data, params, progress_callback=None):
    """Extract surface mesh from label map."""
    from geometry.cgal_wrapper import marching_cubes as mc
    iso_value = params.get("iso_value", 0.5)
    spacing = params.get("spacing", (1, 1, 1))
    result = mc(label_data.array, spacing=spacing, iso_value=iso_value)
    from core.mesh_data import MeshData
    mesh = MeshData(
        vertices=result["vertices"],
        faces=result["faces"],
        name=f"MC_{label_data.name}" if hasattr(label_data, 'name') else "MarchingCubes",
    )
    return AlgorithmResult(algorithm_id="marching_cubes", mesh_data=mesh)


def _flying_edges(label_data, params, progress_callback=None):
    """Faster Marching Cubes via vtkFlyingEdges3D."""
    import vtk
    from vtkmodules.util.numpy_support import numpy_to_vtk
    iso_value = params.get("iso_value", 0.5)

    vtk_data = numpy_to_vtk(label_data.array.ravel(), deep=False, array_type=vtk.VTK_FLOAT)
    vtk_data.SetNumberOfComponents(1)
    img = vtk.vtkImageData()
    img.SetDimensions(label_data.array.shape)
    img.GetPointData().SetScalars(vtk_data)

    fe = vtk.vtkFlyingEdges3D()
    fe.SetInputData(img)
    fe.SetValue(0, iso_value)
    fe.ComputeNormalsOn()
    fe.Update()

    poly = fe.GetOutput()
    verts = []
    faces = []
    for i in range(poly.GetNumberOfPoints()):
        verts.append(poly.GetPoint(i))
    for i in range(poly.GetNumberOfCells()):
        faces.append([poly.GetCell(i).GetPointId(j) for j in range(3)])

    import numpy as np
    from core.mesh_data import MeshData
    mesh = MeshData(vertices=np.array(verts), faces=np.array(faces), name=f"FE_{label_data.name}")
    return AlgorithmResult(algorithm_id="flying_edges", mesh_data=mesh)


def register_reconstruction_algorithms():
    _registry.register(AlgorithmDefinition(
        id="marching_cubes", name="Marching Cubes", category=AlgorithmCategory.RECONSTRUCTION,
        parameters=[AlgorithmParameter("iso_value", "float", "Iso Value", default=0.5, min_val=0.0, max_val=1.0)],
        run_func=_marching_cubes,
        estimated_time_sec=5.0,
        input_parameter="label_data",
    ))
    _registry.register(AlgorithmDefinition(
        id="flying_edges", name="Flying Edges (Fast)", category=AlgorithmCategory.RECONSTRUCTION,
        parameters=[AlgorithmParameter("iso_value", "float", "Iso Value", default=0.5, min_val=0.0, max_val=1.0)],
        run_func=_flying_edges,
        estimated_time_sec=2.0,
        input_parameter="label_data",
    ))
