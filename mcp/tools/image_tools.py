"""
MCP Tools — exposing MedAxis functionality to AI agents.
"""
from ..router import MCPTool
from ..auth import PermissionLevel


def _volume_id(volume) -> str:
    return str(getattr(volume, "id", "") or id(volume))


def _find_volume(controller, volume_id: str):
    volume = getattr(controller, "volumes", {}).get(str(volume_id))
    if volume is None and getattr(controller, "current_volume", None) is not None:
        current = controller.current_volume
        if _volume_id(current) == str(volume_id):
            return current
    if volume is None:
        raise ValueError(f"Volume not found: {volume_id}")
    return volume


# ─── Image I/O Tools ───

def _make_open_dicom_tool(controller) -> MCPTool:
    async def handler(params: dict):
        path = params["path"]
        volume = await controller.file_manager.open_async(path)
        controller.emit_volume_loaded(volume)
        return {
            "volume_id": _volume_id(volume),
            "dimensions": list(volume.dimensions),
            "spacing": list(volume.spacing),
            "modality": getattr(volume, 'modality', 'unknown'),
        }
    return MCPTool(
        name="open_dicom",
        description="Open a DICOM file or folder and return volume metadata.",
        parameters_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to DICOM file or folder"},
                "series_uid": {"type": "string", "description": "Optional specific SeriesInstanceUID to load"},
            },
            "required": ["path"],
        },
        handler=handler,
    )


def _make_open_nifti_tool(controller) -> MCPTool:
    async def handler(params: dict):
        path = params["path"]
        volume = await controller.file_manager.open_async(path)
        controller.emit_volume_loaded(volume)
        return {
            "volume_id": _volume_id(volume),
            "dimensions": list(volume.dimensions),
            "spacing": list(volume.spacing),
        }
    return MCPTool(
        name="open_nifti",
        description="Open a NIfTI file (.nii / .nii.gz) and return metadata.",
        parameters_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to .nii or .nii.gz file"},
            },
            "required": ["path"],
        },
        handler=handler,
    )


def _make_get_metadata_tool(controller) -> MCPTool:
    async def handler(params: dict):
        vid = params["volume_id"]
        volume = _find_volume(controller, vid)
        return {
            "volume_id": _volume_id(volume),
            "name": volume.name,
            "dimensions": list(volume.dimensions),
            "spacing": list(volume.spacing_mm),
            "origin": list(volume.origin_mm),
            "modality": volume.modality,
            "intensity_range": list(volume.get_min_max()),
        }
    return MCPTool(
        name="get_metadata",
        description="Get complete DICOM/NIfTI metadata for a loaded volume.",
        parameters_schema={
            "type": "object",
            "properties": {"volume_id": {"type": "string"}},
            "required": ["volume_id"],
        },
        handler=handler,
    )


# ─── View Control Tools ───

def _make_set_slice_tool(controller) -> MCPTool:
    async def handler(params: dict):
        volume = _find_volume(controller, params["volume_id"])
        axis = params.get("axis", "axial")
        view = getattr(getattr(controller, "main_window", None), "view_container", None)
        viewport = view.get_view(axis) if view is not None else None
        if viewport is None or not hasattr(viewport, "set_slice_index"):
            raise RuntimeError("Slice viewport is unavailable")
        index = int(params.get("index", 0))
        viewport.set_slice_index(index)
        return {"volume_id": _volume_id(volume), "slice_index": index, "axis": axis}
    return MCPTool(
        name="set_slice",
        description="Set the current slice index and orientation.",
        parameters_schema={
            "type": "object",
            "properties": {
                "volume_id": {"type": "string"},
                "axis": {"type": "string", "enum": ["axial", "coronal", "sagittal"]},
                "index": {"type": "integer"},
            },
            "required": ["volume_id", "axis", "index"],
        },
        handler=handler,
    )


def _make_set_window_level_tool(controller) -> MCPTool:
    async def handler(params: dict):
        volume = _find_volume(controller, params["volume_id"])
        view_container = getattr(getattr(controller, "main_window", None), "view_container", None)
        if view_container is not None:
            for viewport in view_container.slice_views():
                viewport.set_window_level(float(params["window"]), float(params["level"]))
        return {"volume_id": _volume_id(volume), "window": params["window"], "level": params["level"]}
    return MCPTool(
        name="set_window_level",
        description="Set window/level for display.",
        parameters_schema={
            "type": "object",
            "properties": {
                "volume_id": {"type": "string"},
                "window": {"type": "number"},
                "level": {"type": "number"},
            },
            "required": ["volume_id", "window", "level"],
        },
        handler=handler,
    )


# ─── AI Tools ───

def _make_run_ai_segmentation_tool(controller) -> MCPTool:
    async def handler(params: dict):
        volume = _find_volume(controller, params["volume_id"])
        result = await controller.ai_client_manager.segment(
            volume=volume,
            model_id=params["model_id"],
            service_url=params.get("service_url"),
            extra_params=params.get("params", {}),
        )
        return result
    return MCPTool(
        name="run_ai_segmentation",
        description="Run AI segmentation on a volume using an external AI service.",
        parameters_schema={
            "type": "object",
            "properties": {
                "volume_id": {"type": "string", "description": "Volume to segment"},
                "model_id": {"type": "string", "description": "Model ID, e.g. 'totalsegmentator-v2'"},
                "service_url": {"type": "string", "description": "Optional AI service URL"},
                "params": {"type": "object", "description": "Additional model parameters"},
            },
            "required": ["volume_id", "model_id"],
        },
        handler=handler,
    )


def _make_list_ai_models_tool(controller) -> MCPTool:
    async def handler(params: dict):
        manager = controller.ai_client_manager
        if manager is None:
            return []
        return await manager.list_all_models()
    return MCPTool(
        name="list_ai_models",
        description="List available AI models from registered services.",
        parameters_schema={
            "type": "object",
            "properties": {
                "service_url": {"type": "string", "description": "Optional specific service URL"},
            },
        },
        handler=handler,
        permission=PermissionLevel.READONLY,
    )


def _make_list_algorithms_tool(controller) -> MCPTool:
    async def handler(_params: dict):
        registry = controller.algorithm_registry
        if registry is None:
            return []
        return [
            {
                "id": definition.id,
                "name": definition.name,
                "category": definition.category.value,
                "description": definition.description,
                "input_parameter": definition.input_parameter,
                "parameters": definition.parameters and [
                    {
                        "name": p.name,
                        "type": p.type,
                        "label": p.label,
                        "default": p.default,
                        "choices": p.choices,
                        "minimum": p.min_val,
                        "maximum": p.max_val,
                    }
                    for p in definition.parameters
                ] or [],
            }
            for definition in registry.list_all()
        ]

    return MCPTool(
        name="list_algorithms",
        description="List registered processing algorithms and their parameters.",
        parameters_schema={"type": "object", "properties": {}},
        handler=handler,
        permission=PermissionLevel.READONLY,
    )


def _make_run_algorithm_tool(controller) -> MCPTool:
    async def handler(params: dict):
        volume = _find_volume(controller, params["volume_id"])
        definition = controller.algorithm_registry.get(params["algorithm_id"])
        if definition is None:
            raise ValueError(f"Algorithm not found: {params['algorithm_id']}")
        input_data = volume
        if definition.input_parameter == "label_data":
            labels = controller.label_manager.for_volume(_volume_id(volume))
            label_id = params.get("label_id")
            input_data = next((label for label in labels if label.id == label_id), labels[-1] if labels else None)
        elif definition.input_parameter == "mesh_data":
            input_data = controller.meshes.get(str(params.get("mesh_id", "")))
        if input_data is None:
            raise ValueError(f"No {definition.input_parameter} available for algorithm")
        result = controller.run_algorithm(params["algorithm_id"], input_data, params.get("params", {}))
        if not result.success:
            raise RuntimeError(result.error or "Algorithm failed")
        return {
            "algorithm_id": result.algorithm_id,
            "execution_time_sec": result.execution_time_sec,
            "label_id": getattr(result.label_data, "id", None),
            "mesh_id": getattr(result.mesh_data, "id", None),
            "statistics": result.statistics,
        }

    return MCPTool(
        name="run_algorithm",
        description="Run a registered algorithm against a loaded volume, label, or mesh.",
        parameters_schema={
            "type": "object",
            "properties": {
                "volume_id": {"type": "string"},
                "algorithm_id": {"type": "string"},
                "label_id": {"type": "string"},
                "mesh_id": {"type": "string"},
                "params": {"type": "object"},
            },
            "required": ["volume_id", "algorithm_id"],
        },
        handler=handler,
    )


def _make_get_label_stats_tool(controller) -> MCPTool:
    async def handler(params: dict):
        label = controller.label_manager.get(params["label_id"])
        if label is None:
            raise ValueError(f"Label not found: {params['label_id']}")
        volume = _find_volume(controller, label.parent_volume_id)
        return label.compute_stats(volume, params.get("label_value")).to_dict()

    return MCPTool(
        name="get_label_stats",
        description="Calculate quantitative statistics for a segmentation label.",
        parameters_schema={
            "type": "object",
            "properties": {
                "label_id": {"type": "string"},
                "label_value": {"type": "integer"},
            },
            "required": ["label_id"],
        },
        handler=handler,
        permission=PermissionLevel.READONLY,
    )


def _make_export_screenshot_tool(controller) -> MCPTool:
    async def handler(params: dict):
        if controller.export_manager is None or controller.main_window is None:
            raise RuntimeError("Screenshot export is unavailable")
        view = controller.main_window.view_container.get_view(params.get("view", "axial"))
        output = controller.export_manager.export_screenshot(view, "png", params["path"])
        return {"path": str(output)}

    return MCPTool(
        name="export_screenshot",
        description="Capture a named viewport to a PNG file.",
        parameters_schema={
            "type": "object",
            "properties": {
                "view": {"type": "string", "enum": ["axial", "coronal", "sagittal", "3d"]},
                "path": {"type": "string"},
            },
            "required": ["path"],
        },
        handler=handler,
    )


def register_all_tools(controller) -> list[MCPTool]:
    """Register all MCP tools with the given AppController."""
    tools = [
        _make_open_dicom_tool(controller),
        _make_open_nifti_tool(controller),
        _make_get_metadata_tool(controller),
        _make_set_slice_tool(controller),
        _make_set_window_level_tool(controller),
        _make_run_ai_segmentation_tool(controller),
        _make_list_ai_models_tool(controller),
        _make_list_algorithms_tool(controller),
        _make_run_algorithm_tool(controller),
        _make_get_label_stats_tool(controller),
        _make_export_screenshot_tool(controller),
    ]
    return tools
