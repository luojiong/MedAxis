"""
MCP Resources — exposing current workspace state to AI agents.
"""
from .router import MCPResource


def _make_patients_resource(controller) -> MCPResource:
    async def handler():
        import json
        if controller.workspace and controller.workspace.active_project:
            patients = controller.workspace.active_project.patients
            return json.dumps([p.to_dict() if hasattr(p, 'to_dict') else {"id": str(i)} for i, p in enumerate(patients)])
        return json.dumps([])
    return MCPResource(
        uri="medaxis://patients",
        name="Patients",
        description="All currently loaded patients.",
        handler=handler,
    )


def _make_volumes_resource(controller) -> MCPResource:
    async def handler():
        import json
        volumes = []
        for volume_id, volume in getattr(controller, "volumes", {}).items():
            volumes.append({
                "volume_id": str(volume_id),
                "name": getattr(volume, "name", "Volume"),
                "dimensions": list(getattr(volume, "dimensions", ())),
                "spacing": list(getattr(volume, "spacing_mm", ())),
                "modality": getattr(volume, "modality", ""),
            })
        return json.dumps(volumes)
    return MCPResource(
        uri="medaxis://volumes",
        name="Volumes",
        description="All loaded volumes with metadata.",
        handler=handler,
    )


def register_all_resources(controller) -> list[MCPResource]:
    return [
        _make_patients_resource(controller),
        _make_volumes_resource(controller),
    ]
