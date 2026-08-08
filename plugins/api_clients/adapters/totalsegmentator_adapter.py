"""
Service adapters for popular AI segmentation APIs.
Each adapter normalizes the external API format to MedAxis standard.
"""
from .rest_client import AIAPIClient, AIServiceConfig, AIModelInfo, RESTClient
from typing import Optional


class TotalSegmentatorAdapter(RESTClient):
    """Adapter for TotalSegmentator REST API."""

    async def list_models(self) -> list[AIModelInfo]:
        return [AIModelInfo(
            id="totalsegmentator-v2",
            name="TotalSegmentator v2",
            modality="CT",
            num_classes=104,
            class_names=["spleen", "kidney_right", "kidney_left", "liver", "stomach", "pancreas", "heart", "lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right", "aorta", "inferior_vena_cava", "portal_vein_and_splenic_vein", "gallbladder", "adrenal_gland_right", "adrenal_gland_left", "esophagus", "trachea", "thyroid_gland", "bladder", "prostate", "uterus", "ovary_right", "ovary_left"],
            estimated_time_sec=45,
            description="Full-body CT multi-organ segmentation (104 classes)",
        )]


class nnUNetAdapter(RESTClient):
    """Adapter for nnU-Net v2 inference server."""

    def __init__(self, config: AIServiceConfig):
        super().__init__(config)
        self._task_name = getattr(config, 'task_name', 'default')

    async def list_models(self) -> list[AIModelInfo]:
        resp = await self._client.get(f"{self.config.base_url}/tasks")
        resp.raise_for_status()
        tasks = resp.json()
        return [AIModelInfo(id=t["name"], name=t["name"], modality=t.get("modality", "CT"), num_classes=t.get("num_classes", 1), description=t.get("description", "")) for t in tasks]


class MedSAMAdapter(RESTClient):
    """Adapter for MedSAM / SAM 2 Promptable Segmentation API."""

    async def list_models(self) -> list[AIModelInfo]:
        return [AIModelInfo(id="medsam", name="MedSAM", modality="Any", description="Promptable medical image segmentation (point/box)")]

    async def segment(self, volume_data: bytes, model_id: str, params: dict = None):
        # MedSAM requires prompts (points or boxes)
        prompts = params.get("prompts", []) if params else []
        files = {"image": ("volume.nii.gz", volume_data)}
        data = {"model_id": model_id, "prompts": str(prompts)}
        resp = await self._client.post(f"{self.config.base_url}/segment", files=files, data=data)
        resp.raise_for_status()
        return resp.json()


class MONAIDeployAdapter(RESTClient):
    """Adapter for MONAI Deploy App SDK."""

    async def list_models(self) -> list[AIModelInfo]:
        resp = await self._client.get(f"{self.config.base_url}/apps")
        resp.raise_for_status()
        apps = resp.json()
        return [AIModelInfo(id=app["name"], name=app["name"], modality=app.get("modality", ""), description=app.get("description", "")) for app in apps]


class GenericAdapter(RESTClient):
    """Adapter for any MedAxis-compatible API."""
    pass


def create_adapter(config: AIServiceConfig) -> Optional[AIAPIClient]:
    """Factory: create the appropriate adapter based on config.adapter_type."""
    adapters = {
        "totalsegmentator": TotalSegmentatorAdapter,
        "nnunet": nnUNetAdapter,
        "medsam": MedSAMAdapter,
        "monai": MONAIDeployAdapter,
        "generic": GenericAdapter,
    }
    adapter_cls = adapters.get(config.adapter_type, GenericAdapter)
    return adapter_cls(config)
