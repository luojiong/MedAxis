"""
AI API Client — base class and REST implementation.
"""
from abc import ABC, abstractmethod
from typing import Optional
from dataclasses import dataclass, field
import httpx
import asyncio


@dataclass
class AIServiceConfig:
    """Configuration for an external AI service."""
    id: str
    name: str
    base_url: str
    adapter_type: str = "generic"    # totalsegmentator / nnunet / medsam / monai / generic
    api_key: Optional[str] = None
    timeout_sec: float = 300.0
    enabled: bool = True


@dataclass
class AIModelInfo:
    """Info about an AI model available from a service."""
    id: str
    name: str
    modality: str = ""
    num_classes: int = 0
    class_names: list[str] = field(default_factory=list)
    estimated_time_sec: float = 60.0
    description: str = ""


class AIAPIClient(ABC):
    """Base class for AI API clients."""

    def __init__(self, config: AIServiceConfig):
        self.config = config

    @abstractmethod
    async def list_models(self) -> list[AIModelInfo]: ...

    @abstractmethod
    async def segment(self, volume_data: bytes, model_id: str, params: dict = None) -> dict: ...

    @abstractmethod
    async def get_job_status(self, job_id: str) -> dict: ...

    @property
    def is_available(self) -> bool:
        return self.config.enabled


class RESTClient(AIAPIClient):
    """REST API client for MedAxis-compatible AI services."""

    def __init__(self, config: AIServiceConfig):
        super().__init__(config)
        self._client = httpx.AsyncClient(timeout=config.timeout_sec)

    async def list_models(self) -> list[AIModelInfo]:
        resp = await self._client.get(f"{self.config.base_url}/api/v1/models")
        resp.raise_for_status()
        data = resp.json()
        return [AIModelInfo(**m) for m in data.get("models", [])]

    async def segment(self, volume_data: bytes, model_id: str, params: dict = None) -> dict:
        files = {"image": ("volume.nii.gz", volume_data, "application/octet-stream")}
        form_data = {"model_id": model_id}
        if params:
            form_data["params"] = str(params)

        resp = await self._client.post(
            f"{self.config.base_url}/api/v1/segment",
            files=files,
            data=form_data,
        )
        resp.raise_for_status()
        return resp.json()

    async def get_job_status(self, job_id: str) -> dict:
        resp = await self._client.get(f"{self.config.base_url}/api/v1/jobs/{job_id}")
        resp.raise_for_status()
        return resp.json()

    async def download_result(self, url: str) -> bytes:
        resp = await self._client.get(url)
        resp.raise_for_status()
        return resp.content

    async def close(self):
        await self._client.aclose()


class AIServiceManager:
    """Manages multiple AI service connections."""

    def __init__(self, controller=None):
        self.controller = controller
        self._services: dict[str, AIServiceConfig] = {}
        self._clients: dict[str, AIAPIClient] = {}

    def add_service(self, config: AIServiceConfig):
        self._services[config.id] = config
        if config.id in self._clients:
            del self._clients[config.id]

    def remove_service(self, service_id: str):
        self._services.pop(service_id, None)
        self._clients.pop(service_id, None)

    def get_client(self, service_id: str) -> Optional[AIAPIClient]:
        if service_id in self._clients:
            return self._clients[service_id]
        config = self._services.get(service_id)
        if config is None:
            return None
        client = RESTClient(config)
        self._clients[service_id] = client
        return client

    async def list_all_models(self) -> list[dict]:
        results = []
        for sid, config in self._services.items():
            if not config.enabled:
                continue
            try:
                client = self.get_client(sid)
                models = await client.list_models()
                for m in models:
                    results.append({"service_id": sid, "service_name": config.name, "model": m.__dict__})
            except Exception as e:
                print(f"Error listing models from {config.name}: {e}")
        return results

    async def segment(self, volume, model_id: str, service_url: str = None, extra_params: dict = None):
        # Find service that has this model, or use specified URL
        for sid, client in self._clients.items():
            try:
                models = await client.list_models()
                if any(m.id == model_id for m in models):
                    import gzip
                    import io
                    buffer = io.BytesIO()
                    with gzip.GzipFile(fileobj=buffer, mode='wb') as f:
                        import numpy as np
                        np.save(f, volume.to_numpy())
                    result = await client.segment(buffer.getvalue(), model_id, extra_params)
                    return result
            except Exception:
                continue
        raise ValueError(f"No service found with model: {model_id}")

    async def shutdown_async(self) -> None:
        """Close all persistent HTTP clients."""
        for client in self._clients.values():
            close = getattr(client, "close", None)
            if close is not None:
                await close()
        self._clients.clear()

    def shutdown(self) -> None:
        """Synchronous lifecycle hook used by AppController."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self.shutdown_async())
