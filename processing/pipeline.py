"""
Processing Pipeline — chain multiple algorithms sequentially.
"""
from typing import Any, Callable, Optional
from dataclasses import dataclass, field
from .algorithm_registry import AlgorithmRegistry, AlgorithmResult, algorithm_registry


@dataclass
class PipelineStep:
    algorithm_id: str
    params: dict = field(default_factory=dict)
    output_label: Optional[str] = None  # name for intermediate result


@dataclass
class PipelineResult:
    steps: list[AlgorithmResult] = field(default_factory=list)
    final_result: Any = None            # LabelData or MeshData
    total_time_sec: float = 0.0
    success: bool = True
    error: Optional[str] = None


class ProcessingPipeline:
    """Chain multiple algorithms: e.g. Smooth → Threshold → Morphology → MarchingCubes."""

    def __init__(self, registry: AlgorithmRegistry = None):
        self.registry = registry or algorithm_registry
        self.steps: list[PipelineStep] = []
        self._progress_callbacks: list[Callable] = []
        self._cancelled = False

    def add_step(self, algorithm_id: str, params: dict = None, output_label: str = None) -> "ProcessingPipeline":
        self.steps.append(PipelineStep(algorithm_id=algorithm_id, params=params or {}, output_label=output_label))
        return self

    def on_progress(self, callback: Callable[[float, str], None]) -> "ProcessingPipeline":
        self._progress_callbacks.append(callback)
        return self

    def cancel(self):
        self._cancelled = True

    def run(self, volume: Any, current_label: Any = None) -> PipelineResult:
        import time
        start = time.perf_counter()
        results = []
        working_volume = volume
        working_label = current_label
        working_mesh = None

        for i, step in enumerate(self.steps):
            if self._cancelled:
                return PipelineResult(steps=results, success=False, error="Cancelled by user")

            progress = i / len(self.steps)
            for cb in self._progress_callbacks:
                cb(progress, f"Running: {step.algorithm_id}")

            definition = self.registry.get(step.algorithm_id)
            if definition is None:
                return PipelineResult(steps=results, success=False,
                                      error=f"Algorithm '{step.algorithm_id}' not found")
            inputs = {
                "volume": working_volume,
                "label_data": working_label,
                "mesh_data": working_mesh,
            }
            input_data = inputs.get(definition.input_parameter)
            if input_data is None:
                return PipelineResult(
                    steps=results,
                    success=False,
                    error=f"{step.algorithm_id} requires {definition.input_parameter}",
                )

            r = self.registry.run(
                step.algorithm_id,
                input_data,
                step.params,
                progress_callback=lambda p: [cb(progress + p / len(self.steps), step.algorithm_id) for cb in self._progress_callbacks]
            )
            results.append(r)

            if not r.success:
                return PipelineResult(steps=results, success=False, error=r.error, total_time_sec=time.perf_counter() - start)

            if r.volume_data is not None:
                from core.volume_data import VolumeData

                if isinstance(r.volume_data, VolumeData):
                    working_volume = r.volume_data
                else:
                    working_volume = VolumeData(
                        array=r.volume_data,
                        spacing=volume.spacing_mm,
                        origin=volume.origin_mm,
                        direction=volume.direction,
                        name=f"{volume.name}_{step.algorithm_id}",
                        modality=volume.modality,
                    )
            if r.label_data is not None:
                working_label = r.label_data
            if r.mesh_data is not None:
                working_mesh = r.mesh_data

        final_result = working_mesh
        if final_result is None:
            final_result = working_label
        if final_result is None:
            final_result = working_volume
        return PipelineResult(steps=results, final_result=final_result,
                              total_time_sec=time.perf_counter() - start)
