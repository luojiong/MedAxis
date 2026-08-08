"""
MedAxis Processing Layer — Algorithm Registry & Pipeline.

All algorithms (classical ITK-based and AI API-based) are registered here.
"""
from typing import Any, Callable, Optional
from dataclasses import dataclass, field
from enum import Enum
import time


class AlgorithmCategory(Enum):
    FILTERING = "filtering"
    MORPHOLOGY = "morphology"
    THRESHOLD = "threshold"
    SEGMENTATION = "segmentation"
    RECONSTRUCTION = "reconstruction"
    SURFACE = "surface"
    REGISTRATION = "registration"
    ARITHMETIC = "arithmetic"
    ENHANCEMENT = "enhancement"
    RADIOMICS = "radiomics"
    AI = "ai"
    MEASUREMENT = "measurement"


@dataclass
class AlgorithmParameter:
    name: str
    type: str                            # "int", "float", "str", "bool", "choice", "range", "point3d"
    label: str
    description: str = ""
    default: Any = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    choices: Optional[list[str]] = None
    required: bool = False


@dataclass
class AlgorithmDefinition:
    id: str
    name: str
    category: AlgorithmCategory
    description: str = ""
    parameters: list[AlgorithmParameter] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    # The actual callable (Python function wrapping ITK filter or API call)
    run_func: Optional[Callable] = None
    estimated_time_sec: float = 1.0
    input_parameter: str = "volume"


@dataclass
class AlgorithmResult:
    algorithm_id: str
    label_data: Any = None         # LabelData
    mesh_data: Any = None          # MeshData
    volume_data: Any = None        # VolumeData or NumPy image result
    statistics: dict = field(default_factory=dict)
    execution_time_sec: float = 0.0
    error: Optional[str] = None
    success: bool = True


class AlgorithmRegistry:
    """Global registry of all available algorithms."""
    _instance: Optional["AlgorithmRegistry"] = None
    _algorithms: dict[str, AlgorithmDefinition] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._algorithms = {}
        return cls._instance

    def register(self, algo: AlgorithmDefinition) -> None:
        if not algo.id or not algo.id.replace("_", "").isalnum():
            raise ValueError("Algorithm ids must contain only letters, numbers and underscores")
        self._algorithms[algo.id] = algo

    def unregister(self, algo_id: str):
        self._algorithms.pop(algo_id, None)

    def get(self, algo_id: str) -> Optional[AlgorithmDefinition]:
        return self._algorithms.get(algo_id)

    def list_all(self) -> list[AlgorithmDefinition]:
        return list(self._algorithms.values())

    def list_by_category(self, category: AlgorithmCategory) -> list[AlgorithmDefinition]:
        return [a for a in self._algorithms.values() if a.category == category]

    def list_by_tag(self, tag: str) -> list[AlgorithmDefinition]:
        return [a for a in self._algorithms.values() if tag in a.tags]

    def run(self, algo_id: str, input_data: Any, params: Optional[dict] = None,
            progress_callback: Optional[Callable[[float], None]] = None) -> AlgorithmResult:
        algo = self._algorithms.get(algo_id)
        if algo is None:
            return AlgorithmResult(algorithm_id=algo_id, success=False, error=f"Algorithm '{algo_id}' not found")
        if algo.run_func is None:
            return AlgorithmResult(algorithm_id=algo_id, success=False, error=f"Algorithm '{algo_id}' has no run_func")

        start = time.perf_counter()
        try:
            from .parameters import ParameterSchema

            validated_params = ParameterSchema(
                algorithm_id=algo.id,
                parameters=algo.parameters,
            ).validate_params(params or {})
            if progress_callback is not None:
                progress_callback(0.0)
            result = algo.run_func(
                **{algo.input_parameter: input_data},
                params=validated_params,
                progress_callback=progress_callback,
            )
            elapsed = time.perf_counter() - start
            if isinstance(result, AlgorithmResult):
                result.execution_time_sec = elapsed
                if progress_callback is not None:
                    progress_callback(1.0)
                return result
            if progress_callback is not None:
                progress_callback(1.0)
            return AlgorithmResult(algorithm_id=algo_id, label_data=result, execution_time_sec=elapsed)
        except Exception as e:
            elapsed = time.perf_counter() - start
            return AlgorithmResult(algorithm_id=algo_id, success=False, error=str(e), execution_time_sec=elapsed)


# global singleton
algorithm_registry = AlgorithmRegistry()
