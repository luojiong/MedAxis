"""
Parameter schema definition and validation using Pydantic.
"""
from pydantic import BaseModel, Field
from .algorithm_registry import AlgorithmParameter


class ParameterSchema(BaseModel):
    """Defines the parameter schema for an algorithm."""
    algorithm_id: str
    parameters: list[AlgorithmParameter] = Field(default_factory=list)

    def validate_params(self, params: dict) -> dict:
        """Validate and fill defaults for input parameters."""
        validated = {}
        param_map = {p.name: p for p in self.parameters}

        for name, p in param_map.items():
            if name in params:
                val = params[name]
                if p.type == "int":
                    val = int(val)
                elif p.type == "float":
                    val = float(val)
                elif p.type == "bool":
                    val = bool(val)
                elif p.type == "str":
                    val = str(val)
                elif p.type == "choice" and p.choices and val not in p.choices:
                    raise ValueError(f"Parameter '{name}': '{val}' not in choices {p.choices}")

                if p.min_val is not None and isinstance(val, (int, float)) and val < p.min_val:
                    raise ValueError(f"Parameter '{name}': {val} < min {p.min_val}")
                if p.max_val is not None and isinstance(val, (int, float)) and val > p.max_val:
                    raise ValueError(f"Parameter '{name}': {val} > max {p.max_val}")

                validated[name] = val
            elif p.required:
                raise ValueError(f"Missing required parameter: '{name}'")
            elif p.default is not None:
                validated[name] = p.default

        return validated

    def to_json_schema(self) -> dict:
        """Generate JSON Schema for MCP Tool description."""
        properties = {}
        required = []
        for p in self.parameters:
            type_map = {"int": "integer", "float": "number", "str": "string", "bool": "boolean"}
            prop = {
                "type": type_map.get(p.type, "string"),
                "description": p.description,
            }
            if p.default is not None:
                prop["default"] = p.default
            if p.choices:
                prop["enum"] = p.choices
            if p.min_val is not None:
                prop["minimum"] = p.min_val
            if p.max_val is not None:
                prop["maximum"] = p.max_val
            properties[p.name] = prop
            if p.required:
                required.append(p.name)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }
