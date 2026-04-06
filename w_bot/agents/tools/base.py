from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any


class Tool(ABC):
    _TYPE_MAP = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    @staticmethod
    def _resolve_type(raw_type: Any) -> str | None:
        if isinstance(raw_type, list):
            for item in raw_type:
                if item != "null":
                    return str(item)
            return None
        return str(raw_type) if isinstance(raw_type, str) else None

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        pass

    @abstractmethod
    async def execute(self, **kwargs: Any) -> Any:
        pass

    async def ainvoke(self, params: dict[str, Any] | None = None) -> Any:
        arguments = params if isinstance(params, dict) else {}
        normalized = self.cast_params(arguments)
        errors = self.validate_params(normalized)
        if errors:
            return "Invalid parameters: " + "; ".join(errors)
        return await self.execute(**normalized)

    def invoke(self, params: dict[str, Any] | None = None) -> Any:
        try:
            asyncio.get_running_loop()
            # Already in async context - run in thread pool to avoid blocking
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, self.ainvoke(params))
                return future.result()
        except RuntimeError:
            return asyncio.run(self.ainvoke(params))

    def cast_params(self, params: dict[str, Any]) -> dict[str, Any]:
        schema = self.parameters or {}
        if schema.get("type", "object") != "object":
            return params
        return self._cast_object(params, schema)

    def _cast_object(self, obj: Any, schema: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(obj, dict):
            return obj
        props = schema.get("properties", {})
        result: dict[str, Any] = {}
        for key, value in obj.items():
            if key in props:
                result[key] = self._cast_value(value, props[key])
            else:
                result[key] = value
        return result

    def _cast_value(self, value: Any, schema: dict[str, Any]) -> Any:
        target_type = self._resolve_type(schema.get("type"))
        if target_type == "boolean" and isinstance(value, bool):
            return value
        if target_type == "integer" and isinstance(value, int) and not isinstance(value, bool):
            return value
        if target_type in self._TYPE_MAP and target_type not in {"boolean", "integer", "array", "object"}:
            expected = self._TYPE_MAP[target_type]
            if isinstance(value, expected):
                return value
        if target_type == "integer" and isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                return value
        if target_type == "number" and isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return value
        if target_type == "boolean" and isinstance(value, str):
            lowered = value.lower()
            if lowered in {"true", "1", "yes"}:
                return True
            if lowered in {"false", "0", "no"}:
                return False
            return value
        if target_type == "string":
            return value if value is None else str(value)
        if target_type == "array" and isinstance(value, list):
            item_schema = schema.get("items")
            return [self._cast_value(item, item_schema) for item in value] if item_schema else value
        if target_type == "object" and isinstance(value, dict):
            return self._cast_object(value, schema)
        return value

    def validate_params(self, params: dict[str, Any]) -> list[str]:
        if not isinstance(params, dict):
            return [f"parameters must be an object, got {type(params).__name__}"]
        schema = self.parameters or {}
        if schema.get("type", "object") != "object":
            raise ValueError(f"Schema must be object type, got {schema.get('type')!r}")
        return self._validate(params, {**schema, "type": "object"}, "")

    def _validate(self, value: Any, schema: dict[str, Any], path: str) -> list[str]:
        raw_type = schema.get("type")
        nullable = (isinstance(raw_type, list) and "null" in raw_type) or schema.get("nullable", False)
        target_type = self._resolve_type(raw_type)
        label = path or "parameter"
        if nullable and value is None:
            return []
        if target_type == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
            return [f"{label} should be integer"]
        if target_type == "number" and (not isinstance(value, self._TYPE_MAP[target_type]) or isinstance(value, bool)):
            return [f"{label} should be number"]
        if target_type in self._TYPE_MAP and target_type not in {"integer", "number"}:
            if not isinstance(value, self._TYPE_MAP[target_type]):
                return [f"{label} should be {target_type}"]

        errors: list[str] = []
        if "enum" in schema and value not in schema["enum"]:
            errors.append(f"{label} must be one of {schema['enum']}")
        if target_type in {"integer", "number"}:
            if "minimum" in schema and value < schema["minimum"]:
                errors.append(f"{label} must be >= {schema['minimum']}")
            if "maximum" in schema and value > schema["maximum"]:
                errors.append(f"{label} must be <= {schema['maximum']}")
        if target_type == "string":
            if "minLength" in schema and len(value) < schema["minLength"]:
                errors.append(f"{label} must be at least {schema['minLength']} chars")
            if "maxLength" in schema and len(value) > schema["maxLength"]:
                errors.append(f"{label} must be at most {schema['maxLength']} chars")
        if target_type == "object":
            props = schema.get("properties", {})
            for key in schema.get("required", []):
                if key not in value:
                    errors.append(f"missing required {path + '.' + key if path else key}")
            for key, item in value.items():
                if key in props:
                    errors.extend(self._validate(item, props[key], path + "." + key if path else key))
        if target_type == "array" and "items" in schema:
            for index, item in enumerate(value):
                child_path = f"{path}[{index}]" if path else f"[{index}]"
                errors.extend(self._validate(item, schema["items"], child_path))
        return errors

    def to_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class FunctionTool(Tool):
    def __init__(
        self,
        *,
        name: str,
        description: str,
        parameters: dict[str, Any],
        func: Callable[..., Any] | None = None,
        coroutine: Callable[..., Awaitable[Any]] | None = None,
    ) -> None:
        self._name = name
        self._description = description
        self._parameters = parameters
        self._func = func
        self._coroutine = coroutine

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._parameters

    async def execute(self, **kwargs: Any) -> Any:
        params = self.cast_params(kwargs)
        errors = self.validate_params(params)
        if errors:
            return "Invalid parameters: " + "; ".join(errors)
        if self._coroutine is not None:
            return await self._coroutine(**params)
        if self._func is not None:
            return await asyncio.to_thread(self._func, **params)
        raise RuntimeError(f"Tool {self._name} has no implementation")
