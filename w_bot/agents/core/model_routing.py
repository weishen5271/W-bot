from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AnyMessage

from .message_utils import _has_native_image_blocks, _route_for_history


@dataclass(frozen=True)
class ModelRouteSelection:
    llm: Any
    route: str
    tool_names: tuple[str, ...]


class ModelRouter:
    """Select the model route and bind tools for the current turn."""

    def __init__(
        self,
        *,
        llm_text: Any,
        llm_image: Any = None,
        llm_audio: Any = None,
        tools_by_name: dict[str, Any] | None = None,
    ) -> None:
        self._llm_text = llm_text
        self._llm_image = llm_image
        self._llm_audio = llm_audio
        self._tools_by_name = tools_by_name or {}
        self._llm_tool_cache: dict[tuple[str, tuple[str, ...]], Any] = {}

    def select(
        self,
        *,
        history: list[AnyMessage],
        messages: list[AnyMessage],
    ) -> ModelRouteSelection:
        route = _route_for_history(history)
        if route == "image" and self._llm_image is None:
            route = "text"
        if route == "audio" and self._llm_audio is None:
            route = "text"
        if route == "image" and not _has_native_image_blocks(messages):
            route = "text"
        tool_names = tuple(sorted(self._tools_by_name))
        return ModelRouteSelection(
            llm=self._llm_for_route_with_tools(route, tool_names),
            route=route,
            tool_names=tool_names,
        )

    def _llm_for_route_with_tools(self, route: str, tool_names: tuple[str, ...]) -> Any:
        base_llm = self._llm_for_route(route)
        if not tool_names:
            return base_llm
        cache_key = (route, tool_names)
        cached = self._llm_tool_cache.get(cache_key)
        if cached is not None:
            return cached
        bind_tools = getattr(base_llm, "bind_tools", None)
        if not callable(bind_tools):
            return base_llm
        selected_tools = [self._tools_by_name[name] for name in tool_names if name in self._tools_by_name]
        bindable_tools = [
            tool.to_schema() if hasattr(tool, "to_schema") and callable(tool.to_schema) else tool
            for tool in selected_tools
        ]
        bound = bind_tools(bindable_tools)
        self._llm_tool_cache[cache_key] = bound
        return bound

    def _llm_for_route(self, route: str) -> Any:
        if route == "image" and self._llm_image is not None:
            return self._llm_image
        if route == "audio" and self._llm_audio is not None:
            return self._llm_audio
        return self._llm_text

