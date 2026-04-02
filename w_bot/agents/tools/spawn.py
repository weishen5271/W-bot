"""Subagent management tools."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from w_bot.agents.tools.base import Tool


def _graph_from_kwargs(kwargs: dict[str, Any]) -> Any | None:
    runtime = kwargs.get("_wbot_tool_context")
    if not isinstance(runtime, dict):
        return None
    return runtime.get("graph")


def _thread_id_from_kwargs(kwargs: dict[str, Any]) -> str:
    runtime = kwargs.get("_wbot_tool_context")
    if not isinstance(runtime, dict):
        return "-"
    thread_id = runtime.get("thread_id")
    return thread_id.strip() if isinstance(thread_id, str) and thread_id.strip() else "-"


def _context_messages_from_kwargs(kwargs: dict[str, Any]) -> list[Any]:
    runtime = kwargs.get("_wbot_tool_context")
    if not isinstance(runtime, dict):
        return []
    messages = runtime.get("state_messages")
    return list(messages) if isinstance(messages, list) else []


def _status_callback_from_kwargs(kwargs: dict[str, Any]) -> Any:
    runtime = kwargs.get("_wbot_tool_context")
    if not isinstance(runtime, dict):
        return None
    callback = runtime.get("status_callback")
    return callback if callable(callback) else None


def _emit_tool_status(kwargs: dict[str, Any], text: str) -> None:
    callback = _status_callback_from_kwargs(kwargs)
    if callable(callback):
        callback(str(text))


class SpawnTool(Tool):
    """Tool to spawn a background subagent."""

    def __init__(self, workspace_root: Path):
        self._workspace_root = workspace_root

    @property
    def name(self) -> str:
        return "spawn"

    @property
    def description(self) -> str:
        return (
            "Spawn a subagent to handle an independent task. "
            "Use list_subagents or wait_subagent to collect results later."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "The task for the subagent to complete"},
                "label": {"type": "string", "description": "Optional short label for the task"},
                "agent_type": {
                    "type": "string",
                    "description": "Subagent profile: worker, explore, plan, or verify",
                    "enum": ["worker", "explore", "plan", "verify"],
                },
            },
            "required": ["task"],
        }

    async def execute(
        self,
        task: str,
        label: str | None = None,
        agent_type: str = "worker",
        **kwargs: Any,
    ) -> str:
        graph = _graph_from_kwargs(kwargs)
        if graph is None:
            return "Error: subagent runtime is not available"
        context_messages = _context_messages_from_kwargs(kwargs)
        result = graph.spawn_subagent(
            agent_type=agent_type,
            task=task,
            label=label or "",
            context_messages=context_messages,
            parent_thread_id=_thread_id_from_kwargs(kwargs),
        )
        return (
            f"Spawned subagent: id={result['id']} status={result['status']} "
            f"type={result['agent_type']} label={result['label'] or '-'}"
        )


class ListSubagentsTool(Tool):
    @property
    def name(self) -> str:
        return "list_subagents"

    @property
    def description(self) -> str:
        return "List spawned subagents and their latest status."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "status": {"type": "string", "description": "Optional status filter"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100},
            },
        }

    async def execute(self, status: str | None = None, limit: int = 20, **kwargs: Any) -> str:
        graph = _graph_from_kwargs(kwargs)
        if graph is None:
            return "Error: subagent runtime is not available"
        jobs = graph.list_subagents(status=status, limit=limit)
        return json.dumps(jobs, ensure_ascii=False, indent=2)


class WaitSubagentTool(Tool):
    @property
    def name(self) -> str:
        return "wait_subagent"

    @property
    def description(self) -> str:
        return "Wait for a spawned subagent to finish and return its current result."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Subagent id"},
                "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 600},
            },
            "required": ["id"],
        }

    async def execute(self, id: str, timeout_seconds: int = 60, **kwargs: Any) -> str:
        graph = _graph_from_kwargs(kwargs)
        if graph is None:
            return "Error: subagent runtime is not available"
        remaining = max(1, int(timeout_seconds))
        result: dict[str, Any] = {"id": id, "status": "timeout"}
        while remaining > 0:
            slice_seconds = min(2, remaining)
            _emit_tool_status(kwargs, f"正在等待子任务 {id[:8]} 完成（剩余超时 {remaining}s）")
            result = graph.wait_for_subagent(id, timeout_seconds=slice_seconds)
            status = str(result.get("status") or "").strip().lower()
            if status in {"completed", "failed", "timeout", "not_found"}:
                _emit_tool_status(kwargs, f"子任务 {id[:8]} 当前状态：{status}")
                break
            remaining -= slice_seconds
        else:
            result = graph.wait_for_subagent(id, timeout_seconds=0)
            _emit_tool_status(kwargs, f"等待子任务 {id[:8]} 超时")
        return json.dumps(result, ensure_ascii=False, indent=2)
