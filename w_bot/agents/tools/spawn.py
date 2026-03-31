"""Spawn tool for creating background subagents."""

import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from w_bot.agents.tools.base import Tool
from w_bot.agents.tools.common import append_jsonl


class SpawnTool(Tool):
    """Tool to spawn a background task."""

    def __init__(self, workspace_root: Path):
        self._workspace_root = workspace_root

    @property
    def name(self) -> str:
        return "spawn"

    @property
    def description(self) -> str:
        return (
            "Spawn a subagent to handle a task in the background. "
            "Use this for complex or time-consuming tasks that can run independently."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "The task for the subagent to complete"},
                "label": {"type": "string", "description": "Optional short label for the task"},
            },
            "required": ["task"],
        }

    async def execute(self, task: str, label: str | None = None, **kwargs: Any) -> str:
        job = {
            "id": uuid.uuid4().hex,
            "task": task,
            "label": label or "",
            "status": "pending",
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        append_jsonl(self._workspace_root / ".w_bot_spawn_jobs.jsonl", job)
        return f"Spawned task: id={job['id']}"
