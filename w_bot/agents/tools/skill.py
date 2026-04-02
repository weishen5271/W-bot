from __future__ import annotations

import json
from typing import Any

from w_bot.agents.skills import SkillsLoader
from w_bot.agents.tools.base import Tool


def _runtime_context(kwargs: dict[str, Any]) -> dict[str, Any]:
    value = kwargs.get("_wbot_tool_context")
    return value if isinstance(value, dict) else {}


class RunSkillTool(Tool):
    def __init__(self, *, skills_loader: SkillsLoader):
        self._skills_loader = skills_loader

    @property
    def name(self) -> str:
        return "run_skill"

    @property
    def description(self) -> str:
        return (
            "Execute a named skill inside a forked subagent. "
            "Use this only when the user explicitly wants isolated, delegated, parallel, or background execution."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "skill_name": {"type": "string", "description": "Exact skill name from the skills summary"},
                "task": {"type": "string", "description": "Concrete task for the skill to perform"},
                "arguments": {"type": "object", "description": "Optional structured arguments for the skill"},
            },
            "required": ["skill_name", "task"],
        }

    async def execute(
        self,
        skill_name: str,
        task: str,
        arguments: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        runtime = _runtime_context(kwargs)
        graph = runtime.get("graph")
        if graph is None:
            return "Error: skill runtime is not available"

        depth = runtime.get("subagent_depth")
        if isinstance(depth, int) and depth > 0:
            return "Error: nested run_skill is not allowed inside a subagent"

        skill = self._skills_loader.get_skill(skill_name)
        if skill is None:
            return f"Error: Skill not found: {skill_name}"

        result = await graph.run_skill_subagent(
            skill_name=skill_name,
            task=task,
            arguments=arguments or {},
            context_messages=list(runtime.get("state_messages") or []),
            thread_id=str(runtime.get("thread_id") or "-"),
            status_callback=runtime.get("status_callback") if callable(runtime.get("status_callback")) else None,
        )
        return json.dumps(result, ensure_ascii=False, indent=2)
