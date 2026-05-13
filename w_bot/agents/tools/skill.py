from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from w_bot.agents.skills import SkillsLoader
from w_bot.agents.tools.base import Tool


def _runtime_context(kwargs: dict[str, Any]) -> dict[str, Any]:
    value = kwargs.get("_wbot_tool_context")
    return value if isinstance(value, dict) else {}


def _emit_status(runtime: dict[str, Any], text: str) -> None:
    callback = runtime.get("status_callback")
    if not callable(callback):
        return
    try:
        callback(str(text))
    except Exception:
        return


def _linked_skill_files(skill_dir: Path) -> list[str]:
    linked_roots = ("references", "templates", "scripts", "assets")
    result: list[str] = []
    for root_name in linked_roots:
        root = skill_dir / root_name
        if not root.exists() or not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file():
                result.append(path.relative_to(skill_dir).as_posix())
    return result


def _resolve_skill_file(skill_dir: Path, file_path: str) -> Path | None:
    candidate = (skill_dir / file_path).resolve()
    try:
        candidate.relative_to(skill_dir.resolve())
    except ValueError:
        return None
    if not candidate.is_file():
        return None
    return candidate


class SkillsListTool(Tool):
    def __init__(self, *, skills_loader: SkillsLoader):
        self._skills_loader = skills_loader

    @property
    def name(self) -> str:
        return "skills_list"

    @property
    def description(self) -> str:
        return (
            "List available skills with name, description, availability, and path. "
            "Use skill_view to load a relevant skill before applying its workflow."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "available_only": {
                    "type": "boolean",
                    "description": "When true, only return skills whose declared requirements are available.",
                },
            },
            "required": [],
        }

    async def execute(self, available_only: bool = False, **kwargs: Any) -> str:
        del kwargs
        skills = self._skills_loader.list_skills(filter_unavailable=available_only)
        payload: list[dict[str, Any]] = []
        for skill in skills:
            check = self._skills_loader.check_requirements(skill)
            item: dict[str, Any] = {
                "name": skill.name,
                "description": skill.description or skill.name,
                "available": check.available,
                "source": skill.source,
                "path": str(skill.path),
            }
            if not check.available:
                item["missing"] = {
                    "bins": list(check.missing_bins),
                    "env": list(check.missing_env),
                }
            payload.append(item)
        return json.dumps(
            {
                "skills": payload,
                "hint": "Call skill_view(name) to load the matching SKILL.md before using general tools.",
            },
            ensure_ascii=False,
            indent=2,
        )


class SkillViewTool(Tool):
    def __init__(self, *, skills_loader: SkillsLoader):
        self._skills_loader = skills_loader

    @property
    def name(self) -> str:
        return "skill_view"

    @property
    def description(self) -> str:
        return (
            "Load a skill's instructions or one of its linked files. "
            "Use this instead of read_file for skill documents so the model follows a stable skill-loading path."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Exact skill name from skills_list or the skills summary."},
                "file_path": {
                    "type": "string",
                    "description": "Optional linked file path inside the skill, such as references/api.md or scripts/setup.py.",
                },
            },
            "required": ["name"],
        }

    async def execute(self, name: str, file_path: str = "", **kwargs: Any) -> str:
        del kwargs
        skill = self._skills_loader.get_skill(name)
        if skill is None:
            return json.dumps(
                {
                    "error": f"Skill not found: {name}",
                    "hint": "Call skills_list() to see exact skill names.",
                },
                ensure_ascii=False,
                indent=2,
            )

        skill_dir = skill.path.parent
        target_path = skill.path
        if file_path:
            resolved = _resolve_skill_file(skill_dir, file_path)
            if resolved is None:
                return json.dumps(
                    {
                        "error": f"Linked file not found or outside skill: {file_path}",
                        "skill": skill.name,
                        "linked_files": _linked_skill_files(skill_dir),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            target_path = resolved

        try:
            content = target_path.read_text(encoding="utf-8")
        except OSError as exc:
            return json.dumps(
                {"error": f"Failed to read skill file: {exc}", "skill": skill.name},
                ensure_ascii=False,
                indent=2,
            )

        check = self._skills_loader.check_requirements(skill)
        return json.dumps(
            {
                "name": skill.name,
                "description": skill.description or skill.name,
                "available": check.available,
                "source": skill.source,
                "path": str(target_path),
                "content": content,
                "linked_files": _linked_skill_files(skill_dir),
                "usage_hint": "Follow the loaded skill instructions in the current agent unless the user asked for isolated/background execution.",
            },
            ensure_ascii=False,
            indent=2,
        )


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

        _emit_status(runtime, f"准备执行 Skill：{skill_name}")
        result = await graph.run_skill_subagent(
            skill_name=skill_name,
            task=task,
            arguments=arguments or {},
            context_messages=list(runtime.get("state_messages") or []),
            thread_id=str(runtime.get("thread_id") or "-"),
            status_callback=runtime.get("status_callback") if callable(runtime.get("status_callback")) else None,
        )
        _emit_status(runtime, f"Skill 执行完成：{skill_name}")
        return json.dumps(result, ensure_ascii=False, indent=2)
