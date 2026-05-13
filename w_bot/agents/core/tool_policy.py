from __future__ import annotations

from pathlib import Path
from typing import Any


READ_ONLY_TOOL_TOKENS = (
    "read",
    "search",
    "fetch",
    "list",
    "get",
    "query",
    "session_search",
    "web",
)

MUTATING_TOOL_TOKENS = (
    "write",
    "edit",
    "patch",
    "delete",
    "remove",
    "move",
    "rename",
    "shell",
    "exec",
    "terminal",
    "memory",
    "cron",
    "spawn",
    "subagent",
    "run_skill",
)

PATH_ARG_NAMES = ("path", "file", "file_path", "target", "working_dir", "cwd")


class ToolPolicy:
    """Decide whether a batch of tool calls can run concurrently."""

    def can_parallelize(self, tool_calls: list[dict[str, Any]]) -> bool:
        if len(tool_calls) <= 1:
            return False
        parsed = [(self._tool_name(item), self._tool_args(item)) for item in tool_calls]
        if not all(self.is_readonly(name, args) for name, args in parsed):
            return False
        paths = [path for _, args in parsed for path in self._extract_paths(args)]
        return len(paths) == len(set(paths))

    def is_readonly(self, tool_name: str, args: dict[str, Any] | None = None) -> bool:
        del args
        normalized = self._normalize_name(tool_name)
        if not normalized:
            return False
        if self.is_mutating(normalized):
            return False
        return any(token in normalized for token in READ_ONLY_TOOL_TOKENS)

    def is_mutating(self, tool_name: str, args: dict[str, Any] | None = None) -> bool:
        del args
        normalized = self._normalize_name(tool_name)
        return any(token in normalized for token in MUTATING_TOOL_TOKENS)

    def requires_serial(self, tool_name: str, args: dict[str, Any] | None = None) -> bool:
        return not self.is_readonly(tool_name, args)

    @staticmethod
    def _tool_name(tool_call: dict[str, Any]) -> str:
        return str(tool_call.get("name") or "").strip()

    @staticmethod
    def _tool_args(tool_call: dict[str, Any]) -> dict[str, Any]:
        args = tool_call.get("args")
        if args is None:
            args = tool_call.get("arguments")
        return args if isinstance(args, dict) else {}

    @staticmethod
    def _normalize_name(tool_name: str) -> str:
        return str(tool_name or "").strip().lower()

    @staticmethod
    def _extract_paths(args: dict[str, Any]) -> list[str]:
        paths: list[str] = []
        for key in PATH_ARG_NAMES:
            value = args.get(key)
            if isinstance(value, str) and value.strip():
                try:
                    paths.append(str(Path(value).expanduser()))
                except Exception:
                    paths.append(value.strip())
        return paths

