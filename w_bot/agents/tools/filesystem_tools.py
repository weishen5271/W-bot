from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from .common import resolve_read_path, resolve_workspace_path


def build_filesystem_tools(*, workspace_root: Path, readonly_roots: list[Path]) -> list[Any]:
    @tool
    def read_file(path: str, start_line: int = 1, end_line: int = 300) -> str:
        """Read a file from the workspace or readonly roots and return selected lines."""
        try:
            target = resolve_read_path(path, readonly_roots=readonly_roots)
        except ValueError as exc:
            return str(exc)
        if not target.exists():
            return f"File not found: {target}"
        if not target.is_file():
            return f"Path is not a file: {target}"

        lines = target.read_text(encoding="utf-8").splitlines()
        start = max(1, start_line)
        end = min(len(lines), max(start, end_line))
        return "\n".join(lines[start - 1 : end])

    @tool
    def modify_file(
        path: str,
        mode: str,
        content: str = "",
        find_text: str = "",
        replace_text: str = "",
        replace_all: bool = False,
        overwrite: bool = True,
    ) -> str:
        """Create, replace, or patch a file inside the workspace."""
        normalized_mode = (mode or "").strip().lower()
        if normalized_mode not in {"create", "replace", "patch"}:
            return "Invalid mode. Supported values: create, replace, patch"

        try:
            target = resolve_workspace_path(path, workspace_root=workspace_root)
        except ValueError as exc:
            return str(exc)

        if normalized_mode in {"create", "replace"}:
            existed_before = target.exists()
            if target.exists() and normalized_mode == "create" and not overwrite:
                return f"File already exists and overwrite=false: {target}"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            action = "Created" if normalized_mode == "create" and not existed_before else "Wrote"
            return f"{action} {len(content)} chars to {target}"

        if not target.exists() or not target.is_file():
            return f"File not found: {target}"
        source = target.read_text(encoding="utf-8")
        if find_text not in source:
            return "No match found"
        if replace_all:
            updated = source.replace(find_text, replace_text)
            changed = source.count(find_text)
        else:
            updated = source.replace(find_text, replace_text, 1)
            changed = 1
        target.write_text(updated, encoding="utf-8")
        return f"Updated {changed} occurrence(s) in {target}"

    @tool
    def list_dir(path: str = ".", recursive: bool = False) -> str:
        """List files or directories inside the workspace."""
        try:
            target = resolve_workspace_path(path, workspace_root=workspace_root)
        except ValueError as exc:
            return str(exc)
        if not target.exists():
            return f"Path not found: {target}"
        if target.is_file():
            stat = target.stat()
            return f"FILE\t{target}\t{stat.st_size}"

        iterator = target.rglob("*") if recursive else target.glob("*")
        output: list[str] = []
        for item in sorted(iterator):
            kind = "DIR" if item.is_dir() else "FILE"
            size = item.stat().st_size if item.is_file() else 0
            output.append(f"{kind}\t{item}\t{size}")
        return "\n".join(output[:2000]) or "(empty)"

    return [read_file, modify_file, list_dir]
