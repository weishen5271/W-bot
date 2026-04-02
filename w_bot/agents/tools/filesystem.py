"""File system tools: read, write, edit, list."""

import difflib
import json
import mimetypes
from pathlib import Path
from typing import Any

from w_bot.agents.escalation import EscalationManager
from w_bot.agents.tools.base import Tool
from w_bot.utils.helpers import build_image_content_blocks, detect_image_mime


def _resolve_path(
    path: str,
    workspace: Path | None = None,
    allowed_dir: Path | None = None,
    extra_allowed_dirs: list[Path] | None = None,
) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute() and workspace:
        p = workspace / p
    resolved = p.resolve()
    if allowed_dir:
        all_dirs = [allowed_dir] + (extra_allowed_dirs or [])
        if not any(_is_under(resolved, d) for d in all_dirs):
            raise PermissionError(f"Path {path} is outside allowed directory {allowed_dir}")
    return resolved


def _is_under(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory.resolve())
        return True
    except ValueError:
        return False


class _FsTool(Tool):
    def __init__(
        self,
        workspace: Path | None = None,
        allowed_dir: Path | None = None,
        extra_allowed_dirs: list[Path] | None = None,
        escalation_manager: EscalationManager | None = None,
    ):
        self._workspace = workspace
        self._allowed_dir = allowed_dir
        self._extra_allowed_dirs = extra_allowed_dirs
        self._escalation_manager = escalation_manager

    def _resolve(
        self,
        path: str,
        *,
        tool_name: str,
        tool_context: dict[str, Any] | None = None,
        justification: str,
        prefix_rule: list[str] | None = None,
    ) -> Path | str:
        session_id = str((tool_context or {}).get("thread_id") or "-").strip() or "-"
        p = Path(path).expanduser()
        if not p.is_absolute() and self._workspace:
            p = self._workspace / p
        resolved = p.resolve()
        if self._allowed_dir:
            all_dirs = [self._allowed_dir] + (self._extra_allowed_dirs or [])
            if not any(_is_under(resolved, d) for d in all_dirs):
                command = f"{tool_name} {resolved}"
                if self._escalation_manager and self._escalation_manager.is_command_approved(
                    session_id=session_id,
                    command=command,
                ):
                    return resolved
                if self._escalation_manager is not None:
                    request = self._escalation_manager.create_request(
                        session_id=session_id,
                        command=command,
                        working_dir=str((self._workspace or Path.cwd()).resolve()),
                        justification=justification,
                        prefix_rule=prefix_rule or [tool_name],
                        risk_type="workspace_path",
                    )
                    payload = {
                        "type": "escalation_request",
                        "request_id": request.id,
                        "session_id": request.session_id,
                        "status": request.status,
                        "risk_type": request.risk_type,
                        "justification": request.justification,
                        "command": request.command,
                        "working_dir": request.working_dir,
                        "prefix_rule": request.prefix_rule,
                        "message": (
                            f"操作需要提权审批。请求ID={request.id}。"
                            "请让用户执行 /escalation 查看详情，并使用 /approve <请求ID> 或 /deny <请求ID> [原因]。"
                        ),
                    }
                    return json.dumps(payload, ensure_ascii=False)
                raise PermissionError(f"Path {path} is outside allowed directory {self._allowed_dir}")
        return resolved


class ReadFileTool(_FsTool):
    _MAX_CHARS = 128_000
    _DEFAULT_LIMIT = 2000

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return "Read the contents of a file. Returns numbered lines. Use offset and limit to paginate through large files."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The file path to read"},
                "offset": {"type": "integer", "description": "Line number to start reading from (1-indexed, default 1)", "minimum": 1},
                "limit": {"type": "integer", "description": "Maximum number of lines to read (default 2000)", "minimum": 1},
            },
            "required": ["path"],
        }

    async def execute(self, path: str | None = None, offset: int = 1, limit: int | None = None, **kwargs: Any) -> Any:
        try:
            if not path:
                return "Error reading file: Unknown path"
            tool_context = kwargs.get("_wbot_tool_context") if isinstance(kwargs.get("_wbot_tool_context"), dict) else {}
            fp = self._resolve(
                path,
                tool_name=self.name,
                tool_context=tool_context,
                justification="读取工作区外文件",
            )
            if isinstance(fp, str):
                return fp
            if not fp.exists():
                return f"Error: File not found: {path}"
            if not fp.is_file():
                return f"Error: Not a file: {path}"

            raw = fp.read_bytes()
            if not raw:
                return f"(Empty file: {path})"

            mime = detect_image_mime(raw) or mimetypes.guess_type(path)[0]
            if mime and mime.startswith("image/"):
                return build_image_content_blocks(raw, mime, str(fp), f"(Image file: {path})")

            try:
                text_content = raw.decode("utf-8")
            except UnicodeDecodeError:
                return f"Error: Cannot read binary file {path} (MIME: {mime or 'unknown'}). Only UTF-8 text and images are supported."

            all_lines = text_content.splitlines()
            total = len(all_lines)
            if offset < 1:
                offset = 1
            if offset > total:
                return f"Error: offset {offset} is beyond end of file ({total} lines)"

            start = offset - 1
            end = min(start + (limit or self._DEFAULT_LIMIT), total)
            numbered = [f"{start + i + 1}| {line}" for i, line in enumerate(all_lines[start:end])]
            result = "\n".join(numbered)

            if len(result) > self._MAX_CHARS:
                trimmed, chars = [], 0
                for line in numbered:
                    chars += len(line) + 1
                    if chars > self._MAX_CHARS:
                        break
                    trimmed.append(line)
                end = start + len(trimmed)
                result = "\n".join(trimmed)

            if end < total:
                result += f"\n\n(Showing lines {offset}-{end} of {total}. Use offset={end + 1} to continue.)"
            else:
                result += f"\n\n(End of file: {total} lines total)"
            return result
        except PermissionError as exc:
            return f"Error: {exc}"
        except Exception as exc:
            return f"Error reading file: {exc}"


class WriteFileTool(_FsTool):
    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return "Write content to a file at the given path. Creates parent directories if needed."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The file path to write to"},
                "content": {"type": "string", "description": "The content to write"},
            },
            "required": ["path", "content"],
        }

    async def execute(self, path: str | None = None, content: str | None = None, **kwargs: Any) -> str:
        try:
            if not path:
                raise ValueError("Unknown path")
            if content is None:
                raise ValueError("Unknown content")
            tool_context = kwargs.get("_wbot_tool_context") if isinstance(kwargs.get("_wbot_tool_context"), dict) else {}
            fp = self._resolve(
                path,
                tool_name=self.name,
                tool_context=tool_context,
                justification="写入工作区外文件",
            )
            if isinstance(fp, str):
                return fp
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_text(content, encoding="utf-8")
            return f"Successfully wrote {len(content)} bytes to {fp}"
        except PermissionError as exc:
            return f"Error: {exc}"
        except Exception as exc:
            return f"Error writing file: {exc}"


def _find_match(content: str, old_text: str) -> tuple[str | None, int]:
    if old_text in content:
        return old_text, content.count(old_text)

    old_lines = old_text.splitlines()
    if not old_lines:
        return None, 0
    stripped_old = [line.strip() for line in old_lines]
    content_lines = content.splitlines()

    candidates = []
    for index in range(len(content_lines) - len(stripped_old) + 1):
        window = content_lines[index : index + len(stripped_old)]
        if [line.strip() for line in window] == stripped_old:
            candidates.append("\n".join(window))

    if candidates:
        return candidates[0], len(candidates)
    return None, 0


class EditFileTool(_FsTool):
    @property
    def name(self) -> str:
        return "edit_file"

    @property
    def description(self) -> str:
        return "Edit a file by replacing old_text with new_text. Supports minor whitespace or line-ending differences. Set replace_all=true to replace every occurrence."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The file path to edit"},
                "old_text": {"type": "string", "description": "The text to find and replace"},
                "new_text": {"type": "string", "description": "The text to replace with"},
                "replace_all": {"type": "boolean", "description": "Replace all occurrences (default false)"},
            },
            "required": ["path", "old_text", "new_text"],
        }

    async def execute(
        self,
        path: str | None = None,
        old_text: str | None = None,
        new_text: str | None = None,
        replace_all: bool = False,
        **kwargs: Any,
    ) -> str:
        try:
            if not path:
                raise ValueError("Unknown path")
            if old_text is None:
                raise ValueError("Unknown old_text")
            if new_text is None:
                raise ValueError("Unknown new_text")

            tool_context = kwargs.get("_wbot_tool_context") if isinstance(kwargs.get("_wbot_tool_context"), dict) else {}
            fp = self._resolve(
                path,
                tool_name=self.name,
                tool_context=tool_context,
                justification="修改工作区外文件",
            )
            if isinstance(fp, str):
                return fp
            if not fp.exists():
                return f"Error: File not found: {path}"

            raw = fp.read_bytes()
            uses_crlf = b"\r\n" in raw
            content = raw.decode("utf-8").replace("\r\n", "\n")
            match, count = _find_match(content, old_text.replace("\r\n", "\n"))

            if match is None:
                return self._not_found_msg(old_text, content, path)
            if count > 1 and not replace_all:
                return f"Warning: old_text appears {count} times. Provide more context to make it unique, or set replace_all=true."

            norm_new = new_text.replace("\r\n", "\n")
            new_content = content.replace(match, norm_new) if replace_all else content.replace(match, norm_new, 1)
            if uses_crlf:
                new_content = new_content.replace("\n", "\r\n")

            fp.write_bytes(new_content.encode("utf-8"))
            return f"Successfully edited {fp}"
        except PermissionError as exc:
            return f"Error: {exc}"
        except Exception as exc:
            return f"Error editing file: {exc}"

    @staticmethod
    def _not_found_msg(old_text: str, content: str, path: str) -> str:
        lines = content.splitlines(keepends=True)
        old_lines = old_text.splitlines(keepends=True)
        window = len(old_lines)

        best_ratio, best_start = 0.0, 0
        for index in range(max(1, len(lines) - window + 1)):
            ratio = difflib.SequenceMatcher(None, old_lines, lines[index : index + window]).ratio()
            if ratio > best_ratio:
                best_ratio, best_start = ratio, index

        if best_ratio > 0.5:
            diff = "\n".join(difflib.unified_diff(old_lines, lines[best_start : best_start + window], fromfile="old_text (provided)", tofile=f"{path} (actual, line {best_start + 1})", lineterm=""))
            return f"Error: old_text not found in {path}.\nBest match ({best_ratio:.0%} similar) at line {best_start + 1}:\n{diff}"
        return f"Error: old_text not found in {path}. No similar text found. Verify the file content."


class ListDirTool(_FsTool):
    _DEFAULT_MAX = 200
    _IGNORE_DIRS = {
        ".git", "node_modules", "__pycache__", ".venv", "venv",
        "dist", "build", ".tox", ".mypy_cache", ".pytest_cache",
        ".ruff_cache", ".coverage", "htmlcov",
    }

    @property
    def name(self) -> str:
        return "list_dir"

    @property
    def description(self) -> str:
        return "List the contents of a directory. Set recursive=true to explore nested structure. Common noise directories are auto-ignored."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The directory path to list"},
                "recursive": {"type": "boolean", "description": "Recursively list all files (default false)"},
                "max_entries": {"type": "integer", "description": "Maximum entries to return (default 200)", "minimum": 1},
            },
            "required": ["path"],
        }

    async def execute(self, path: str | None = None, recursive: bool = False, max_entries: int | None = None, **kwargs: Any) -> str:
        try:
            if path is None:
                raise ValueError("Unknown path")
            tool_context = kwargs.get("_wbot_tool_context") if isinstance(kwargs.get("_wbot_tool_context"), dict) else {}
            dp = self._resolve(
                path,
                tool_name=self.name,
                tool_context=tool_context,
                justification="列出工作区外目录",
            )
            if isinstance(dp, str):
                return dp
            if not dp.exists():
                return f"Error: Directory not found: {path}"
            if not dp.is_dir():
                return f"Error: Not a directory: {path}"

            cap = max_entries or self._DEFAULT_MAX
            items: list[str] = []
            total = 0

            if recursive:
                for item in sorted(dp.rglob("*")):
                    if any(part in self._IGNORE_DIRS for part in item.parts):
                        continue
                    total += 1
                    if len(items) < cap:
                        rel = item.relative_to(dp)
                        items.append(f"{rel}/" if item.is_dir() else str(rel))
            else:
                for item in sorted(dp.iterdir()):
                    if item.name in self._IGNORE_DIRS:
                        continue
                    total += 1
                    if len(items) < cap:
                        prefix = "[DIR] " if item.is_dir() else "[FILE] "
                        items.append(f"{prefix}{item.name}")

            if not items and total == 0:
                return f"Directory {path} is empty"

            result = "\n".join(items)
            if total > cap:
                result += f"\n\n(truncated, showing first {cap} of {total} entries)"
            return result
        except PermissionError as exc:
            return f"Error: {exc}"
        except Exception as exc:
            return f"Error listing directory: {exc}"
