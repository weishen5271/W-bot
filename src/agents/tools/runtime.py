from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib import error as url_error
from urllib import request as url_request

from langchain_core.tools import StructuredTool, tool

from ..logging_config import get_logger
from ..memory import LongTermMemoryStore

logger = get_logger(__name__)


def build_tools(
    *,
    memory_store: LongTermMemoryStore,
    user_id: str,
    tavily_api_key: str,
    enable_exec_tool: bool,
    enable_cron_service: bool,
    mcp_servers: list[dict[str, Any]] | None,
    extra_readonly_dirs: list[str] | None = None,
) -> list[Any]:
    logger.info("Building tools for user_id=%s", user_id)
    workspace_root = Path.cwd().resolve()
    sandbox_root = workspace_root / ".sandbox"
    readonly_roots = [workspace_root]
    for candidate in extra_readonly_dirs or []:
        try:
            readonly_roots.append(Path(candidate).resolve())
        except OSError:
            logger.warning("Skip invalid readonly root: %s", candidate)
    tools: list[Any] = []

    @tool
    def read_file(path: str, start_line: int = 1, end_line: int = 300) -> str:
        """Read a text file from workspace. Supports line window via start_line/end_line."""

        try:
            target = _resolve_read_path(path, readonly_roots=readonly_roots)
        except ValueError as exc:
            return str(exc)
        if not target.exists():
            return f"File not found: {target}"
        if not target.is_file():
            return f"Path is not a file: {target}"

        lines = target.read_text(encoding="utf-8").splitlines()
        start = max(1, start_line)
        end = min(len(lines), max(start, end_line))
        snippet = lines[start - 1 : end]
        return "\n".join(snippet)

    @tool
    def write_file(path: str, content: str, overwrite: bool = True) -> str:
        """Write text file in workspace. Creates parent directories automatically."""

        try:
            target = _resolve_workspace_path(path, workspace_root=workspace_root)
        except ValueError as exc:
            return str(exc)
        if target.exists() and not overwrite:
            return f"File already exists and overwrite=false: {target}"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} chars to {target}"

    @tool
    def edit_file(path: str, find_text: str, replace_text: str, replace_all: bool = False) -> str:
        """Edit file by replacing text."""

        try:
            target = _resolve_workspace_path(path, workspace_root=workspace_root)
        except ValueError as exc:
            return str(exc)
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
        """List files/directories under workspace path."""

        try:
            target = _resolve_workspace_path(path, workspace_root=workspace_root)
        except ValueError as exc:
            return str(exc)
        if not target.exists():
            return f"Path not found: {target}"
        if target.is_file():
            stat = target.stat()
            return f"FILE\t{target}\t{stat.st_size}"

        output: list[str] = []
        if recursive:
            iterator = target.rglob("*")
        else:
            iterator = target.glob("*")
        for item in sorted(iterator):
            kind = "DIR" if item.is_dir() else "FILE"
            size = item.stat().st_size if item.is_file() else 0
            output.append(f"{kind}\t{item}\t{size}")
        return "\n".join(output[:2000]) or "(empty)"

    @tool
    def web_search(query: str, max_results: int = 5) -> str:
        """Search the web via Tavily and return brief results."""

        if not tavily_api_key:
            return "TAVILY_API_KEY is not configured."

        payload = _http_post_json(
            url="https://api.tavily.com/search",
            payload={
                "api_key": tavily_api_key,
                "query": query,
                "max_results": max(1, max_results),
            },
            timeout=20,
        )
        if isinstance(payload, str):
            return payload

        items = payload.get("results")
        if not isinstance(items, list) or not items:
            return "No results"

        lines: list[str] = []
        for item in items[: max(1, max_results)]:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "(no title)")
            url = str(item.get("url") or "")
            content = str(item.get("content") or "").strip()
            snippet = content[:220] + ("..." if len(content) > 220 else "")
            lines.append(f"- {title}\n  {url}\n  {snippet}")
        return "\n".join(lines) if lines else "No results"

    @tool
    def web_fetch(url: str, max_chars: int = 8000) -> str:
        """Fetch a URL and return a cleaned text excerpt."""

        req = url_request.Request(url, headers={"User-Agent": "W-bot/1.0"})
        try:
            with url_request.urlopen(req, timeout=20) as response:
                raw = response.read().decode("utf-8", errors="ignore")
        except Exception as exc:
            return f"Fetch failed: {type(exc).__name__}: {exc}"
        text = _strip_html(raw)
        return text[: max(200, max_chars)]

    @tool
    def message(recipient: str, content: str) -> str:
        """Send an internal message to a named recipient queue."""

        msg = {
            "id": uuid.uuid4().hex,
            "recipient": recipient,
            "content": content,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        _append_jsonl(workspace_root / ".w_bot_messages.jsonl", msg)
        return f"Message queued: id={msg['id']} recipient={recipient}"

    @tool
    def spawn(task: str, context: str = "") -> str:
        """Create a child task record for later processing."""

        job = {
            "id": uuid.uuid4().hex,
            "task": task,
            "context": context,
            "status": "pending",
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        _append_jsonl(workspace_root / ".w_bot_spawn_jobs.jsonl", job)
        return f"Spawned task: id={job['id']}"

    @tool
    def execute_python(code: str) -> str:
        """Run Python code in a local lightweight sandbox under workspace/.sandbox."""

        logger.info("Executing Python code in local sandbox, code_len=%s", len(code))
        return _run_python_in_local_sandbox(code=code, sandbox_root=sandbox_root)

    @tool
    def save_memory(text: str, memory_type: str = "experience") -> str:
        """Persist long-term memory."""

        doc_id = memory_store.save(user_id=user_id, text=text, memory_type=memory_type)
        if not doc_id:
            return "Memory backend unavailable"
        return f"Memory saved, id={doc_id}"

    tools.extend(
        [
            read_file,
            write_file,
            edit_file,
            list_dir,
            web_search,
            web_fetch,
            message,
            spawn,
            execute_python,
            save_memory,
        ]
    )

    if enable_exec_tool:
        tools.append(_build_exec_tool(workspace_root=workspace_root))

    if enable_cron_service:
        tools.append(_build_cron_tool(workspace_root=workspace_root))

    tools.extend(_build_mcp_tools(mcp_servers or []))

    logger.info("Registered tools: %s", [getattr(t, "name", str(t)) for t in tools])
    return tools


def _build_exec_tool(*, workspace_root: Path) -> StructuredTool:
    def _exec(command: str, timeout_sec: int = 30) -> str:
        """Execute shell command in workspace."""

        try:
            completed = subprocess.run(
                command,
                shell=True,
                cwd=str(workspace_root),
                capture_output=True,
                text=True,
                timeout=max(1, timeout_sec),
            )
        except subprocess.TimeoutExpired:
            return "Command timed out"
        except Exception as exc:
            return f"Command failed: {type(exc).__name__}: {exc}"

        chunks: list[str] = [
            f"exit_code={completed.returncode}",
            f"stdout:\n{completed.stdout.strip()}",
            f"stderr:\n{completed.stderr.strip()}",
        ]
        return "\n\n".join(chunks).strip()

    return StructuredTool.from_function(
        func=_exec,
        name="exec",
        description="Execute a shell command in local workspace.",
    )


def _run_python_in_local_sandbox(*, code: str, sandbox_root: Path, timeout_sec: int = 15) -> str:
    sandbox_root.mkdir(parents=True, exist_ok=True)

    script_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".py",
            prefix="snippet_",
            dir=str(sandbox_root),
            delete=False,
        ) as script_file:
            script_file.write(code)
            script_path = Path(script_file.name)

        completed = subprocess.run(
            [sys.executable, "-I", str(script_path)],
            cwd=str(sandbox_root),
            capture_output=True,
            text=True,
            timeout=max(1, timeout_sec),
            env=_sandbox_env(sandbox_root),
            preexec_fn=_sandbox_preexec_fn(),
        )
    except subprocess.TimeoutExpired:
        return f"Execution timed out after {timeout_sec}s"
    except Exception as exc:
        logger.exception("Local sandbox execution failed")
        return f"Execution failed: {type(exc).__name__}: {exc}"
    finally:
        if script_path and script_path.exists():
            try:
                script_path.unlink()
            except OSError:
                logger.warning("Failed to remove sandbox script: %s", script_path)

    chunks: list[str] = [f"exit_code={completed.returncode}"]
    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    if stdout:
        chunks.append(f"stdout:\n{stdout}")
    if stderr:
        chunks.append(f"stderr:\n{stderr}")
    if len(chunks) == 1:
        chunks.append("No output")
    return "\n\n".join(chunks)


def _sandbox_env(sandbox_root: Path) -> dict[str, str]:
    allowed = ["PATH", "LANG", "LC_ALL", "TZ"]
    env = {k: v for k, v in os.environ.items() if k in allowed}
    env["HOME"] = str(sandbox_root)
    env["PYTHONNOUSERSITE"] = "1"
    return env


def _sandbox_preexec_fn() -> Any | None:
    # On Unix, add soft limits for CPU time and address space.
    try:
        import resource
    except ImportError:
        return None

    def _set_limits() -> None:
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
        except Exception:
            pass
        try:
            # 512 MB virtual memory cap for lightweight isolation.
            resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
        except Exception:
            pass

    return _set_limits


def _build_cron_tool(*, workspace_root: Path) -> StructuredTool:
    def _cron(
        action: str,
        task_name: str,
        schedule: str = "",
        payload: str = "",
    ) -> str:
        """Manage simple cron task records when cron service is enabled."""

        jobs_file = workspace_root / ".w_bot_cron_jobs.json"
        jobs = _read_json_file(jobs_file, default=[])
        if not isinstance(jobs, list):
            jobs = []

        if action == "list":
            return json.dumps(jobs, ensure_ascii=False, indent=2)

        if action == "create":
            if not schedule:
                return "schedule is required for create"
            job = {
                "id": uuid.uuid4().hex,
                "task_name": task_name,
                "schedule": schedule,
                "payload": payload,
                "enabled": True,
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }
            jobs.append(job)
            jobs_file.write_text(json.dumps(jobs, ensure_ascii=False, indent=2), encoding="utf-8")
            return f"Cron job created: id={job['id']}"

        if action == "delete":
            new_jobs = [j for j in jobs if j.get("task_name") != task_name]
            jobs_file.write_text(json.dumps(new_jobs, ensure_ascii=False, indent=2), encoding="utf-8")
            return f"Cron jobs removed: {len(jobs) - len(new_jobs)}"

        return "Unsupported action. Use list/create/delete."

    return StructuredTool.from_function(
        func=_cron,
        name="cron",
        description="Create/list/delete cron jobs when cron service is enabled.",
    )


def _build_mcp_tools(mcp_servers: list[dict[str, Any]]) -> list[StructuredTool]:
    tools: list[StructuredTool] = []
    for server in mcp_servers:
        if not isinstance(server, dict):
            continue
        if server.get("enabled", True) is False:
            continue

        server_name = _sanitize_tool_token(str(server.get("name") or "server"))
        base_url = str(server.get("base_url") or "").strip().rstrip("/")
        if not base_url:
            continue

        discovery_path = str(server.get("discovery_path") or "/tools")
        invoke_path_template = str(server.get("invoke_path_template") or "/tools/{tool}")
        headers = server.get("headers") if isinstance(server.get("headers"), dict) else {}

        discovered = _discover_mcp_tools(base_url=base_url, discovery_path=discovery_path, headers=headers)
        for item in discovered:
            tool_name = _sanitize_tool_token(item.get("name", "tool"))
            tool_desc = item.get("description") or f"MCP tool {tool_name} from {server_name}"
            full_name = f"mcp_{server_name}_{tool_name}"
            tools.append(
                _make_mcp_tool(
                    full_name=full_name,
                    description=tool_desc,
                    base_url=base_url,
                    invoke_path_template=invoke_path_template,
                    remote_tool_name=item.get("name", tool_name),
                    headers=headers,
                )
            )
    return tools


def _make_mcp_tool(
    *,
    full_name: str,
    description: str,
    base_url: str,
    invoke_path_template: str,
    remote_tool_name: str,
    headers: dict[str, Any],
) -> StructuredTool:
    def _call(arguments_json: str = "{}") -> str:
        """Call MCP tool with JSON arguments."""

        try:
            arguments = json.loads(arguments_json or "{}")
        except json.JSONDecodeError:
            return "arguments_json must be valid JSON"

        path = invoke_path_template.replace("{tool}", remote_tool_name)
        url = f"{base_url}{path}"
        payload = {"arguments": arguments}
        raw = _http_post_json(url=url, payload=payload, headers=headers, timeout=20)
        if isinstance(raw, str):
            return raw
        return json.dumps(raw, ensure_ascii=False)

    return StructuredTool.from_function(func=_call, name=full_name, description=description)


def _discover_mcp_tools(*, base_url: str, discovery_path: str, headers: dict[str, Any]) -> list[dict[str, str]]:
    url = f"{base_url}{discovery_path}"
    raw = _http_get_json(url=url, headers=headers, timeout=10)
    if isinstance(raw, str):
        logger.warning("Failed to discover MCP tools from %s: %s", base_url, raw)
        return []

    tools = raw.get("tools") if isinstance(raw, dict) else raw
    if not isinstance(tools, list):
        return []

    result: list[dict[str, str]] = []
    for item in tools:
        if isinstance(item, str):
            result.append({"name": item, "description": ""})
            continue
        if isinstance(item, dict) and item.get("name"):
            result.append(
                {
                    "name": str(item["name"]),
                    "description": str(item.get("description") or ""),
                }
            )
    return result


def _http_get_json(
    *,
    url: str,
    headers: dict[str, Any] | None = None,
    timeout: int = 10,
) -> dict[str, Any] | str:
    req_headers = {"User-Agent": "W-bot/1.0"}
    if headers:
        req_headers.update({str(k): str(v) for k, v in headers.items()})
    req = url_request.Request(url, headers=req_headers)
    try:
        with url_request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8", errors="ignore"))
    except (url_error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
        return f"HTTP GET failed: {type(exc).__name__}: {exc}"


def _http_post_json(
    *,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, Any] | None = None,
    timeout: int = 10,
) -> dict[str, Any] | str:
    body = json.dumps(payload).encode("utf-8")
    req_headers = {"Content-Type": "application/json", "User-Agent": "W-bot/1.0"}
    if headers:
        req_headers.update({str(k): str(v) for k, v in headers.items()})
    req = url_request.Request(url, data=body, headers=req_headers, method="POST")
    try:
        with url_request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8", errors="ignore"))
    except (url_error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
        return f"HTTP POST failed: {type(exc).__name__}: {exc}"


def _resolve_workspace_path(path: str, *, workspace_root: Path) -> Path:
    candidate = Path(path)
    resolved = candidate.resolve() if candidate.is_absolute() else (workspace_root / candidate).resolve()
    if not _is_relative_to(resolved, workspace_root):
        raise ValueError(f"Path escapes workspace: {path}")
    return resolved


def _resolve_read_path(path: str, *, readonly_roots: list[Path]) -> Path:
    if not readonly_roots:
        raise ValueError("No readonly roots configured")
    candidate = Path(path)
    resolved = candidate.resolve() if candidate.is_absolute() else (readonly_roots[0] / candidate).resolve()
    for root in readonly_roots:
        if _is_relative_to(resolved, root):
            return resolved
    raise ValueError(f"Path escapes readonly roots: {path}")


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def _sanitize_tool_token(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_").lower()
    return cleaned or "tool"


def _strip_html(raw: str) -> str:
    without_script = re.sub(r"<script[\s\S]*?</script>", " ", raw, flags=re.IGNORECASE)
    without_style = re.sub(r"<style[\s\S]*?</style>", " ", without_script, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", without_style)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False))
        f.write("\n")


def _read_json_file(path: Path, *, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default
