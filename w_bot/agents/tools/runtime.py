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
    enable_cron_service: bool,
    mcp_servers: list[dict[str, Any]] | None,
    extra_readonly_dirs: list[str] | None = None,
) -> list[Any]:
    """构建并返回目标对象。
    
    Args:
        memory_store: 长期记忆存储实例，用于检索与保存记忆。
        user_id: 业务对象唯一标识。
        tavily_api_key: Tavily 搜索服务 API Key。
        enable_cron_service: 是否启用定时服务工具。
        mcp_servers: MCP 服务配置列表。
        extra_readonly_dirs: 额外只读目录列表。
    """
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
        """处理read/file相关逻辑并返回结果。
        
        Args:
            path: 文件路径。
            start_line: 读取片段的起始行号。
            end_line: 读取片段的结束行号。
        """

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
    def modify_file(
        path: str,
        mode: str,
        content: str = "",
        find_text: str = "",
        replace_text: str = "",
        replace_all: bool = False,
        overwrite: bool = True,
    ) -> str:
        """创建、覆盖或按查找替换方式修改工作区文件。
        
        Args:
            path: 文件路径。
            mode: 修改模式，仅支持 create、replace、patch。
            content: create/replace 模式下写入的完整内容。
            find_text: patch 模式下要查找的文本。
            replace_text: patch 模式下替换成的文本。
            replace_all: patch 模式下是否替换全部命中。
            overwrite: create 模式下若文件已存在是否覆盖。
        """

        normalized_mode = (mode or "").strip().lower()
        if normalized_mode not in {"create", "replace", "patch"}:
            return "Invalid mode. Supported values: create, replace, patch"

        try:
            target = _resolve_workspace_path(path, workspace_root=workspace_root)
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
        """处理list/dir相关逻辑并返回结果。
        
        Args:
            path: 文件路径。
            recursive: 是否递归扫描子目录。
        """

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
        """处理web/search相关逻辑并返回结果。
        
        Args:
            query: 检索查询文本。
            max_results: 数值限制参数，用于控制处理规模。
        """

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
        """处理web/fetch相关逻辑并返回结果。
        
        Args:
            url: HTTP 请求地址。
            max_chars: 返回文本片段允许的最大字符数。
        """

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
        """处理message相关逻辑并返回结果。
        
        Args:
            recipient: 消息接收方标识。
            content: 消息内容主体。
        """

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
        """处理spawn相关逻辑并返回结果。
        
        Args:
            task: 任务描述文本。
            context: 文本参数，作为本次处理的输入内容。
        """

        job = {
            "id": uuid.uuid4().hex,
            "task": task,
            "context": context,
            "status": "pending",
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        _append_jsonl(workspace_root / ".w_bot_spawn_jobs.jsonl", job)
        return f"Spawned task: id={job['id']}"

    tools.extend(
        [
            read_file,
            modify_file,
            list_dir,
            web_search,
            web_fetch,
            message,
            spawn,
        ]
    )

    tools.append(_build_exec_tool(workspace_root=workspace_root, sandbox_root=sandbox_root))

    if enable_cron_service:
        tools.append(_build_cron_tool(workspace_root=workspace_root))

    tools.extend(_build_mcp_tools(mcp_servers or []))

    logger.info("Registered tools: %s", [getattr(t, "name", str(t)) for t in tools])
    return tools


def _build_exec_tool(*, workspace_root: Path, sandbox_root: Path) -> StructuredTool:
    """构建并返回目标对象。
    
    Args:
        workspace_root: 工作区根目录路径。
    """
    def _exec(command: str, timeout_sec: int = 30) -> str:
        """处理exec相关逻辑并返回结果。
        
        Args:
            command: 待执行的命令字符串。
            timeout_sec: 请求超时时间（秒）。
        """

        normalized = (command or "").strip()
        if not normalized:
            return "Command is empty"
        if _looks_like_python_source(normalized):
            logger.info("Executing inline Python via exec tool, code_len=%s", len(normalized))
            return _run_python_in_local_sandbox(code=normalized, sandbox_root=sandbox_root)

        try:
            completed = subprocess.run(
                normalized,
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
        description=(
            "Execute a shell command in the local workspace and return its output. "
            "If the input is raw Python source code instead of a shell command, "
            "run it in the local Python sandbox."
        ),
    )


def _run_python_in_local_sandbox(*, code: str, sandbox_root: Path, timeout_sec: int = 15) -> str:
    """处理run/python/in/local/sandbox相关逻辑并返回结果。
    
    Args:
        code: 待执行的 Python 代码。
        sandbox_root: 沙箱根目录。
        timeout_sec: 请求超时时间（秒）。
    """
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


def _looks_like_python_source(command: str) -> bool:
    snippet = (command or "").lstrip()
    if not snippet:
        return False
    if snippet.startswith(("python ", "python3 ", "uv run ", "bash ", "sh ", "zsh ")):
        return False
    indicators = (
        "\n",
        "import ",
        "from ",
        "print(",
        "def ",
        "class ",
        "for ",
        "while ",
        "if ",
        " = ",
    )
    return any(token in snippet for token in indicators)


def _sandbox_env(sandbox_root: Path) -> dict[str, str]:
    """处理sandbox/env相关逻辑并返回结果。
    
    Args:
        sandbox_root: 沙箱根目录。
    """
    allowed = ["PATH", "LANG", "LC_ALL", "TZ"]
    env = {k: v for k, v in os.environ.items() if k in allowed}
    env["HOME"] = str(sandbox_root)
    env["PYTHONNOUSERSITE"] = "1"
    return env


def _sandbox_preexec_fn() -> Any | None:
    # On Unix, add soft limits for CPU time and address space.
    """处理sandbox/preexec/fn相关逻辑并返回结果。
    """
    try:
        import resource
    except ImportError:
        return None

    def _set_limits() -> None:
        """处理set/limits相关逻辑并返回结果。
        """
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
    """构建并返回目标对象。
    
    Args:
        workspace_root: 工作区根目录路径。
    """
    def _cron(
        action: str,
        task_name: str,
        schedule: str = "",
        payload: str = "",
    ) -> str:
        """处理cron相关逻辑并返回结果。
        
        Args:
            action: 定时任务执行动作。
            task_name: 名称参数，用于标识目标对象。
            schedule: 任务调度配置。
            payload: 输入载荷字典，包含请求字段与元数据。
        """

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
    """构建并返回目标对象。
    
    Args:
        mcp_servers: MCP 服务配置列表。
    """
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
    """处理make/mcp/tool相关逻辑并返回结果。
    
    Args:
        full_name: 名称参数，用于标识目标对象。
        description: 描述文本，用于记录或展示。
        base_url: 地址参数，用于请求远端资源。
        invoke_path_template: 目标路径参数，用于定位文件或目录。
        remote_tool_name: 名称参数，用于标识目标对象。
        headers: HTTP 请求头字典。
    """
    def _call(arguments_json: str = "{}") -> str:
        """处理call相关逻辑并返回结果。
        
        Args:
            arguments_json: 工具参数 JSON 字符串。
        """

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
    """处理discover/mcp/tools相关逻辑并返回结果。
    
    Args:
        base_url: 地址参数，用于请求远端资源。
        discovery_path: 目标路径参数，用于定位文件或目录。
        headers: HTTP 请求头字典。
    """
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
    """处理http/get/json相关逻辑并返回结果。
    
    Args:
        url: HTTP 请求地址。
        headers: HTTP 请求头字典。
        timeout: 请求超时时间（秒）。
    """
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
    """处理http/post/json相关逻辑并返回结果。
    
    Args:
        url: HTTP 请求地址。
        payload: 输入载荷字典，包含请求字段与元数据。
        headers: HTTP 请求头字典。
        timeout: 请求超时时间（秒）。
    """
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
    """处理resolve/workspace/path相关逻辑并返回结果。
    
    Args:
        path: 文件路径。
        workspace_root: 工作区根目录路径。
    """
    candidate = Path(path)
    resolved = candidate.resolve() if candidate.is_absolute() else (workspace_root / candidate).resolve()
    if not _is_relative_to(resolved, workspace_root):
        raise ValueError(f"Path escapes workspace: {path}")
    return resolved


def _resolve_read_path(path: str, *, readonly_roots: list[Path]) -> Path:
    """处理resolve/read/path相关逻辑并返回结果。
    
    Args:
        path: 文件路径。
        readonly_roots: 只读目录白名单。
    """
    if not readonly_roots:
        raise ValueError("No readonly roots configured")
    candidate = Path(path)
    resolved = candidate.resolve() if candidate.is_absolute() else (readonly_roots[0] / candidate).resolve()
    for root in readonly_roots:
        if _is_relative_to(resolved, root):
            return resolved
    raise ValueError(f"Path escapes readonly roots: {path}")


def _is_relative_to(path: Path, base: Path) -> bool:
    """判断条件是否满足。
    
    Args:
        path: 文件路径。
        base: 基准路径或基准配置对象。
    """
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def _sanitize_tool_token(value: str) -> str:
    """处理sanitize/tool/token相关逻辑并返回结果。
    
    Args:
        value: 待转换或校验的值。
    """
    cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_").lower()
    return cleaned or "tool"


def _strip_html(raw: str) -> str:
    """处理strip/html相关逻辑并返回结果。
    
    Args:
        raw: 原始输入内容。
    """
    without_script = re.sub(r"<script[\s\S]*?</script>", " ", raw, flags=re.IGNORECASE)
    without_style = re.sub(r"<style[\s\S]*?</style>", " ", without_script, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", without_style)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    """处理append/jsonl相关逻辑并返回结果。
    
    Args:
        path: 文件路径。
        payload: 输入载荷字典，包含请求字段与元数据。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False))
        f.write("\n")


def _read_json_file(path: Path, *, default: Any) -> Any:
    """处理read/json/file相关逻辑并返回结果。
    
    Args:
        path: 文件路径。
        default: 缺失配置时使用的默认值。
    """
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default
