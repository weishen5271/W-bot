"""Shell execution tool."""

import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from w_bot.agents.core.escalation import EscalationManager
from w_bot.agents.core.logging_config import get_logger
from w_bot.agents.tools.base import Tool
from w_bot.security.network import contains_internal_url

logger = get_logger(__name__)


def _emit_tool_status(tool_context: dict[str, Any] | None, text: str) -> None:
    if not isinstance(tool_context, dict):
        return
    callback = tool_context.get("status_callback")
    if callable(callback):
        try:
            callback(str(text))
        except Exception:
            logger.debug("Tool status callback failed", exc_info=True)


class ExecTool(Tool):
    """Tool to execute shell commands."""

    def __init__(
        self,
        timeout: int = 60,
        working_dir: str | None = None,
        deny_patterns: list[str] | None = None,
        allow_patterns: list[str] | None = None,
        restrict_to_workspace: bool = False,
        path_append: str = "",
        escalation_manager: EscalationManager | None = None,
    ):
        self.timeout = timeout
        self.working_dir = working_dir
        self.deny_patterns = deny_patterns or [
            r"\brm\s+-[rf]{1,2}\b",
            r"\bdel\s+/[fq]\b",
            r"\brmdir\s+/s\b",
            r"(?:^|[;&|]\s*)format\b",
            r"\b(mkfs|diskpart)\b",
            r"\bdd\s+if=",
            r">\s*/dev/sd",
            r"\b(shutdown|reboot|poweroff)\b",
            r":\(\)\s*\{.*\};\s*:",
        ]
        self.allow_patterns = allow_patterns or []
        self.restrict_to_workspace = restrict_to_workspace
        self.path_append = path_append
        self.escalation_manager = escalation_manager

    @property
    def name(self) -> str:
        return "exec"

    _MAX_TIMEOUT = 600
    _MAX_OUTPUT = 10_000

    @property
    def description(self) -> str:
        return (
            "Execute a shell command and return its output. "
            "When the command needs workspace-external access, this tool can create an escalation request for user approval."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The shell command to execute"},
                "working_dir": {"type": "string", "description": "Optional working directory for the command"},
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds. Increase for long-running commands like compilation or installation (default 60, max 600).",
                    "minimum": 1,
                    "maximum": 600,
                },
                "justification": {
                    "type": "string",
                    "description": "Short reason shown to the user if this command needs an escalation approval.",
                },
                "prefix_rule": {
                    "type": "array",
                    "description": "Optional reusable command prefix for future approvals, such as ['git', 'pull'].",
                    "items": {"type": "string"},
                },
            },
            "required": ["command"],
        }

    async def execute(
        self,
        command: str,
        working_dir: str | None = None,
        timeout: int | None = None,
        justification: str | None = None,
        prefix_rule: list[str] | None = None,
        **kwargs: Any,
    ) -> str:
        cwd = working_dir or self.working_dir or os.getcwd()
        tool_context = kwargs.get("_wbot_tool_context") if isinstance(kwargs.get("_wbot_tool_context"), dict) else {}
        session_id = str(tool_context.get("thread_id") or "-").strip() or "-"
        display_command = " ".join(command.strip().split())
        if len(display_command) > 120:
            display_command = display_command[:117] + "..."
        guard_error = self._guard_command(
            command,
            cwd,
            session_id=session_id,
            justification=str(justification or "").strip(),
            prefix_rule=prefix_rule,
        )
        if guard_error:
            return guard_error

        effective_timeout = min(timeout or self.timeout, self._MAX_TIMEOUT)
        env = os.environ.copy()
        if self.path_append:
            env["PATH"] = env.get("PATH", "") + os.pathsep + self.path_append

        try:
            _emit_tool_status(tool_context, f"正在执行命令：{display_command}")
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env,
            )

            try:
                started_at = asyncio.get_running_loop().time()
                heartbeat_interval = 2.0
                while True:
                    try:
                        await asyncio.wait_for(process.wait(), timeout=heartbeat_interval)
                        break
                    except asyncio.TimeoutError:
                        elapsed = int(asyncio.get_running_loop().time() - started_at)
                        _emit_tool_status(tool_context, f"命令仍在执行：{display_command}（已等待 {elapsed}s）")
                        if elapsed >= effective_timeout:
                            raise
                stdout, stderr = await process.communicate()
            except asyncio.TimeoutError:
                process.kill()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    pass
                finally:
                    if sys.platform != "win32":
                        try:
                            os.waitpid(process.pid, os.WNOHANG)
                        except (ProcessLookupError, ChildProcessError) as exc:
                            logger.debug("Process already reaped or not found: %s", exc)
                _emit_tool_status(tool_context, f"命令执行超时：{display_command}")
                return f"Error: Command timed out after {effective_timeout} seconds"

            output_parts = []
            if stdout:
                output_parts.append(stdout.decode("utf-8", errors="replace"))
            if stderr:
                stderr_text = stderr.decode("utf-8", errors="replace")
                if stderr_text.strip():
                    output_parts.append(f"STDERR:\n{stderr_text}")
            output_parts.append(f"\nExit code: {process.returncode}")

            result = "\n".join(output_parts) if output_parts else "(no output)"
            if len(result) > self._MAX_OUTPUT:
                half = self._MAX_OUTPUT // 2
                result = result[:half] + f"\n\n... ({len(result) - self._MAX_OUTPUT:,} chars truncated) ...\n\n" + result[-half:]
            _emit_tool_status(tool_context, f"命令执行完成：{display_command}（exit={process.returncode}）")
            return result
        except Exception as exc:
            _emit_tool_status(tool_context, f"命令执行失败：{display_command}")
            return f"Error executing command: {exc}"

    def _guard_command(
        self,
        command: str,
        cwd: str,
        *,
        session_id: str,
        justification: str,
        prefix_rule: list[str] | None,
    ) -> str | None:
        cmd = command.strip()
        lower = cmd.lower()

        for pattern in self.deny_patterns:
            if re.search(pattern, lower):
                return "Error: Command blocked by safety guard (dangerous pattern detected)"

        if self.allow_patterns and not any(re.search(pattern, lower) for pattern in self.allow_patterns):
            return "Error: Command blocked by safety guard (not in allowlist)"

        if contains_internal_url(cmd):
            return "Error: Command blocked by safety guard (internal/private URL detected)"

        if self.restrict_to_workspace:
            if "..\\" in cmd or "../" in cmd:
                return "Error: Command blocked by safety guard (path traversal detected)"

            cwd_path = Path(cwd).resolve()
            for raw in self._extract_absolute_paths(cmd):
                try:
                    expanded = os.path.expandvars(raw.strip())
                    resolved = Path(expanded).expanduser().resolve()
                except Exception:
                    continue
                if resolved.is_absolute() and cwd_path not in resolved.parents and resolved != cwd_path:
                    if self.escalation_manager and self.escalation_manager.is_command_approved(
                        session_id=session_id,
                        command=cmd,
                    ):
                        return None
                    return self._build_escalation_response(
                        session_id=session_id,
                        command=cmd,
                        cwd=str(cwd_path),
                        justification=justification or "访问工作区外路径",
                        prefix_rule=prefix_rule,
                        risk_type="workspace_path",
                    )

        return None

    def _build_escalation_response(
        self,
        *,
        session_id: str,
        command: str,
        cwd: str,
        justification: str,
        prefix_rule: list[str] | None,
        risk_type: str,
    ) -> str:
        if self.escalation_manager is None:
            return "Error: Command blocked by safety guard (path outside working dir)"
        request = self.escalation_manager.create_request(
            session_id=session_id,
            command=command,
            working_dir=cwd,
            justification=justification,
            prefix_rule=prefix_rule,
            risk_type=risk_type,
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
                f"命令需要提权审批。请求ID={request.id}。"
                "请让用户执行 /escalation 查看详情，并使用 /approve <请求ID> 或 /deny <请求ID> [原因]。"
            ),
        }
        return json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def _extract_absolute_paths(command: str) -> list[str]:
        win_paths = re.findall(r"[A-Za-z]:\\[^\s\"'|><;]+", command)
        posix_paths = re.findall(r"(?:^|[\s|>'\"])(/[^\s\"'>;|<]+)", command)
        home_paths = re.findall(r"(?:^|[\s|>'\"])(~[^\s\"'>;|<]*)", command)
        return win_paths + posix_paths + home_paths
