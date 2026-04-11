from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage

from w_bot.utils.helpers import _tool_result_to_text

from ..core.logging_config import get_logger
from ..tools.base import Tool
from ..tools.common import append_jsonl
from .skills import SkillsLoader, SkillSpec
from .subagent_builtins import BUILTIN_SUBAGENTS, BuiltinSubagentDefinition

logger = get_logger(__name__)


def _tool_args_preview(tool_name: str, args: dict[str, Any]) -> str:
    candidates = [
        args.get("url"),
        args.get("query"),
        args.get("path"),
        args.get("command"),
        args.get("task"),
        args.get("id"),
        args.get("working_dir"),
    ]
    for item in candidates:
        text = str(item or "").strip()
        if text:
            compact = " ".join(text.split())
            return compact[:96] + ("..." if len(compact) > 96 else "")
    if not args:
        return tool_name
    try:
        raw = json.dumps(args, ensure_ascii=False, sort_keys=True)
    except Exception:
        raw = str(args)
    compact = " ".join(raw.split())
    return compact[:96] + ("..." if len(compact) > 96 else "")


def _tool_progress_emoji(tool_name: str) -> str:
    normalized = (tool_name or "").strip().lower()
    if any(token in normalized for token in ["browser", "navigate", "web"]):
        return "🌐"
    if any(token in normalized for token in ["search", "grep", "find"]):
        return "🔎"
    if any(token in normalized for token in ["read", "fetch", "load"]):
        return "📖"
    if any(token in normalized for token in ["write", "edit", "patch"]):
        return "✍"
    if any(token in normalized for token in ["exec", "shell", "command"]):
        return "⚙"
    if any(token in normalized for token in ["spawn", "subagent", "wait"]):
        return "🧩"
    return "⚡"


def _tool_progress_action(tool_name: str) -> str:
    normalized = (tool_name or "").strip().lower()
    for token, label in [
        ("navigate", "navigate"),
        ("search", "search"),
        ("fetch", "fetch"),
        ("read", "read"),
        ("write", "write"),
        ("edit", "edit"),
        ("exec", "exec"),
        ("shell", "exec"),
        ("spawn", "spawn"),
        ("wait", "wait"),
    ]:
        if token in normalized:
            return label
    compact = normalized.replace("mcp_", "").replace("_tool", "")
    return compact[:18] if compact else "run"


def _tool_progress_line(
    *,
    event_type: str,
    tool_name: str,
    preview: str,
    elapsed_seconds: float | None,
    ok: bool | None,
) -> str:
    if event_type == "tool.started":
        return f"  ┊ ⚡ preparing {tool_name}..."
    label = " ".join((preview or tool_name).split())
    if len(label) > 88:
        label = label[:85] + "..."
    duration = f"  {elapsed_seconds:.1f}s" if elapsed_seconds is not None else ""
    suffix = " [error]" if ok is False else ""
    return f"  ┊ {_tool_progress_emoji(tool_name)} {_tool_progress_action(tool_name)}  {label}{duration}{suffix}"


def _emit_tool_progress(
    callback: Any | None,
    *,
    event_type: str,
    tool_name: str,
    preview: str = "",
    elapsed_seconds: float | None = None,
    ok: bool | None = None,
    function_args: dict[str, Any] | None = None,
) -> None:
    if not callable(callback):
        return
    try:
        callback(
            _tool_progress_line(
                event_type=event_type,
                tool_name=tool_name,
                preview=preview,
                elapsed_seconds=elapsed_seconds,
                ok=ok,
            )
        )
    except Exception:
        logger.debug("Subagent tool progress line callback failed", exc_info=True)


@dataclass(frozen=True)
class SubagentConfig:
    agent_type: str
    name: str
    description: str
    system_prompt: str = ""
    allowed_tools: list[str] | None = None
    disallowed_tools: list[str] = field(default_factory=list)
    max_turns: int = 12
    timeout_seconds: int = 300
    inherit_system_prompt: bool = True
    model: str | None = None


@dataclass
class SubagentResult:
    success: bool
    final_response: str
    messages: list[BaseMessage] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    duration_seconds: float = 0.0
    usage: dict[str, Any] | None = None


@dataclass
class SubagentJobRecord:
    id: str
    label: str
    task: str
    agent_type: str
    status: str
    created_at: str
    updated_at: str
    thread_id: str
    config: SubagentConfig
    context_messages: list[BaseMessage] = field(default_factory=list)
    status_callback: Any | None = None
    result: SubagentResult | None = None
    error: str | None = None

    def summary(self) -> dict[str, Any]:
        result = self.result
        return {
            "id": self.id,
            "label": self.label,
            "task": self.task,
            "agent_type": self.agent_type,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "thread_id": self.thread_id,
            "error": self.error or (result.error if result else None),
            "final_response": result.final_response if result else "",
            "duration_seconds": result.duration_seconds if result else 0.0,
            "tool_calls": result.tool_calls if result else [],
        }


class SubagentManager:
    def __init__(
        self,
        *,
        parent_graph: Any,
        tools: list[Tool],
        workspace_root: Path,
        skills_loader: SkillsLoader | None = None,
    ) -> None:
        self._parent = parent_graph
        self._tools = list(tools)
        self._workspace_root = workspace_root
        self._skills_loader = skills_loader
        self._lock = threading.RLock()
        self._jobs: dict[str, SubagentJobRecord] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._events: dict[str, threading.Event] = {}
        self._journal_path = self._workspace_root / ".w_bot_spawn_jobs.jsonl"

    def spawn(
        self,
        *,
        agent_type: str,
        task: str,
        label: str = "",
        context_messages: list[BaseMessage] | None = None,
        parent_thread_id: str = "-",
        status_callback: Any | None = None,
    ) -> dict[str, Any]:
        definition = self._resolve_definition(agent_type)
        job_id = uuid.uuid4().hex
        now = _now_iso()
        job = SubagentJobRecord(
            id=job_id,
            label=label.strip(),
            task=task,
            agent_type=definition.key,
            status="pending",
            created_at=now,
            updated_at=now,
            thread_id=parent_thread_id,
            config=self._build_config(definition),
            context_messages=list(context_messages or []),
            status_callback=status_callback if callable(status_callback) else None,
        )
        done = threading.Event()
        worker = threading.Thread(
            target=self._run_job,
            args=(job_id,),
            name=f"subagent-{job_id[:8]}",
            daemon=True,
        )
        with self._lock:
            self._jobs[job_id] = job
            self._events[job_id] = done
            self._threads[job_id] = worker
        self._append_event(job, event="created")
        worker.start()
        return job.summary()

    def list_jobs(self, *, status: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        normalized_status = status.strip().lower() if isinstance(status, str) and status.strip() else ""
        with self._lock:
            jobs = list(self._jobs.values())
        jobs.sort(key=lambda item: item.created_at, reverse=True)
        if normalized_status:
            jobs = [job for job in jobs if job.status.lower() == normalized_status]
        return [job.summary() for job in jobs[: max(1, limit)]]

    def wait_for(self, job_id: str, *, timeout_seconds: int = 60) -> dict[str, Any]:
        with self._lock:
            event = self._events.get(job_id)
            job = self._jobs.get(job_id)
        if job is None or event is None:
            return {"id": job_id, "status": "not_found", "error": f"Unknown subagent id: {job_id}"}
        event.wait(max(1, timeout_seconds))
        with self._lock:
            current = self._jobs.get(job_id) or job
        return current.summary()

    async def execute_skill(
        self,
        *,
        skill_name: str,
        task: str,
        arguments: dict[str, Any] | None = None,
        context_messages: list[BaseMessage] | None = None,
        thread_id: str = "-",
        status_callback: Any | None = None,
    ) -> SubagentResult:
        if self._skills_loader is None:
            return SubagentResult(success=False, final_response="", error="Skills are not enabled")
        skill = self._skills_loader.get_skill(skill_name)
        if skill is None:
            return SubagentResult(success=False, final_response="", error=f"Skill not found: {skill_name}")
        requirement = self._skills_loader.check_requirements(skill)
        if not requirement.available:
            parts: list[str] = []
            if requirement.missing_bins:
                parts.append("missing bins: " + ", ".join(requirement.missing_bins))
            if requirement.missing_env:
                parts.append("missing env: " + ", ".join(requirement.missing_env))
            return SubagentResult(
                success=False,
                final_response="",
                error=f"Skill requirements not satisfied for {skill_name} ({'; '.join(parts)})",
            )
        config = self._build_skill_config(skill)
        prompt = self._build_skill_prompt(skill=skill, task=task, arguments=arguments or {})
        job = SubagentJobRecord(
            id=uuid.uuid4().hex,
            label=skill_name,
            task=prompt,
            agent_type=f"skill:{skill_name}",
            status="running",
            created_at=_now_iso(),
            updated_at=_now_iso(),
            thread_id=thread_id,
            config=config,
            context_messages=list(context_messages or []),
            status_callback=status_callback if callable(status_callback) else None,
        )
        return await self._execute_job(job)

    def _run_job(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            event = self._events[job_id]
            job.status = "running"
            job.updated_at = _now_iso()
        self._append_event(job, event="started")
        try:
            result = asyncio.run(
                asyncio.wait_for(
                    self._execute_job(job),
                    timeout=max(1, int(job.config.timeout_seconds)),
                )
            )
            with self._lock:
                job.result = result
                job.status = "completed" if result.success else "failed"
                job.error = result.error
                job.updated_at = _now_iso()
        except asyncio.TimeoutError:
            with self._lock:
                job.status = "timeout"
                job.error = f"Subagent timed out after {job.config.timeout_seconds}s"
                job.updated_at = _now_iso()
        except Exception as exc:
            logger.exception("Subagent job failed: id=%s", job_id)
            with self._lock:
                job.status = "failed"
                job.error = f"{type(exc).__name__}: {exc}"
                job.updated_at = _now_iso()
        finally:
            self._append_event(job, event=job.status)
            event.set()

    async def _execute_job(self, job: SubagentJobRecord) -> SubagentResult:
        config = job.config
        tools = self._resolve_tools(config)
        tool_map = {tool.name: tool for tool in tools}
        llm = self._resolve_llm(config)
        system_prompt = self._build_system_prompt(config)
        messages: list[BaseMessage] = []
        if system_prompt.strip():
            messages.append(SystemMessage(content=system_prompt))
        messages.extend(_sanitize_context_messages(job.context_messages))
        messages.append(HumanMessage(content=job.task))

        start_time = time.monotonic()
        emitted: list[BaseMessage] = []
        all_tool_calls: list[dict[str, Any]] = []
        last_usage: dict[str, Any] | None = None

        for _turn in range(max(1, config.max_turns)):
            self._emit_status(job, f"Skill 子任务执行中：{job.label or job.agent_type}")
            model = llm.bind_tools([tool.to_schema() for tool in tools]) if tools else llm
            response = await asyncio.to_thread(model.invoke, messages)
            emitted.append(response)
            messages.append(response)
            usage = getattr(response, "usage_metadata", None)
            if isinstance(usage, dict):
                last_usage = usage

            tool_calls = getattr(response, "tool_calls", None) or []
            if not tool_calls:
                final_text = _message_text(response)
                self._emit_status(job, f"Skill 子任务已完成：{job.label or job.agent_type}")
                return SubagentResult(
                    success=True,
                    final_response=final_text,
                    messages=emitted,
                    tool_calls=all_tool_calls,
                    duration_seconds=time.monotonic() - start_time,
                    usage=last_usage,
                )

            tool_messages = await asyncio.gather(
                *(
                    self._execute_tool_call(
                        tool_map=tool_map,
                        tool_call=tool_call,
                        thread_id=job.thread_id,
                        status_callback=job.status_callback,
                    )
                    for tool_call in tool_calls
                )
            )
            for tool_call in tool_calls:
                all_tool_calls.append(
                    {
                        "name": str(tool_call.get("name") or ""),
                        "args": tool_call.get("args") or tool_call.get("arguments") or {},
                    }
                )
            emitted.extend(tool_messages)
            messages.extend(tool_messages)

        return SubagentResult(
            success=False,
            final_response="",
            messages=emitted,
            tool_calls=all_tool_calls,
            error=f"Reached max_turns={config.max_turns} without a final response",
            duration_seconds=time.monotonic() - start_time,
            usage=last_usage,
        )

    async def _execute_tool_call(
        self,
        *,
        tool_map: dict[str, Tool],
        tool_call: dict[str, Any],
        thread_id: str,
        status_callback: Any | None = None,
    ) -> ToolMessage:
        name = str(tool_call.get("name") or "").strip()
        tool_call_id = str(tool_call.get("id") or uuid.uuid4().hex)
        tool = tool_map.get(name)
        if tool is None:
            return ToolMessage(content=f"Tool not found: {name}", tool_call_id=tool_call_id, name=name or "unknown")

        params = tool_call.get("args")
        if params is None:
            params = tool_call.get("arguments")
        if not isinstance(params, dict):
            params = {}
        preview = _tool_args_preview(name, params)
        _emit_tool_progress(
            status_callback,
            event_type="tool.started",
            tool_name=name,
            preview=preview,
            function_args=params,
        )
        started_at = time.monotonic()
        params = {
            **params,
            "_wbot_tool_context": {
                "graph": self._parent,
                "thread_id": thread_id,
                "subagent_depth": 1,
                "status_callback": status_callback if callable(status_callback) else None,
                "tool_progress_callback": status_callback if callable(status_callback) else None,
            },
        }
        try:
            result = await tool.ainvoke(params)
            content = _tool_result_to_text(result)
        except Exception as exc:
            logger.exception("Subagent tool execution failed: tool=%s", name)
            content = f"Tool execution failed: {type(exc).__name__}: {exc}"
        _emit_tool_progress(
            status_callback,
            event_type="tool.completed",
            tool_name=name,
            preview=preview,
            elapsed_seconds=time.monotonic() - started_at,
            ok=not content.lower().startswith(("error:", "tool execution failed:", "invalid parameters:", "tool not found:")),
            function_args=params,
        )
        return ToolMessage(content=content, tool_call_id=tool_call_id, name=name)

    @staticmethod
    def _emit_status(job: SubagentJobRecord, text: str) -> None:
        callback = job.status_callback if callable(job.status_callback) else None
        if callback is None:
            return
        try:
            callback(str(text))
        except Exception:
            logger.debug("Subagent status callback failed", exc_info=True)

    def _resolve_definition(self, agent_type: str) -> BuiltinSubagentDefinition:
        normalized = agent_type.strip().lower() if agent_type.strip() else "worker"
        definition = BUILTIN_SUBAGENTS.get(normalized)
        if definition is None:
            raise ValueError(
                f"Unknown subagent type: {agent_type}. Available: {', '.join(sorted(BUILTIN_SUBAGENTS))}"
            )
        return definition

    def _build_config(self, definition: BuiltinSubagentDefinition) -> SubagentConfig:
        return SubagentConfig(
            agent_type=definition.key,
            name=definition.name,
            description=definition.description,
            system_prompt=definition.system_prompt,
            allowed_tools=definition.allowed_tools,
            disallowed_tools=list(definition.disallowed_tools),
            max_turns=definition.max_turns,
        )

    def _build_skill_config(self, skill: SkillSpec) -> SubagentConfig:
        return SubagentConfig(
            agent_type=f"skill:{skill.name}",
            name=skill.name,
            description=skill.description or f"Skill {skill.name}",
            system_prompt=(
                "You are executing a reusable skill in a forked subagent.\n"
                "Follow the skill instructions carefully, use tools when needed, "
                "and return a concise execution summary."
            ),
            allowed_tools=None,
            disallowed_tools=["spawn", "list_subagents", "wait_subagent", "run_skill"],
            max_turns=10,
        )

    def _resolve_tools(self, config: SubagentConfig) -> list[Tool]:
        tools = list(self._tools)
        if config.allowed_tools is not None:
            tools = [tool for tool in tools if tool.name in config.allowed_tools]
        if config.disallowed_tools:
            tools = [tool for tool in tools if tool.name not in set(config.disallowed_tools)]
        return [tool for tool in tools if tool.name not in {"spawn", "wait_subagent", "list_subagents"}]

    def _resolve_llm(self, config: SubagentConfig) -> Any:
        base_llm = self._parent._llm_text_base
        if not config.model:
            return base_llm
        try:
            return base_llm.model_copy(update={"model_name": config.model, "model": config.model})
        except Exception:
            logger.warning("Failed to clone subagent LLM with model=%s; fallback to inherited model", config.model)
            return base_llm

    def _build_system_prompt(self, config: SubagentConfig) -> str:
        parts: list[str] = []
        if config.inherit_system_prompt:
            parts.append(str(self._parent._prepared_system_prompt_base or "").strip())
        if config.system_prompt.strip():
            parts.append(config.system_prompt.strip())
        parts.append(
            "Subagent rules:\n"
            "- Focus only on the delegated task.\n"
            "- Do not ask the parent to do work you can complete yourself.\n"
            "- Return concrete findings or outcomes, not process chatter."
        )
        return "\n\n---\n\n".join(part for part in parts if part)

    @staticmethod
    def _build_skill_prompt(
        *,
        skill: SkillSpec,
        task: str,
        arguments: dict[str, Any],
    ) -> str:
        args_json = json.dumps(arguments, ensure_ascii=False, indent=2)
        body = skill.content.strip() or "(empty skill content)"
        return (
            f"Execute skill `{skill.name}`.\n\n"
            f"Skill description: {skill.description or skill.name}\n\n"
            "Skill content:\n"
            f"{body}\n\n"
            "Delegated task:\n"
            f"{task.strip() or '(none)'}\n\n"
            "Arguments:\n"
            f"{args_json}\n\n"
            "Return what you did, key outputs, and any blockers."
        )

    def _append_event(self, job: SubagentJobRecord, *, event: str) -> None:
        payload = {
            "event": event,
            "job": {
                **job.summary(),
                "config": asdict(job.config),
            },
            "timestamp": _now_iso(),
        }
        append_jsonl(self._journal_path, payload)


def _sanitize_context_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    filtered = [message for message in messages if not isinstance(message, SystemMessage)]
    try:
        from .message_utils import sanitize_messages_for_llm

        sanitized = sanitize_messages_for_llm(filtered)
        return [message for message in sanitized if not isinstance(message, SystemMessage)]
    except Exception:
        return filtered


def _message_text(message: BaseMessage) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(content)


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")
