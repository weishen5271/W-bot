from __future__ import annotations

import asyncio
import os
import select
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from rich.console import Console
from rich.live import Live
from rich.text import Text

from .agent import WBotGraph, clear_runtime_callbacks, set_runtime_callbacks
from .config import Settings, load_settings
from .escalation import EscalationManager, EscalationRequest
from .file_checkpointer import WorkspaceFileCheckpointer, resolve_short_term_memory_path
from .logging_config import get_logger, setup_logging
from .memory import LongTermMemoryStore
from .openclaw_profile import OpenClawProfileLoader
from .runtime_status import RuntimeStatusSnapshot
from .session_store import (
    RECENT_SESSIONS_LIMIT,
    SessionRecord,
    SessionStateStore,
)
from .skills import SkillsLoader
from .streaming import latest_non_tool_ai_reply, normalize_display_text
from .text_sanitizer import sanitize_user_text
from .token_tracker import extract_token_usage
from .tools.runtime import build_tools

console = Console()
logger = get_logger(__name__)
_SAVED_TERM_ATTRS: Any = None

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.application import Application
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.enums import EditingMode
    from prompt_toolkit.formatted_text import HTML
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.key_binding.key_processor import KeyPressEvent
    from prompt_toolkit.layout import Layout
    from prompt_toolkit.layout.containers import HSplit
    from prompt_toolkit.patch_stdout import patch_stdout
    from prompt_toolkit.layout.dimension import Dimension
    from prompt_toolkit.styles import Style
    from prompt_toolkit.widgets import Box, Button, Dialog, Frame, Label, TextArea
except Exception:  # pragma: no cover - graceful fallback when dependency is unavailable
    PromptSession = None
    Application = None
    FileHistory = None
    KeyBindings = None
    KeyPressEvent = Any
    HTML = None
    Layout = None
    HSplit = None
    Dimension = None
    Style = None
    patch_stdout = None
    Box = None
    Button = None
    Dialog = None
    Frame = None
    Label = None
    TextArea = None
    EditingMode = None
    Completer = object  # type: ignore[assignment]
    Completion = None

@dataclass
class CliAppState:
    session_id: str
    last_user_text: str = ""
    vim_mode: bool = False
    recent_sessions: list[SessionRecord] = field(default_factory=list)
    runtime_status: RuntimeStatusSnapshot | None = None
    deferred_escalation_ids: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class CliSlashCommand:
    name: str
    description: str
    handler: Any
    aliases: tuple[str, ...] = ()
    argument_hint: str = ""


@dataclass(frozen=True)
class CliCommandResult:
    handled: bool = True
    should_exit: bool = False
    clear_screen: bool = False
    message: str | None = None


@dataclass
class CliCommandContext:
    app_state: CliAppState
    graph: Any
    settings: Settings
    session_store: "SessionStateStore"
    escalation_manager: EscalationManager
    skills_loader: SkillsLoader | None = None
    input_reader: "CliInputReader | None" = None


class CliThinkingSpinner:
    def __init__(self, console_obj: Console, *, runtime_status: RuntimeStatusSnapshot | None = None) -> None:
        self._console = console_obj
        self._runtime_status = runtime_status
        self._status = self._console.status("[dim]W-bot 正在思考...[/dim]", spinner="dots")
        self._active = False

    def start(self) -> None:
        if self._active:
            return
        self._status.start()
        self._active = True

    def update(self, text: str) -> None:
        if not self._active:
            return
        phase = self._runtime_status.spinner_text() if self._runtime_status is not None else _friendly_cli_phase(text)
        if phase:
            self._status.update(f"[dim]{phase}[/dim]")

    def stop(self) -> None:
        if not self._active:
            return
        self._status.stop()
        self._active = False


class CliStreamRenderer:
    def __init__(
        self,
        console_obj: Console,
        *,
        render_markdown: bool = True,
        runtime_status: RuntimeStatusSnapshot | None = None,
    ) -> None:
        self._console = console_obj
        self._render_markdown = render_markdown
        self._runtime_status = runtime_status
        self._spinner = CliThinkingSpinner(console_obj, runtime_status=runtime_status)
        self._spinner.start()
        self._buffer = ""
        self._segments: list[tuple[str, str]] = []
        self._live: Live | None = None
        self._last_refresh_at = 0.0
        self._last_token_at = time.monotonic()
        self._status_line = ""
        self._last_status_line = ""
        self._closed = threading.Event()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name="cli-stream-heartbeat",
            daemon=True,
        )
        self._heartbeat_thread.start()
        self.stream_started = False

    def update_status(self, text: str) -> None:
        if self._runtime_status is not None:
            self._runtime_status.record_status_message(text)
        phase = self._runtime_status.spinner_text() if self._runtime_status is not None else _friendly_cli_phase(text)
        self._status_line = phase.strip()
        self._spinner.update(text)
        if self.stream_started:
            self._refresh_live(force=True)

    def _renderable(self, *, final: bool = False) -> Any:
        rendered = Text()
        if self._segments:
            for kind, payload in self._segments:
                style = "cyan" if kind == "reasoning" else "white"
                rendered.append(payload, style=style)
        else:
            rendered = Text(self._buffer or "")
        if not final and self.stream_started and self._status_line:
            rendered.append("\n\n")
            rendered.append(f"[状态] {self._status_line}", style="dim")
        return rendered

    def on_delta(self, delta: Any) -> None:
        kind = "answer"
        payload = delta
        if isinstance(delta, dict):
            kind = str(delta.get("kind") or "answer").strip().lower() or "answer"
            payload = delta.get("text") or ""
        payload = payload or ""
        if not payload:
            return
        self._last_token_at = time.monotonic()
        self._buffer += payload
        self._segments.append((kind, payload))
        if self._live is None:
            if not self._buffer.strip():
                return
            self._spinner.stop()
            self._console.print()
            self._console.print("[bold cyan]W-bot[/bold cyan]")
            self._live = Live(self._renderable(final=False), console=self._console, auto_refresh=False)
            self._live.start()
            self.stream_started = True
        self._refresh_live(force="\n" in payload)

    def _refresh_live(self, *, force: bool = False) -> None:
        if self._live is None:
            return
        now = time.monotonic()
        if not force and (now - self._last_refresh_at) <= 0.02:
            return
        self._live.update(self._renderable(final=False))
        self._live.refresh()
        self._last_refresh_at = now

    def _heartbeat_loop(self) -> None:
        while not self._closed.wait(1.0):
            if self._live is None:
                continue
            idle_seconds = time.monotonic() - self._last_token_at
            if idle_seconds < 3.0:
                continue
            if self._runtime_status is not None:
                status_text = self._runtime_status.spinner_text().strip()
            else:
                status_text = self._status_line.strip()
            if not status_text:
                continue
            if status_text == self._last_status_line and idle_seconds < 8.0:
                continue
            self._status_line = f"{status_text} | 无新输出 {int(idle_seconds)}s"
            self._last_status_line = self._status_line
            self._refresh_live(force=True)

    def finish(self, final_text: str = "") -> None:
        self._closed.set()
        self._status_line = ""
        final_payload = final_text or self._buffer
        if self._live is not None:
            if final_payload and final_payload != self._buffer:
                self._buffer = final_payload
                self._segments = [("answer", final_payload)]
            self._live.update(self._renderable(final=True))
            self._live.refresh()
            self._live.stop()
            self._live = None
            self._console.print()
            self._spinner.stop()
            return
        self._spinner.stop()
        self._buffer = final_payload
        if final_payload:
            self._segments = [("answer", final_payload)]
        self._console.print()
        self._console.print("[bold cyan]W-bot[/bold cyan]")
        self._console.print(self._renderable(final=True))
        self._console.print()

def run_cli(
    config_path: str = "configs/app.json",
    *,
    session_id: str | None = None,
    force_new_session: bool = False,
) -> None:
    """执行主流程并返回处理结果。
    
    Args:
        config_path: 目标路径参数，用于定位文件或目录。
    """
    overrides: dict[str, Any] = {}
    if session_id:
        overrides["sessionId"] = session_id
    settings = load_settings(config_path=config_path, overrides=overrides or None)
    setup_logging(enable_console_logs=settings.enable_console_logs)
    logger.info("Starting W-bot CLI runtime")
    llm_text = build_llm(settings, model_name=settings.model_routing.text_model_name)
    llm_image = (
        build_llm(settings, model_name=settings.model_routing.image_model_name)
        if settings.model_routing.image_model_name
        else None
    )
    llm_audio = (
        build_llm(settings, model_name=settings.model_routing.audio_model_name)
        if settings.model_routing.audio_model_name
        else None
    )
    openclaw_profile_loader = OpenClawProfileLoader(
        root_dir=settings.openclaw_profile_root_dir,
        enabled=settings.enable_openclaw_profile,
        auto_init=settings.openclaw_auto_init,
    )
    openclaw_profile_loader.prepare_startup()
    memory_file_path = openclaw_profile_loader.resolve_memory_file_path(settings.memory_file_path)
    memory_store = LongTermMemoryStore(memory_file_path=memory_file_path)
    skills_loader = (
        SkillsLoader(
            workspace_skills_dir=settings.skills_workspace_dir,
            builtin_skills_dir=settings.skills_builtin_dir or None,
        )
        if settings.enable_skills
        else None
    )
    escalation_manager = EscalationManager(settings.escalation_state_file_path)
    tools = build_tools(
        memory_store=memory_store,
        user_id=settings.user_id,
        tavily_api_key=settings.tavily_api_key,
        enable_cron_service=settings.enable_cron_service,
        mcp_servers=settings.mcp_servers,
        escalation_manager=escalation_manager,
        skills_loader=skills_loader,
        extra_readonly_dirs=[str(skills_loader.builtin_skills_dir)] if skills_loader else None,
        restrict_to_workspace=settings.restrict_to_workspace,
    )

    short_term_memory_path = resolve_short_term_memory_path(settings.short_term_memory_path)
    logger.info("Initializing workspace short-term checkpointer: %s", short_term_memory_path)
    if settings.short_term_memory_optimization.enabled:
        logger.warning("shortTermMemoryOptimization is ignored in workspace file mode")

    with WorkspaceFileCheckpointer(short_term_memory_path) as checkpointer:
        if hasattr(checkpointer, "setup"):
            logger.info("Running checkpointer setup")
            checkpointer.setup()

        graph = WBotGraph(
            llm=llm_text,
            tools=tools,
            memory_store=memory_store,
            retrieve_top_k=settings.retrieve_top_k,
            user_id=settings.user_id,
            checkpointer=checkpointer,
            skills_loader=skills_loader,
            openclaw_profile_loader=openclaw_profile_loader,
            multimodal_settings=settings.multimodal,
            model_name=settings.model_routing.text_model_name,
            llm_image=llm_image,
            llm_audio=llm_audio,
            image_model_name=settings.model_routing.image_model_name,
            audio_model_name=settings.model_routing.audio_model_name,
            token_optimization_settings=settings.token_optimization,
            max_tool_steps_per_turn=settings.loop_guard.max_tool_steps_per_turn,
            max_same_tool_call_repeats=settings.loop_guard.max_same_tool_call_repeats,
        ).app

        logger.info("Graph ready, entering REPL loop")
        _repl(
            graph=graph,
            settings=settings,
            escalation_manager=escalation_manager,
            skills_loader=skills_loader,
            force_new_session=force_new_session,
        )


def build_llm(settings: Settings, *, model_name: str) -> ChatOpenAI:
    """构建并返回目标对象。
    
    Args:
        settings: 全局设置对象。
        model_name: 当前使用的模型名称。
    """
    kwargs: dict[str, Any] = {
        "model": model_name,
        "api_key": settings.llm_api_key,
        "base_url": settings.llm_base_url,
        "temperature": settings.llm_temperature,
        "streaming": True,
    }
    if settings.llm_extra_headers:
        kwargs["default_headers"] = settings.llm_extra_headers
    return ChatOpenAI(**kwargs)


def _repl(
    graph: Any,
    settings: Settings,
    *,
    escalation_manager: EscalationManager,
    skills_loader: SkillsLoader | None = None,
    force_new_session: bool = False,
) -> None:
    """处理repl相关逻辑并返回结果。
    
    Args:
        graph: 对话图执行器实例。
        settings: 全局设置对象。
    """
    console.print("[bold cyan]W-bot CLI[/bold cyan] | type quit/exit to leave")

    session_store = SessionStateStore(settings.session_state_file_path)
    current_session_id = settings.session_id if force_new_session else (session_store.load() or settings.session_id)
    runtime_status = RuntimeStatusSnapshot(session_id=current_session_id)
    session_store.save(current_session_id, workspace_root=str(Path.cwd().resolve()))
    app_state = CliAppState(
        session_id=current_session_id,
        recent_sessions=session_store.list_recent(),
        runtime_status=runtime_status,
    )
    logger.info("Loaded session_id=%s", current_session_id)
    console.print(
        f"[bold green]Current session:[/bold green] {current_session_id}  ([bold]/new[/bold] for new session)"
    )

    _render_existing_session_history(graph=graph, session_id=current_session_id)
    input_reader = CliInputReader(settings=settings, commands=_build_slash_commands(), app_state=app_state)

    while True:
        user_text = sanitize_user_text(input_reader.read())
        if not user_text.strip():
            continue

        app_state.session_id = current_session_id
        if app_state.runtime_status is not None:
            app_state.runtime_status.set_session(current_session_id)
            app_state.runtime_status.refresh_tasks(graph.list_subagents(limit=20))
        app_state.recent_sessions = session_store.list_recent()
        if user_text.startswith("/"):
            command_result = _handle_slash_command(
                raw_text=user_text,
                context=CliCommandContext(
                    app_state=app_state,
                    graph=graph,
                    settings=settings,
                    session_store=session_store,
                    escalation_manager=escalation_manager,
                    skills_loader=skills_loader,
                    input_reader=input_reader,
                ),
            )
            current_session_id = app_state.session_id
            app_state.recent_sessions = session_store.list_recent()
            if command_result.clear_screen:
                console.clear()
                console.print("[bold cyan]W-bot CLI[/bold cyan] | type quit/exit to leave")
                console.print(
                    f"[bold green]Current session:[/bold green] {current_session_id}  ([bold]/new[/bold] for new session)"
                )
            if command_result.message:
                console.print(command_result.message)
            if command_result.should_exit:
                logger.info("User requested exit")
                _restore_terminal_state()
                console.print("[bold yellow]Session closed.[/bold yellow]")
                return
            continue

        if user_text.strip().lower() in {"quit", "exit"}:
            logger.info("User requested exit")
            _restore_terminal_state()
            console.print("[bold yellow]Session closed.[/bold yellow]")
            return

        logger.info("Received user input, len=%s", len(user_text))
        _run_agent_turn(
            graph=graph,
            settings=settings,
            session_store=session_store,
            app_state=app_state,
            escalation_manager=escalation_manager,
            user_text=user_text,
        )


def _shorten_text(text: str, limit: int = 120) -> str:
    compact = " ".join(str(text or "").split())
    if len(compact) <= limit:
        return compact
    return f"{compact[: max(0, limit - 3)]}..."


def _render_escalation_request(request: EscalationRequest) -> str:
    lines = [
        f"[bold cyan]Escalation[/bold cyan]: {request.id}",
        f"- status: {request.status}",
        f"- risk_type: {request.risk_type}",
        f"- working_dir: {request.working_dir}",
        f"- command: {request.command}",
    ]
    if request.justification:
        lines.append(f"- justification: {request.justification}")
    if request.prefix_rule:
        lines.append(f"- prefix_rule: {' '.join(request.prefix_rule)}")
    if request.denial_reason:
        lines.append(f"- denial_reason: {request.denial_reason}")
    return "\n".join(lines)


def _run_agent_turn(
    *,
    graph: Any,
    settings: Settings,
    session_store: SessionStateStore,
    app_state: CliAppState,
    escalation_manager: EscalationManager | None = None,
    user_text: str,
    recent_action_hint: str | None = None,
) -> str:
    app_state.last_user_text = user_text
    if app_state.runtime_status is not None:
        app_state.runtime_status.set_phase(
            "analyzing",
            "分析需求中",
            recent_action=recent_action_hint or _shorten_text(user_text, 120),
        )
        app_state.runtime_status.refresh_tasks(graph.list_subagents(limit=20))
    renderer = CliStreamRenderer(console, runtime_status=app_state.runtime_status)
    latest_ai_text = ""

    def emit_status(text: str) -> None:
        renderer.update_status(text)

    def emit_token(text: str) -> None:
        payload = text or ""
        if not payload:
            return
        renderer.on_delta(payload)

    config = {
        "configurable": {
            "thread_id": app_state.session_id,
            "status_callback": emit_status,
            "token_callback": emit_token,
            "defer_summary_update": True,
        },
        "recursion_limit": settings.loop_guard.recursion_limit,
    }
    inputs = {"messages": [HumanMessage(content=user_text)]}

    set_runtime_callbacks(token_callback=emit_token, debug_callback=None)
    try:
        result = graph.invoke(inputs, config=config)
        latest_ai_text = _latest_ai_reply_from_result(result)
        _refresh_runtime_usage(graph=graph, app_state=app_state)
        _persist_session_snapshot(session_store=session_store, app_state=app_state)
    except Exception as exc:
        logger.exception("Conversation round failed but REPL will continue")
        detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        if app_state.runtime_status is not None:
            app_state.runtime_status.mark_failed(detail or str(exc))
            app_state.runtime_status.refresh_tasks(graph.list_subagents(limit=20))
        _persist_session_snapshot(session_store=session_store, app_state=app_state)
        renderer.finish("")
        console.print(
            "[bold red]本轮对话出现异常，已跳过本轮并保持程序继续运行。[/bold red]"
        )
        _print_failure_report(app_state)
        if detail:
            console.print(f"[red]异常详情：{detail}[/red]")
        return ""
    finally:
        clear_runtime_callbacks()

    if not latest_ai_text:
        latest_ai_text = "我收到了你的消息，但暂时没有生成可用回复。"
    if app_state.runtime_status is not None:
        app_state.runtime_status.set_phase("rendering", "整理结果中", recent_action="汇总本轮输出")
        app_state.runtime_status.refresh_tasks(graph.list_subagents(limit=20))
    renderer.finish("" if renderer.stream_started else latest_ai_text)
    if app_state.runtime_status is not None:
        app_state.runtime_status.set_phase("idle", "空闲", recent_action="等待下一条输入")
        _persist_session_snapshot(session_store=session_store, app_state=app_state)
    if escalation_manager is not None:
        _maybe_prompt_escalation_choices(
            graph=graph,
            settings=settings,
            session_store=session_store,
            app_state=app_state,
            escalation_manager=escalation_manager,
        )
    return latest_ai_text


def _maybe_prompt_escalation_choices(
    *,
    graph: Any,
    settings: Settings,
    session_store: SessionStateStore,
    app_state: CliAppState,
    escalation_manager: EscalationManager,
) -> None:
    pending_requests = escalation_manager.list_requests(
        session_id=app_state.session_id,
        status="pending",
        limit=20,
    )
    if not pending_requests:
        # Backward-compatible fallback for requests created before thread_id propagation was fixed.
        pending_requests = escalation_manager.list_requests(
            session_id="-",
            status="pending",
            limit=20,
        )
    available_requests = [
        item for item in pending_requests if item.id not in app_state.deferred_escalation_ids
    ]
    if not available_requests:
        return
    request = available_requests[0]
    action = _prompt_escalation_action(request)
    if action == "approve":
        app_state.deferred_escalation_ids.discard(request.id)
        approved = escalation_manager.approve_request(request_id=request.id)
        if approved is None:
            console.print(f"[red]提权请求批准失败:[/red] {request.id}")
            return
        console.print(f"[bold green]已批准提权请求[/bold green] {approved.id}")
        console.print(_render_escalation_request(approved))
        followup = (
            f"系统通知：提权请求 {approved.id} 已获批准。\n"
            f"已批准命令：{approved.command}\n"
            "请继续当前任务；如果需要执行该命令，直接调用对应工具，不要重复申请提权。"
        )
        _run_agent_turn(
            graph=graph,
            settings=settings,
            session_store=session_store,
            app_state=app_state,
            escalation_manager=escalation_manager,
            user_text=followup,
            recent_action_hint=f"批准提权请求 {approved.id}",
        )
        return
    if action == "deny":
        app_state.deferred_escalation_ids.discard(request.id)
        denied = escalation_manager.deny_request(request_id=request.id, reason="用户在交互式审批面板中拒绝")
        if denied is None:
            console.print(f"[red]拒绝提权请求失败:[/red] {request.id}")
            return
        console.print(f"[bold yellow]已拒绝提权请求[/bold yellow] {denied.id}")
        console.print(_render_escalation_request(denied))
        return
    app_state.deferred_escalation_ids.add(request.id)


def _prompt_escalation_action(request: EscalationRequest) -> str:
    title = "提权审批"
    text = "\n".join(
        [
            "本轮操作需要提权，请选择下一步：",
            "",
            f"请求ID: {request.id}",
            f"风险类型: {request.risk_type}",
            f"工作目录: {request.working_dir}",
            f"命令/操作: {request.command}",
            f"用途说明: {request.justification or '[none]'}",
            f"授权前缀: {' '.join(request.prefix_rule) if request.prefix_rule else '[none]'}",
        ]
    )
    if all(
        item is not None
        for item in (Application, Layout, HSplit, Box, Button, Dialog, Frame, Label, TextArea, Style)
    ):
        try:
            return _run_escalation_tui(title=title, text=text)
        except Exception:
            logger.exception("Failed to render prompt_toolkit escalation TUI")

    console.print(f"\n[bold yellow]{title}[/bold yellow]")
    console.print(_render_escalation_request(request))
    console.print("请选择: [1] 批准并继续  [2] 拒绝  [3] 稍后处理")
    while True:
        try:
            choice = input("Select > ").strip()
        except EOFError:
            return "later"
        except KeyboardInterrupt:
            return "later"
        if choice in {"1", "approve", "a"}:
            return "approve"
        if choice in {"2", "deny", "d"}:
            return "deny"
        if choice in {"3", "later", "l", ""}:
            return "later"
        console.print("[yellow]请输入 1、2 或 3。[/yellow]")


def _run_escalation_tui(*, title: str, text: str) -> str:
    assert Application is not None
    assert Layout is not None
    assert HSplit is not None
    assert Box is not None
    assert Button is not None
    assert Dialog is not None
    assert Frame is not None
    assert Label is not None
    assert TextArea is not None
    assert Style is not None
    assert KeyBindings is not None

    result: dict[str, str] = {"value": "later"}

    def _submit(action: str) -> None:
        result["value"] = action
        app.exit(result=action)

    detail_area = TextArea(
        text=text,
        read_only=True,
        scrollbar=True,
        focusable=True,
        wrap_lines=True,
    )
    hint = Label(
        text="Tab 切换焦点，方向键滚动详情，Enter 激活按钮，Esc 关闭并稍后处理。",
        dont_extend_height=True,
    )
    approve_button = Button(text="批准并继续", handler=lambda: _submit("approve"))
    deny_button = Button(text="拒绝", handler=lambda: _submit("deny"))
    later_button = Button(text="稍后处理", handler=lambda: _submit("later"))

    body = HSplit(
        [
            Frame(body=detail_area, title="请求详情"),
            Box(body=hint, padding_top=1, padding_bottom=0),
        ],
        padding=1,
    )
    dialog = Dialog(
        title=title,
        body=body,
        buttons=[approve_button, deny_button, later_button],
        width=Dimension(preferred=100),
        modal=True,
    )
    bindings = KeyBindings()

    @bindings.add("escape")
    @bindings.add("c-c")
    def _(event: KeyPressEvent) -> None:
        del event
        _submit("later")

    style = Style.from_dict(
        {
            "dialog": "bg:#20242b",
            "dialog frame.label": "bg:#20242b #d7e3ff bold",
            "dialog.body": "bg:#20242b #f5f7fa",
            "button": "bg:#31425a #f5f7fa",
            "button.focused": "bg:#7dd3fc #0b1020 bold",
            "text-area": "bg:#11161d #f5f7fa",
            "frame.border": "#5b708f",
            "label": "#d0d7e2",
        }
    )
    app = Application(
        layout=Layout(dialog, focused_element=approve_button),
        key_bindings=bindings,
        full_screen=True,
        mouse_support=True,
        style=style,
    )
    app.run()
    return result["value"]


def _refresh_runtime_usage(*, graph: Any, app_state: CliAppState) -> None:
    status = app_state.runtime_status
    if status is None:
        return
    stats = _collect_session_snapshot_stats(graph=graph, session_id=app_state.session_id)
    estimate = _estimate_session_cost(stats)
    status.update_usage(
        input_tokens=int(stats.get("input_tokens", 0) or 0),
        output_tokens=int(stats.get("output_tokens", 0) or 0),
        total_cost=estimate if estimate is not None else status.total_cost,
    )
    status.refresh_tasks(graph.list_subagents(limit=20))


def _persist_session_snapshot(*, session_store: SessionStateStore, app_state: CliAppState) -> None:
    status = app_state.runtime_status
    session_store.save(
        app_state.session_id,
        title=_shorten_text(app_state.last_user_text, 80),
        workspace_root=str(Path.cwd().resolve()),
        last_phase=status.phase_label if status is not None else "",
        last_action=status.recent_action if status is not None else "",
        last_error=status.last_error if status is not None else "",
        task_count=(status.tasks.running + status.tasks.pending) if status is not None else 0,
    )
    app_state.recent_sessions = session_store.list_recent()


def _print_failure_report(app_state: CliAppState) -> None:
    status = app_state.runtime_status
    if status is None:
        return
    lines = [
        "[bold red][执行失败][/bold red]",
        f"阶段: {status.last_error_phase or status.phase_label}",
        f"最后动作: {status.recent_action or '[unknown]'}",
    ]
    if status.last_error:
        lines.append(f"建议关注: {_shorten_text(status.last_error, 160)}")
    console.print("\n".join(lines))


def _render_existing_session_history(
    *,
    graph: Any,
    session_id: str,
) -> None:
    """将数据渲染为目标文本或展示格式。
    
    Args:
        graph: 对话图执行器实例。
        session_id: 业务对象唯一标识。
        seen_message_signatures: 消息签名集合，用于去重渲染。
    """
    config = {"configurable": {"thread_id": session_id}}
    try:
        snapshot = graph.get_state(config)
    except Exception:
        logger.exception("Failed to load session history for thread_id=%s", session_id)
        return

    values = getattr(snapshot, "values", None) or {}
    messages = values.get("messages", [])
    if not messages:
        console.print("[dim]No previous messages in this session.[/dim]")
        return

    console.print(f"[bold cyan]Restored {len(messages)} message(s) from previous session.[/bold cyan]")


def _message_to_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return normalize_display_text(content)
    if isinstance(content, list):
        lines: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                text = str(block.get("text") or "").strip()
                if text:
                    lines.append(text)
        if lines:
            return normalize_display_text("\n".join(lines))
    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            return normalize_display_text(text)
    return normalize_display_text(str(content))


def _latest_ai_reply_from_result(result: Any) -> str:
    values = result if isinstance(result, dict) else {}
    messages = values.get("messages", []) if isinstance(values.get("messages", []), list) else []
    return latest_non_tool_ai_reply(messages, content_to_text=lambda content: _message_to_text(content))


def _friendly_cli_phase(text: str) -> str:
    normalized = str(text or "").strip()
    if not normalized:
        return "W-bot 正在思考..."
    replacements = [
        ("正在检索长期记忆上下文", "正在回忆相关上下文..."),
        ("已加载", "已找到"),
        ("条长期记忆", "条相关记忆"),
        ("未命中长期记忆，继续直接回答。", "没有历史线索，直接开始回答..."),
        ("正在整理对话上下文", "正在整理问题上下文..."),
        ("已选择模型路由：text。", "已选择文本模型，准备回答..."),
        ("已选择模型路由：image。", "已选择图像模型，准备回答..."),
        ("已选择模型路由：audio。", "已选择音频模型，准备回答..."),
        ("正在生成回复", "正在生成回复..."),
        ("准备执行工具调用：", "正在调用工具: "),
        ("检测到重复工具调用", "检测到重复工具调用，正在停止重试..."),
        ("工具调用已达上限", "工具调用次数过多，正在停止本轮自动重试..."),
        ("触发兼容回退：改用纯文本上下文重试。", "正在切换兼容模式后重试..."),
        ("模型调用失败，已返回兜底提示。", "模型调用失败，正在返回兜底结果..."),
        ("回复已生成。", "回复已生成。"),
    ]
    friendly = normalized
    for source, target in replacements:
        friendly = friendly.replace(source, target)
    return friendly


class CliSlashCommandCompleter(Completer):
    def __init__(self, commands: list[CliSlashCommand]) -> None:
        self._commands = commands

    def get_completions(self, document: Any, complete_event: Any) -> Any:
        del complete_event
        text = document.text_before_cursor.lstrip()
        if not text.startswith("/"):
            return

        fragment = text[1:]
        for command in self._commands:
            names = (command.name, *command.aliases)
            for candidate in names:
                if not candidate.startswith(fragment):
                    continue
                display = f"/{command.name}"
                if command.argument_hint:
                    display = f"{display} {command.argument_hint}"
                meta = command.description
                yield Completion(
                    text=f"/{candidate}",
                    start_position=-len(fragment) - 1,
                    display=display,
                    display_meta=meta,
                )


class CliInputReader:
    def __init__(self, *, settings: Settings, commands: list[CliSlashCommand], app_state: CliAppState) -> None:
        self._settings = settings
        self._commands = commands
        self._app_state = app_state
        _save_terminal_state()
        self._session = self._build_prompt_session()

    def read(self) -> str:
        if self._session is None:
            return input("\nYou > ")
        try:
            _flush_pending_tty_input()
            if patch_stdout is not None and HTML is not None:
                with patch_stdout():
                    return asyncio.run(
                        self._session.prompt_async(HTML("<b fg='ansicyan'>You &gt;</b> "))
                    )
            return asyncio.run(self._session.prompt_async())
        except KeyboardInterrupt:
            return ""
        except EOFError:
            _restore_terminal_state()
            return "exit"

    def _build_prompt_session(self) -> Any:
        if (
            PromptSession is None
            or FileHistory is None
            or KeyBindings is None
            or EditingMode is None
        ):
            return None

        history_path = _resolve_prompt_history_path(self._settings)
        history_path.parent.mkdir(parents=True, exist_ok=True)

        bindings = KeyBindings()

        @bindings.add("c-l")
        def _(event: Any) -> None:
            event.app.renderer.clear()

        return PromptSession(
            history=FileHistory(str(history_path)),
            key_bindings=bindings,
            enable_history_search=True,
            enable_open_in_editor=False,
            multiline=False,
            editing_mode=EditingMode.VI if self._app_state.vim_mode else EditingMode.EMACS,
        )

    def refresh(self) -> None:
        self._session = self._build_prompt_session()


def _save_terminal_state() -> None:
    global _SAVED_TERM_ATTRS
    if _SAVED_TERM_ATTRS is not None:
        return
    try:
        import termios

        _SAVED_TERM_ATTRS = termios.tcgetattr(sys.stdin.fileno())
    except Exception:
        _SAVED_TERM_ATTRS = None


def _restore_terminal_state() -> None:
    if _SAVED_TERM_ATTRS is None:
        return
    try:
        import termios

        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, _SAVED_TERM_ATTRS)
    except Exception:
        return


def _flush_pending_tty_input() -> None:
    try:
        fd = sys.stdin.fileno()
        if not os.isatty(fd):
            return
    except Exception:
        return

    try:
        import termios

        termios.tcflush(fd, termios.TCIFLUSH)
        return
    except Exception:
        pass

    try:
        while True:
            ready, _, _ = select.select([fd], [], [], 0)
            if not ready:
                break
            if not os.read(fd, 4096):
                break
    except Exception:
        return

def _resolve_prompt_history_path(settings: Settings) -> Path:
    session_state_path = Path(settings.session_state_file_path).expanduser()
    if not session_state_path.is_absolute():
        session_state_path = Path.cwd() / session_state_path
    return session_state_path.with_name(".w_bot_prompt_history")
def _build_slash_commands() -> list[CliSlashCommand]:
    return [
        CliSlashCommand("help", "显示可用的 CLI 命令", _cmd_help),
        CliSlashCommand("new", "创建一个新的会话", _cmd_new, argument_hint="[session_id]"),
        CliSlashCommand("resume", "恢复指定会话", _cmd_resume, argument_hint="<session_id>"),
        CliSlashCommand("session", "列出最近会话和当前会话", _cmd_session, aliases=("sessions",)),
        CliSlashCommand("escalation", "查看提权请求状态", _cmd_escalation, aliases=("esc",), argument_hint="[pending|approved|denied|request_id]"),
        CliSlashCommand("approve", "批准提权请求并继续执行", _cmd_approve, argument_hint="[request_id]"),
        CliSlashCommand("deny", "拒绝提权请求", _cmd_deny, argument_hint="<request_id> [reason]"),
        CliSlashCommand("test-escalation-ui", "强制弹出提权 TUI 测试窗口", _cmd_test_escalation_ui),
        CliSlashCommand("status", "查看当前运行状态与最近动作", _cmd_status),
        CliSlashCommand("tasks", "查看子任务与后台任务状态", _cmd_tasks, argument_hint="[status|task_id]"),
        CliSlashCommand("history", "查看当前会话最近消息摘要", _cmd_history, argument_hint="[count]"),
        CliSlashCommand("stats", "查看当前会话统计信息", _cmd_stats),
        CliSlashCommand("cost", "查看 token 使用和成本估算", _cmd_cost),
        CliSlashCommand("vim", "切换或查看 Vim 输入模式", _cmd_vim, argument_hint="[on|off|toggle|status]"),
        CliSlashCommand("config", "查看当前 CLI 关键配置", _cmd_config),
        CliSlashCommand("skills", "查看技能列表或技能详情", _cmd_skills, aliases=("skill",), argument_hint="[skill_name]"),
        CliSlashCommand("clear", "清空当前终端显示", _cmd_clear),
        CliSlashCommand("exit", "退出 CLI", _cmd_exit, aliases=("quit",)),
    ]


def _handle_slash_command(raw_text: str, context: CliCommandContext) -> CliCommandResult:
    normalized = raw_text.strip()
    if not normalized.startswith("/"):
        return CliCommandResult(handled=False)

    parts = normalized[1:].split(maxsplit=1)
    name = (parts[0] if parts else "").strip().lower()
    args = parts[1].strip() if len(parts) > 1 else ""
    for command in _build_slash_commands():
        if name == command.name or name in command.aliases:
            return command.handler(args, context)
    return CliCommandResult(message=f"[yellow]Unknown command:[/yellow] {normalized}\n输入 [bold]/help[/bold] 查看可用命令。")


def _cmd_help(args: str, context: CliCommandContext) -> CliCommandResult:
    del args, context
    lines = ["[bold cyan]Available commands[/bold cyan]"]
    for command in _build_slash_commands():
        hint = f" {command.argument_hint}" if command.argument_hint else ""
        alias_text = f" [dim](aliases: {', '.join('/' + alias for alias in command.aliases)})[/dim]" if command.aliases else ""
        lines.append(f"[bold]/{command.name}[/bold]{hint} - {command.description}{alias_text}")
    lines.append("[dim]提示：支持方向键历史、自动补全、Ctrl+L 清屏。[/dim]")
    return CliCommandResult(message="\n".join(lines))


def _cmd_new(args: str, context: CliCommandContext) -> CliCommandResult:
    session_id = args.strip() or datetime.now().strftime("cli_session_%Y%m%d_%H%M%S")
    context.app_state.session_id = session_id
    if context.app_state.runtime_status is not None:
        context.app_state.runtime_status.set_session(session_id)
        context.app_state.runtime_status.set_phase("idle", "空闲", recent_action="已切换到新会话")
    context.session_store.save(session_id, workspace_root=str(Path.cwd().resolve()))
    return CliCommandResult(message=f"[bold green]Started new session:[/bold green] {session_id}")


def _cmd_resume(args: str, context: CliCommandContext) -> CliCommandResult:
    session_id = args.strip()
    if not session_id:
        return CliCommandResult(message="[yellow]Usage:[/yellow] /resume <session_id>")
    context.app_state.session_id = session_id
    if context.app_state.runtime_status is not None:
        context.app_state.runtime_status.set_session(session_id)
        context.app_state.runtime_status.set_phase("idle", "空闲", recent_action="已恢复会话")
    context.session_store.save(session_id, workspace_root=str(Path.cwd().resolve()))
    _render_existing_session_history(graph=context.graph, session_id=session_id)
    return CliCommandResult(message=f"[bold green]Resumed session:[/bold green] {session_id}")


def _cmd_escalation(args: str, context: CliCommandContext) -> CliCommandResult:
    query = args.strip()
    manager = context.escalation_manager
    if query and query not in {"pending", "approved", "denied", "all"}:
        request = manager.get_request(query)
        if request is None or request.session_id not in {context.app_state.session_id, "-"}:
            return CliCommandResult(message=f"[yellow]未找到提权请求:[/yellow] {query}")
        return CliCommandResult(message=_render_escalation_request(request))

    status = None if query in {"", "all"} else query
    requests = manager.list_requests(
        session_id=context.app_state.session_id,
        status=status,
        limit=10,
    )
    if not requests:
        requests = manager.list_requests(
            session_id="-",
            status=status,
            limit=10,
        )
    if not requests:
        return CliCommandResult(message="[dim]当前会话没有匹配的提权请求。[/dim]")
    lines = ["[bold cyan]Escalation Requests[/bold cyan]"]
    for item in requests:
        lines.append(
            f"- {item.id} [{item.status}] {item.risk_type} | {_shorten_text(item.command, 80)}"
        )
        if item.justification:
            lines.append(f"  reason: {_shorten_text(item.justification, 100)}")
    return CliCommandResult(message="\n".join(lines))


def _cmd_approve(args: str, context: CliCommandContext) -> CliCommandResult:
    request_id = args.strip()
    manager = context.escalation_manager
    request: EscalationRequest | None = None
    if request_id:
        request = manager.get_request(request_id)
    else:
        pending = manager.list_requests(
            session_id=context.app_state.session_id,
            status="pending",
            limit=1,
        )
        if not pending:
            pending = manager.list_requests(
                session_id="-",
                status="pending",
                limit=1,
            )
        request = pending[0] if pending else None
    if request is None or request.session_id not in {context.app_state.session_id, "-"}:
        return CliCommandResult(message="[yellow]当前会话没有可批准的提权请求。[/yellow]")
    approved = manager.approve_request(request_id=request.id)
    if approved is None:
        return CliCommandResult(message=f"[red]提权请求批准失败:[/red] {request.id}")

    console.print(f"[bold green]已批准提权请求[/bold green] {approved.id}")
    console.print(_render_escalation_request(approved))
    followup = (
        f"系统通知：提权请求 {approved.id} 已获批准。\n"
        f"已批准命令：{approved.command}\n"
        "请继续当前任务；如果需要执行该命令，直接调用 exec，不要重复申请提权。"
    )
    _run_agent_turn(
        graph=context.graph,
        settings=context.settings,
        session_store=context.session_store,
        app_state=context.app_state,
        escalation_manager=context.escalation_manager,
        user_text=followup,
        recent_action_hint=f"批准提权请求 {approved.id}",
    )
    return CliCommandResult(handled=True)


def _cmd_deny(args: str, context: CliCommandContext) -> CliCommandResult:
    parts = args.strip().split(maxsplit=1)
    if not parts:
        return CliCommandResult(message="[yellow]Usage:[/yellow] /deny <request_id> [reason]")
    request_id = parts[0].strip()
    reason = parts[1].strip() if len(parts) > 1 else ""
    request = context.escalation_manager.get_request(request_id)
    if request is None or request.session_id not in {context.app_state.session_id, "-"}:
        return CliCommandResult(message=f"[yellow]未找到提权请求:[/yellow] {request_id}")
    denied = context.escalation_manager.deny_request(request_id=request_id, reason=reason)
    if denied is None:
        return CliCommandResult(message=f"[red]拒绝提权请求失败:[/red] {request_id}")
    lines = [f"[bold yellow]已拒绝提权请求[/bold yellow] {denied.id}"]
    if denied.denial_reason:
        lines.append(f"原因: {denied.denial_reason}")
    return CliCommandResult(message="\n".join(lines))


def _cmd_test_escalation_ui(args: str, context: CliCommandContext) -> CliCommandResult:
    del args
    request = EscalationRequest(
        id="test-escalation-ui",
        session_id=context.app_state.session_id,
        command="read_file C:/Windows/win.ini",
        working_dir=str(Path.cwd().resolve()),
        justification="这是一个用于验证终端 TUI 是否能正常显示与选择的测试请求。",
        prefix_rule=["read_file"],
        risk_type="workspace_path",
        status="pending",
        created_at=datetime.now().isoformat(timespec="seconds"),
        updated_at=datetime.now().isoformat(timespec="seconds"),
    )
    action = _prompt_escalation_action(request)
    lines = [
        "[bold cyan]Escalation UI Test[/bold cyan]",
        f"- selected_action: {action}",
    ]
    return CliCommandResult(message="\n".join(lines))


def _cmd_session(args: str, context: CliCommandContext) -> CliCommandResult:
    del args
    recent = context.session_store.list_recent()
    lines = [
        f"[bold green]Current session:[/bold green] {context.app_state.session_id}",
        "[bold cyan]Recent sessions[/bold cyan]",
    ]
    if not recent:
        lines.append("[dim]No saved sessions yet.[/dim]")
    else:
        for record in recent:
            marker = " [green](current)[/green]" if record.session_id == context.app_state.session_id else ""
            detail_parts = [f"[dim]{record.updated_at}[/dim]"]
            if record.last_phase:
                detail_parts.append(f"[dim]{record.last_phase}[/dim]")
            if record.task_count:
                detail_parts.append(f"[dim]tasks={record.task_count}[/dim]")
            lines.append(f"- {record.session_id} {' '.join(detail_parts)}{marker}")
            if record.title:
                lines.append(f"  title: {record.title}")
            if record.last_action:
                lines.append(f"  last_action: {_shorten_text(record.last_action, 100)}")
    return CliCommandResult(message="\n".join(lines))


def _cmd_status(args: str, context: CliCommandContext) -> CliCommandResult:
    del args
    status = context.app_state.runtime_status
    if status is None:
        return CliCommandResult(message="[dim]当前运行时状态不可用。[/dim]")
    status.refresh_tasks(context.graph.list_subagents(limit=20))
    _refresh_runtime_usage(graph=context.graph, app_state=context.app_state)
    lines = [
        "[bold cyan]Runtime Status[/bold cyan]",
        f"- session: {context.app_state.session_id}",
        f"- phase: {status.phase_label}",
        f"- recent_action: {status.recent_action or '[none]'}",
        f"- tasks: running={status.tasks.running} pending={status.tasks.pending} completed={status.tasks.completed} failed={status.tasks.failed}",
        f"- tokens: input={status.input_tokens} output={status.output_tokens}",
    ]
    if status.total_cost > 0:
        lines.append(f"- cost: ${status.total_cost:.6f} USD")
    if status.last_error:
        lines.append(f"- last_error: {status.last_error_phase or status.phase} | {status.last_error}")
    if status.tasks.highlighted_tasks:
        lines.append("- highlighted_tasks:")
        for item in status.tasks.highlighted_tasks:
            lines.append(f"  - {item}")
    return CliCommandResult(message="\n".join(lines))


def _cmd_tasks(args: str, context: CliCommandContext) -> CliCommandResult:
    query = args.strip().lower()
    jobs = context.graph.list_subagents(limit=20)
    if query:
        jobs = [job for job in jobs if query == str(job.get("status") or "").strip().lower() or query in str(job.get("id") or "").lower()]
    if context.app_state.runtime_status is not None:
        context.app_state.runtime_status.refresh_tasks(jobs)
    if not jobs:
        return CliCommandResult(message="[dim]当前没有匹配的后台任务。[/dim]")
    lines = ["[bold cyan]Background Tasks[/bold cyan]"]
    for job in jobs:
        job_id = str(job.get("id") or "-")
        label = str(job.get("label") or job.get("agent_type") or "-")
        status = str(job.get("status") or "-")
        task = _shorten_text(str(job.get("task") or ""), 80)
        duration = job.get("duration_seconds") or 0
        lines.append(f"- {job_id[:8]} [{status}] {label} ({duration}s)")
        if task:
            lines.append(f"  task: {task}")
        error = str(job.get("error") or "").strip()
        if error:
            lines.append(f"  error: {_shorten_text(error, 120)}")
        final_response = str(job.get("final_response") or "").strip()
        if final_response and status == "completed":
            lines.append(f"  result: {_shorten_text(final_response, 120)}")
    return CliCommandResult(message="\n".join(lines))


def _cmd_history(args: str, context: CliCommandContext) -> CliCommandResult:
    count = 6
    if args.strip():
        try:
            count = max(1, min(int(args.strip()), 20))
        except ValueError:
            return CliCommandResult(message="[yellow]Usage:[/yellow] /history [count]")
    preview = _session_history_preview(graph=context.graph, session_id=context.app_state.session_id, limit=count)
    return CliCommandResult(message=preview)


def _cmd_clear(args: str, context: CliCommandContext) -> CliCommandResult:
    del args, context
    return CliCommandResult(clear_screen=True)


def _cmd_exit(args: str, context: CliCommandContext) -> CliCommandResult:
    del args, context
    return CliCommandResult(should_exit=True)


def _cmd_stats(args: str, context: CliCommandContext) -> CliCommandResult:
    del args
    stats = _collect_session_snapshot_stats(graph=context.graph, session_id=context.app_state.session_id)
    lines = [
        f"[bold green]Session[/bold green]: {context.app_state.session_id}",
        f"[bold cyan]Messages[/bold cyan]: total={stats['message_count']} user={stats['user_messages']} assistant={stats['assistant_messages']} tool={stats['tool_messages']}",
        f"[bold cyan]Summary[/bold cyan]: summarized={stats['summarized_message_count']} compact_level={stats['context_compaction_level']}",
        f"[bold cyan]Tokens[/bold cyan]: input={stats['input_tokens']} output={stats['output_tokens']} cache_write={stats['cache_creation_input_tokens']} cache_read={stats['cache_read_input_tokens']} total={stats['total_tokens']}",
        f"[bold cyan]Budget[/bold cyan]: used={stats['used_tokens']} threshold={stats['threshold_tokens']} headroom={stats['percent_left']}%",
    ]
    if stats["warnings"]:
        lines.append(f"[yellow]State[/yellow]: {stats['warnings']}")
    return CliCommandResult(message="\n".join(lines))


def _cmd_cost(args: str, context: CliCommandContext) -> CliCommandResult:
    del args
    stats = _collect_session_snapshot_stats(graph=context.graph, session_id=context.app_state.session_id)
    estimate = _estimate_session_cost(stats)
    lines = [
        f"[bold green]Session[/bold green]: {context.app_state.session_id}",
        f"[bold cyan]Model[/bold cyan]: {context.settings.model_routing.text_model_name}",
        f"[bold cyan]Token usage[/bold cyan]: input={stats['input_tokens']} output={stats['output_tokens']} cache_write={stats['cache_creation_input_tokens']} cache_read={stats['cache_read_input_tokens']} total={stats['total_tokens']}",
    ]
    if estimate is None:
        lines.append(
            "[dim]Cost estimate unavailable. Set env vars `WBOT_INPUT_COST_PER_1M`, `WBOT_OUTPUT_COST_PER_1M`, `WBOT_CACHE_WRITE_COST_PER_1M`, `WBOT_CACHE_READ_COST_PER_1M` to enable pricing.[/dim]"
        )
    else:
        lines.append(f"[bold cyan]Estimated cost[/bold cyan]: ${estimate:.6f} USD")
    return CliCommandResult(message="\n".join(lines))


def _cmd_vim(args: str, context: CliCommandContext) -> CliCommandResult:
    action = (args.strip().lower() or "status")
    if action == "status":
        status = "on" if context.app_state.vim_mode else "off"
        return CliCommandResult(message=f"[bold cyan]Vim mode[/bold cyan]: {status}")
    if action not in {"on", "off", "toggle"}:
        return CliCommandResult(message="[yellow]Usage:[/yellow] /vim [on|off|toggle|status]")

    if action == "toggle":
        context.app_state.vim_mode = not context.app_state.vim_mode
    else:
        context.app_state.vim_mode = action == "on"
    if context.input_reader is not None:
        context.input_reader.refresh()
    status = "on" if context.app_state.vim_mode else "off"
    return CliCommandResult(message=f"[bold green]Vim mode[/bold green] is now {status}")


def _cmd_config(args: str, context: CliCommandContext) -> CliCommandResult:
    del args
    settings = context.settings
    lines = [
        "[bold cyan]CLI config[/bold cyan]",
        f"- provider: {settings.model_provider}",
        f"- text_model: {settings.model_routing.text_model_name}",
        f"- image_model: {settings.model_routing.image_model_name or '[disabled]'}",
        f"- audio_model: {settings.model_routing.audio_model_name or '[disabled]'}",
        f"- memory_file: {settings.memory_file_path}",
        f"- short_term_memory: {settings.short_term_memory_path}",
        f"- session_state_file: {settings.session_state_file_path}",
        f"- openclaw_profile_root: {settings.openclaw_profile_root_dir}",
        f"- skills_enabled: {settings.enable_skills}",
        f"- mcp_servers: {len(settings.mcp_servers)}",
        f"- token_optimization: {settings.token_optimization.enabled}",
        f"- streaming: {settings.enable_streaming}",
        f"- vim_mode: {context.app_state.vim_mode}",
    ]
    return CliCommandResult(message="\n".join(lines))


def _cmd_skills(args: str, context: CliCommandContext) -> CliCommandResult:
    loader = context.skills_loader
    if loader is None:
        return CliCommandResult(message="[dim]Skills are disabled in current config.[/dim]")

    skill_name = args.strip()
    if not skill_name:
        skills = loader.list_skills(filter_unavailable=False)
        if not skills:
            return CliCommandResult(message="[dim]No skills found.[/dim]")
        lines = ["[bold cyan]Skills[/bold cyan]"]
        for skill in skills:
            check = loader.check_requirements(skill)
            status = "available" if check.available else "unavailable"
            lines.append(f"- {skill.name} [{skill.source}/{status}] {skill.description or ''}".rstrip())
        return CliCommandResult(message="\n".join(lines))

    skill = loader.get_skill(skill_name)
    if skill is None:
        return CliCommandResult(message=f"[yellow]Skill not found:[/yellow] {skill_name}")
    check = loader.check_requirements(skill)
    lines = [
        f"[bold cyan]Skill[/bold cyan]: {skill.name}",
        f"- source: {skill.source}",
        f"- path: {skill.path}",
        f"- available: {check.available}",
        f"- always: {skill.always}",
        f"- requires_bins: {', '.join(skill.requires_bins) if skill.requires_bins else '[none]'}",
        f"- requires_env: {', '.join(skill.requires_env) if skill.requires_env else '[none]'}",
    ]
    if skill.description:
        lines.append(f"- description: {skill.description}")
    if not check.available:
        missing_parts: list[str] = []
        if check.missing_bins:
            missing_parts.append(f"missing bins: {', '.join(check.missing_bins)}")
        if check.missing_env:
            missing_parts.append(f"missing env: {', '.join(check.missing_env)}")
        lines.append(f"- unmet requirements: {'; '.join(missing_parts)}")
    return CliCommandResult(message="\n".join(lines))


def _session_history_preview(*, graph: Any, session_id: str, limit: int = 6) -> str:
    config = {"configurable": {"thread_id": session_id}}
    try:
        snapshot = graph.get_state(config)
    except Exception:
        logger.exception("Failed to load session history preview for thread_id=%s", session_id)
        return "[red]无法读取会话历史。[/red]"

    values = getattr(snapshot, "values", None) or {}
    messages = values.get("messages", [])
    if not messages:
        return "[dim]当前会话还没有历史消息。[/dim]"

    lines = [f"[bold cyan]Recent messages for {session_id}[/bold cyan]"]
    for message in messages[-limit:]:
        role = getattr(message, "type", None) or getattr(message, "role", None) or message.__class__.__name__
        label = "You" if str(role).lower() in {"human", "user"} else "W-bot"
        text = _message_to_text(message).strip()
        compact = " ".join(text.split())
        if len(compact) > 120:
            compact = f"{compact[:117]}..."
        lines.append(f"- [bold]{label}[/bold]: {compact or '[empty]'}")
    return "\n".join(lines)


def _collect_session_snapshot_stats(*, graph: Any, session_id: str) -> dict[str, Any]:
    config = {"configurable": {"thread_id": session_id}}
    try:
        snapshot = graph.get_state(config)
    except Exception:
        logger.exception("Failed to collect session stats for thread_id=%s", session_id)
        return {
            "message_count": 0,
            "user_messages": 0,
            "assistant_messages": 0,
            "tool_messages": 0,
            "summarized_message_count": 0,
            "context_compaction_level": "unknown",
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "total_tokens": 0,
            "used_tokens": 0,
            "threshold_tokens": 0,
            "percent_left": 0,
            "warnings": "stats unavailable",
        }

    values = getattr(snapshot, "values", None) or {}
    messages = values.get("messages", [])
    message_count = len(messages) if isinstance(messages, list) else 0
    user_messages = 0
    assistant_messages = 0
    tool_messages = 0
    for message in messages if isinstance(messages, list) else []:
        role = str(getattr(message, "type", None) or getattr(message, "role", None) or "").lower()
        if role in {"human", "user"}:
            user_messages += 1
        elif role == "tool":
            tool_messages += 1
        else:
            assistant_messages += 1

    usage = extract_token_usage(values.get("session_token_usage") or {})
    budget_state = values.get("token_budget_state") if isinstance(values.get("token_budget_state"), dict) else {}
    warnings = _describe_budget_state(budget_state)
    return {
        "message_count": message_count,
        "user_messages": user_messages,
        "assistant_messages": assistant_messages,
        "tool_messages": tool_messages,
        "summarized_message_count": int(values.get("summarized_message_count") or 0),
        "context_compaction_level": str(values.get("context_compaction_level") or "none"),
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_creation_input_tokens": usage.cache_creation_input_tokens,
        "cache_read_input_tokens": usage.cache_read_input_tokens,
        "total_tokens": usage.total,
        "used_tokens": int(budget_state.get("used_tokens", 0) or 0),
        "threshold_tokens": int(budget_state.get("threshold_tokens", 0) or 0),
        "percent_left": int(budget_state.get("percent_left", 0) or 0),
        "warnings": warnings,
    }


def _describe_budget_state(budget_state: dict[str, Any]) -> str:
    if not budget_state:
        return ""
    if bool(budget_state.get("is_at_blocking_limit")):
        return "near blocking limit"
    if bool(budget_state.get("is_above_auto_compact_threshold")):
        return "above auto-compact threshold"
    if bool(budget_state.get("is_above_error_threshold")):
        return "in elevated range"
    if bool(budget_state.get("is_above_warning_threshold")):
        return "in warning range"
    return "healthy"


def _estimate_session_cost(stats: dict[str, Any]) -> float | None:
    env_keys = {
        "input_tokens": "WBOT_INPUT_COST_PER_1M",
        "output_tokens": "WBOT_OUTPUT_COST_PER_1M",
        "cache_creation_input_tokens": "WBOT_CACHE_WRITE_COST_PER_1M",
        "cache_read_input_tokens": "WBOT_CACHE_READ_COST_PER_1M",
    }
    pricing: dict[str, float] = {}
    for key, env_name in env_keys.items():
        raw = os.getenv(env_name)
        if raw is None:
            continue
        try:
            pricing[key] = max(0.0, float(raw))
        except ValueError:
            logger.warning("Invalid cost env value ignored: %s=%s", env_name, raw)

    if not pricing:
        return None

    total = 0.0
    for stat_key, price_per_million in pricing.items():
        total += (max(0, int(stats.get(stat_key, 0) or 0)) / 1_000_000.0) * price_per_million
    return total


if __name__ == "__main__":
    run_cli()
