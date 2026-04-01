from __future__ import annotations

import os
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
from .file_checkpointer import WorkspaceFileCheckpointer, resolve_short_term_memory_path
from .logging_config import get_logger, setup_logging
from .memory import LongTermMemoryStore
from .openclaw_profile import OpenClawProfileLoader
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

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.enums import EditingMode
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.key_binding import KeyBindings
except Exception:  # pragma: no cover - graceful fallback when dependency is unavailable
    PromptSession = None
    AutoSuggestFromHistory = None
    FileHistory = None
    KeyBindings = None
    EditingMode = None
    Completer = object  # type: ignore[assignment]
    Completion = None

@dataclass
class CliAppState:
    session_id: str
    last_user_text: str = ""
    vim_mode: bool = False
    recent_sessions: list[SessionRecord] = field(default_factory=list)


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
    skills_loader: SkillsLoader | None = None
    input_reader: "CliInputReader | None" = None


class CliThinkingSpinner:
    def __init__(self, console_obj: Console) -> None:
        self._console = console_obj
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
        phase = _friendly_cli_phase(text)
        if phase:
            self._status.update(f"[dim]{phase}[/dim]")

    def stop(self) -> None:
        if not self._active:
            return
        self._status.stop()
        self._active = False


class CliStreamRenderer:
    def __init__(self, console_obj: Console, *, render_markdown: bool = True) -> None:
        self._console = console_obj
        self._render_markdown = render_markdown
        self._spinner = CliThinkingSpinner(console_obj)
        self._spinner.start()
        self._buffer = ""
        self._segments: list[tuple[str, str]] = []
        self._live: Live | None = None
        self._last_refresh_at = 0.0
        self.stream_started = False

    def update_status(self, text: str) -> None:
        if self.stream_started:
            return
        self._spinner.update(text)

    def _renderable(self, *, final: bool = False) -> Any:
        del final
        rendered = Text()
        if self._segments:
            for kind, payload in self._segments:
                style = "cyan" if kind == "reasoning" else "white"
                rendered.append(payload, style=style)
            return rendered
        return Text(self._buffer or "")

    def on_delta(self, delta: Any) -> None:
        kind = "answer"
        payload = delta
        if isinstance(delta, dict):
            kind = str(delta.get("kind") or "answer").strip().lower() or "answer"
            payload = delta.get("text") or ""
        payload = payload or ""
        if not payload:
            return
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
        now = time.monotonic()
        if "\n" in payload or (now - self._last_refresh_at) > 0.02:
            self._live.update(self._renderable(final=False))
            self._live.refresh()
            self._last_refresh_at = now

    def finish(self, final_text: str = "") -> None:
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
    tools = build_tools(
        memory_store=memory_store,
        user_id=settings.user_id,
        tavily_api_key=settings.tavily_api_key,
        enable_cron_service=settings.enable_cron_service,
        mcp_servers=settings.mcp_servers,
        skills_loader=skills_loader,
        extra_readonly_dirs=[str(skills_loader.builtin_skills_dir)] if skills_loader else None,
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
    session_store.save(current_session_id)
    app_state = CliAppState(session_id=current_session_id, recent_sessions=session_store.list_recent())
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
        app_state.recent_sessions = session_store.list_recent()
        if user_text.startswith("/"):
            command_result = _handle_slash_command(
                raw_text=user_text,
                context=CliCommandContext(
                    app_state=app_state,
                    graph=graph,
                    settings=settings,
                    session_store=session_store,
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
                console.print("[bold yellow]Session closed.[/bold yellow]")
                return
            continue

        if user_text.strip().lower() in {"quit", "exit"}:
            logger.info("User requested exit")
            console.print("[bold yellow]Session closed.[/bold yellow]")
            return

        logger.info("Received user input, len=%s", len(user_text))
        app_state.last_user_text = user_text
        renderer = CliStreamRenderer(console)
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
                "thread_id": current_session_id,
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
        except Exception as exc:
            logger.exception("Conversation round failed but REPL will continue")
            renderer.finish("")
            console.print(
                "[bold red]本轮对话出现异常，已跳过本轮并保持程序继续运行。[/bold red]"
            )
            detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            if detail:
                console.print(f"[red]异常详情：{detail}[/red]")
            continue
        finally:
            clear_runtime_callbacks()

        if not latest_ai_text:
            latest_ai_text = "我收到了你的消息，但暂时没有生成可用回复。"
        renderer.finish("" if renderer.stream_started else latest_ai_text)


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
        self._session = self._build_prompt_session()

    def read(self) -> str:
        if self._session is None:
            return input("\nYou > ")
        try:
            return self._session.prompt()
        except KeyboardInterrupt:
            return ""
        except EOFError:
            return "exit"

    def _build_prompt_session(self) -> Any:
        if (
            PromptSession is None
            or FileHistory is None
            or AutoSuggestFromHistory is None
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
            message="\nYou > ",
            completer=CliSlashCommandCompleter(self._commands),
            complete_while_typing=True,
            complete_in_thread=True,
            reserve_space_for_menu=8,
            auto_suggest=AutoSuggestFromHistory(),
            history=FileHistory(str(history_path)),
            key_bindings=bindings,
            enable_history_search=True,
            editing_mode=EditingMode.VI if self._app_state.vim_mode else EditingMode.EMACS,
        )

    def refresh(self) -> None:
        self._session = self._build_prompt_session()

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
    context.session_store.save(session_id)
    return CliCommandResult(message=f"[bold green]Started new session:[/bold green] {session_id}")


def _cmd_resume(args: str, context: CliCommandContext) -> CliCommandResult:
    session_id = args.strip()
    if not session_id:
        return CliCommandResult(message="[yellow]Usage:[/yellow] /resume <session_id>")
    context.app_state.session_id = session_id
    context.session_store.save(session_id)
    _render_existing_session_history(graph=context.graph, session_id=session_id)
    return CliCommandResult(message=f"[bold green]Resumed session:[/bold green] {session_id}")


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
            lines.append(f"- {record.session_id} [dim]{record.updated_at}[/dim]{marker}")
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
