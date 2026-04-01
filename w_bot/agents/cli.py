from __future__ import annotations

import json
import os
import time
import traceback
from datetime import datetime
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
from .skills import SkillsLoader
from .streaming import latest_non_tool_ai_reply, normalize_display_text
from .text_sanitizer import sanitize_user_text
from .tools.runtime import build_tools

console = Console()
logger = get_logger(__name__)


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


class SessionStateStore:
    def __init__(self, file_path: str) -> None:
        """初始化对象并保存运行所需依赖。
        
        Args:
            file_path: 文件路径对象或路径字符串。
        """
        self._file_path = file_path

    def load(self) -> str | None:
        """加载目标配置或数据并返回。
        """
        if not os.path.exists(self._file_path):
            return None

        try:
            with open(self._file_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            logger.exception("Failed to load session state file: %s", self._file_path)
            return None

        session_id = payload.get("session_id") if isinstance(payload, dict) else None
        if isinstance(session_id, str) and session_id.strip():
            return session_id.strip()
        return None

    def save(self, session_id: str) -> None:
        """保存数据到持久化存储。
        
        Args:
            session_id: 业务对象唯一标识。
        """
        folder = os.path.dirname(os.path.abspath(self._file_path))
        if folder and not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

        with open(self._file_path, "w", encoding="utf-8") as f:
            json.dump({"session_id": session_id}, f, ensure_ascii=True, indent=2)


def run_cli(config_path: str = "configs/app.json") -> None:
    """执行主流程并返回处理结果。
    
    Args:
        config_path: 目标路径参数，用于定位文件或目录。
    """
    settings = load_settings(config_path=config_path)
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
        _repl(graph=graph, settings=settings)


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


def _repl(graph: Any, settings: Settings) -> None:
    """处理repl相关逻辑并返回结果。
    
    Args:
        graph: 对话图执行器实例。
        settings: 全局设置对象。
    """
    console.print("[bold cyan]W-bot CLI[/bold cyan] | type quit/exit to leave")

    session_store = SessionStateStore(settings.session_state_file_path)
    current_session_id = session_store.load() or settings.session_id
    session_store.save(current_session_id)
    logger.info("Loaded session_id=%s", current_session_id)
    console.print(
        f"[bold green]Current session:[/bold green] {current_session_id}  ([bold]/new[/bold] for new session)"
    )

    _render_existing_session_history(graph=graph, session_id=current_session_id)

    while True:
        user_text = sanitize_user_text(input("\nYou > "))
        if not user_text.strip():
            continue

        if user_text.strip().lower() == "/new":
            current_session_id = datetime.now().strftime("cli_session_%Y%m%d_%H%M%S")
            session_store.save(current_session_id)
            logger.info("Created new session via /new: %s", current_session_id)
            console.print(f"[bold green]Started new session:[/bold green] {current_session_id}")
            continue

        if user_text.strip().lower() in {"quit", "exit"}:
            logger.info("User requested exit")
            console.print("[bold yellow]Session closed.[/bold yellow]")
            return

        logger.info("Received user input, len=%s", len(user_text))
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


if __name__ == "__main__":
    run_cli()
