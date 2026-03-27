from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from rich.console import Console
from rich.panel import Panel

from .agent import WBotGraph, message_kind
from .config import Settings, load_settings
from .logging_config import get_logger, setup_logging
from .memory import LongTermMemoryStore
from .short_memory_optimizer import (
    ShortTermMemoryOptimizationSettings,
    start_short_memory_optimizer_worker,
)
from .skills import SkillsLoader
from .tools.runtime import build_tools

console = Console()
logger = get_logger(__name__)


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
    setup_logging()
    logger.info("Starting W-bot CLI runtime")

    try:
        from langgraph.checkpoint.postgres import PostgresSaver
    except ImportError as exc:  # pragma: no cover - environment dependent
        logger.exception("Failed to import PostgresSaver")
        raise RuntimeError(
            "PostgresSaver import failed. Please install psycopg first: "
            "pip install 'psycopg[binary]'"
        ) from exc

    settings = load_settings(config_path=config_path)
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
    memory_store = LongTermMemoryStore(memory_file_path=settings.memory_file_path)
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
        enable_exec_tool=settings.enable_exec_tool,
        enable_cron_service=settings.enable_cron_service,
        mcp_servers=settings.mcp_servers,
        extra_readonly_dirs=[str(skills_loader.builtin_skills_dir)] if skills_loader else None,
    )

    logger.info("Initializing Postgres checkpointer")
    with PostgresSaver.from_conn_string(settings.postgres_dsn) as checkpointer:
        if hasattr(checkpointer, "setup"):
            logger.info("Running checkpointer setup")
            checkpointer.setup()

        optimizer_settings = ShortTermMemoryOptimizationSettings(
            enabled=settings.short_term_memory_optimization.enabled,
            run_on_startup=settings.short_term_memory_optimization.run_on_startup,
            interval_minutes=settings.short_term_memory_optimization.interval_minutes,
            keep_recent_checkpoints=settings.short_term_memory_optimization.keep_recent_checkpoints,
            summary_batch_size=settings.short_term_memory_optimization.summary_batch_size,
            max_threads_per_run=settings.short_term_memory_optimization.max_threads_per_run,
            max_checkpoints_per_thread=settings.short_term_memory_optimization.max_checkpoints_per_thread,
            archive_before_delete=settings.short_term_memory_optimization.archive_before_delete,
            compress_level=settings.short_term_memory_optimization.compress_level,
        )
        _, optimizer_stop_event = start_short_memory_optimizer_worker(
            postgres_dsn=settings.postgres_dsn,
            settings=optimizer_settings,
        )

        graph = WBotGraph(
            llm=llm_text,
            tools=tools,
            memory_store=memory_store,
            retrieve_top_k=settings.retrieve_top_k,
            user_id=settings.user_id,
            checkpointer=checkpointer,
            skills_loader=skills_loader,
            multimodal_settings=settings.multimodal,
            model_name=settings.model_routing.text_model_name,
            llm_image=llm_image,
            llm_audio=llm_audio,
            image_model_name=settings.model_routing.image_model_name,
            audio_model_name=settings.model_routing.audio_model_name,
            token_optimization_settings=settings.token_optimization,
        ).app

        logger.info("Graph ready, entering REPL loop")
        try:
            _repl(graph=graph, settings=settings)
        finally:
            if optimizer_stop_event is not None:
                optimizer_stop_event.set()


def build_llm(settings: Settings, *, model_name: str) -> ChatOpenAI:
    """构建并返回目标对象。
    
    Args:
        settings: 全局设置对象。
        model_name: 当前使用的模型名称。
    """
    return ChatOpenAI(
        model=model_name,
        api_key=settings.dashscope_api_key,
        base_url=settings.bailian_base_url,
        temperature=0.2,
    )


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

    seen_message_ids: set[str] = set()
    _render_existing_session_history(
        graph=graph,
        session_id=current_session_id,
        seen_message_ids=seen_message_ids,
    )

    while True:
        user_text = input("\nYou > ").strip()
        if not user_text:
            continue

        if user_text.lower() == "/new":
            current_session_id = datetime.now().strftime("cli_session_%Y%m%d_%H%M%S")
            session_store.save(current_session_id)
            seen_message_ids.clear()
            logger.info("Created new session via /new: %s", current_session_id)
            console.print(f"[bold green]Started new session:[/bold green] {current_session_id}")
            continue

        if user_text.lower() in {"quit", "exit"}:
            logger.info("User requested exit")
            console.print("[bold yellow]Session closed.[/bold yellow]")
            return

        logger.info("Received user input, len=%s", len(user_text))
        config = {
            "configurable": {
                "thread_id": current_session_id,
            }
        }
        inputs = {"messages": [HumanMessage(content=user_text)]}

        for event in graph.stream(inputs, config=config, stream_mode="values"):
            messages = event.get("messages", [])
            if not messages:
                continue
            last = messages[-1]
            msg_id = getattr(last, "id", None) or f"{len(messages)}-{message_kind(last)}"
            if msg_id in seen_message_ids:
                continue
            seen_message_ids.add(msg_id)
            logger.debug("Rendering message: kind=%s, id=%s", message_kind(last), msg_id)
            _render_message(last)


def _render_existing_session_history(
    *,
    graph: Any,
    session_id: str,
    seen_message_ids: set[str],
) -> None:
    """将数据渲染为目标文本或展示格式。
    
    Args:
        graph: 对话图执行器实例。
        session_id: 业务对象唯一标识。
        seen_message_ids: 消息相关参数，用于定位或处理消息。
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

    console.print(f"[bold cyan]Restored {len(messages)} message(s) from previous session:[/bold cyan]")
    for idx, message in enumerate(messages, start=1):
        msg_id = getattr(message, "id", None) or f"{idx}-{message_kind(message)}"
        seen_message_ids.add(msg_id)
        _render_message(message)


def _render_message(message: Any) -> None:
    """将数据渲染为目标文本或展示格式。
    
    Args:
        message: 单条消息对象。
    """
    kind = message_kind(message)
    title = {
        "thought": "Thought",
        "action": "Action",
        "tool": "Observation",
        "human": "Human",
    }.get(kind, "Message")

    style = {
        "thought": "cyan",
        "action": "magenta",
        "tool": "green",
        "human": "white",
    }.get(kind, "white")

    content = message.content if isinstance(message.content, str) else str(message.content)

    if kind == "action":
        content = f"{content}\n\nTool Calls: {message.tool_calls}"

    console.print(Panel(content, title=title, border_style=style, expand=True))


if __name__ == "__main__":
    run_cli()
