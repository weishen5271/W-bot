from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from rich.console import Console
from rich.panel import Panel

if __package__ in (None, ""):
    _SRC_DIR = os.path.dirname(os.path.dirname(__file__))
    if _SRC_DIR not in sys.path:
        sys.path.insert(0, _SRC_DIR)
    from agents.agent import CyberCoreGraph, message_kind
    from agents.config import Settings, load_settings
    from agents.logging_config import get_logger, setup_logging
    from agents.memory import LongTermMemoryStore
    from agents.tools.runtime import build_tools
else:
    from .agent import CyberCoreGraph, message_kind
    from .config import Settings, load_settings
    from .logging_config import get_logger, setup_logging
    from .memory import LongTermMemoryStore
    from .tools.runtime import build_tools

console = Console()
logger = get_logger(__name__)


class SessionStateStore:
    def __init__(self, file_path: str) -> None:
        self._file_path = file_path

    def load(self) -> str | None:
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
        folder = os.path.dirname(os.path.abspath(self._file_path))
        if folder and not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

        with open(self._file_path, "w", encoding="utf-8") as f:
            json.dump({"session_id": session_id}, f, ensure_ascii=True, indent=2)


def run_cli() -> None:
    setup_logging()
    logger.info("Starting CyberCore CLI runtime")

    try:
        from langgraph.checkpoint.postgres import PostgresSaver
    except ImportError as exc:  # pragma: no cover - environment dependent
        logger.exception("Failed to import PostgresSaver")
        raise RuntimeError(
            "PostgresSaver import failed. Please install psycopg first: "
            "pip install 'psycopg[binary]'"
        ) from exc

    settings = load_settings()
    llm = build_llm(settings)
    memory_store = LongTermMemoryStore(memory_file_path=settings.memory_file_path)
    tools = build_tools(
        memory_store=memory_store,
        user_id=settings.user_id,
        e2b_api_key=settings.e2b_api_key,
        tavily_api_key=settings.tavily_api_key,
        enable_exec_tool=settings.enable_exec_tool,
        enable_cron_service=settings.enable_cron_service,
        mcp_servers=settings.mcp_servers,
    )

    logger.info("Initializing Postgres checkpointer")
    with PostgresSaver.from_conn_string(settings.postgres_dsn) as checkpointer:
        if hasattr(checkpointer, "setup"):
            logger.info("Running checkpointer setup")
            checkpointer.setup()

        graph = CyberCoreGraph(
            llm=llm,
            tools=tools,
            memory_store=memory_store,
            retrieve_top_k=settings.retrieve_top_k,
            user_id=settings.user_id,
            checkpointer=checkpointer,
        ).app

        logger.info("Graph ready, entering REPL loop")
        _repl(graph=graph, settings=settings)


def build_llm(settings: Settings) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.bailian_model_name,
        api_key=settings.dashscope_api_key,
        base_url=settings.bailian_base_url,
        temperature=0.2,
    )


def _repl(graph: Any, settings: Settings) -> None:
    console.print("[bold cyan]CyberCore CLI[/bold cyan] | type quit/exit to leave")

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
