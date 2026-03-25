from __future__ import annotations

import os
import sys
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


def run_cli() -> None:
    setup_logging()
    logger.info("Starting CyberCore CLI runtime")

    try:
        from langgraph.checkpoint.postgres import PostgresSaver
    except ImportError as exc:  # pragma: no cover - environment dependent
        logger.exception("Failed to import PostgresSaver")
        raise RuntimeError(
            "无法导入 PostgresSaver。请安装 psycopg 二进制依赖，例如："
            " pip install 'psycopg[binary]'，并确保本机可用 libpq。"
        ) from exc

    settings = load_settings()
    llm = build_llm(settings)
    memory_store = LongTermMemoryStore(
        milvus_uri=settings.milvus_uri,
        collection_name=settings.memory_collection,
    )
    tools = build_tools(
        memory_store=memory_store,
        user_id=settings.user_id,
        e2b_api_key=settings.e2b_api_key,
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
    console.print("[bold cyan]CyberCore CLI[/bold cyan] 已启动，输入 quit/exit 退出。")
    seen_message_ids: set[str] = set()

    while True:
        user_text = input("\nYou > ").strip()
        if not user_text:
            continue
        if user_text.lower() in {"quit", "exit"}:
            logger.info("User requested exit")
            console.print("[bold yellow]已退出。[/bold yellow]")
            return

        logger.info("Received user input, len=%s", len(user_text))
        config = {
            "configurable": {
                "thread_id": settings.thread_id,
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
