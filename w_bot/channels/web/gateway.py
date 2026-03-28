from __future__ import annotations

import argparse
import asyncio
import json
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel

from w_bot.agents.agent import message_kind
from w_bot.agents.agent import WBotGraph
from w_bot.agents.config import default_app_config, load_settings
from w_bot.agents.logging_config import get_logger, setup_logging
from w_bot.agents.memory import LongTermMemoryStore
from w_bot.agents.openclaw_profile import OpenClawProfileLoader
from w_bot.agents.short_memory_optimizer import (
    ShortTermMemoryOptimizationSettings,
    start_short_memory_optimizer_worker,
)
from w_bot.agents.skills import SkillsLoader
from w_bot.agents.text_sanitizer import sanitize_user_text
from w_bot.agents.tools.runtime import build_tools

logger = get_logger(__name__)


@dataclass(frozen=True)
class WebConfig:
    enabled: bool
    host: str
    port: int

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "WebConfig":
        return WebConfig(
            enabled=bool(_pick(payload, "enabled", default=True)),
            host=str(_pick(payload, "host", default="127.0.0.1")).strip() or "127.0.0.1",
            port=_safe_port(_pick(payload, "port", default=8000)),
        )


@dataclass(frozen=True)
class GatewayConfig:
    web: WebConfig
    thread_prefix: str


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    reply: str


class SessionResponse(BaseModel):
    session_id: str


class HistoryResponse(BaseModel):
    session_id: str
    messages: list[dict[str, str]]


def run_web_gateway(config_path: str = "configs/app.json") -> None:
    settings = load_settings(config_path=config_path)
    setup_logging(enable_console_logs=settings.enable_console_logs)
    cfg = load_gateway_config(config_path)
    logger.info("Building graph for Web gateway")

    if not cfg.web.enabled:
        logger.warning("Web config loaded but channels.web.enabled=false")

    try:
        from langgraph.checkpoint.postgres import PostgresSaver
    except ImportError as exc:  # pragma: no cover - environment dependent
        logger.exception("Failed to import PostgresSaver")
        raise RuntimeError(
            "PostgresSaver import failed. Please install psycopg first: pip install 'psycopg[binary]'"
        ) from exc

    llm_text = _build_llm(settings, model_name=settings.model_routing.text_model_name)
    llm_image = (
        _build_llm(settings, model_name=settings.model_routing.image_model_name)
        if settings.model_routing.image_model_name
        else None
    )
    llm_audio = (
        _build_llm(settings, model_name=settings.model_routing.audio_model_name)
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
        enable_exec_tool=settings.enable_exec_tool,
        enable_cron_service=settings.enable_cron_service,
        mcp_servers=settings.mcp_servers,
        extra_readonly_dirs=[str(skills_loader.builtin_skills_dir)] if skills_loader else None,
    )

    with PostgresSaver.from_conn_string(settings.postgres_dsn) as checkpointer:
        if hasattr(checkpointer, "setup"):
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

        app = _build_app(
            graph=graph,
            thread_prefix=cfg.thread_prefix,
            expose_step_logs=settings.expose_step_logs,
            recursion_limit=settings.loop_guard.recursion_limit,
        )
        try:
            uvicorn.run(app, host=cfg.web.host, port=cfg.web.port, log_level="info")
        finally:
            if optimizer_stop_event is not None:
                optimizer_stop_event.set()


def _build_app(
    *,
    graph: Any,
    thread_prefix: str,
    expose_step_logs: bool,
    recursion_limit: int,
) -> FastAPI:
    app = FastAPI(title="W-bot Web Gateway")
    graph_lock = threading.Lock()
    static_root = Path(__file__).resolve().parent / "static"
    index_path = static_root / "index.html"

    @app.get("/")
    def read_index() -> FileResponse:
        if not index_path.exists():
            raise HTTPException(status_code=500, detail="Web UI file is missing: index.html")
        return FileResponse(
            index_path,
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/session/new", response_model=SessionResponse)
    def new_session() -> SessionResponse:
        return SessionResponse(session_id=_new_session_id(thread_prefix))

    @app.get("/api/history", response_model=HistoryResponse)
    def get_history(session_id: str) -> HistoryResponse:
        if not session_id.strip():
            raise HTTPException(status_code=400, detail="session_id is required")
        config = {"configurable": {"thread_id": session_id.strip()}}

        with graph_lock:
            try:
                snapshot = graph.get_state(config)
            except Exception as exc:
                logger.exception("Failed to load history for session_id=%s", session_id)
                raise HTTPException(status_code=500, detail="Failed to load history") from exc

        values = getattr(snapshot, "values", None) or {}
        messages = values.get("messages", [])
        normalized: list[dict[str, str]] = []
        for message in messages:
            kind = message_kind(message)
            normalized.append(
                {
                    "role": kind,
                    "content": _message_to_text(getattr(message, "content", "")),
                }
            )

        return HistoryResponse(session_id=session_id.strip(), messages=normalized)

    @app.post("/api/chat", response_model=ChatResponse)
    def chat(payload: ChatRequest) -> ChatResponse:
        message = sanitize_user_text(payload.message)
        if not message.strip():
            raise HTTPException(status_code=400, detail="message is required")

        session_id = (payload.session_id or "").strip() or _new_session_id(thread_prefix)
        config = {
            "configurable": {"thread_id": session_id},
            "recursion_limit": recursion_limit,
        }
        inputs = {"messages": [HumanMessage(content=message)]}
        latest_ai_text = ""
        ai_order: list[str] = []
        ai_text_by_id: dict[str, str] = {}

        with graph_lock:
            try:
                for event in graph.stream(inputs, config=config, stream_mode="values"):
                    messages = event.get("messages", []) if isinstance(event, dict) else []
                    if not messages:
                        continue
                    last = messages[-1]
                    if isinstance(last, AIMessage) and not last.tool_calls:
                        msg_id = getattr(last, "id", None) or f"{len(messages)}-thought"
                        text = _message_to_text(last.content).strip()
                        if text:
                            prev = ai_text_by_id.get(msg_id, "")
                            if msg_id not in ai_text_by_id:
                                ai_order.append(msg_id)
                            if text != prev:
                                ai_text_by_id[msg_id] = text
                            latest_ai_text = text
            except Exception as exc:
                logger.exception("Failed to process web chat request")
                raise HTTPException(status_code=500, detail="Chat processing failed") from exc

        if ai_order:
            latest_ai_text = "\n\n".join(
                ai_text_by_id[msg_id] for msg_id in ai_order if ai_text_by_id.get(msg_id, "").strip()
            )
        if not latest_ai_text:
            latest_ai_text = "我收到了你的消息，但暂时没有生成可用回复。"
        return ChatResponse(session_id=session_id, reply=latest_ai_text)

    @app.post("/api/chat/stream")
    def chat_stream(payload: ChatRequest) -> StreamingResponse:
        message = sanitize_user_text(payload.message)
        if not message.strip():
            raise HTTPException(status_code=400, detail="message is required")

        session_id = (payload.session_id or "").strip() or _new_session_id(thread_prefix)
        token_queue: queue.Queue[tuple[str, str]] = queue.Queue()
        done_event = threading.Event()
        emitted_text = ""
        token_event_count = 0
        stream_id = 1
        token_buffer = ""
        token_flush_interval = 0.05
        last_token_flush_at = time.monotonic()
        streamed_any_token = False
        current_segment_streamed = False
        seen_status: set[str] = set()
        token_queue.put(("thinking", "Wbot is thinking...."))

        def flush_token_buffer(*, force: bool) -> None:
            nonlocal token_buffer
            nonlocal last_token_flush_at
            nonlocal token_event_count
            nonlocal streamed_any_token
            if not token_buffer:
                return
            now = time.monotonic()
            should_flush = force or ("\n" in token_buffer) or (now - last_token_flush_at >= token_flush_interval)
            if not should_flush:
                return
            token_event_count += 1
            streamed_any_token = True
            payload_json = json.dumps({"text": token_buffer, "stream_id": stream_id}, ensure_ascii=False)
            token_queue.put(("token", payload_json))
            if token_event_count == 1 or token_event_count % 20 == 0:
                logger.info(
                    "Web stream token enqueued: session_id=%s count=%s delta_len=%s stream_id=%s",
                    session_id,
                    token_event_count,
                    len(token_buffer),
                    stream_id,
                )
            token_buffer = ""
            last_token_flush_at = now

        def emit_stream_meta(*, stream_end: bool, resuming: bool) -> None:
            token_queue.put(
                (
                    "stream_meta",
                    json.dumps(
                        {
                            "stream_id": stream_id,
                            "_stream_end": stream_end,
                            "_resuming": resuming,
                        },
                        ensure_ascii=False,
                    ),
                )
            )

        def emit_token(text: str) -> None:
            nonlocal token_buffer
            nonlocal emitted_text
            nonlocal current_segment_streamed
            if text:
                delta = text
                if text.startswith(emitted_text):
                    delta = text[len(emitted_text):]
                if delta:
                    emitted_text += delta
                if delta:
                    current_segment_streamed = True
                    token_buffer += delta
                    flush_token_buffer(force=False)

        def emit_status(text: str) -> None:
            normalized = text.strip()
            if not normalized:
                return
            if normalized in seen_status:
                return
            seen_status.add(normalized)
            logger.info("Web step status: session_id=%s status=%s", session_id, normalized)
            token_queue.put(("status", normalized))

        def worker() -> None:
            nonlocal stream_id
            nonlocal emitted_text
            nonlocal streamed_any_token
            nonlocal current_segment_streamed
            if expose_step_logs:
                emit_status("请求已接收，开始处理。")
            config = {
                "configurable": {
                    "thread_id": session_id,
                    "token_callback": emit_token,
                    "status_callback": emit_status if expose_step_logs else None,
                },
                "recursion_limit": recursion_limit,
            }
            inputs = {"messages": [HumanMessage(content=message)]}
            latest_ai_text = ""
            ai_order: list[str] = []
            ai_text_by_id: dict[str, str] = {}
            seen_tool_call_ids: set[str] = set()
            try:
                with graph_lock:
                    for event in graph.stream(inputs, config=config, stream_mode="values"):
                        messages = event.get("messages", []) if isinstance(event, dict) else []
                        if not messages:
                            continue
                        last = messages[-1]
                        msg_id = str(getattr(last, "id", "") or f"{len(messages)}-{type(last).__name__}")
                        if isinstance(last, AIMessage) and last.tool_calls:
                            if msg_id not in seen_tool_call_ids:
                                seen_tool_call_ids.add(msg_id)
                                flush_token_buffer(force=True)
                                if current_segment_streamed:
                                    emit_stream_meta(stream_end=True, resuming=False)
                                stream_id += 1
                                emitted_text = ""
                                current_segment_streamed = False
                                emit_stream_meta(stream_end=False, resuming=True)
                            if expose_step_logs:
                                emit_status(f"正在调用工具：{_summarize_tool_names(last.tool_calls)}")
                        elif expose_step_logs and isinstance(last, ToolMessage):
                            emit_status("工具调用已返回，正在整合结果。")
                        if isinstance(last, AIMessage) and not last.tool_calls:
                            ai_msg_id = getattr(last, "id", None) or f"{len(messages)}-thought"
                            text = _message_to_text(last.content).strip()
                            if text:
                                prev = ai_text_by_id.get(ai_msg_id, "")
                                if ai_msg_id not in ai_text_by_id:
                                    ai_order.append(ai_msg_id)
                                if text == prev:
                                    continue
                                ai_text_by_id[ai_msg_id] = text
                                latest_ai_text = text
                                if not (streamed_any_token or current_segment_streamed or token_buffer):
                                    payload_json = json.dumps(
                                        {"id": ai_msg_id, "text": text},
                                        ensure_ascii=False,
                                    )
                                    token_queue.put(("ai_message", payload_json))
                if expose_step_logs:
                    emit_status("处理完成。")
            except Exception:
                logger.exception("Failed to process streaming web chat request")
                token_queue.put(("error", "Chat processing failed"))
            finally:
                flush_token_buffer(force=True)
                if ai_order:
                    latest_ai_text = "\n\n".join(
                        ai_text_by_id[msg_id] for msg_id in ai_order if ai_text_by_id.get(msg_id, "").strip()
                    )
                if not latest_ai_text:
                    latest_ai_text = "我收到了你的消息，但暂时没有生成可用回复。"
                if current_segment_streamed:
                    emit_stream_meta(stream_end=True, resuming=False)
                if not streamed_any_token:
                    token_queue.put(
                        (
                            "ai_message",
                            json.dumps({"id": "final", "text": latest_ai_text}, ensure_ascii=False),
                        )
                    )
                token_queue.put(
                    (
                        "done",
                        json.dumps({"reply": latest_ai_text, "_streamed": streamed_any_token}, ensure_ascii=False),
                    )
                )
                done_event.set()

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        async def event_stream() -> Any:
            emitted_any_token = False
            # Some clients/proxies buffer tiny chunks; send an initial SSE comment to force flush.
            yield ": stream-open\n\n"
            yield _sse_event("session", json.dumps({"session_id": session_id}, ensure_ascii=False))
            heartbeat_at = time.monotonic()
            while True:
                try:
                    event, data = token_queue.get_nowait()
                except queue.Empty:
                    if done_event.is_set():
                        break
                    now = time.monotonic()
                    if now - heartbeat_at >= 1.0:
                        heartbeat_at = now
                        yield ": ping\n\n"
                    await asyncio.sleep(0.05)
                    continue

                if event == "token":
                    emitted_any_token = True
                    yield _sse_event("token", data)
                    await asyncio.sleep(0)
                    continue
                if event == "stream_meta":
                    yield _sse_event("stream_meta", data)
                    await asyncio.sleep(0)
                    continue
                if event == "thinking":
                    payload_json = json.dumps({"text": data}, ensure_ascii=False)
                    yield _sse_event("thinking", payload_json)
                    await asyncio.sleep(0)
                    continue
                if event == "status":
                    payload_json = json.dumps({"text": data}, ensure_ascii=False)
                    yield _sse_event("status", payload_json)
                    await asyncio.sleep(0)
                    continue
                if event == "ai_message":
                    yield _sse_event("ai_message", data)
                    await asyncio.sleep(0)
                    continue
                if event == "error":
                    payload_json = json.dumps({"message": data}, ensure_ascii=False)
                    yield _sse_event("error", payload_json)
                    await asyncio.sleep(0)
                    continue
                if event == "done":
                    parsed_done = json.loads(data) if data else {}
                    done_reply = str(parsed_done.get("reply") or "")
                    done_streamed = bool(parsed_done.get("_streamed"))
                    if not emitted_any_token and done_reply.strip():
                        # Fallback: if upstream callback tokens were not observed, still stream in chunks.
                        for chunk in _chunk_text(done_reply, size=18):
                            payload_json = json.dumps({"text": chunk}, ensure_ascii=False)
                            yield _sse_event("token", payload_json)
                            await asyncio.sleep(0.01)
                    payload_json = json.dumps({"reply": done_reply, "_streamed": done_streamed}, ensure_ascii=False)
                    yield _sse_event("done", payload_json)
                    await asyncio.sleep(0)
                    break

        headers = {
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
        return StreamingResponse(event_stream(), media_type="text/event-stream", headers=headers)

    return app


def load_gateway_config(config_path: str) -> GatewayConfig:
    target = Path(config_path)
    if not target.is_absolute():
        target = Path.cwd() / target

    if not target.exists():
        _write_default_config(target)
        raise FileNotFoundError(
            f"Config not found. A template has been generated at: {target}. "
            "Please fill required fields and retry."
        )

    payload = json.loads(target.read_text(encoding="utf-8"))
    channels = payload.get("channels") if isinstance(payload.get("channels"), dict) else {}
    web_raw = channels.get("web") if isinstance(channels.get("web"), dict) else {}
    thread_prefix = str(_pick(payload, "threadPrefix", "thread_prefix", default="web")).strip() or "web"
    return GatewayConfig(web=WebConfig.from_dict(web_raw), thread_prefix=thread_prefix)


def _write_default_config(target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    template = default_app_config()
    target.write_text(json.dumps(template, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _build_llm(settings: Any, *, model_name: str) -> Any:
    from langchain_openai import ChatOpenAI

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


def _pick(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return default


def _safe_port(value: Any) -> int:
    try:
        port = int(value)
    except (TypeError, ValueError):
        return 8000
    return min(65535, max(1, port))


def _new_session_id(thread_prefix: str) -> str:
    prefix = thread_prefix.strip() or "web"
    return f"{prefix}:{int(time.time() * 1000)}"


def _message_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    texts.append(text.strip())
        return "\n".join(texts)
    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            return text
        return str(content)
    return str(content)


def _sse_event(event: str, data: str) -> str:
    return f"event: {event}\ndata: {data}\n\n"


def _summarize_tool_names(tool_calls: list[dict[str, Any]]) -> str:
    names: list[str] = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        name = str(tool_call.get("name") or "").strip()
        if name:
            names.append(name)
    if not names:
        return f"{len(tool_calls)} 个工具"
    return ", ".join(names)


def _chunk_text(text: str, *, size: int) -> list[str]:
    payload = text or ""
    if not payload:
        return []
    return [payload[i: i + size] for i in range(0, len(payload), size)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run W-bot Web gateway")
    parser.add_argument(
        "--config",
        default="configs/app.json",
        help="Path to gateway config JSON (default: configs/app.json)",
    )
    args = parser.parse_args()
    run_web_gateway(config_path=args.config)


if __name__ == "__main__":
    main()
