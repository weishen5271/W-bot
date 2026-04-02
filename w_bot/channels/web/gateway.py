from __future__ import annotations

import argparse
import asyncio
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from w_bot.agents.agent import message_kind
from w_bot.agents.agent import WBotGraph
from w_bot.agents.config import default_app_config, load_settings
from w_bot.agents.escalation import EscalationManager, EscalationRequest
from w_bot.agents.file_checkpointer import WorkspaceFileCheckpointer, resolve_short_term_memory_path
from w_bot.agents.logging_config import get_logger, setup_logging
from w_bot.agents.memory import LongTermMemoryStore
from w_bot.agents.openclaw_profile import OpenClawProfileLoader
from w_bot.agents.skills import SkillsLoader
from w_bot.agents.streaming import StreamTextAssembler, latest_non_tool_ai_reply, normalize_display_text
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


class EscalationActionRequest(BaseModel):
    session_id: str
    request_id: str
    reason: str | None = None


class EscalationItemResponse(BaseModel):
    id: str
    session_id: str
    status: str
    risk_type: str
    command: str
    working_dir: str
    justification: str
    prefix_rule: list[str]
    created_at: str
    updated_at: str
    denial_reason: str


def run_web_gateway(config_path: str = "configs/app.json") -> None:
    settings = load_settings(config_path=config_path)
    setup_logging(enable_console_logs=settings.enable_console_logs)
    cfg = load_gateway_config(config_path)
    logger.info("Building graph for Web gateway")

    if not cfg.web.enabled:
        logger.warning("Web config loaded but channels.web.enabled=false")

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
    )

    short_term_memory_path = resolve_short_term_memory_path(settings.short_term_memory_path)
    if settings.short_term_memory_optimization.enabled:
        logger.warning("shortTermMemoryOptimization is ignored in workspace file mode")

    with WorkspaceFileCheckpointer(short_term_memory_path) as checkpointer:
        if hasattr(checkpointer, "setup"):
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

        app = _build_app(
            graph=graph,
            thread_prefix=cfg.thread_prefix,
            expose_step_logs=settings.expose_step_logs,
            recursion_limit=settings.loop_guard.recursion_limit,
            escalation_manager=escalation_manager,
        )
        uvicorn.run(app, host=cfg.web.host, port=cfg.web.port, log_level="info")


def _build_app(
    *,
    graph: Any,
    thread_prefix: str,
    expose_step_logs: bool,
    recursion_limit: int,
    escalation_manager: EscalationManager,
) -> FastAPI:
    app = FastAPI(title="W-bot Web Gateway")
    session_locks: dict[str, threading.Lock] = {}
    session_locks_guard = threading.Lock()
    static_root = Path(__file__).resolve().parent / "static"
    index_path = static_root / "index.html"

    def _session_lock(session_id: str) -> threading.Lock:
        normalized = session_id.strip() or "_default"
        with session_locks_guard:
            lock = session_locks.get(normalized)
            if lock is None:
                lock = threading.Lock()
                session_locks[normalized] = lock
            return lock

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
        normalized_session_id = session_id.strip()
        config = {"configurable": {"thread_id": normalized_session_id}}

        with _session_lock(normalized_session_id):
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
            "configurable": {
                "thread_id": session_id,
                "defer_summary_update": True,
            },
            "recursion_limit": recursion_limit,
        }
        inputs = {"messages": [HumanMessage(content=message)]}

        with _session_lock(session_id):
            try:
                result = graph.invoke(inputs, config=config)
            except Exception as exc:
                logger.exception("Failed to process web chat request")
                raise HTTPException(status_code=500, detail="Chat processing failed") from exc

        latest_ai_text = _latest_ai_reply_from_result(result)
        if not latest_ai_text:
            latest_ai_text = "我收到了你的消息，但暂时没有生成可用回复。"
        return ChatResponse(session_id=session_id, reply=latest_ai_text)

    @app.get("/api/escalations", response_model=list[EscalationItemResponse])
    def list_escalations(session_id: str, status: str | None = None) -> list[EscalationItemResponse]:
        if not session_id.strip():
            raise HTTPException(status_code=400, detail="session_id is required")
        items = escalation_manager.list_requests(session_id=session_id.strip(), status=status, limit=20)
        return [_serialize_escalation_item(item) for item in items]

    @app.post("/api/escalations/approve", response_model=EscalationItemResponse)
    def approve_escalation(payload: EscalationActionRequest) -> EscalationItemResponse:
        session_id = payload.session_id.strip()
        request_id = payload.request_id.strip()
        if not session_id or not request_id:
            raise HTTPException(status_code=400, detail="session_id and request_id are required")
        request = escalation_manager.get_request(request_id)
        if request is None or request.session_id != session_id:
            raise HTTPException(status_code=404, detail="Escalation request not found")
        approved = escalation_manager.approve_request(request_id=request_id)
        if approved is None:
            raise HTTPException(status_code=500, detail="Failed to approve escalation request")
        return _serialize_escalation_item(approved)

    @app.post("/api/escalations/deny", response_model=EscalationItemResponse)
    def deny_escalation(payload: EscalationActionRequest) -> EscalationItemResponse:
        session_id = payload.session_id.strip()
        request_id = payload.request_id.strip()
        if not session_id or not request_id:
            raise HTTPException(status_code=400, detail="session_id and request_id are required")
        request = escalation_manager.get_request(request_id)
        if request is None or request.session_id != session_id:
            raise HTTPException(status_code=404, detail="Escalation request not found")
        denied = escalation_manager.deny_request(request_id=request_id, reason=payload.reason or "")
        if denied is None:
            raise HTTPException(status_code=500, detail="Failed to deny escalation request")
        return _serialize_escalation_item(denied)

    @app.post("/api/chat/stream")
    def chat_stream(payload: ChatRequest) -> StreamingResponse:
        message = sanitize_user_text(payload.message)
        if not message.strip():
            raise HTTPException(status_code=400, detail="message is required")

        session_id = (payload.session_id or "").strip() or _new_session_id(thread_prefix)
        async def event_stream() -> Any:
            loop = asyncio.get_running_loop()
            event_queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
            worker_done = threading.Event()
            event_queue.put_nowait(("thinking", "Wbot is thinking...."))

            def enqueue_event(event_name: str, payload: str) -> None:
                loop.call_soon_threadsafe(event_queue.put_nowait, (event_name, payload))

            def worker() -> None:
                stream_assemblers = {
                    "reasoning": StreamTextAssembler(),
                    "answer": StreamTextAssembler(),
                }
                token_event_count = 0
                stream_id = 1
                token_buffer = ""
                token_flush_interval = 0.015
                last_token_flush_at = time.monotonic()
                streamed_any_token = False
                seen_status: set[str] = set()

                def flush_token_buffer(*, force: bool) -> None:
                    nonlocal token_buffer
                    nonlocal last_token_flush_at
                    nonlocal token_event_count
                    nonlocal streamed_any_token
                    if not token_buffer:
                        return
                    now = time.monotonic()
                    should_flush = force or ("\n" in token_buffer) or (
                        now - last_token_flush_at >= token_flush_interval
                    )
                    if not should_flush:
                        return
                    token_event_count += 1
                    streamed_any_token = True
                    payload_json = json.dumps(
                        {"tokens": [json.loads(line) for line in token_buffer.splitlines() if line.strip()], "stream_id": stream_id},
                        ensure_ascii=False,
                    )
                    enqueue_event("token", payload_json)
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
                    enqueue_event(
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

                def emit_token(token: Any) -> None:
                    nonlocal token_buffer
                    kind = "answer"
                    text = token
                    if isinstance(token, dict):
                        kind = str(token.get("kind") or "answer").strip().lower() or "answer"
                        text = token.get("text") or ""
                    if not text:
                        return
                    assembler = stream_assemblers.get(kind)
                    if assembler is None:
                        assembler = StreamTextAssembler()
                        stream_assemblers[kind] = assembler
                    delta = assembler.consume(text)
                    if not delta:
                        return
                    token_buffer += json.dumps({"text": delta, "kind": kind}, ensure_ascii=False) + "\n"
                    should_force_flush = ("\n" in delta) or any(mark in delta for mark in ("。", "！", "？", ".", "!", "?"))
                    flush_token_buffer(force=should_force_flush)

                def emit_status(text: str) -> None:
                    normalized = text.strip()
                    if not normalized or normalized in seen_status:
                        return
                    seen_status.add(normalized)
                    logger.info("Web step status: session_id=%s status=%s", session_id, normalized)
                    enqueue_event("status", normalized)

                if expose_step_logs:
                    emit_status("请求已接收，开始处理。")
                config = {
                    "configurable": {
                        "thread_id": session_id,
                        "token_callback": emit_token,
                        "status_callback": emit_status if expose_step_logs else None,
                        "defer_summary_update": True,
                    },
                    "recursion_limit": recursion_limit,
                }
                inputs = {"messages": [HumanMessage(content=message)]}
                try:
                    with _session_lock(session_id):
                        result = graph.invoke(inputs, config=config)
                    if expose_step_logs:
                        emit_status("处理完成。")
                except Exception:
                    logger.exception("Failed to process streaming web chat request")
                    enqueue_event("error", "Chat processing failed")
                    result = None
                finally:
                    flush_token_buffer(force=True)
                    latest_ai_text = _latest_ai_reply_from_result(result)
                    if not latest_ai_text:
                        latest_ai_text = "我收到了你的消息，但暂时没有生成可用回复。"
                    if not streamed_any_token:
                        enqueue_event(
                            "ai_message",
                            json.dumps({"id": "final", "text": latest_ai_text}, ensure_ascii=False),
                        )
                    enqueue_event(
                        "done",
                        json.dumps(
                            {"reply": latest_ai_text, "_streamed": streamed_any_token},
                            ensure_ascii=False,
                        ),
                    )
                    worker_done.set()

            thread = threading.Thread(target=worker, daemon=True)
            thread.start()

            emitted_any_token = False
            # Some clients/proxies buffer tiny chunks; send an initial SSE comment to force flush.
            yield ": stream-open\n\n"
            yield _sse_event("session", json.dumps({"session_id": session_id}, ensure_ascii=False))
            heartbeat_at = time.monotonic()
            while True:
                try:
                    event, data = await asyncio.wait_for(event_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    if worker_done.is_set():
                        break
                    now = time.monotonic()
                    if now - heartbeat_at >= 1.0:
                        heartbeat_at = now
                        yield ": ping\n\n"
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
                            payload_json = json.dumps({"text": chunk, "kind": "answer"}, ensure_ascii=False)
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
        return normalize_display_text(content)
    if isinstance(content, list):
        texts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    texts.append(text.strip())
        return normalize_display_text("\n".join(texts))
    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            return normalize_display_text(text)
        return normalize_display_text(str(content))
    return normalize_display_text(str(content))


def _sse_event(event: str, data: str) -> str:
    return f"event: {event}\ndata: {data}\n\n"


def _latest_ai_reply_from_result(result: Any) -> str:
    values = result if isinstance(result, dict) else {}
    messages = values.get("messages", []) if isinstance(values.get("messages", []), list) else []
    return latest_non_tool_ai_reply(messages, content_to_text=_message_to_text)


def _chunk_text(text: str, *, size: int) -> list[str]:
    payload = text or ""
    if not payload:
        return []
    return [payload[i: i + size] for i in range(0, len(payload), size)]


def _serialize_escalation_item(item: EscalationRequest) -> EscalationItemResponse:
    return EscalationItemResponse(
        id=item.id,
        session_id=item.session_id,
        status=item.status,
        risk_type=item.risk_type,
        command=item.command,
        working_dir=item.working_dir,
        justification=item.justification,
        prefix_rule=item.prefix_rule,
        created_at=item.created_at,
        updated_at=item.updated_at,
        denial_reason=item.denial_reason,
    )


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
