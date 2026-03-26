from __future__ import annotations

import argparse
import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from agents.agent import CyberCoreGraph
from agents.config import default_app_config, load_settings
from agents.logging_config import get_logger, setup_logging
from agents.memory import LongTermMemoryStore
from agents.tools.runtime import build_tools

logger = get_logger(__name__)


@dataclass(frozen=True)
class FeishuConfig:
    enabled: bool
    app_id: str
    app_secret: str
    encrypt_key: str
    verification_token: str
    allow_from: list[str]
    react_emoji: str
    group_policy: str
    reply_to_message: bool

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "FeishuConfig":
        return FeishuConfig(
            enabled=bool(_pick(payload, "enabled", default=False)),
            app_id=str(_pick(payload, "appId", "app_id", default="")).strip(),
            app_secret=str(_pick(payload, "appSecret", "app_secret", default="")).strip(),
            encrypt_key=str(_pick(payload, "encryptKey", "encrypt_key", default="")).strip(),
            verification_token=str(
                _pick(payload, "verificationToken", "verification_token", default="")
            ).strip(),
            allow_from=_coerce_list(_pick(payload, "allowFrom", "allow_from", default=["*"])),
            react_emoji=str(_pick(payload, "reactEmoji", "react_emoji", default="THUMBSUP")).strip(),
            group_policy=str(_pick(payload, "groupPolicy", "group_policy", default="mention")).strip(),
            reply_to_message=bool(
                _pick(payload, "replyToMessage", "reply_to_message", default=True)
            ),
        )


@dataclass(frozen=True)
class GatewayConfig:
    feishu: FeishuConfig
    thread_prefix: str


class FeishuGateway:
    def __init__(self, *, graph: Any, config: FeishuConfig, thread_prefix: str) -> None:
        self._graph = graph
        self._config = config
        self._thread_prefix = thread_prefix
        self._seen_message_ids: OrderedDict[str, None] = OrderedDict()
        self._seen_lock = threading.Lock()
        self._graph_lock = threading.Lock()
        self._client: Any | None = None
        self._lark: Any | None = None

    def start(self) -> None:
        if not self._config.enabled:
            logger.warning("Feishu channel is disabled by config")
            return

        if not self._config.app_id or not self._config.app_secret:
            logger.error("Feishu app_id/app_secret is empty, cannot start")
            return

        try:
            import lark_oapi as lark
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency lark-oapi. Install with: pip install lark-oapi"
            ) from exc

        self._lark = lark
        self._client = lark.Client.builder().app_id(self._config.app_id).app_secret(
            self._config.app_secret
        ).build()

        event_handler = (
            lark.EventDispatcherHandler.builder(
                self._config.verification_token or "",
                self._config.encrypt_key or "",
            )
            .register_p2_im_message_receive_v1(self._on_message)
            .build()
        )

        logger.info("Feishu gateway started with WebSocket long connection")
        while True:
            try:
                ws_client = lark.ws.Client(
                    self._config.app_id,
                    self._config.app_secret,
                    event_handler=event_handler,
                    log_level=lark.LogLevel.INFO,
                )
                ws_client.start()
            except Exception:
                logger.exception("Feishu WebSocket loop crashed; retry after 5 seconds")
                time.sleep(5)

    def _on_message(self, data: Any) -> None:
        try:
            self._handle_message(data)
        except Exception:
            logger.exception("Failed to handle Feishu inbound message")

    def _handle_message(self, data: Any) -> None:
        payload = self._as_dict(data)
        event = payload.get("event", payload)
        if not isinstance(event, dict):
            return

        message = event.get("message") if isinstance(event.get("message"), dict) else {}
        sender = event.get("sender") if isinstance(event.get("sender"), dict) else {}
        sender_id = sender.get("sender_id") if isinstance(sender.get("sender_id"), dict) else {}

        message_id = str(message.get("message_id") or "").strip()
        chat_id = str(message.get("chat_id") or "").strip()
        chat_type = str(message.get("chat_type") or "").strip().lower()
        message_type = str(message.get("message_type") or message.get("msg_type") or "").strip().lower()
        sender_type = str(sender.get("sender_type") or "").strip().lower()
        open_id = str(sender_id.get("open_id") or "").strip()

        if not message_id or not chat_id:
            return

        if self._is_duplicate(message_id):
            logger.debug("Skip duplicated message: %s", message_id)
            return

        if sender_type == "bot":
            logger.debug("Skip bot self message: %s", message_id)
            return

        if not self._is_allowed(open_id):
            logger.info("Skip message due to allow_from rule: open_id=%s", open_id)
            return

        text = self._extract_text(content_raw=message.get("content"), message_type=message_type)
        if not text.strip():
            logger.info("Skip non-text message currently unsupported: type=%s", message_type)
            return

        if chat_type == "group" and self._config.group_policy == "mention":
            if not self._is_group_mentioned(event=event, text=text):
                logger.debug("Skip group message without mention under mention policy")
                return

        if self._config.react_emoji:
            self._add_reaction(message_id=message_id, emoji_type=self._config.react_emoji)

        session_key = chat_id if chat_type == "group" else (open_id or chat_id)
        session_id = f"{self._thread_prefix}:{session_key}"
        reply_text = self._ask_agent(text=text, session_id=session_id)
        if not reply_text.strip():
            reply_text = "我收到了你的消息，但暂时没有生成可用回复。"

        self._send_text(
            chat_id=chat_id,
            open_id=open_id,
            text=reply_text,
            reply_to_message_id=message_id if self._config.reply_to_message else "",
        )

    def _ask_agent(self, *, text: str, session_id: str) -> str:
        config = {"configurable": {"thread_id": session_id}}
        inputs = {"messages": [HumanMessage(content=text)]}
        latest_ai_text = ""

        with self._graph_lock:
            for event in self._graph.stream(inputs, config=config, stream_mode="values"):
                messages = event.get("messages", []) if isinstance(event, dict) else []
                if not messages:
                    continue
                last = messages[-1]
                if isinstance(last, AIMessage) and not last.tool_calls:
                    content = _message_to_text(last.content).strip()
                    if content:
                        latest_ai_text = content

        return latest_ai_text

    def _send_text(
        self,
        *,
        chat_id: str,
        open_id: str,
        text: str,
        reply_to_message_id: str,
    ) -> None:
        if self._client is None or self._lark is None:
            return

        content = json.dumps({"text": text}, ensure_ascii=False)

        if reply_to_message_id:
            try:
                self._reply_message(message_id=reply_to_message_id, content=content)
                return
            except Exception:
                logger.exception("Reply API failed; fallback to message.create")

        receive_id_type = "chat_id" if chat_id.startswith("oc_") else "open_id"
        receive_id = chat_id if receive_id_type == "chat_id" else (open_id or chat_id)
        self._create_message(receive_id_type=receive_id_type, receive_id=receive_id, content=content)

    def _create_message(self, *, receive_id_type: str, receive_id: str, content: str) -> None:
        assert self._lark is not None
        assert self._client is not None

        imv1 = self._lark.im.v1
        body = (
            imv1.CreateMessageRequestBody.builder()
            .receive_id(receive_id)
            .msg_type("text")
            .content(content)
            .build()
        )
        request = (
            imv1.CreateMessageRequest.builder()
            .receive_id_type(receive_id_type)
            .request_body(body)
            .build()
        )
        response = self._client.im.v1.message.create(request)
        if not self._response_ok(response):
            raise RuntimeError(f"send failed: {self._response_summary(response)}")

    def _reply_message(self, *, message_id: str, content: str) -> None:
        assert self._lark is not None
        assert self._client is not None

        imv1 = self._lark.im.v1
        body = imv1.ReplyMessageRequestBody.builder().content(content).msg_type("text").build()
        request = imv1.ReplyMessageRequest.builder().message_id(message_id).request_body(body).build()
        response = self._client.im.v1.message.reply(request)
        if not self._response_ok(response):
            raise RuntimeError(f"reply failed: {self._response_summary(response)}")

    def _add_reaction(self, *, message_id: str, emoji_type: str) -> None:
        if self._client is None or self._lark is None:
            return

        try:
            imv1 = self._lark.im.v1
            body = (
                imv1.CreateMessageReactionRequestBody.builder().reaction_type(emoji_type).build()
            )
            request = (
                imv1.CreateMessageReactionRequest.builder()
                .message_id(message_id)
                .request_body(body)
                .build()
            )
            response = self._client.im.v1.message_reaction.create(request)
            if not self._response_ok(response):
                logger.debug("Create reaction failed: %s", self._response_summary(response))
        except Exception:
            logger.debug("Create reaction API not available in current lark-oapi version", exc_info=True)

    @staticmethod
    def _response_ok(response: Any) -> bool:
        if response is None:
            return False
        if hasattr(response, "success") and callable(response.success):
            try:
                return bool(response.success())
            except Exception:
                return False
        code = getattr(response, "code", None)
        if code is None and hasattr(response, "get_code") and callable(response.get_code):
            try:
                code = response.get_code()
            except Exception:
                code = None
        return code == 0

    @staticmethod
    def _response_summary(response: Any) -> str:
        if response is None:
            return "response=None"
        code = getattr(response, "code", None)
        if code is None and hasattr(response, "get_code") and callable(response.get_code):
            try:
                code = response.get_code()
            except Exception:
                code = None
        msg = getattr(response, "msg", None)
        if msg is None and hasattr(response, "get_msg") and callable(response.get_msg):
            try:
                msg = response.get_msg()
            except Exception:
                msg = None
        return f"code={code}, msg={msg}"

    def _is_duplicate(self, message_id: str) -> bool:
        with self._seen_lock:
            if message_id in self._seen_message_ids:
                return True
            self._seen_message_ids[message_id] = None
            if len(self._seen_message_ids) > 1000:
                self._seen_message_ids.popitem(last=False)
        return False

    def _is_allowed(self, open_id: str) -> bool:
        allow = self._config.allow_from
        if not allow:
            return False
        if "*" in allow:
            return True
        return bool(open_id) and open_id in allow

    @staticmethod
    def _extract_text(*, content_raw: Any, message_type: str) -> str:
        if not isinstance(content_raw, str) or not content_raw.strip():
            return ""

        try:
            content = json.loads(content_raw)
        except json.JSONDecodeError:
            return content_raw

        if not isinstance(content, dict):
            return str(content)

        if message_type == "text":
            return str(content.get("text") or "")

        if message_type == "post":
            return _flatten_post_text(content)

        return str(content)

    @staticmethod
    def _is_group_mentioned(*, event: dict[str, Any], text: str) -> bool:
        message = event.get("message") if isinstance(event.get("message"), dict) else {}
        mentions = message.get("mentions")
        if isinstance(mentions, list) and mentions:
            return True

        lowered = text.lower()
        return "@_all" in lowered or "<at " in lowered

    def _as_dict(self, data: Any) -> dict[str, Any]:
        if isinstance(data, dict):
            return data

        if self._lark is not None:
            try:
                raw = self._lark.JSON.marshal(data)
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass

        if hasattr(data, "__dict__"):
            payload = data.__dict__
            if isinstance(payload, dict):
                return payload

        return {}


def run_feishu_gateway(config_path: str = "configs/app.json") -> None:
    setup_logging()

    cfg = load_gateway_config(config_path)
    if not cfg.feishu.enabled:
        logger.warning("Feishu config loaded but channels.feishu.enabled=false")

    settings = load_settings(config_path=config_path)
    logger.info("Building graph for Feishu gateway")

    try:
        from langgraph.checkpoint.postgres import PostgresSaver
    except ImportError as exc:  # pragma: no cover - environment dependent
        logger.exception("Failed to import PostgresSaver")
        raise RuntimeError(
            "PostgresSaver import failed. Please install psycopg first: pip install 'psycopg[binary]'"
        ) from exc

    llm = _build_llm(settings)
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

    with PostgresSaver.from_conn_string(settings.postgres_dsn) as checkpointer:
        if hasattr(checkpointer, "setup"):
            checkpointer.setup()

        graph = CyberCoreGraph(
            llm=llm,
            tools=tools,
            memory_store=memory_store,
            retrieve_top_k=settings.retrieve_top_k,
            user_id=settings.user_id,
            checkpointer=checkpointer,
        ).app

        gateway = FeishuGateway(graph=graph, config=cfg.feishu, thread_prefix=cfg.thread_prefix)
        gateway.start()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CyberCore Feishu gateway")
    parser.add_argument(
        "--config",
        default="configs/app.json",
        help="Path to gateway config JSON (default: configs/app.json)",
    )
    args = parser.parse_args()
    run_feishu_gateway(config_path=args.config)


def load_gateway_config(config_path: str) -> GatewayConfig:
    target = Path(config_path)
    if not target.is_absolute():
        target = Path.cwd() / target

    if not target.exists():
        _write_default_config(target)
        raise FileNotFoundError(
            f"Config not found. A template has been generated at: {target}. "
            "Please fill appId/appSecret and retry."
        )

    payload = json.loads(target.read_text(encoding="utf-8"))
    channels = payload.get("channels") if isinstance(payload.get("channels"), dict) else {}
    feishu_raw = channels.get("feishu") if isinstance(channels.get("feishu"), dict) else {}
    thread_prefix = str(_pick(payload, "threadPrefix", "thread_prefix", default="feishu")).strip()
    if not thread_prefix:
        thread_prefix = "feishu"

    return GatewayConfig(feishu=FeishuConfig.from_dict(feishu_raw), thread_prefix=thread_prefix)


def _write_default_config(target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    template = default_app_config()
    target.write_text(json.dumps(template, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _build_llm(settings: Any) -> Any:
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.bailian_model_name,
        api_key=settings.dashscope_api_key,
        base_url=settings.bailian_base_url,
        temperature=0.2,
    )


def _pick(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return default


def _coerce_list(value: Any) -> list[str]:
    if isinstance(value, list):
        out = [str(item).strip() for item in value if str(item).strip()]
        return out or ["*"]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return ["*"]


def _flatten_post_text(content: dict[str, Any]) -> str:
    post = content.get("post") if isinstance(content.get("post"), dict) else content
    if not isinstance(post, dict):
        return ""

    locale_payload = post.get("zh_cn") or post.get("en_us")
    if not isinstance(locale_payload, dict):
        return ""

    lines = locale_payload.get("content")
    if not isinstance(lines, list):
        return ""

    texts: list[str] = []
    for line in lines:
        if not isinstance(line, list):
            continue
        for part in line:
            if not isinstance(part, dict):
                continue
            txt = part.get("text")
            if isinstance(txt, str) and txt.strip():
                texts.append(txt.strip())
    return "\n".join(texts)


def _message_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content)


if __name__ == "__main__":
    main()
