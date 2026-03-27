from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from agents.agent import WBotGraph
from agents.config import default_app_config, load_settings
from agents.logging_config import get_logger, setup_logging
from agents.memory import LongTermMemoryStore
from agents.skills import SkillsLoader
from agents.tools.runtime import build_tools
from channels.models import InboundMedia, InboundMessage

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
    def __init__(
        self,
        *,
        graph: Any,
        config: FeishuConfig,
        thread_prefix: str,
        media_root_dir: str = "media",
    ) -> None:
        self._graph = graph
        self._config = config
        self._thread_prefix = thread_prefix
        self._seen_message_ids: OrderedDict[str, None] = OrderedDict()
        self._seen_lock = threading.Lock()
        self._graph_lock = threading.Lock()
        self._session_lock = threading.Lock()
        self._session_overrides: dict[str, str] = {}
        self._client: Any | None = None
        self._lark: Any | None = None
        self._media_root = Path(media_root_dir).expanduser()
        if not self._media_root.is_absolute():
            self._media_root = Path.cwd() / self._media_root
        self._media_root.mkdir(parents=True, exist_ok=True)

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
        media = self._extract_media(
            message_id=message_id,
            message_type=message_type,
            content_raw=message.get("content"),
        )
        if not text.strip() and not media:
            logger.info("Skip empty/unsupported message: type=%s", message_type)
            return

        if chat_type == "group" and self._config.group_policy == "mention":
            if not self._is_group_mentioned(event=event, text=text):
                logger.debug("Skip group message without mention under mention policy")
                return

        if self._config.react_emoji:
            self._add_reaction(message_id=message_id, emoji_type=self._config.react_emoji)

        session_key = chat_id if chat_type == "group" else (open_id or chat_id)
        if self._is_new_session_command(text):
            session_id = f"{self._thread_prefix}:{session_key}:{int(time.time() * 1000)}"
            with self._session_lock:
                self._session_overrides[session_key] = session_id
            self._send_text(
                chat_id=chat_id,
                open_id=open_id,
                text=f"已为当前会话创建新上下文：`{session_id}`",
                reply_to_message_id=message_id if self._config.reply_to_message else "",
            )
            return

        with self._session_lock:
            session_id = self._session_overrides.get(
                session_key,
                f"{self._thread_prefix}:{session_key}",
            )
        inbound = InboundMessage(content=text, media=media)
        reply_text = self._ask_agent(inbound=inbound, session_id=session_id)
        if not reply_text.strip():
            reply_text = "我收到了你的消息，但暂时没有生成可用回复。"

        self._send_text(
            chat_id=chat_id,
            open_id=open_id,
            text=reply_text,
            reply_to_message_id=message_id if self._config.reply_to_message else "",
        )

    def _ask_agent(self, *, inbound: InboundMessage, session_id: str) -> str:
        config = {"configurable": {"thread_id": session_id}}
        media_payload = [item.to_dict() for item in inbound.media]
        inputs = {
            "messages": [
                HumanMessage(
                    content=inbound.content or "",
                    additional_kwargs={"media": media_payload} if media_payload else {},
                )
            ]
        }
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
    def _is_new_session_command(text: str) -> bool:
        normalized = text.strip().lower()
        if normalized == "/new":
            return True
        return any(line.strip().lower() == "/new" for line in text.splitlines())

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

    def _extract_media(
        self,
        *,
        message_id: str,
        message_type: str,
        content_raw: Any,
    ) -> list[InboundMedia]:
        if not isinstance(content_raw, str) or not content_raw.strip():
            return []

        try:
            content = json.loads(content_raw)
        except json.JSONDecodeError:
            return []

        if not isinstance(content, dict):
            return []

        message_type = message_type.lower().strip()
        if message_type == "image":
            image_key = str(content.get("image_key") or "").strip()
            if not image_key:
                return []
            item = self._download_media_item(
                message_id=message_id,
                resource_key=image_key,
                resource_type="image",
                suggested_name=f"{image_key}.png",
                kind="image",
                meta={},
            )
            return [item] if item is not None else []

        if message_type == "audio":
            file_key = str(content.get("file_key") or "").strip()
            if not file_key:
                return []
            item = self._download_media_item(
                message_id=message_id,
                resource_key=file_key,
                resource_type="audio",
                suggested_name=str(content.get("file_name") or f"{file_key}.mp3"),
                kind="audio",
                meta={"duration": content.get("duration")},
            )
            return [item] if item is not None else []

        if message_type in {"file", "media"}:
            file_key = str(content.get("file_key") or "").strip()
            if not file_key:
                return []
            filename = str(content.get("file_name") or content.get("name") or f"{file_key}.bin")
            kind = _guess_kind_from_filename(filename)
            item = self._download_media_item(
                message_id=message_id,
                resource_key=file_key,
                resource_type="file",
                suggested_name=filename,
                kind=kind,
                meta={},
            )
            return [item] if item is not None else []

        if message_type == "video":
            file_key = str(content.get("file_key") or "").strip()
            if not file_key:
                return []
            item = self._download_media_item(
                message_id=message_id,
                resource_key=file_key,
                resource_type="video",
                suggested_name=str(content.get("file_name") or f"{file_key}.mp4"),
                kind="video",
                meta={"duration": content.get("duration")},
            )
            return [item] if item is not None else []

        return []

    def _download_media_item(
        self,
        *,
        message_id: str,
        resource_key: str,
        resource_type: str,
        suggested_name: str,
        kind: str,
        meta: dict[str, Any],
    ) -> InboundMedia | None:
        token = self._fetch_app_access_token()
        if not token:
            logger.warning("Cannot download media because app access token is unavailable")
            return None

        content, mime = self._download_message_resource(
            message_id=message_id,
            resource_key=resource_key,
            resource_type=resource_type,
            access_token=token,
        )
        if content is None:
            return None

        target = self._media_root / time.strftime("%Y%m%d")
        target.mkdir(parents=True, exist_ok=True)
        safe_name = _sanitize_filename(suggested_name)
        file_path = target / f"{uuid.uuid4().hex}_{safe_name}"
        file_path.write_bytes(content)

        digest = hashlib.sha256(content).hexdigest()
        return InboundMedia(
            id=resource_key,
            path=str(file_path),
            mime=mime,
            kind=kind,
            size_bytes=len(content),
            sha256=digest,
            meta=dict(meta),
        )

    def _fetch_app_access_token(self) -> str:
        url = "https://open.feishu.cn/open-apis/auth/v3/app_access_token/internal"
        payload = {
            "app_id": self._config.app_id,
            "app_secret": self._config.app_secret,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url=url,
            method="POST",
            data=data,
            headers={"Content-Type": "application/json; charset=utf-8"},
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                body = resp.read().decode("utf-8")
        except (urllib.error.URLError, TimeoutError):
            logger.exception("Failed to fetch Feishu app access token")
            return ""

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            logger.error("Invalid token response from Feishu")
            return ""

        if int(parsed.get("code", -1)) != 0:
            logger.error("Token API returned error: code=%s msg=%s", parsed.get("code"), parsed.get("msg"))
            return ""
        token = parsed.get("app_access_token")
        return str(token or "")

    def _download_message_resource(
        self,
        *,
        message_id: str,
        resource_key: str,
        resource_type: str,
        access_token: str,
    ) -> tuple[bytes | None, str]:
        encoded_key = urllib.parse.quote(resource_key, safe="")
        encoded_message_id = urllib.parse.quote(message_id, safe="")
        query = urllib.parse.urlencode({"type": resource_type})
        url = (
            "https://open.feishu.cn/open-apis/im/v1/messages/"
            f"{encoded_message_id}/resources/{encoded_key}?{query}"
        )
        req = urllib.request.Request(
            url=url,
            method="GET",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                body = resp.read()
                mime = resp.headers.get("Content-Type", "application/octet-stream")
                return body, mime
        except (urllib.error.URLError, TimeoutError):
            logger.exception(
                "Failed to download Feishu media resource, message_id=%s, key=%s, type=%s",
                message_id,
                resource_key,
                resource_type,
            )
            return None, "application/octet-stream"

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

    with PostgresSaver.from_conn_string(settings.postgres_dsn) as checkpointer:
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
            multimodal_settings=settings.multimodal,
            model_name=settings.model_routing.text_model_name,
            llm_image=llm_image,
            llm_audio=llm_audio,
            image_model_name=settings.model_routing.image_model_name,
            audio_model_name=settings.model_routing.audio_model_name,
        ).app

        gateway = FeishuGateway(
            graph=graph,
            config=cfg.feishu,
            thread_prefix=cfg.thread_prefix,
            media_root_dir=settings.multimodal.media_root_dir,
        )
        gateway.start()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run W-bot Feishu gateway")
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


def _build_llm(settings: Any, *, model_name: str) -> Any:
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model_name,
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


def _sanitize_filename(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in name)
    cleaned = cleaned.strip("._")
    return cleaned or "file.bin"


def _guess_kind_from_filename(name: str) -> str:
    lowered = name.lower()
    if lowered.endswith((".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")):
        return "image"
    if lowered.endswith((".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac", ".opus")):
        return "audio"
    if lowered.endswith((".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")):
        return "video"
    if lowered.endswith(
        (".txt", ".md", ".json", ".csv", ".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx")
    ):
        return "document"
    return "other"


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
