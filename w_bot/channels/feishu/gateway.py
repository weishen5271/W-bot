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
from w_bot.agents.tools.runtime import build_tools
from w_bot.channels.models import InboundMedia, InboundMessage

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
        """从输入数据构建目标对象。
        
        Args:
            payload: 输入载荷字典，包含请求字段与元数据。
        """
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
        expose_step_logs: bool = True,
    ) -> None:
        """初始化对象并保存运行所需依赖。
        
        Args:
            graph: 对话图执行器实例。
            config: 配置对象或配置字典。
            thread_prefix: 线程前缀，用于生成会话 thread_id。
            media_root_dir: 媒体文件落盘根目录。
        """
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
        self._expose_step_logs = expose_step_logs
        if not self._media_root.is_absolute():
            self._media_root = Path.cwd() / self._media_root
        self._media_root.mkdir(parents=True, exist_ok=True)

    def start(self) -> None:
        """启动对应服务或流程。
        """
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
        """处理事件回调并分发到内部处理逻辑。
        
        Args:
            data: 飞书事件回调原始对象。
        """
        try:
            self._handle_message(data)
        except Exception:
            logger.exception("Failed to handle Feishu inbound message")

    def _handle_message(self, data: Any) -> None:
        """处理指定输入并执行业务分支。
        
        Args:
            data: 已接收的飞书消息事件对象。
        """
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
        """处理ask/agent相关逻辑并返回结果。
        
        Args:
            inbound: 归一化后的入站消息对象。
            session_id: 业务对象唯一标识。
        """
        status_lines: list[str] = []

        def emit_status(text: str) -> None:
            normalized = text.strip()
            if normalized and normalized not in status_lines:
                status_lines.append(normalized)
                logger.info("Feishu step status: session_id=%s status=%s", session_id, normalized)

        config = {
            "configurable": {
                "thread_id": session_id,
                "status_callback": emit_status if self._expose_step_logs else None,
            }
        }
        media_payload = [item.to_dict() for item in inbound.media]
        prompt_text = _build_inbound_prompt_text(inbound)
        inputs = {
            "messages": [
                HumanMessage(
                    content=prompt_text,
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

        if self._expose_step_logs and status_lines:
            status_block = "\n".join(f"- {line}" for line in status_lines)
            if latest_ai_text.strip():
                return f"步骤状态：\n{status_block}\n\n最终回复：\n{latest_ai_text}"
            return f"步骤状态：\n{status_block}"

        return latest_ai_text

    def _send_text(
        self,
        *,
        chat_id: str,
        open_id: str,
        text: str,
        reply_to_message_id: str,
    ) -> None:
        """处理send/text相关逻辑并返回结果。
        
        Args:
            chat_id: 聊天会话 ID。
            open_id: 业务对象唯一标识。
            text: 待处理文本。
            reply_to_message_id: 业务对象唯一标识。
        """
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
        """处理create/message相关逻辑并返回结果。
        
        Args:
            receive_id_type: 接收方 ID 类型（如 chat_id、open_id）。
            receive_id: 业务对象唯一标识。
            content: 消息内容主体。
        """
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
        """处理reply/message相关逻辑并返回结果。
        
        Args:
            message_id: 消息唯一 ID。
            content: 消息内容主体。
        """
        assert self._lark is not None
        assert self._client is not None

        imv1 = self._lark.im.v1
        body = imv1.ReplyMessageRequestBody.builder().content(content).msg_type("text").build()
        request = imv1.ReplyMessageRequest.builder().message_id(message_id).request_body(body).build()
        response = self._client.im.v1.message.reply(request)
        if not self._response_ok(response):
            raise RuntimeError(f"reply failed: {self._response_summary(response)}")

    def _add_reaction(self, *, message_id: str, emoji_type: str) -> None:
        """处理add/reaction相关逻辑并返回结果。
        
        Args:
            message_id: 消息唯一 ID。
            emoji_type: 要添加的表情反应类型。
        """
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
        """处理response/ok相关逻辑并返回结果。
        
        Args:
            response: HTTP 响应对象。
        """
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
        """处理response/summary相关逻辑并返回结果。
        
        Args:
            response: HTTP 响应对象。
        """
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
        """判断条件是否满足。
        
        Args:
            message_id: 消息唯一 ID。
        """
        with self._seen_lock:
            if message_id in self._seen_message_ids:
                return True
            self._seen_message_ids[message_id] = None
            if len(self._seen_message_ids) > 1000:
                self._seen_message_ids.popitem(last=False)
        return False

    def _is_allowed(self, open_id: str) -> bool:
        """判断条件是否满足。
        
        Args:
            open_id: 业务对象唯一标识。
        """
        allow = self._config.allow_from
        if not allow:
            return False
        if "*" in allow:
            return True
        return bool(open_id) and open_id in allow

    @staticmethod
    def _is_new_session_command(text: str) -> bool:
        """判断条件是否满足。
        
        Args:
            text: 待处理文本。
        """
        normalized = text.strip().lower()
        if normalized == "/new":
            return True
        return any(line.strip().lower() == "/new" for line in text.splitlines())

    @staticmethod
    def _extract_text(*, content_raw: Any, message_type: str) -> str:
        """从输入中提取所需信息。
        
        Args:
            content_raw: 原始内容字段（通常为 JSON 字符串）。
            message_type: 消息类型标识。
        """
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
        """从输入中提取所需信息。
        
        Args:
            message_id: 消息唯一 ID。
            message_type: 消息类型标识。
            content_raw: 原始内容字段（通常为 JSON 字符串）。
        """
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
        """处理download/media/item相关逻辑并返回结果。
        
        Args:
            message_id: 消息唯一 ID。
            resource_key: 飞书资源标识，用于下载媒体文件。
            resource_type: 飞书资源类型（如 image/file/audio/video）。
            suggested_name: 建议使用的文件名。
            kind: 媒体类型标识。
            meta: 补充元数据字典。
        """
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
        """处理fetch/app/access/token相关逻辑并返回结果。
        """
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
        """处理download/message/resource相关逻辑并返回结果。
        
        Args:
            message_id: 消息唯一 ID。
            resource_key: 飞书资源标识，用于下载媒体文件。
            resource_type: 飞书资源类型（如 image/file/audio/video）。
            access_token: 飞书接口访问令牌。
        """
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
        """判断条件是否满足。
        
        Args:
            event: 事件对象，包含飞书回调字段。
            text: 待处理文本。
        """
        message = event.get("message") if isinstance(event.get("message"), dict) else {}
        mentions = message.get("mentions")
        if isinstance(mentions, list) and mentions:
            return True

        lowered = text.lower()
        return "@_all" in lowered or "<at " in lowered

    def _as_dict(self, data: Any) -> dict[str, Any]:
        """处理as/dict相关逻辑并返回结果。
        
        Args:
            data: 任意输入对象，尝试转换为字典。
        """
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
    """启动飞书网关并开始处理入站消息。
    
    Args:
        config_path: 目标路径参数，用于定位文件或目录。
    """
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
        ).app

        gateway = FeishuGateway(
            graph=graph,
            config=cfg.feishu,
            thread_prefix=cfg.thread_prefix,
            media_root_dir=settings.multimodal.media_root_dir,
            expose_step_logs=settings.expose_step_logs,
        )
        try:
            gateway.start()
        finally:
            if optimizer_stop_event is not None:
                optimizer_stop_event.set()


def main() -> None:
    """程序主入口。
    """
    parser = argparse.ArgumentParser(description="Run W-bot Feishu gateway")
    parser.add_argument(
        "--config",
        default="configs/app.json",
        help="Path to gateway config JSON (default: configs/app.json)",
    )
    args = parser.parse_args()
    run_feishu_gateway(config_path=args.config)


def load_gateway_config(config_path: str) -> GatewayConfig:
    """加载目标配置或数据并返回。
    
    Args:
        config_path: 目标路径参数，用于定位文件或目录。
    """
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
    """处理write/default/config相关逻辑并返回结果。
    
    Args:
        target: 目标对象或目标路径。
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    template = default_app_config()
    target.write_text(json.dumps(template, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _build_llm(settings: Any, *, model_name: str) -> Any:
    """构建并返回目标对象。
    
    Args:
        settings: 全局设置对象。
        model_name: 当前使用的模型名称。
    """
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model_name,
        api_key=settings.dashscope_api_key,
        base_url=settings.bailian_base_url,
        temperature=0.2,
    )


def _pick(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """处理pick相关逻辑并返回结果。
    
    Args:
        data: 输入字典对象，用于按键名读取配置值。
        keys: 候选键名列表，按顺序尝试读取。
        default: 缺失配置时使用的默认值。
    """
    for key in keys:
        if key in data:
            return data[key]
    return default


def _coerce_list(value: Any) -> list[str]:
    """处理coerce/list相关逻辑并返回结果。
    
    Args:
        value: 待转换或校验的值。
    """
    if isinstance(value, list):
        out = [str(item).strip() for item in value if str(item).strip()]
        return out or ["*"]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return ["*"]


def _sanitize_filename(name: str) -> str:
    """处理sanitize/filename相关逻辑并返回结果。
    
    Args:
        name: 名称参数，用于标识目标对象。
    """
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in name)
    cleaned = cleaned.strip("._")
    return cleaned or "file.bin"


def _guess_kind_from_filename(name: str) -> str:
    """处理guess/kind/from/filename相关逻辑并返回结果。
    
    Args:
        name: 名称参数，用于标识目标对象。
    """
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


def _build_inbound_prompt_text(inbound: InboundMessage) -> str:
    """构建并返回目标对象。
    
    Args:
        inbound: 归一化后的入站消息对象。
    """
    user_text = inbound.content.strip()
    if not inbound.media:
        return user_text

    kinds = sorted({item.kind for item in inbound.media})
    lines = [
        "用户本轮发送了附件，请不要按纯文本聊天方式处理。",
        f"附件类型: {', '.join(kinds)}",
    ]
    for kind in kinds:
        lines.append(f"- {kind}: {_kind_strategy(kind)}")

    if user_text:
        lines.append(f"用户文本诉求: {user_text}")
    else:
        lines.append("用户未提供额外文本诉求，请先基于附件给出处理结果和下一步建议。")
    return "\n".join(lines)


def _kind_strategy(kind: str) -> str:
    """处理kind/strategy相关逻辑并返回结果。
    
    Args:
        kind: 媒体类型标识。
    """
    if kind == "image":
        return "先识别图片关键信息，再结合用户目标执行。"
    if kind == "audio":
        return "先转写/总结音频内容，再执行后续任务。"
    if kind == "video":
        return "先提取视频关键内容（可说明限制），再执行任务。"
    if kind == "document":
        return "先读取并理解文档内容，再进行分析、总结或改写。"
    return "先说明可处理范围，并请求用户给出明确目标。"


def _flatten_post_text(content: dict[str, Any]) -> str:
    """处理flatten/post/text相关逻辑并返回结果。
    
    Args:
        content: 消息内容主体。
    """
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
    """处理message/to/text相关逻辑并返回结果。
    
    Args:
        content: 消息内容主体。
    """
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
