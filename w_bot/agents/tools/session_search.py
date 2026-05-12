from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from ..core.logging_config import get_logger
from ..core.message_utils import _to_text_content
from .base import Tool

logger = get_logger(__name__)

MAX_SESSION_CHARS = 100_000
_HIDDEN_SESSION_SOURCES = ("tool",)


class SessionSearchTool(Tool):
    def __init__(self, db: Any) -> None:
        self._db = db

    @property
    def name(self) -> str:
        return "session_search"

    @property
    def description(self) -> str:
        return (
            "检索历史会话，或在 query 为空时浏览最近会话。"
            "当用户提到上次、之前、继续某个历史问题，或需要确认过去怎么处理过类似任务时主动使用。"
            "关键词检索支持 OR、短语和前缀查询，返回按会话聚合后的摘要。"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词；留空时返回最近会话列表。",
                },
                "role_filter": {
                    "type": "string",
                    "description": "可选，逗号分隔角色过滤，例如 user,assistant。",
                },
                "limit": {
                    "type": "integer",
                    "description": "最多返回会话数，默认 3，最大 5。",
                    "default": 3,
                    "minimum": 1,
                    "maximum": 5,
                },
            },
            "required": [],
        }

    async def execute(
        self,
        query: str = "",
        role_filter: str | None = None,
        limit: int = 3,
        _wbot_tool_context: dict[str, Any] | None = None,
    ) -> str:
        current_session_id = ""
        graph = None
        if isinstance(_wbot_tool_context, dict):
            current_session_id = str(_wbot_tool_context.get("thread_id") or "").strip()
            graph = _wbot_tool_context.get("graph")

        normalized_limit = min(5, max(1, int(limit or 3)))
        if not query or not str(query).strip():
            return self._list_recent_sessions(normalized_limit, current_session_id)

        role_list = _parse_role_filter(role_filter)
        try:
            raw_results = self._db.search_messages(
                query=str(query).strip(),
                role_filter=role_list,
                exclude_sources=list(_HIDDEN_SESSION_SOURCES),
                limit=50,
                offset=0,
            )
        except Exception as exc:
            logger.exception("Session search failed")
            return _json_error(f"Search failed: {exc}")

        if not raw_results:
            return json.dumps(
                {
                    "success": True,
                    "query": str(query).strip(),
                    "results": [],
                    "count": 0,
                    "message": "未找到匹配的历史会话。",
                },
                ensure_ascii=False,
            )

        current_root = self._resolve_to_parent(current_session_id) if current_session_id else ""
        seen: dict[str, dict[str, Any]] = {}
        for item in raw_results:
            raw_sid = str(item.get("session_id") or "")
            resolved_sid = self._resolve_to_parent(raw_sid)
            if not resolved_sid:
                continue
            if current_root and resolved_sid == current_root:
                continue
            if current_session_id and raw_sid == current_session_id:
                continue
            if resolved_sid not in seen:
                copied = dict(item)
                copied["session_id"] = resolved_sid
                seen[resolved_sid] = copied
            if len(seen) >= normalized_limit:
                break

        tasks: list[tuple[str, dict[str, Any], str, dict[str, Any]]] = []
        for session_id, match_info in seen.items():
            try:
                messages = self._db.get_messages_as_conversation(session_id)
                if not messages:
                    continue
                session_meta = self._db.get_session(session_id) or {}
                transcript = _truncate_around_matches(
                    _format_conversation(messages),
                    str(query),
                )
                tasks.append((session_id, match_info, transcript, session_meta))
            except Exception:
                logger.warning("Failed to prepare session search result: %s", session_id, exc_info=True)

        summaries = await asyncio.gather(
            *(
                self._summarize_session(graph=graph, transcript=transcript, query=str(query), meta=meta)
                for _, _, transcript, meta in tasks
            ),
            return_exceptions=True,
        )

        results: list[dict[str, Any]] = []
        for (session_id, match_info, transcript, _), summary in zip(tasks, summaries):
            entry = {
                "session_id": session_id,
                "when": _format_timestamp(match_info.get("session_started")),
                "source": match_info.get("source", "unknown"),
                "model": match_info.get("model"),
            }
            if isinstance(summary, BaseException):
                logger.warning("Session summarization failed: %s", summary)
                summary = ""
            if summary:
                entry["summary"] = str(summary)
            else:
                preview = transcript[:500] + ("\n...[truncated]" if len(transcript) > 500 else "")
                entry["summary"] = f"[原始预览：摘要不可用]\n{preview or '无可用预览。'}"
            results.append(entry)

        return json.dumps(
            {
                "success": True,
                "query": str(query).strip(),
                "results": results,
                "count": len(results),
                "sessions_searched": len(seen),
            },
            ensure_ascii=False,
        )

    def _list_recent_sessions(self, limit: int, current_session_id: str = "") -> str:
        try:
            sessions = self._db.list_sessions_rich(
                limit=limit + 5,
                exclude_sources=list(_HIDDEN_SESSION_SOURCES),
            )
        except Exception as exc:
            logger.exception("Failed to list recent sessions")
            return _json_error(f"Failed to list recent sessions: {exc}")

        current_root = self._resolve_to_parent(current_session_id) if current_session_id else ""
        results: list[dict[str, Any]] = []
        for item in sessions:
            session_id = str(item.get("id") or "")
            if current_root and session_id == current_root:
                continue
            if current_session_id and session_id == current_session_id:
                continue
            if item.get("parent_session_id"):
                continue
            results.append(
                {
                    "session_id": session_id,
                    "title": item.get("title") or None,
                    "source": item.get("source", ""),
                    "started_at": _format_timestamp(item.get("started_at")),
                    "last_active": _format_timestamp(item.get("last_active")),
                    "message_count": item.get("message_count", 0),
                    "preview": item.get("preview", ""),
                }
            )
            if len(results) >= limit:
                break

        return json.dumps(
            {
                "success": True,
                "mode": "recent",
                "results": results,
                "count": len(results),
                "message": f"已返回 {len(results)} 个最近会话；传入 query 可搜索具体主题。",
            },
            ensure_ascii=False,
        )

    def _resolve_to_parent(self, session_id: str) -> str:
        visited: set[str] = set()
        sid = session_id
        while sid and sid not in visited:
            visited.add(sid)
            session = self._db.get_session(sid)
            if not session:
                break
            parent = session.get("parent_session_id")
            if not parent:
                break
            sid = str(parent)
        return sid

    async def _summarize_session(
        self,
        *,
        graph: Any,
        transcript: str,
        query: str,
        meta: dict[str, Any],
    ) -> str:
        llm = getattr(graph, "session_search_llm", None) if graph is not None else None
        if llm is None:
            return ""
        system_prompt = (
            "你正在回顾一段历史会话，帮助主 Agent 准确回忆过去发生过什么。"
            "请围绕搜索主题输出中文事实摘要，覆盖：用户目标、采取的操作、结果、关键决策、"
            "重要命令/文件/错误信息，以及未解决事项。不要编造 transcript 中没有的信息。"
        )
        user_prompt = (
            f"搜索主题：{query}\n"
            f"会话来源：{meta.get('source', 'unknown')}\n"
            f"会话时间：{_format_timestamp(meta.get('started_at'))}\n\n"
            f"历史会话 transcript：\n{transcript}\n\n"
            "请输出面向后续继续工作的摘要："
        )
        try:
            result = await asyncio.to_thread(
                llm.invoke,
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ],
            )
        except Exception:
            logger.debug("Session search summarizer unavailable", exc_info=True)
            return ""
        return _to_text_content(getattr(result, "content", "")).strip()


def _format_conversation(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for msg in messages:
        role = str(msg.get("role") or "unknown").upper()
        content = str(msg.get("content") or "")
        tool_name = msg.get("tool_name")
        if role == "TOOL" and tool_name:
            if len(content) > 500:
                content = content[:250] + "\n...[truncated]...\n" + content[-250:]
            parts.append(f"[TOOL:{tool_name}]: {content}")
            continue
        if role == "ASSISTANT":
            calls = msg.get("tool_calls")
            if isinstance(calls, list) and calls:
                names = []
                for call in calls:
                    if isinstance(call, dict):
                        names.append(str(call.get("name") or call.get("function", {}).get("name") or "?"))
                if names:
                    parts.append(f"[ASSISTANT]: [Called: {', '.join(names)}]")
            if content:
                parts.append(f"[ASSISTANT]: {content}")
            continue
        parts.append(f"[{role}]: {content}")
    return "\n\n".join(parts)


def _truncate_around_matches(full_text: str, query: str, max_chars: int = MAX_SESSION_CHARS) -> str:
    if len(full_text) <= max_chars:
        return full_text
    terms = [term.strip('"').lower() for term in query.split() if term.upper() not in {"OR", "AND", "NOT"}]
    lowered = full_text.lower()
    first_match = min((lowered.find(term) for term in terms if term and lowered.find(term) >= 0), default=0)
    half = max_chars // 2
    start = max(0, first_match - half)
    end = min(len(full_text), start + max_chars)
    if end - start < max_chars:
        start = max(0, end - max_chars)
    prefix = "...[earlier conversation truncated]...\n\n" if start > 0 else ""
    suffix = "\n\n...[later conversation truncated]..." if end < len(full_text) else ""
    return prefix + full_text[start:end] + suffix


def _format_timestamp(value: Any) -> str:
    if value is None:
        return "unknown"
    try:
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value).strftime("%Y-%m-%d %H:%M:%S")
        text = str(value).strip()
        if text.replace(".", "", 1).isdigit():
            return datetime.fromtimestamp(float(text)).strftime("%Y-%m-%d %H:%M:%S")
        return text
    except (OSError, OverflowError, ValueError):
        return str(value)


def _parse_role_filter(value: str | None) -> list[str] | None:
    if not value:
        return None
    roles = [item.strip() for item in value.split(",") if item.strip()]
    return roles or None


def _json_error(message: str) -> str:
    return json.dumps({"success": False, "error": message}, ensure_ascii=False)
