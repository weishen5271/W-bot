"""
Tool analysis utilities for W-bot agent.

This module contains functions for analyzing tool calls, detecting failures,
and building retry messages during agent execution.
"""

import json
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage

from .logging_config import get_logger
from .message_utils import _to_text_content

logger = get_logger(__name__)


def _summarize_tool_calls(tool_calls: list[dict[str, Any]]) -> str:
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


def _count_tool_steps_since_last_human(messages: list[AnyMessage]) -> int:
    count = 0
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            break
        if isinstance(message, AIMessage) and message.tool_calls:
            count += 1
    return count


def _count_named_tool_calls_since_last_human(messages: list[AnyMessage], tool_name: str) -> int:
    count = 0
    target = (tool_name or "").strip().lower()
    if not target:
        return 0
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            break
        if not isinstance(message, AIMessage) or not message.tool_calls:
            continue
        for tool_call in message.tool_calls:
            if str(tool_call.get("name") or "").strip().lower() == target:
                count += 1
    return count


def _same_tool_call_streak(messages: list[AnyMessage]) -> tuple[str, int]:
    signatures: list[str] = []
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            break
        if isinstance(message, ToolMessage):
            continue
        if isinstance(message, AIMessage) and message.tool_calls:
            signature = _tool_call_signature(message.tool_calls)
            if signature:
                signatures.append(signature)
            continue
        break

    if not signatures:
        return "", 0

    latest = signatures[0]
    repeat_count = 0
    for signature in signatures:
        if signature != latest:
            break
        repeat_count += 1
    return latest, repeat_count


def _tool_call_signature(tool_calls: list[dict[str, Any]]) -> str:
    normalized: list[str] = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        name = str(tool_call.get("name") or "").strip()
        args = tool_call.get("args")
        if args is None:
            args = tool_call.get("arguments")
        try:
            args_text = json.dumps(args, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            args_text = str(args)
        normalized.append(f"{name}:{args_text}")
    return "|".join(normalized)


def _is_tool_failure_content(content: Any) -> bool:
    text = str(content or "").strip()
    if not text:
        return False
    lowered = text.lower()
    failure_prefixes = (
        "error:",
        "stderr:",
        "tool execution failed:",
        "tool not found:",
        "invalid parameters:",
    )
    if lowered.startswith(failure_prefixes):
        return True
    exit_code = _extract_exit_code(text)
    if exit_code is not None and exit_code != 0:
        return True
    if '"error"' in lowered:
        try:
            parsed = json.loads(text)
        except (TypeError, ValueError, json.JSONDecodeError):
            return False
        if isinstance(parsed, dict) and parsed.get("error"):
            return True
    return False


def _extract_tool_failure_summary(content: Any) -> str:
    text = str(content or "").strip()
    if not text:
        return ""
    exit_code = _extract_exit_code(text)
    if exit_code is not None and exit_code != 0:
        return f"Exit code: {exit_code}"
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return text[:240]
    if isinstance(parsed, dict):
        error = parsed.get("error")
        if error:
            return str(error)[:240]
    return text[:240]


def _extract_exit_code(text: str) -> int | None:
    marker = "Exit code:"
    if marker not in text:
        return None
    tail = text.rsplit(marker, 1)[-1].strip().splitlines()[0].strip()
    try:
        return int(tail)
    except (TypeError, ValueError):
        return None


def _build_text_only_retry_messages(
    *,
    system_prompt: str,
    history: list[AnyMessage],
) -> list[AnyMessage]:
    """Build and return target object.

    Args:
        system_prompt: System prompt text.
        history: List of historical messages.
    """
    user_text = ""
    for message in reversed(history):
        if not isinstance(message, HumanMessage):
            continue
        user_text = _to_text_content(message.content).strip()
        if user_text:
            break

    if not user_text:
        user_text = "用户发送了一条消息，请给出可执行回复。"

    return [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_text),
    ]


def _runtime_error_reply_text(exc: BaseException | None = None) -> str:
    detail = _format_exception_brief(exc)
    suffix = f"\n异常详情：{detail}" if detail else ""
    return (
        "这次处理出现了临时异常，但服务没有中断。"
        "你可以继续对话，或调整一下问题后再试。"
        f"{suffix}"
    )


def _format_exception_brief(exc: BaseException | None) -> str:
    if exc is None:
        return ""
    try:
        return f"{type(exc).__name__}: {exc}"
    except Exception:
        return str(exc)
