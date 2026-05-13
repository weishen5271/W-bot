from __future__ import annotations

import json
from typing import Any, TypedDict

from langchain_core.messages import AnyMessage

from ..intent.intent_detection import (  # noqa: F401 - compatibility re-exports
    _continue_current_task_prompt,
    _has_tool_messages_since_last_human,
    _looks_like_capability_question,
    _looks_like_casual_chat,
    _looks_like_cron_request,
    _looks_like_exec_request,
    _looks_like_file_edit_request,
    _looks_like_file_read_request,
    _looks_like_message_request,
    _looks_like_project_inspection_request,
    _looks_like_spawn_request,
    _looks_like_web_request,
    _response_looks_incomplete,
    _should_check_completion_for_turn,
    _should_enable_tools_for_text,
    _should_expose_run_skill,
)
from .message_utils import (  # noqa: F401 - compatibility re-exports
    _extract_last_user_message,
    _merge_token_usage_dicts,
    _truncate_text_preserving_edges,
)
from .tool_analysis import (  # noqa: F401 - compatibility re-exports
    _count_tool_steps_since_last_human,
    _summarize_tool_calls,
)


class AgentState(TypedDict):
    messages: list[AnyMessage]
    long_term_context: str
    conversation_summary: str
    summarized_message_count: int
    prepared_system_prompt_base: str
    latest_token_usage: dict[str, Any]
    session_token_usage: dict[str, Any]
    token_budget_state: dict[str, Any]
    context_compaction_level: str
    last_tool_failed: bool
    consecutive_tool_failures: int
    last_tool_name: str
    last_tool_error: str


def _tool_args_preview(tool_name: str, args: dict[str, Any]) -> str:
    candidates = [
        args.get("url"),
        args.get("query"),
        args.get("path"),
        args.get("command"),
        args.get("task"),
        args.get("id"),
        args.get("working_dir"),
    ]
    for item in candidates:
        text = str(item or "").strip()
        if text:
            compact = " ".join(text.split())
            return compact[:96] + ("..." if len(compact) > 96 else "")
    if not args:
        return tool_name
    try:
        raw = json.dumps(args, ensure_ascii=False, sort_keys=True)
    except Exception:
        raw = str(args)
    compact = " ".join(raw.split())
    return compact[:96] + ("..." if len(compact) > 96 else "")


def _tool_progress_action(tool_name: str) -> str:
    normalized = (tool_name or "").strip().lower()
    for token, label in [
        ("navigate", "navigate"),
        ("search", "search"),
        ("fetch", "fetch"),
        ("read", "read"),
        ("write", "write"),
        ("edit", "edit"),
        ("exec", "exec"),
        ("shell", "exec"),
        ("spawn", "spawn"),
        ("subagent", "delegate"),
        ("wait", "wait"),
    ]:
        if token in normalized:
            return label
    return "run"


def _tool_progress_emoji(tool_name: str) -> str:
    normalized = (tool_name or "").strip().lower()
    for token, emoji in [
        ("browser", "🌐"),
        ("navigate", "🌐"),
        ("web", "🌐"),
        ("search", "🔎"),
        ("grep", "🔎"),
        ("find", "🔎"),
        ("read", "📖"),
        ("fetch", "📖"),
        ("load", "📖"),
        ("write", "✍"),
        ("edit", "✍"),
        ("patch", "✍"),
        ("exec", "⚙"),
        ("shell", "⚙"),
        ("command", "⚙"),
        ("spawn", "🧩"),
        ("subagent", "🧩"),
        ("wait", "🧩"),
    ]:
        if token in normalized:
            return emoji
    return "⚡"


__all__ = [
    "AgentState",
    "_continue_current_task_prompt",
    "_count_tool_steps_since_last_human",
    "_extract_last_user_message",
    "_has_tool_messages_since_last_human",
    "_looks_like_capability_question",
    "_looks_like_casual_chat",
    "_looks_like_cron_request",
    "_looks_like_exec_request",
    "_looks_like_file_edit_request",
    "_looks_like_file_read_request",
    "_looks_like_message_request",
    "_looks_like_project_inspection_request",
    "_looks_like_spawn_request",
    "_looks_like_web_request",
    "_merge_token_usage_dicts",
    "_response_looks_incomplete",
    "_should_check_completion_for_turn",
    "_should_enable_tools_for_text",
    "_should_expose_run_skill",
    "_summarize_tool_calls",
    "_tool_args_preview",
    "_tool_progress_action",
    "_tool_progress_emoji",
    "_truncate_text_preserving_edges",
]
