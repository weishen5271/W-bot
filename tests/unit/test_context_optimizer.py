from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from w_bot.agents.core.config import TokenOptimizationSettings
from w_bot.agents.core.context_optimizer import ContextOptimizer


def _settings(**overrides) -> TokenOptimizationSettings:
    payload = {
        "enabled": True,
        "max_recent_user_turns": 2,
        "summary_trigger_messages": 2,
        "summary_max_chars": 200,
        "context_window_tokens": 4096,
        "auto_compact_buffer_tokens": 1000,
        "warning_threshold_buffer_tokens": 1000,
        "error_threshold_buffer_tokens": 1000,
        "blocking_buffer_tokens": 100,
        "enable_dynamic_system_context": True,
        "enable_git_status": False,
        "git_status_max_chars": 2000,
        "enable_project_instruction_scan": False,
        "project_instruction_files": (),
    }
    payload.update(overrides)
    return TokenOptimizationSettings(**payload)


def test_context_optimizer_disabled_returns_full_history() -> None:
    optimizer = ContextOptimizer(settings=_settings(enabled=False))
    history = [HumanMessage(content="a"), AIMessage(content="b")]

    result = optimizer.prepare(history=history)

    assert result.recent_messages == history
    assert result.context_compaction_level == "off"
    assert result.token_budget_state == {}


def test_context_optimizer_updates_summary_when_not_deferred() -> None:
    optimizer = ContextOptimizer(settings=_settings(summary_trigger_messages=1))
    history = [
        HumanMessage(content="目标 A"),
        AIMessage(content="完成 A"),
        HumanMessage(content="目标 B"),
        AIMessage(content="完成 B"),
        HumanMessage(content="目标 C"),
    ]

    result = optimizer.prepare(
        history=history,
        summarized_message_count=0,
        defer_summary_update=False,
    )

    assert result.summarized_message_count > 0
    assert "【目标与约束】" in result.conversation_summary
    assert result.recent_messages[-1].content == "目标 C"
