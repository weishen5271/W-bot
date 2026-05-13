from __future__ import annotations

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage

from w_bot.agents.core.config import TokenOptimizationSettings
from w_bot.agents.core.context_optimizer import ContextOptimizer
from w_bot.agents.core.runtime import AgentRuntime
from w_bot.agents.core.session_db import SessionStore
from w_bot.agents.core.session_models import RuntimeConfig


def test_agent_runtime_updates_summary_when_not_deferred(tmp_path) -> None:
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content="ok")
    store = SessionStore(str(tmp_path / "sessions.sqlite"))
    settings = TokenOptimizationSettings(
        enabled=True,
        max_recent_user_turns=1,
        summary_trigger_messages=1,
        summary_max_chars=200,
        context_window_tokens=4096,
        auto_compact_buffer_tokens=1000,
        warning_threshold_buffer_tokens=1000,
        error_threshold_buffer_tokens=1000,
        blocking_buffer_tokens=100,
        enable_dynamic_system_context=True,
        enable_git_status=False,
        git_status_max_chars=2000,
        enable_project_instruction_scan=False,
        project_instruction_files=(),
    )
    runtime = AgentRuntime(
        llm=llm,
        session_store=store,
        context_optimizer=ContextOptimizer(settings=settings),
    )

    runtime.run_turn(
        session_id="s1",
        inbound_messages=[
            HumanMessage(content="第一轮"),
            AIMessage(content="第一轮回答"),
            HumanMessage(content="第二轮"),
            AIMessage(content="第二轮回答"),
            HumanMessage(content="第三轮"),
        ],
        config=RuntimeConfig(defer_summary_update=False),
    )

    summary, count = store.get_summary("s1")
    assert count > 0
    assert "【目标与约束】" in summary
