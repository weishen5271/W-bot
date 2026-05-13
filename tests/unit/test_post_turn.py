from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from w_bot.agents.core.config import TokenOptimizationSettings
from w_bot.agents.core.context_optimizer import ContextOptimizer
from w_bot.agents.core.post_turn import PostTurnProcessor
from w_bot.agents.core.session_db import SessionStore
from w_bot.agents.core.session_models import RuntimeConfig


class FakeSessionSearchDB:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def sync_langchain_messages(self, **kwargs) -> None:
        self.calls.append(kwargs)


def test_post_turn_flushes_session_search_index(tmp_path) -> None:
    store = SessionStore(str(tmp_path / "sessions.sqlite"))
    store.ensure_session(session_id="s1")
    store.append_message("s1", HumanMessage(content="hello"))
    search_db = FakeSessionSearchDB()
    processor = PostTurnProcessor(
        session_store=store,
        session_search_db=search_db,
        source="unit",
        user_id="tester",
        model="mock",
    )

    processor.after_turn(session_id="s1")

    assert len(search_db.calls) == 1
    call = search_db.calls[0]
    assert call["session_id"] == "s1"
    assert call["source"] == "unit"
    assert call["user_id"] == "tester"
    assert call["model"] == "mock"
    assert call["messages"][0].content == "hello"


def test_post_turn_refreshes_deferred_summary_from_session_store(tmp_path) -> None:
    store = SessionStore(str(tmp_path / "sessions.sqlite"))
    store.ensure_session(session_id="s1")
    store.append_messages(
        "s1",
        [
            HumanMessage(content="请调研 W-bot 架构"),
            AIMessage(content="已确认需要重构入口。"),
            HumanMessage(content="继续实现"),
            AIMessage(content="已切换 Web 和飞书入口。"),
        ],
    )
    optimizer = ContextOptimizer(
        settings=TokenOptimizationSettings(
            enabled=True,
            max_recent_user_turns=1,
            summary_trigger_messages=1,
            summary_max_chars=300,
            context_window_tokens=4096,
            auto_compact_buffer_tokens=512,
            warning_threshold_buffer_tokens=1024,
            error_threshold_buffer_tokens=2048,
            blocking_buffer_tokens=256,
            enable_dynamic_system_context=False,
            enable_git_status=False,
            git_status_max_chars=0,
            enable_project_instruction_scan=False,
            project_instruction_files=(),
        )
    )
    processor = PostTurnProcessor(session_store=store, context_optimizer=optimizer)

    processor.after_turn(session_id="s1", config=RuntimeConfig(defer_summary_update=True))

    summary, summarized_count = store.get_summary("s1")
    assert "【目标与约束】" in summary
    assert summarized_count > 0
