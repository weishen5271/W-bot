from __future__ import annotations

from unittest.mock import MagicMock

from langchain_core.messages import HumanMessage

from w_bot.agents.core.runtime import AgentRuntime
from w_bot.agents.core.session_db import SessionStore


def test_runtime_compat_get_state_returns_session_messages(tmp_path) -> None:
    store = SessionStore(str(tmp_path / "sessions.sqlite"))
    runtime = AgentRuntime(llm=MagicMock(), session_store=store)
    store.append_message("s1", HumanMessage(content="hello"))
    store.update_summary("s1", summary="summary", summarized_message_count=1)

    snapshot = runtime.compat_adapter.get_state({"configurable": {"thread_id": "s1"}})

    assert snapshot.values["messages"][0].content == "hello"
    assert snapshot.values["conversation_summary"] == "summary"
    assert snapshot.values["summarized_message_count"] == 1


def test_runtime_compat_exposes_session_search_llm(tmp_path) -> None:
    llm = MagicMock()
    runtime = AgentRuntime(llm=llm, session_store=SessionStore(str(tmp_path / "sessions.sqlite")))

    assert runtime.compat_adapter.session_search_llm is llm

