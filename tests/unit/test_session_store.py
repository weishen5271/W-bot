from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from w_bot.agents.core.session_db import SessionStore


def test_session_store_round_trips_messages(tmp_path) -> None:
    store = SessionStore(str(tmp_path / "sessions.sqlite"))
    session_id = "test-session"
    messages = [
        HumanMessage(content="你好", additional_kwargs={"media": [{"type": "text"}]}),
        AIMessage(
            content="我需要查一下",
            tool_calls=[
                {
                    "name": "search",
                    "args": {"query": "W-bot"},
                    "id": "call_1",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(content="结果", tool_call_id="call_1", name="search"),
    ]

    store.ensure_session(session_id=session_id, source="unit", user_id="tester", model="mock")
    store.append_messages(session_id, messages)

    loaded = store.get_messages(session_id)
    assert len(loaded) == 3
    assert isinstance(loaded[0], HumanMessage)
    assert loaded[0].content == "你好"
    assert loaded[0].additional_kwargs["media"][0]["type"] == "text"
    assert isinstance(loaded[1], AIMessage)
    assert loaded[1].tool_calls[0]["name"] == "search"
    assert loaded[1].tool_calls[0]["args"]["query"] == "W-bot"
    assert isinstance(loaded[2], ToolMessage)
    assert loaded[2].tool_call_id == "call_1"
    assert loaded[2].name == "search"


def test_session_store_updates_summary(tmp_path) -> None:
    store = SessionStore(str(tmp_path / "sessions.sqlite"))
    store.ensure_session(session_id="s1")
    store.update_summary("s1", summary="摘要", summarized_message_count=2)

    summary, count = store.get_summary("s1")
    assert summary == "摘要"
    assert count == 2

