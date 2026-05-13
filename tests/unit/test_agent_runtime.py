from __future__ import annotations

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from w_bot.agents.core.runtime import AgentRuntime
from w_bot.agents.core.session_db import SessionStore
from w_bot.agents.core.session_models import RuntimeConfig


def test_agent_runtime_minimal_turn_persists_messages(tmp_path) -> None:
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content="你好，我是 W-bot")
    store = SessionStore(str(tmp_path / "sessions.sqlite"))
    statuses: list[str] = []
    runtime = AgentRuntime(
        llm=llm,
        session_store=store,
        user_id="tester",
        source="unit",
        model_name="mock",
    )

    result = runtime.run_turn(
        session_id="s1",
        inbound_messages=[HumanMessage(content="你好")],
        config=RuntimeConfig(status_callback=statuses.append),
    )

    assert result.completed is True
    assert result.final_response == "你好，我是 W-bot"
    loaded = store.get_messages("s1")
    assert len(loaded) == 2
    assert loaded[0].content == "你好"
    assert loaded[1].content == "你好，我是 W-bot"
    assert statuses


def test_agent_runtime_flushes_session_search_after_turn(tmp_path) -> None:
    class FakeSessionSearchDB:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def sync_langchain_messages(self, **kwargs) -> None:
            self.calls.append(kwargs)

    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content="已记录")
    search_db = FakeSessionSearchDB()
    store = SessionStore(str(tmp_path / "sessions.sqlite"))
    runtime = AgentRuntime(
        llm=llm,
        session_store=store,
        session_search_db=search_db,
        user_id="tester",
        source="unit",
        model_name="mock",
    )

    runtime.run_turn(session_id="s1", inbound_messages=[HumanMessage(content="hello")])

    assert len(search_db.calls) == 1
    call = search_db.calls[0]
    assert call["session_id"] == "s1"
    assert call["source"] == "unit"
    assert call["user_id"] == "tester"
    assert call["model"] == "mock"
    assert [message.content for message in call["messages"]] == ["hello", "已记录"]


class FakeSearchTool:
    name = "web_search"

    def invoke(self, args: dict) -> str:
        return f"result for {args['query']}"


class FailingTool:
    name = "web_search"

    def invoke(self, args: dict) -> str:
        del args
        raise RuntimeError("boom")


def test_agent_runtime_executes_tool_calls_and_continues(tmp_path) -> None:
    llm = MagicMock()
    llm.bind_tools.return_value = llm
    llm.invoke.side_effect = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "web_search",
                    "args": {"query": "W-bot"},
                    "id": "call_1",
                    "type": "tool_call",
                }
            ],
        ),
        AIMessage(content="查到了 W-bot 信息"),
    ]
    store = SessionStore(str(tmp_path / "sessions.sqlite"))
    runtime = AgentRuntime(
        llm=llm,
        session_store=store,
        tools=[FakeSearchTool()],
    )

    result = runtime.run_turn(
        session_id="s1",
        inbound_messages=[HumanMessage(content="查一下 W-bot")],
        config=RuntimeConfig(recursion_limit=4),
    )

    assert result.completed is True
    assert result.tool_calls == 1
    assert result.final_response == "查到了 W-bot 信息"
    loaded = store.get_messages("s1")
    assert len(loaded) == 4
    assert isinstance(loaded[2], ToolMessage)
    assert loaded[2].content == "result for W-bot"
    assert loaded[3].content == "查到了 W-bot 信息"


def test_agent_runtime_stops_at_recursion_limit(tmp_path) -> None:
    llm = MagicMock()
    llm.bind_tools.return_value = llm
    llm.invoke.return_value = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "web_search",
                "args": {"query": "loop"},
                "id": "call_1",
                "type": "tool_call",
            }
        ],
    )
    store = SessionStore(str(tmp_path / "sessions.sqlite"))
    runtime = AgentRuntime(llm=llm, session_store=store, tools=[FakeSearchTool()])

    result = runtime.run_turn(
        session_id="s1",
        inbound_messages=[HumanMessage(content="循环")],
        config=RuntimeConfig(recursion_limit=2),
    )

    assert result.completed is False
    assert result.tool_calls == 2
    assert "最大迭代次数" in result.final_response
    assert result.metadata["stopped_by"] == "recursion_limit"


def test_agent_runtime_stops_when_tool_steps_exceed_limit(tmp_path) -> None:
    llm = MagicMock()
    llm.bind_tools.return_value = llm
    llm.invoke.side_effect = [
        AIMessage(
            content="",
            tool_calls=[
                {"name": "web_search", "args": {"query": "one"}, "id": "call_1", "type": "tool_call"}
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                {"name": "web_search", "args": {"query": "two"}, "id": "call_2", "type": "tool_call"}
            ],
        ),
    ]
    store = SessionStore(str(tmp_path / "sessions.sqlite"))
    runtime = AgentRuntime(llm=llm, session_store=store, tools=[FakeSearchTool()])

    result = runtime.run_turn(
        session_id="s1",
        inbound_messages=[HumanMessage(content="查两次")],
        config=RuntimeConfig(recursion_limit=4, max_tool_steps_per_turn=1),
    )

    assert result.completed is False
    assert result.metadata["stopped_by"] == "tool_guard"
    assert "工具调用次数已达上限" in result.final_response


def test_agent_runtime_stops_repeated_tool_calls(tmp_path) -> None:
    llm = MagicMock()
    llm.bind_tools.return_value = llm
    repeated_call = {
        "name": "web_search",
        "args": {"query": "same"},
        "id": "call_same",
        "type": "tool_call",
    }
    llm.invoke.side_effect = [
        AIMessage(content="", tool_calls=[repeated_call]),
        AIMessage(content="", tool_calls=[repeated_call]),
    ]
    store = SessionStore(str(tmp_path / "sessions.sqlite"))
    runtime = AgentRuntime(llm=llm, session_store=store, tools=[FakeSearchTool()])

    result = runtime.run_turn(
        session_id="s1",
        inbound_messages=[HumanMessage(content="重复查")],
        config=RuntimeConfig(recursion_limit=4, max_same_tool_call_repeats=2),
    )

    assert result.completed is False
    assert result.metadata["stopped_by"] == "tool_guard"
    assert "同一工具调用连续重复" in result.final_response


def test_agent_runtime_recovers_after_consecutive_tool_failures(tmp_path) -> None:
    llm = MagicMock()
    llm.bind_tools.return_value = llm
    llm.invoke.return_value = AIMessage(
        content="",
        tool_calls=[
            {"name": "web_search", "args": {"query": "fail"}, "id": "call_fail", "type": "tool_call"}
        ],
    )
    store = SessionStore(str(tmp_path / "sessions.sqlite"))
    runtime = AgentRuntime(llm=llm, session_store=store, tools=[FailingTool()])

    result = runtime.run_turn(
        session_id="s1",
        inbound_messages=[HumanMessage(content="失败")],
        config=RuntimeConfig(recursion_limit=4, max_consecutive_tool_failures=2),
    )

    assert result.completed is False
    assert result.metadata["stopped_by"] == "consecutive_tool_failures"
    assert "连续执行失败 2 次" in result.final_response
    assert "RuntimeError: boom" in result.final_response


def test_agent_runtime_returns_model_failure_fallback(tmp_path) -> None:
    llm = MagicMock()
    llm.invoke.side_effect = RuntimeError("model down")
    store = SessionStore(str(tmp_path / "sessions.sqlite"))
    runtime = AgentRuntime(llm=llm, session_store=store)

    result = runtime.run_turn(
        session_id="s1",
        inbound_messages=[HumanMessage(content="你好")],
    )

    assert result.completed is False
    assert "模型调用失败" in result.final_response
    assert result.error == "model down"
