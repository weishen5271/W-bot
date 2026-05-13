from __future__ import annotations

import asyncio
import time

from langchain_core.messages import HumanMessage

from w_bot.agents.core.session_models import RuntimeConfig
from w_bot.agents.core.tool_executor import ToolExecutor
from w_bot.agents.core.tool_policy import ToolPolicy


class FakeTool:
    def __init__(self, name: str, result: str = "ok", delay: float = 0.0, fail: bool = False) -> None:
        self.name = name
        self.result = result
        self.delay = delay
        self.fail = fail
        self.calls: list[dict] = []

    def invoke(self, args: dict) -> str:
        self.calls.append(args)
        if self.delay:
            time.sleep(self.delay)
        if self.fail:
            raise RuntimeError("boom")
        return self.result


def test_tool_policy_parallelizes_readonly_calls() -> None:
    policy = ToolPolicy()

    assert policy.can_parallelize(
        [
            {"name": "read_file", "args": {"path": "a.txt"}},
            {"name": "web_search", "args": {"query": "hello"}},
        ]
    )


def test_tool_policy_serializes_mutating_calls() -> None:
    policy = ToolPolicy()

    assert not policy.can_parallelize(
        [
            {"name": "read_file", "args": {"path": "a.txt"}},
            {"name": "write_file", "args": {"path": "b.txt"}},
        ]
    )


def test_tool_executor_runs_readonly_batch_in_parallel() -> None:
    first = FakeTool("read_file", result="a", delay=0.05)
    second = FakeTool("web_search", result="b", delay=0.05)
    executor = ToolExecutor(tools_by_name={"read_file": first, "web_search": second})

    started = time.monotonic()
    results = asyncio.run(
        executor.execute(
            tool_calls=[
                {"name": "read_file", "args": {"path": "a.txt"}, "id": "call_a"},
                {"name": "web_search", "args": {"query": "q"}, "id": "call_b"},
            ],
            messages=[HumanMessage(content="hi")],
            session_id="s1",
            config=RuntimeConfig(),
        )
    )
    elapsed = time.monotonic() - started

    assert [item.content for item in results] == ["a", "b"]
    assert elapsed < 0.09


def test_tool_executor_converts_failures_to_tool_messages() -> None:
    tool = FakeTool("read_file", fail=True)
    executor = ToolExecutor(tools_by_name={"read_file": tool})

    results = asyncio.run(
        executor.execute(
            tool_calls=[{"name": "read_file", "args": {"path": "a.txt"}, "id": "call_a"}],
            messages=[],
            session_id="s1",
        )
    )

    assert len(results) == 1
    assert "Tool execution failed" in results[0].content
    assert results[0].tool_call_id == "call_a"

