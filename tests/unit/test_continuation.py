from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from w_bot.agents.core.continuation import ContinuationController


def test_continuation_controller_continues_incomplete_reply() -> None:
    controller = ContinuationController(max_attempts=1)

    decision = controller.decide(
        user_goal="请修改文件",
        history=[HumanMessage(content="请修改文件")],
        response=AIMessage(content="我先检查一下。"),
    )

    assert decision.should_continue is True
    assert decision.reason == "incomplete_markers"
    assert "继续执行当前任务" in decision.prompt


def test_continuation_controller_skips_casual_chat() -> None:
    controller = ContinuationController(max_attempts=1)

    decision = controller.decide(
        user_goal="你好",
        history=[HumanMessage(content="你好")],
        response=AIMessage(content="你好！"),
    )

    assert decision.should_continue is False

