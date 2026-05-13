from __future__ import annotations

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage

from w_bot.agents.core.model_runner import ModelRunner
from w_bot.agents.core.session_models import RuntimeConfig


def test_model_runner_invokes_llm() -> None:
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content="ok")
    runner = ModelRunner(llm=llm)

    result = runner.invoke(messages=[HumanMessage(content="hi")], config=RuntimeConfig())

    assert result.completed is True
    assert result.message.content == "ok"
    llm.invoke.assert_called_once()


def test_model_runner_returns_fallback_message_on_error() -> None:
    llm = MagicMock()
    llm.invoke.side_effect = RuntimeError("boom")
    runner = ModelRunner(llm=llm)

    result = runner.invoke(messages=[HumanMessage(content="hi")], config=RuntimeConfig())

    assert result.completed is False
    assert "模型调用失败" in result.message.content
    assert result.error == "boom"


def test_model_runner_retries_route_failure_with_fallback_llm() -> None:
    route_llm = MagicMock()
    route_llm.invoke.side_effect = RuntimeError("route down")
    fallback_llm = MagicMock()
    fallback_llm.invoke.return_value = AIMessage(content="fallback ok")
    runner = ModelRunner(llm=route_llm)

    result = runner.invoke(
        messages=[HumanMessage(content="hi")],
        config=RuntimeConfig(),
        llm=route_llm,
        fallback_llm=fallback_llm,
    )

    assert result.completed is True
    assert result.message.content == "fallback ok"
    fallback_llm.invoke.assert_called_once()


def test_model_runner_uses_text_only_retry_for_message_length_error() -> None:
    route_llm = MagicMock()
    route_llm.invoke.side_effect = RuntimeError("messages parameter length invalid")
    fallback_llm = MagicMock()
    fallback_llm.invoke.return_value = AIMessage(content="text only ok")
    runner = ModelRunner(llm=route_llm)

    result = runner.invoke(
        messages=[HumanMessage(content="hi")],
        config=RuntimeConfig(),
        llm=route_llm,
        fallback_llm=fallback_llm,
        system_prompt="system",
    )

    assert result.completed is True
    assert result.message.content == "text only ok"
    retry_messages = fallback_llm.invoke.call_args.args[0]
    assert retry_messages[0].content == "system"
