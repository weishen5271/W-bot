from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from langchain_core.messages import HumanMessage

from w_bot.agents.core.session_models import RuntimeConfig
from w_bot.channels.feishu.gateway import FeishuConfig, FeishuGateway
from w_bot.channels.models import InboundMedia, InboundMessage


def _feishu_config() -> FeishuConfig:
    return FeishuConfig(
        enabled=True,
        app_id="app",
        app_secret="secret",
        encrypt_key="",
        verification_token="",
        allow_from=["*"],
        react_emoji="",
        group_policy="mention",
        reply_to_message=True,
    )


def test_feishu_ask_agent_uses_runtime_directly(tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    class DummyRuntime:
        def run_turn(
            self,
            *,
            session_id: str,
            inbound_messages: list[object],
            config: RuntimeConfig,
        ) -> SimpleNamespace:
            calls.append(
                {
                    "session_id": session_id,
                    "messages": inbound_messages,
                    "config": config,
                }
            )
            return SimpleNamespace(final_response="飞书 runtime 回复")

    gateway = FeishuGateway(
        runtime=DummyRuntime(),
        config=_feishu_config(),
        thread_prefix="feishu",
        media_root_dir=str(tmp_path / "media"),
        expose_step_logs=False,
        recursion_limit=7,
        max_tool_steps_per_turn=3,
        max_same_tool_call_repeats=2,
    )
    inbound = InboundMessage(
        content="  你好  ",
        media=[
            InboundMedia(
                id="m1",
                path=str(tmp_path / "image.png"),
                mime="image/png",
                kind="image",
                size_bytes=12,
                sha256="abc",
            )
        ],
    )

    reply = gateway._ask_agent(inbound=inbound, session_id="s1")

    assert reply == "飞书 runtime 回复"
    assert calls[0]["session_id"] == "s1"
    runtime_config = calls[0]["config"]
    assert isinstance(runtime_config, RuntimeConfig)
    assert runtime_config.recursion_limit == 7
    assert runtime_config.max_tool_steps_per_turn == 3
    assert runtime_config.max_same_tool_call_repeats == 2
    message = calls[0]["messages"][0]
    assert isinstance(message, HumanMessage)
    assert "你好" in str(message.content)
    assert message.additional_kwargs["media"][0]["id"] == "m1"
