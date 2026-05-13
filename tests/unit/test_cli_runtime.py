from __future__ import annotations

from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage

from w_bot.agents.core import cli
from w_bot.agents.core.session_db import SessionStore
from w_bot.agents.core.session_models import RuntimeConfig
from w_bot.agents.core.session_store import SessionStateStore


class DummyCliStreamRenderer:
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.finished_text = ""

    def update_status(self, text: str) -> None:
        del text

    def on_delta(self, text: str) -> None:
        del text

    def on_tool_progress(self, *args: object, **kwargs: object) -> None:
        del args, kwargs

    def finish(self, text: str) -> None:
        self.finished_text = text


def test_live_render_disabled_by_default_on_macos(monkeypatch) -> None:
    monkeypatch.delenv("WBOT_DISABLE_LIVE", raising=False)
    monkeypatch.delenv("WBOT_FORCE_LIVE", raising=False)
    monkeypatch.setattr(cli.sys, "platform", "darwin")

    console = SimpleNamespace(is_terminal=True, is_interactive=True, color_system="truecolor")

    assert cli._supports_live_render(console) is False


def test_live_render_force_env_overrides_macos_default(monkeypatch) -> None:
    monkeypatch.delenv("WBOT_DISABLE_LIVE", raising=False)
    monkeypatch.setenv("WBOT_FORCE_LIVE", "1")
    monkeypatch.setattr(cli.sys, "platform", "darwin")

    console = SimpleNamespace(is_terminal=False, is_interactive=False, color_system=None)

    assert cli._supports_live_render(console) is True


def test_cli_run_agent_turn_uses_runtime(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(cli, "CliStreamRenderer", DummyCliStreamRenderer)
    calls: list[dict[str, object]] = []

    class DummyRuntime:
        @property
        def compat_adapter(self) -> SimpleNamespace:
            return SimpleNamespace(list_subagents=lambda limit=20: [])

        def run_turn(
            self,
            *,
            session_id: str,
            inbound_messages: list[object],
            config: RuntimeConfig,
        ) -> SimpleNamespace:
            calls.append({"session_id": session_id, "messages": inbound_messages, "config": config})
            return SimpleNamespace(final_response="cli runtime reply")

    settings = SimpleNamespace(
        loop_guard=SimpleNamespace(
            recursion_limit=6,
            max_tool_steps_per_turn=4,
            max_same_tool_call_repeats=2,
        )
    )
    app_state = cli.CliAppState(session_id="cli-s1")
    session_store = SessionStateStore(str(tmp_path / "sessions.json"))

    reply = cli._run_agent_turn(
        runtime=DummyRuntime(),
        settings=settings,
        session_store=session_store,
        app_state=app_state,
        user_text="hello",
    )

    assert reply == "cli runtime reply"
    assert calls[0]["session_id"] == "cli-s1"
    assert isinstance(calls[0]["messages"][0], HumanMessage)
    runtime_config = calls[0]["config"]
    assert isinstance(runtime_config, RuntimeConfig)
    assert runtime_config.recursion_limit == 6
    assert runtime_config.max_tool_steps_per_turn == 4
    assert runtime_config.max_same_tool_call_repeats == 2


def test_cli_history_and_stats_read_runtime_session_store(tmp_path) -> None:
    store = SessionStore(str(tmp_path / "session_store.sqlite"))
    store.ensure_session(session_id="cli-s1", source="cli", user_id="tester", model="mock")
    store.append_messages(
        "cli-s1",
        [
            HumanMessage(content="你好"),
            AIMessage(content="你好，我是 W-bot"),
        ],
    )

    class DummyRuntime:
        session_store = store

        def get_session_messages(self, session_id: str) -> list[object]:
            return store.get_messages(session_id)

    preview = cli._session_history_preview(runtime=DummyRuntime(), session_id="cli-s1", limit=2)
    stats = cli._collect_session_snapshot_stats(runtime=DummyRuntime(), session_id="cli-s1")

    assert "你好" in preview
    assert stats["message_count"] == 2
    assert stats["user_messages"] == 1
    assert stats["assistant_messages"] == 1
