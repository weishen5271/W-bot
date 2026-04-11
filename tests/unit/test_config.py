"""Unit tests for config module."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


class TestSettings:
    """Tests for Settings dataclass."""

    def test_settings_creation(self) -> None:
        """Test creating a Settings instance."""
        from w_bot.agents.core.config import Settings

        settings = Settings(
            model_provider="openai",
            llm_api_key="test-key",
            llm_base_url="https://api.openai.com/v1",
            llm_extra_headers={},
            llm_temperature=0.0,
            dashscope_api_key="",
            bailian_base_url="",
            bailian_model_name="",
            tavily_api_key="",
            memory_file_path="/tmp/memory.json",
            short_term_memory_path="/tmp/short_term.pkl",
            user_id="test-user",
            session_id="test-session",
            session_state_file_path="/tmp/session.json",
            escalation_state_file_path="/tmp/escalations.json",
            retrieve_top_k=5,
            enable_cron_service=False,
            mcp_servers=[],
            enable_skills=True,
            skills_workspace_dir="/tmp/skills",
            skills_builtin_dir="",
            model_routing=MagicMock(),
            multimodal=MagicMock(),
            token_optimization=MagicMock(),
            short_term_memory_optimization=MagicMock(),
            expose_step_logs=False,
            enable_openclaw_profile=False,
            openclaw_profile_root_dir="~/.wbot",
            openclaw_auto_init=True,
            loop_guard=MagicMock(),
            intent_classification=MagicMock(),
            enable_streaming=True,
            enable_console_logs=True,
            restrict_to_workspace=True,
        )

        assert settings.model_provider == "openai"
        assert settings.llm_api_key == "test-key"
        assert settings.user_id == "test-user"

    def test_settings_immutable(self) -> None:
        """Test Settings is a frozen dataclass (immutable)."""
        from dataclasses import FrozenInstanceError

        from w_bot.agents.core.config import Settings

        settings = Settings(
            model_provider="openai",
            llm_api_key="test-key",
            llm_base_url="https://api.openai.com/v1",
            llm_extra_headers={},
            llm_temperature=0.0,
            dashscope_api_key="",
            bailian_base_url="",
            bailian_model_name="",
            tavily_api_key="",
            memory_file_path="/tmp/memory.json",
            short_term_memory_path="/tmp/short_term.pkl",
            user_id="test-user",
            session_id="test-session",
            session_state_file_path="/tmp/session.json",
            escalation_state_file_path="/tmp/escalations.json",
            retrieve_top_k=5,
            enable_cron_service=False,
            mcp_servers=[],
            enable_skills=True,
            skills_workspace_dir="/tmp/skills",
            skills_builtin_dir="",
            model_routing=MagicMock(),
            multimodal=MagicMock(),
            token_optimization=MagicMock(),
            short_term_memory_optimization=MagicMock(),
            expose_step_logs=False,
            enable_openclaw_profile=False,
            openclaw_profile_root_dir="~/.wbot",
            openclaw_auto_init=True,
            loop_guard=MagicMock(),
            intent_classification=MagicMock(),
            enable_streaming=True,
            enable_console_logs=True,
            restrict_to_workspace=True,
        )

        with pytest.raises(FrozenInstanceError):
            settings.user_id = "changed"


class TestDefaultPaths:
    """Tests for default path constants."""

    def test_default_configs_dir(self) -> None:
        """Test DEFAULT_CONFIGS_DIR is defined."""
        from w_bot.agents.core.config import DEFAULT_CONFIGS_DIR
        assert DEFAULT_CONFIGS_DIR == "configs"

    def test_default_app_config_path(self) -> None:
        """Test DEFAULT_APP_CONFIG_PATH is defined."""
        from w_bot.agents.core.config import DEFAULT_APP_CONFIG_PATH
        assert "app.json" in DEFAULT_APP_CONFIG_PATH

    def test_default_session_state_file_path(self) -> None:
        """Test DEFAULT_SESSION_STATE_FILE_PATH is defined."""
        from w_bot.agents.core.config import DEFAULT_SESSION_STATE_FILE_PATH
        assert "session_state.json" in DEFAULT_SESSION_STATE_FILE_PATH

    def test_default_escalation_state_file_path(self) -> None:
        """Test DEFAULT_ESCALATION_STATE_FILE_PATH is defined."""
        from w_bot.agents.core.config import DEFAULT_ESCALATION_STATE_FILE_PATH
        assert "escalations.json" in DEFAULT_ESCALATION_STATE_FILE_PATH
