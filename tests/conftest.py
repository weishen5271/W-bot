"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Generator
from unittest.mock import MagicMock

import pytest


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an event loop for the entire test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_llm() -> MagicMock:
    """Mock LLM for testing."""
    mock = MagicMock()
    mock.invoke.return_value = MagicMock(
        content="Mocked response",
        tool_calls=[],
        usage=MagicMock(input_tokens=10, output_tokens=20),
    )
    return mock


@pytest.fixture
def mock_tools() -> list[MagicMock]:
    """Mock tools list for testing."""
    return [
        MagicMock(
            name="test_tool",
            description="A test tool",
            parameters={"type": "object", "properties": {}},
            invoke=MagicMock(return_value="tool result"),
            ainvoke=MagicMock(return_value="tool result"),
        ),
    ]


@pytest.fixture
def temp_workspace(tmp_path: Path) -> Path:
    """Create a temporary workspace for tests."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "test_file.txt").write_text("test content\nline 2\nline 3")
    (workspace / "subdir").mkdir()
    (workspace / "subdir" / "nested.txt").write_text("nested content")
    return workspace


@pytest.fixture
def temp_config_dir(tmp_path: Path) -> Path:
    """Create a temporary config directory for tests."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    return config_dir


@pytest.fixture
def sample_config(temp_config_dir: Path) -> dict[str, Any]:
    """Create a sample config dict for testing."""
    return {
        "model_provider": "openai",
        "llm_api_key": "test-key",
        "llm_base_url": "https://api.openai.com/v1",
        "llm_extra_headers": {},
        "llm_temperature": 0.0,
        "dashscope_api_key": "",
        "bailian_base_url": "",
        "bailian_model_name": "",
        "tavily_api_key": "",
        "memory_file_path": str(temp_config_dir / "memory.json"),
        "short_term_memory_path": str(temp_config_dir / "short_term.pkl"),
        "user_id": "test-user",
        "session_id": "test-session",
        "session_state_file_path": str(temp_config_dir / "session_state.json"),
        "escalation_state_file_path": str(temp_config_dir / "escalations.json"),
        "retrieve_top_k": 5,
        "enable_cron_service": False,
        "mcp_servers": [],
        "enable_skills": True,
        "skills_workspace_dir": str(temp_config_dir / "skills"),
        "skills_builtin_dir": "",
        "model_routing": {
            "text_model_name": "gpt-4",
            "image_model_name": "gpt-4-vision",
            "audio_model_name": "gpt-4-audio",
        },
        "multimodal": {
            "enabled": True,
            "max_file_bytes": 20 * 1024 * 1024,
            "max_total_bytes_per_turn": 50 * 1024 * 1024,
            "max_files_per_turn": 10,
            "audio_mode": "transcribe",
            "video_keyframe_interval_sec": 5,
            "video_max_frames": 100,
            "document_max_chars": 50_000,
            "temp_ttl_hours": 24,
            "media_root_dir": str(temp_config_dir / "media"),
        },
        "token_optimization": {
            "enabled": True,
            "max_recent_user_turns": 20,
            "summary_trigger_messages": 50,
            "summary_max_chars": 4000,
            "context_window_tokens": 128_000,
            "auto_compact_buffer_tokens": 10_000,
            "warning_threshold_buffer_tokens": 15_000,
            "error_threshold_buffer_tokens": 20_000,
            "blocking_buffer_tokens": 25_000,
            "enable_dynamic_system_context": True,
            "enable_git_status": True,
            "git_status_max_chars": 2000,
        },
        "short_term_memory_optimization": {
            "enabled": True,
            "checkpoint_interval_messages": 20,
            "archive_after_turns": 100,
            "max_checkpoint_history": 50,
        },
        "expose_step_logs": False,
        "enable_openclaw_profile": False,
        "openclaw_profile_root_dir": "~/.wbot",
        "openclaw_auto_init": True,
        "loop_guard": {
            "max_tool_steps_per_turn": 50,
            "max_same_tool_call_repeats": 5,
        },
        "intent_classification": {
            "enabled": True,
            "use_llm_fallback": True,
        },
        "enable_streaming": True,
        "enable_console_logs": True,
        "restrict_to_workspace": True,
    }


@pytest.fixture
def mock_runnable_config() -> dict[str, Any]:
    """Create a mock LangChain RunnableConfig for testing."""
    return {
        "configurable": {
            "thread_id": "test-thread",
            "session_id": "test-session",
        }
    }
