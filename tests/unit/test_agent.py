"""Unit tests for core agent module."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from w_bot.agents.core.agent import (
    _tool_args_preview,
    _tool_progress_emoji,
    _tool_progress_action,
    AgentState,
)


class TestToolHelpers:
    """Tests for tool-related helper functions."""

    def test_tool_args_preview_with_url(self) -> None:
        """Test preview extracts URL from args."""
        result = _tool_args_preview("web_fetch", {"url": "https://example.com/path"})
        assert "example.com" in result

    def test_tool_args_preview_with_path(self) -> None:
        """Test preview extracts path from args."""
        result = _tool_args_preview("read_file", {"path": "/home/user/file.txt"})
        assert "file.txt" in result

    def test_tool_args_preview_with_command(self) -> None:
        """Test preview extracts command from args."""
        result = _tool_args_preview("exec", {"command": "ls -la /tmp"})
        assert "ls" in result

    def test_tool_args_preview_with_task(self) -> None:
        """Test preview extracts task from args."""
        result = _tool_args_preview("spawn", {"task": "分析这个代码库"})
        assert "分析" in result

    def test_tool_args_preview_empty_args(self) -> None:
        """Test preview with empty args returns tool name."""
        result = _tool_args_preview("some_tool", {})
        assert result == "some_tool"

    def test_tool_args_preview_long_content_truncated(self) -> None:
        """Test preview truncates long content."""
        long_path = "/".join(["dir"] * 30)
        result = _tool_args_preview("read_file", {"path": long_path})
        assert len(result) <= 99  # 96 + "..."
        assert result.endswith("...")

    def test_tool_args_preview_json_fallback(self) -> None:
        """Test preview falls back to JSON for complex args."""
        complex_args = {"key1": "value1", "key2": "value2", "nested": {"a": 1, "b": 2}}
        result = _tool_args_preview("complex_tool", complex_args)
        assert "key1" in result or "value1" in result


class TestToolProgressEmoji:
    """Tests for _tool_progress_emoji function."""

    @pytest.mark.parametrize("tool_name,expected_emoji", [
        ("browser", "🌐"),
        ("navigate", "🌐"),
        ("web_fetch", "🌐"),
        ("search", "🔎"),
        ("grep", "🔎"),
        ("find", "🔎"),
        ("read_file", "📖"),
        ("fetch", "📖"),
        ("load", "📖"),
        ("write_file", "✍"),
        ("edit_file", "✍"),
        ("patch", "✍"),
        ("exec", "⚙"),
        ("shell", "⚙"),
        ("command", "⚙"),
        ("spawn", "🧩"),
        ("subagent", "🧩"),
        ("wait", "🧩"),
    ])
    def test_emoji_mapping(self, tool_name: str, expected_emoji: str) -> None:
        """Test emoji is correctly mapped for each tool type."""
        result = _tool_progress_emoji(tool_name)
        assert result == expected_emoji

    def test_default_emoji(self) -> None:
        """Test default emoji for unknown tools."""
        result = _tool_progress_emoji("unknown_tool")
        assert result == "⚡"

    def test_case_insensitive(self) -> None:
        """Test emoji lookup is case insensitive."""
        result = _tool_progress_emoji("SEARCH")
        assert result == "🔎"


class TestToolProgressAction:
    """Tests for _tool_progress_action function."""

    def test_returns_string(self) -> None:
        """Test action returns a string."""
        result = _tool_progress_action("read_file")
        assert isinstance(result, str)

    def test_empty_input(self) -> None:
        """Test action handles empty input."""
        result = _tool_progress_action("")
        assert isinstance(result, str)


class TestAgentState:
    """Tests for AgentState TypedDict."""

    def test_agent_state_structure(self) -> None:
        """Test AgentState has correct keys."""
        state: AgentState = {
            "messages": [],
            "long_term_context": "",
            "conversation_summary": "",
            "summarized_message_count": 0,
            "prepared_system_prompt_base": "",
            "latest_token_usage": {},
            "session_token_usage": {},
            "token_budget_state": {},
            "context_compaction_level": "none",
            "last_tool_failed": False,
            "consecutive_tool_failures": 0,
            "last_tool_name": "",
            "last_tool_error": "",
        }
        assert "messages" in state
        assert "long_term_context" in state
        assert "conversation_summary" in state
        assert state["context_compaction_level"] == "none"
        assert state["last_tool_failed"] is False


class TestAgentImports:
    """Tests to verify agent module imports correctly."""

    def test_agent_module_imports(self) -> None:
        """Test that key classes and functions are importable."""
        from w_bot.agents.core.agent import (
            AgentState,
            ScheduledGraphApp,
            WBotGraph,
        )
        assert AgentState is not None
        assert ScheduledGraphApp is not None

    def test_intent_detection_imports(self) -> None:
        """Test intent detection functions are importable."""
        from w_bot.agents.core.agent import (
            _should_enable_tools_for_text,
            _should_check_completion_for_turn,
            _has_tool_messages_since_last_human,
            _response_looks_incomplete,
            _looks_like_casual_chat,
            _looks_like_capability_question,
            _looks_like_file_read_request,
            _looks_like_file_edit_request,
            _looks_like_web_request,
            _looks_like_exec_request,
            _looks_like_spawn_request,
        )
        assert callable(_should_enable_tools_for_text)
        assert callable(_should_check_completion_for_turn)

    def test_message_utils_imports(self) -> None:
        """Test message utility functions are importable."""
        from w_bot.agents.core.agent import (
            _extract_last_user_message,
            _merge_token_usage_dicts,
            _format_token_budget_snapshot,
            _truncate_text_preserving_edges,
            _last_human_index,
            _build_summary_fallback,
        )
        assert callable(_extract_last_user_message)
        assert callable(_merge_token_usage_dicts)

    def test_tool_analysis_imports(self) -> None:
        """Test tool analysis functions are importable."""
        from w_bot.agents.core.agent import (
            _summarize_tool_calls,
            _count_tool_steps_since_last_human,
            _same_tool_call_streak,
            _is_tool_failure_content,
            _extract_tool_failure_summary,
            _extract_exit_code,
        )
        assert callable(_summarize_tool_calls)
        assert callable(_count_tool_steps_since_last_human)


class TestMessageUtils:
    """Tests for message utility functions."""

    def test_extract_last_user_message(self) -> None:
        """Test extracting last user message from message list."""
        from w_bot.agents.core.agent import _extract_last_user_message
        from langchain_core.messages import HumanMessage, AIMessage

        messages = [
            AIMessage(content="Hello"),
            HumanMessage(content="What is 2+2?"),
            AIMessage(content="2+2 is 4"),
        ]
        result = _extract_last_user_message(messages)
        assert "2+2" in result

    def test_extract_last_user_message_empty(self) -> None:
        """Test extracting from empty message list."""
        from w_bot.agents.core.agent import _extract_last_user_message

        result = _extract_last_user_message([])
        assert result == ""

    def test_truncate_text_preserving_edges(self) -> None:
        """Test text truncation preserves edges."""
        from w_bot.agents.core.agent import _truncate_text_preserving_edges

        long_text = "start " * 100 + "end"
        result = _truncate_text_preserving_edges(long_text, max_chars=50, preserve_tail=True)
        assert "start" in result
        assert "end" in result

    def test_truncate_text_short_text(self) -> None:
        """Test truncation doesn't modify short text."""
        from w_bot.agents.core.agent import _truncate_text_preserving_edges

        short_text = "short text"
        result = _truncate_text_preserving_edges(short_text, max_chars=50, preserve_tail=True)
        assert result == short_text


class TestIntentDetection:
    """Tests for intent detection helper functions."""

    def test_looks_like_casual_chat(self) -> None:
        """Test casual chat detection."""
        from w_bot.agents.core.agent import _looks_like_casual_chat

        # Short smalltalk should match
        assert _looks_like_casual_chat("hello") is True
        assert _looks_like_casual_chat("hi") is True
        assert _looks_like_casual_chat("你好") is True
        assert _looks_like_casual_chat("谢谢") is True
        assert _looks_like_casual_chat("bye") is True

        # Empty text returns True
        assert _looks_like_casual_chat("") is True

        # "hi there" still matches because "hi" is in the token list and len <= 12
        assert _looks_like_casual_chat("hi there") is True
        # "how are you?" doesn't match - no short_smalltalk and "how" not in tokens
        assert _looks_like_casual_chat("how are you?") is False

    def test_looks_like_capability_question(self) -> None:
        """Test capability question detection."""
        from w_bot.agents.core.agent import _looks_like_capability_question

        # Questions about capability should match
        assert _looks_like_capability_question("can you read files?") is True
        assert _looks_like_capability_question("do you support web search?") is True
        assert _looks_like_capability_question("能读取文件吗?") is True

        # Empty text returns False
        assert _looks_like_capability_question("") is False

        # Explicit execution requests should not match capability questions
        assert _looks_like_capability_question("read this file") is False
        assert _looks_like_capability_question("帮我读文件") is False

    def test_looks_like_file_read_request(self) -> None:
        """Test file read request detection."""
        from w_bot.agents.core.agent import _looks_like_file_read_request

        # Read requests with specific keywords should match
        assert _looks_like_file_read_request("read the file") is True
        assert _looks_like_file_read_request("打开文件") is True
        assert _looks_like_file_read_request("list directory") is True
        assert _looks_like_file_read_request("inspect file.py") is True

        # "show me the contents" doesn't match - no specific keyword
        assert _looks_like_file_read_request("show me the contents") is False

        # Non-read requests should not match
        assert _looks_like_file_read_request("write a file") is False

    def test_looks_like_file_edit_request(self) -> None:
        """Test file edit request detection."""
        from w_bot.agents.core.agent import _looks_like_file_edit_request

        # Edit requests with specific keywords should match
        assert _looks_like_file_edit_request("edit the file") is True
        assert _looks_like_file_edit_request("修改文件") is True
        assert _looks_like_file_edit_request("patch the code") is True

        # "modify contents" doesn't match - no specific keyword
        assert _looks_like_file_edit_request("modify contents") is False

        # Non-edit requests should not match
        assert _looks_like_file_edit_request("read the file") is False

    def test_looks_like_exec_request(self) -> None:
        """Test exec request detection."""
        from w_bot.agents.core.agent import _looks_like_exec_request

        # Exec requests with specific keywords should match
        assert _looks_like_exec_request("run command") is True
        assert _looks_like_exec_request("execute ls") is True
        assert _looks_like_exec_request("运行命令") is True

        # Non-exec requests should not match
        assert _looks_like_exec_request("read a file") is False

    def test_looks_like_spawn_request(self) -> None:
        """Test spawn request detection."""
        from w_bot.agents.core.agent import _looks_like_spawn_request

        # Spawn requests with specific keywords should match
        assert _looks_like_spawn_request("spawn a subagent") is True
        assert _looks_like_spawn_request("后台任务") is True
        assert _looks_like_spawn_request("并行执行") is True

        # "run in background" doesn't match - no specific keyword
        assert _looks_like_spawn_request("run in background") is False

        # Non-spawn requests should not match
        assert _looks_like_spawn_request("read a file") is False
