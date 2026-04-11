"""Unit tests for tool_analysis module."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from w_bot.agents.core.tool_analysis import (
    _summarize_tool_calls,
    _count_tool_steps_since_last_human,
    _count_named_tool_calls_since_last_human,
    _same_tool_call_streak,
    _tool_call_signature,
    _is_tool_failure_content,
    _extract_tool_failure_summary,
    _extract_exit_code,
)


class TestSummarizeToolCalls:
    """Tests for _summarize_tool_calls function."""

    def test_empty_list(self) -> None:
        """Test empty tool calls list."""
        result = _summarize_tool_calls([])
        assert "0 个工具" in result

    def test_single_tool(self) -> None:
        """Test single tool call."""
        tool_calls = [{"name": "read_file", "args": {}}]
        result = _summarize_tool_calls(tool_calls)
        assert "read_file" in result

    def test_multiple_tools(self) -> None:
        """Test multiple tool calls."""
        tool_calls = [
            {"name": "read_file", "args": {}},
            {"name": "write_file", "args": {}},
        ]
        result = _summarize_tool_calls(tool_calls)
        assert "read_file" in result
        assert "write_file" in result

    def test_non_dict_items_skipped(self) -> None:
        """Test non-dict items are skipped."""
        tool_calls = [{"name": "read_file"}, "invalid", 123, None]
        result = _summarize_tool_calls(tool_calls)
        assert "read_file" in result

    def test_empty_name_skipped(self) -> None:
        """Test tool calls with empty names are skipped."""
        tool_calls = [{"name": "read_file"}, {"name": ""}, {"name": "  "}]
        result = _summarize_tool_calls(tool_calls)
        assert result.count("read_file") == 1


class TestCountToolStepsSinceLastHuman:
    """Tests for _count_tool_steps_since_last_human function."""

    def test_empty_messages(self) -> None:
        """Test empty message list."""
        result = _count_tool_steps_since_last_human([])
        assert result == 0

    def test_no_human_message(self) -> None:
        """Test messages without human message."""
        messages = [
            AIMessage(content="Hello", tool_calls=[{"name": "tool1", "args": {}, "id": "1"}]),
            AIMessage(content="World", tool_calls=[{"name": "tool2", "args": {}, "id": "2"}]),
        ]
        result = _count_tool_steps_since_last_human(messages)
        assert result == 2

    def test_counts_after_last_human(self) -> None:
        """Test counts only tool calls after last human message."""
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi", tool_calls=[{"name": "tool1", "args": {}, "id": "1"}]),
            AIMessage(content="There", tool_calls=[{"name": "tool2", "args": {}, "id": "2"}]),
        ]
        result = _count_tool_steps_since_last_human(messages)
        assert result == 2

    def test_ignores_messages_before_human(self) -> None:
        """Test messages before human are ignored."""
        messages = [
            AIMessage(content="Before", tool_calls=[{"name": "tool1", "args": {}, "id": "1"}]),
            HumanMessage(content="Human input"),
            AIMessage(content="After", tool_calls=[{"name": "tool2", "args": {}, "id": "2"}]),
        ]
        result = _count_tool_steps_since_last_human(messages)
        assert result == 1


class TestCountNamedToolCallsSinceLastHuman:
    """Tests for _count_named_tool_calls_since_last_human function."""

    def test_empty_messages(self) -> None:
        """Test empty message list."""
        result = _count_named_tool_calls_since_last_human([], "read_file")
        assert result == 0

    def test_counts_matching_tool(self) -> None:
        """Test counts only matching tool name."""
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="", tool_calls=[{"name": "read_file", "args": {}, "id": "1"}]),
            AIMessage(content="", tool_calls=[{"name": "read_file", "args": {}, "id": "2"}]),
            AIMessage(content="", tool_calls=[{"name": "write_file", "args": {}, "id": "3"}]),
        ]
        result = _count_named_tool_calls_since_last_human(messages, "read_file")
        assert result == 2

    def test_case_insensitive(self) -> None:
        """Test tool name matching is case insensitive."""
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="", tool_calls=[{"name": "Read_File", "args": {}, "id": "1"}]),
        ]
        result = _count_named_tool_calls_since_last_human(messages, "read_file")
        assert result == 1

    def test_empty_tool_name(self) -> None:
        """Test empty tool name returns 0."""
        messages = [HumanMessage(content="Hello")]
        result = _count_named_tool_calls_since_last_human(messages, "")
        assert result == 0


class TestSameToolCallStreak:
    """Tests for _same_tool_call_streak function."""

    def test_empty_messages(self) -> None:
        """Test empty message list."""
        name, count = _same_tool_call_streak([])
        assert name == ""
        assert count == 0

    def test_no_tool_calls(self) -> None:
        """Test messages without tool calls."""
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi"),
        ]
        name, count = _same_tool_call_streak(messages)
        assert name == ""
        assert count == 0

    def test_single_tool_call(self) -> None:
        """Test single tool call."""
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="", tool_calls=[{"name": "read_file", "args": {}, "id": "1"}]),
        ]
        name, count = _same_tool_call_streak(messages)
        # name includes args, so it will be "read_file:{}"
        assert name.startswith("read_file")
        assert count == 1

    def test_repeated_same_tool(self) -> None:
        """Test repeated same tool call with same args."""
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="", tool_calls=[{"name": "read_file", "args": {}, "id": "1"}]),
            ToolMessage(content="result a", tool_call_id="1", name="read_file"),
            AIMessage(content="", tool_calls=[{"name": "read_file", "args": {}, "id": "2"}]),
            ToolMessage(content="result b", tool_call_id="2", name="read_file"),
            AIMessage(content="", tool_calls=[{"name": "read_file", "args": {}, "id": "3"}]),
        ]
        name, count = _same_tool_call_streak(messages)
        # With same args, all signatures are identical
        assert name.startswith("read_file")
        assert count == 3  # All three AIMessage tool calls have same signature


class TestToolCallSignature:
    """Tests for _tool_call_signature function."""

    def test_empty_tool_calls(self) -> None:
        """Test empty tool calls."""
        result = _tool_call_signature([])
        assert result == ""

    def test_single_tool_call(self) -> None:
        """Test single tool call signature includes name and args."""
        tool_calls = [{"name": "read_file", "args": {"path": "/tmp/test"}}]
        result = _tool_call_signature(tool_calls)
        assert "read_file" in result
        assert "/tmp/test" in result

    def test_multiple_tool_calls(self) -> None:
        """Test multiple tool calls are joined with pipe."""
        tool_calls = [
            {"name": "read_file", "args": {"path": "/a"}},
            {"name": "write_file", "args": {"path": "/b"}},
        ]
        result = _tool_call_signature(tool_calls)
        assert "read_file" in result
        assert "write_file" in result
        assert "|" in result


class TestIsToolFailureContent:
    """Tests for _is_tool_failure_content function."""

    def test_empty_content(self) -> None:
        """Test empty content is not failure."""
        result = _is_tool_failure_content("")
        assert result is False

    def test_error_prefix(self) -> None:
        """Test content starting with Error is failure."""
        result = _is_tool_failure_content("Error: file not found")
        assert result is True

    def test_stderr_prefix(self) -> None:
        """Test content starting with stderr is failure."""
        result = _is_tool_failure_content("stderr: command not found")
        assert result is True

    def test_tool_execution_failed_prefix(self) -> None:
        """Test content starting with 'Tool execution failed' is failure."""
        result = _is_tool_failure_content("Tool execution failed: timeout")
        assert result is True

    def test_tool_not_found_prefix(self) -> None:
        """Test content starting with 'Tool not found' is failure."""
        result = _is_tool_failure_content("Tool not found: read_file")
        assert result is True

    def test_json_error(self) -> None:
        """Test JSON with error key is failure."""
        result = _is_tool_failure_content('{"error": "something went wrong"}')
        assert result is True

    def test_nonzero_exit_code(self) -> None:
        """Test content with non-zero exit code is failure."""
        result = _is_tool_failure_content("Command failed. Exit code: 1")
        assert result is True


class TestExtractToolFailureSummary:
    """Tests for _extract_tool_failure_summary function."""

    def test_empty_content(self) -> None:
        """Test empty content returns empty."""
        result = _extract_tool_failure_summary("")
        assert result == ""

    def test_extracts_error_message(self) -> None:
        """Test extracts error message."""
        content = "Error: file not found at /path/to/file"
        result = _extract_tool_failure_summary(content)
        assert "file not found" in result.lower() or "error" in result.lower()


class TestExtractExitCode:
    """Tests for _extract_exit_code function."""

    def test_no_exit_code(self) -> None:
        """Test content without 'Exit code:' returns None."""
        result = _extract_exit_code("Some error message")
        assert result is None

    def test_exit_code_0(self) -> None:
        """Test exit code 0 returns None (success)."""
        result = _extract_exit_code("Command succeeded. Exit code: 0")
        assert result == 0

    def test_exit_code_1(self) -> None:
        """Test exit code 1 is returned."""
        content = "Command failed. Exit code: 1"
        result = _extract_exit_code(content)
        assert result == 1

    def test_exit_code_127(self) -> None:
        """Test exit code 127 is returned."""
        content = "Command not found. Exit code: 127"
        result = _extract_exit_code(content)
        assert result == 127
