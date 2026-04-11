"""Unit tests for message_utils module."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from w_bot.agents.core.message_utils import (
    _extract_last_user_message,
    _determine_compaction_level,
    _last_human_index,
    message_kind,
    sanitize_messages_for_llm,
    normalize_messages_for_llm,
)


class TestExtractLastUserMessage:
    """Tests for _extract_last_user_message function."""

    def test_empty_messages(self) -> None:
        """Test empty message list returns empty string."""
        result = _extract_last_user_message([])
        assert result == ""

    def test_human_message_content(self) -> None:
        """Test extracting content from human message."""
        messages = [
            AIMessage(content="Hello"),
            HumanMessage(content="What is 2+2?"),
        ]
        result = _extract_last_user_message(messages)
        assert "2+2" in result

    def test_last_human_message(self) -> None:
        """Test extracting last human message when multiple exist."""
        messages = [
            HumanMessage(content="First question"),
            AIMessage(content="Answer"),
            HumanMessage(content="Second question"),
        ]
        result = _extract_last_user_message(messages)
        assert "Second question" in result

    def test_no_human_message(self) -> None:
        """Test no human message returns empty string."""
        messages = [
            AIMessage(content="Hello"),
            SystemMessage(content="System"),
        ]
        result = _extract_last_user_message(messages)
        assert result == ""


class TestDetermineCompactionLevel:
    """Tests for _determine_compaction_level function."""

    def test_blocking_limit(self) -> None:
        """Test blocking level when at blocking limit."""
        budget_state = {"is_at_blocking_limit": True}
        result = _determine_compaction_level(budget_state)
        assert result == "blocking"

    def test_aggressive_compaction(self) -> None:
        """Test aggressive level when above auto-compact threshold."""
        budget_state = {"is_above_auto_compact_threshold": True}
        result = _determine_compaction_level(budget_state)
        assert result == "aggressive"

    def test_elevated_level(self) -> None:
        """Test elevated level when above error threshold."""
        budget_state = {"is_above_error_threshold": True}
        result = _determine_compaction_level(budget_state)
        assert result == "elevated"

    def test_warning_level(self) -> None:
        """Test warning level when above warning threshold."""
        budget_state = {"is_above_warning_threshold": True}
        result = _determine_compaction_level(budget_state)
        assert result == "warning"

    def test_normal_level(self) -> None:
        """Test normal level when no thresholds exceeded."""
        budget_state = {}
        result = _determine_compaction_level(budget_state)
        assert result == "normal"

    def test_empty_state(self) -> None:
        """Test empty budget state returns normal."""
        result = _determine_compaction_level({})
        assert result == "normal"


class TestLastHumanIndex:
    """Tests for _last_human_index function."""

    def test_empty_messages(self) -> None:
        """Test empty message list returns -1."""
        result = _last_human_index([])
        assert result == -1

    def test_finds_last_human(self) -> None:
        """Test finds index of last human message."""
        messages = [
            HumanMessage(content="First"),
            AIMessage(content="Second"),
            HumanMessage(content="Last"),
        ]
        result = _last_human_index(messages)
        assert result == 2

    def test_no_human_message(self) -> None:
        """Test no human message returns -1."""
        messages = [
            AIMessage(content="First"),
            SystemMessage(content="System"),
        ]
        result = _last_human_index(messages)
        assert result == -1

    def test_single_human_message(self) -> None:
        """Test single human message at index 0."""
        messages = [HumanMessage(content="Only")]
        result = _last_human_index(messages)
        assert result == 0


class TestMessageKind:
    """Tests for message_kind function."""

    def test_human_message(self) -> None:
        """Test HumanMessage returns 'human'."""
        msg = HumanMessage(content="Hello")
        result = message_kind(msg)
        assert result == "human"

    def test_tool_message(self) -> None:
        """Test ToolMessage returns 'tool'."""
        msg = ToolMessage(content="result", tool_call_id="123", name="test")
        result = message_kind(msg)
        assert result == "tool"

    def test_ai_message_with_tool_calls(self) -> None:
        """Test AIMessage with tool_calls returns 'action'."""
        # Create minimal AI message with tool calls
        msg = AIMessage(content="", tool_calls=[{"name": "test", "args": {}, "id": "1"}])
        result = message_kind(msg)
        assert result == "action"

    def test_ai_message_without_tool_calls(self) -> None:
        """Test AIMessage without tool_calls returns 'thought'."""
        msg = AIMessage(content="Hello")
        result = message_kind(msg)
        assert result == "thought"

    def test_unknown_type(self) -> None:
        """Test unknown message type returns 'other'."""
        result = message_kind("not a message")
        assert result == "other"


class TestSanitizeMessagesForLlm:
    """Tests for sanitize_messages_for_llm function."""

    def test_empty_messages(self) -> None:
        """Test empty message list returns empty list."""
        result = sanitize_messages_for_llm([])
        assert result == []

    def test_passes_through_normal_messages(self) -> None:
        """Test normal messages pass through unchanged."""
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi"),
        ]
        result = sanitize_messages_for_llm(messages)
        assert len(result) == 2

    def test_human_message_preserved(self) -> None:
        """Test human message is preserved."""
        messages = [HumanMessage(content="Hello")]
        result = sanitize_messages_for_llm(messages)
        assert len(result) == 1
        assert result[0].content == "Hello"


class TestNormalizeMessagesForLlm:
    """Tests for normalize_messages_for_llm function (requires normalizer)."""

    def test_empty_messages(self) -> None:
        """Test empty message list returns empty list."""
        result = normalize_messages_for_llm([], normalizer=None)
        assert result == []

    def test_without_normalizer_passes_through(self) -> None:
        """Test messages pass through when normalizer is None."""
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi"),
        ]
        result = normalize_messages_for_llm(messages, normalizer=None)
        assert len(result) == 2

    def test_ai_message_preserved(self) -> None:
        """Test AI message content is preserved."""
        messages = [AIMessage(content="Hello")]
        result = normalize_messages_for_llm(messages, normalizer=None)
        assert len(result) == 1
        assert result[0].content == "Hello"
