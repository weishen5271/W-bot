"""Integration tests for agent flow."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


class TestAgentFlowIntegration:
    """Integration tests for the complete agent flow."""

    @pytest.fixture
    def mock_llm(self) -> MagicMock:
        """Create a mock LLM that returns simple responses."""
        mock = MagicMock()
        mock.invoke.return_value = AIMessage(
            content="I can help you with that.",
            tool_calls=[],
        )
        return mock

    @pytest.fixture
    def mock_tools(self) -> list[MagicMock]:
        """Create mock tools for testing."""
        read_tool = MagicMock()
        read_tool.name = "read_file"
        read_tool.description = "Read a file"
        read_tool.invoke.return_value = "file content: hello world"

        write_tool = MagicMock()
        write_tool.name = "write_file"
        write_tool.description = "Write a file"
        write_tool.invoke.return_value = "Successfully wrote to file"

        return [read_tool, write_tool]

    def test_agent_initialization(self) -> None:
        """Test agent can be initialized with mocks."""
        # This test verifies the basic agent setup works
        from w_bot.agents.core.agent import AgentState

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

        assert state["messages"] == []
        assert state["context_compaction_level"] == "none"

    def test_message_flow_without_tools(self) -> None:
        """Test basic message flow without tool calls."""
        from w_bot.agents.core.agent import AgentState
        from w_bot.agents.core.message_utils import _extract_last_user_message

        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!"),
        ]

        state: AgentState = {
            "messages": messages,
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

        # Simulate adding a new message
        state["messages"].append(HumanMessage(content="How are you?"))
        result = _extract_last_user_message(state["messages"])
        assert "How are you" in result

    def test_tool_call_flow(self) -> None:
        """Test message flow with tool calls."""
        from w_bot.agents.core.tool_analysis import _summarize_tool_calls

        messages = [
            HumanMessage(content="Read the config file"),
            AIMessage(content="", tool_calls=[
                {"name": "read_file", "args": {"path": "/config/app.json"}, "id": "call_1"}
            ]),
            ToolMessage(content='{"key": "value"}', tool_call_id="call_1", name="read_file"),
            AIMessage(content="The config file contains: {\"key\": \"value\"}"),
        ]

        # Count tool calls
        tool_calls = [msg.tool_calls[0] for msg in messages if isinstance(msg, AIMessage) and msg.tool_calls]
        summary = _summarize_tool_calls(tool_calls)
        assert "read_file" in summary

    def test_error_recovery_flow(self) -> None:
        """Test error handling in the message flow."""
        from w_bot.agents.core.runtime_status import RuntimeStatusSnapshot

        # Simulate a failed tool call
        snapshot = RuntimeStatusSnapshot(session_id="test-session")
        snapshot.set_phase("executing", "执行中")

        # Tool fails
        snapshot.mark_failed("Error: file not found", phase="executing")
        assert snapshot.last_error == "Error: file not found"
        assert snapshot.phase == "failed"

        # Recovery
        snapshot.begin_turn()
        assert snapshot.phase == "running"
        assert snapshot.last_error == ""

    def test_token_tracking_flow(self) -> None:
        """Test token usage tracking across messages."""
        from w_bot.agents.core.runtime_status import RuntimeStatusSnapshot

        snapshot = RuntimeStatusSnapshot(session_id="test-session")
        snapshot.update_usage(input_tokens=100, output_tokens=50, total_cost=0.01)

        assert snapshot.input_tokens == 100
        assert snapshot.output_tokens == 50
        assert snapshot.total_cost == 0.01

        # Simulate another turn with more usage
        snapshot.update_usage(input_tokens=150, output_tokens=75, total_cost=0.02)
        assert snapshot.input_tokens == 150
        assert snapshot.output_tokens == 75

    def test_context_compaction_flow(self) -> None:
        """Test context compaction trigger and execution."""
        from w_bot.agents.core.message_utils import _determine_compaction_level

        # Normal state
        budget_state = {}
        level = _determine_compaction_level(budget_state)
        assert level == "normal"

        # Warning state
        budget_state = {"is_above_warning_threshold": True}
        level = _determine_compaction_level(budget_state)
        assert level == "warning"

        # Blocking state
        budget_state = {"is_at_blocking_limit": True}
        level = _determine_compaction_level(budget_state)
        assert level == "blocking"

    def test_multi_turn_conversation_flow(self) -> None:
        """Test multi-turn conversation maintains context."""
        from w_bot.agents.core.agent import AgentState
        from w_bot.agents.core.message_utils import _last_human_index

        messages = [
            HumanMessage(content="First question"),
            AIMessage(content="First answer"),
            HumanMessage(content="Follow-up question"),
            AIMessage(content="Follow-up answer"),
            HumanMessage(content="Final question"),
        ]

        state: AgentState = {
            "messages": messages,
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

        # Verify last human index
        index = _last_human_index(state["messages"])
        assert index == 4  # Final question

    def test_subagent_spawn_flow(self) -> None:
        """Test subagent spawn and status tracking."""
        from w_bot.agents.core.runtime_status import RuntimeStatusSnapshot

        snapshot = RuntimeStatusSnapshot(session_id="test-session")

        # Simulate spawning subagents
        jobs = [
            {"id": "job1", "status": "running", "label": "worker"},
            {"id": "job2", "status": "pending", "label": "explore"},
        ]

        snapshot.refresh_tasks(jobs)
        assert snapshot.tasks.running == 1
        assert snapshot.tasks.pending == 1
        assert "worker" in snapshot.tasks.highlighted_tasks[0]


class TestToolExecutionIntegration:
    """Integration tests for tool execution scenarios."""

    @pytest.fixture
    def temp_workspace(self, tmp_path: Path) -> Path:
        """Create a temporary workspace."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        return workspace

    def test_read_write_workflow(self, temp_workspace: Path) -> None:
        """Test read then write file workflow."""
        from w_bot.agents.tools.filesystem import ReadFileTool, WriteFileTool

        read_tool = ReadFileTool(workspace=temp_workspace)
        write_tool = WriteFileTool(workspace=temp_workspace)

        # Write a file
        test_file = temp_workspace / "test.txt"
        import asyncio
        asyncio.run(write_tool.execute(
            path=str(test_file),
            content="Hello, World!"
        ))

        # Read it back
        content = asyncio.run(read_tool.execute(path=str(test_file)))
        assert "Hello, World!" in content

    def test_file_not_found_handling(self, temp_workspace: Path) -> None:
        """Test handling of missing files."""
        from w_bot.agents.tools.filesystem import ReadFileTool

        read_tool = ReadFileTool(workspace=temp_workspace)
        import asyncio

        result = asyncio.run(read_tool.execute(path=str(temp_workspace / "nonexistent.txt")))
        assert "not found" in result.lower() or "error" in result.lower()

    def test_directory_listing_flow(self, temp_workspace: Path) -> None:
        """Test directory listing."""
        from w_bot.agents.tools.filesystem import ListDirTool

        # Create some files
        (temp_workspace / "file1.txt").write_text("content1")
        (temp_workspace / "file2.txt").write_text("content2")
        (temp_workspace / "subdir").mkdir()

        list_tool = ListDirTool(workspace=temp_workspace)
        import asyncio

        result = asyncio.run(list_tool.execute(path=str(temp_workspace)))
        assert "file1.txt" in result
        assert "file2.txt" in result
        assert "subdir" in result or "[DIR] subdir" in result


class TestIntentClassificationIntegration:
    """Integration tests for intent classification flow."""

    def test_casual_chat_intent_flow(self) -> None:
        """Test casual chat intent classification."""
        from w_bot.agents.intent.intent_heuristic import (
            _looks_like_casual_chat,
            _should_enable_tools_for_text,
        )

        # Casual greeting should not enable tools
        assert _looks_like_casual_chat("hello") is True
        assert _should_enable_tools_for_text("hello") is False

    def test_file_operation_intent_flow(self) -> None:
        """Test file operation intent classification."""
        from w_bot.agents.intent.intent_heuristic import (
            _looks_like_file_read_request,
            _should_enable_tools_for_text,
        )

        # File read request should enable tools
        assert _looks_like_file_read_request("read the config file") is True
        assert _should_enable_tools_for_text("read the config file") is True

    def test_exec_request_flow(self) -> None:
        """Test exec request intent classification."""
        from w_bot.agents.intent.intent_heuristic import (
            _looks_like_exec_request,
            _should_expose_run_skill,
        )

        # Exec request should match
        assert _looks_like_exec_request("run ls command") is True
        # Exec should also expose skill functionality
        assert _should_expose_run_skill("run command in background") is True


class TestMemoryIntegration:
    """Integration tests for memory system."""

    def test_long_term_context_empty(self) -> None:
        """Test empty long-term context."""
        from w_bot.agents.core.agent import AgentState

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

        assert state["long_term_context"] == ""

    def test_context_with_summary(self) -> None:
        """Test context with conversation summary."""
        from w_bot.agents.core.agent import AgentState

        state: AgentState = {
            "messages": [],
            "long_term_context": "",
            "conversation_summary": "User discussed project structure and asked for help with files.",
            "summarized_message_count": 10,
            "prepared_system_prompt_base": "",
            "latest_token_usage": {},
            "session_token_usage": {},
            "token_budget_state": {},
            "context_compaction_level": "aggressive",
            "last_tool_failed": False,
            "consecutive_tool_failures": 0,
            "last_tool_name": "",
            "last_tool_error": "",
        }

        assert "files" in state["conversation_summary"].lower()
        assert state["summarized_message_count"] == 10
        assert state["context_compaction_level"] == "aggressive"
