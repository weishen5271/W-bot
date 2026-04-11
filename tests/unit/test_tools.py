"""Unit tests for filesystem tools."""

from __future__ import annotations

from pathlib import Path

import pytest

from w_bot.agents.tools.base import FunctionTool, Tool
from w_bot.agents.tools.filesystem import (
    EditFileTool,
    ListDirTool,
    ReadFileTool,
    WriteFileTool,
    _find_match,
    _is_under,
    _resolve_path,
)


class TestResolvePath:
    """Tests for _resolve_path helper function."""

    def test_absolute_path_unchanged(self) -> None:
        """Test absolute paths are not modified."""
        result = _resolve_path("/tmp/test", workspace=None)
        # On macOS, /tmp is symlinked to /private/tmp
        assert result == Path("/tmp/test") or result == Path("/private/tmp/test")

    def test_relative_path_with_workspace(self) -> None:
        """Test relative paths are resolved relative to workspace."""
        workspace = Path("/workspace")
        result = _resolve_path("file.txt", workspace=workspace)
        assert result == workspace / "file.txt"

    def test_path_expands_user(self) -> None:
        """Test ~ is expanded to user home."""
        result = _resolve_path("~/test.txt")
        assert result.expanduser() == Path.home() / "test.txt"


class TestIsUnder:
    """Tests for _is_under helper function."""

    def test_file_under_directory(self) -> None:
        """Test file under directory returns True."""
        assert _is_under(Path("/a/b/file.txt"), Path("/a/b")) is True

    def test_file_outside_directory(self) -> None:
        """Test file outside directory returns False."""
        assert _is_under(Path("/a/c/file.txt"), Path("/a/b")) is False

    def test_sibling_directories(self) -> None:
        """Test sibling directories return False."""
        assert _is_under(Path("/a/b/file.txt"), Path("/a/c")) is False


class TestFindMatch:
    """Tests for _find_match helper function."""

    def test_exact_match(self) -> None:
        """Test exact string match."""
        content = "line1\nline2\nline3"
        match, count = _find_match(content, "line2")
        assert match == "line2"
        assert count == 1

    def test_no_match(self) -> None:
        """Test no match returns None."""
        content = "line1\nline2\nline3"
        match, count = _find_match(content, "nonexistent")
        assert match is None
        assert count == 0

    def test_multiple_matches(self) -> None:
        """Test multiple matches returns first and count."""
        content = "line\nline\nline"
        match, count = _find_match(content, "line")
        assert match == "line"
        assert count == 3

    def test_match_with_whitespace_differences(self) -> None:
        """Test match works with whitespace differences."""
        content = "  line1  \n  line2  \n  line3  "
        match, count = _find_match(content, "line1")
        assert match is not None

    def test_empty_old_text(self) -> None:
        """Test empty old_text matches at beginning with count = len+1."""
        content = "line1\nline2"
        match, count = _find_match(content, "")
        # Empty string is "in" any string, count("") = len(content) + 1
        assert match == ""
        assert count == len(content) + 1


class TestReadFileTool:
    """Tests for ReadFileTool."""

    @pytest.fixture
    def tool(self, temp_workspace: Path) -> ReadFileTool:
        """Create a ReadFileTool instance with workspace."""
        return ReadFileTool(workspace=temp_workspace)

    def test_name(self, tool: ReadFileTool) -> None:
        """Test tool has correct name."""
        assert tool.name == "read_file"

    def test_description(self, tool: ReadFileTool) -> None:
        """Test tool has description."""
        assert len(tool.description) > 0

    def test_parameters(self, tool: ReadFileTool) -> None:
        """Test tool has correct parameters schema."""
        params = tool.parameters
        assert params["type"] == "object"
        assert "path" in params["required"]
        assert "offset" in params["properties"]
        assert "limit" in params["properties"]

    @pytest.mark.asyncio
    async def test_read_existing_file(self, tool: ReadFileTool, temp_workspace: Path) -> None:
        """Test reading an existing file."""
        result = await tool.execute(path=str(temp_workspace / "test_file.txt"))
        assert "test content" in result
        assert "line 2" in result

    @pytest.mark.asyncio
    async def test_read_nonexistent_file(self, tool: ReadFileTool) -> None:
        """Test reading a nonexistent file returns error."""
        result = await tool.execute(path="/nonexistent/file.txt")
        assert "Error" in result or "not found" in result.lower()

    @pytest.mark.asyncio
    async def test_read_with_offset(self, tool: ReadFileTool, temp_workspace: Path) -> None:
        """Test reading with offset parameter."""
        result = await tool.execute(path=str(temp_workspace / "test_file.txt"), offset=2)
        assert "line 2" in result

    @pytest.mark.asyncio
    async def test_read_with_pagination(self, tool: ReadFileTool, temp_workspace: Path) -> None:
        """Test reading large file shows pagination."""
        result = await tool.execute(path=str(temp_workspace / "test_file.txt"), offset=1, limit=1)
        assert "Showing lines 1-1" in result

    @pytest.mark.asyncio
    async def test_read_empty_file(self, tool: ReadFileTool, temp_workspace: Path) -> None:
        """Test reading empty file."""
        empty_file = temp_workspace / "empty.txt"
        empty_file.write_text("")
        result = await tool.execute(path=str(empty_file))
        assert "Empty file" in result


class TestWriteFileTool:
    """Tests for WriteFileTool."""

    @pytest.fixture
    def tool(self, temp_workspace: Path) -> WriteFileTool:
        """Create a WriteFileTool instance with workspace."""
        return WriteFileTool(workspace=temp_workspace)

    def test_name(self, tool: WriteFileTool) -> None:
        """Test tool has correct name."""
        assert tool.name == "write_file"

    def test_parameters(self, tool: WriteFileTool) -> None:
        """Test tool has correct parameters schema."""
        params = tool.parameters
        assert params["type"] == "object"
        assert "path" in params["required"]
        assert "content" in params["required"]

    @pytest.mark.asyncio
    async def test_write_new_file(self, tool: WriteFileTool, temp_workspace: Path) -> None:
        """Test writing a new file."""
        new_file = temp_workspace / "new_file.txt"
        result = await tool.execute(path=str(new_file), content="new content")
        assert "Successfully wrote" in result
        assert new_file.exists()
        assert new_file.read_text() == "new content"

    @pytest.mark.asyncio
    async def test_write_creates_parent_dirs(self, tool: WriteFileTool, temp_workspace: Path) -> None:
        """Test writing creates parent directories."""
        new_file = temp_workspace / "subdir" / "new_file.txt"
        result = await tool.execute(path=str(new_file), content="nested content")
        assert "Successfully wrote" in result
        assert new_file.exists()

    @pytest.mark.asyncio
    async def test_write_overwrites_existing(self, tool: WriteFileTool, temp_workspace: Path) -> None:
        """Test writing overwrites existing file."""
        existing_file = temp_workspace / "test_file.txt"
        result = await tool.execute(path=str(existing_file), content="overwritten content")
        assert "Successfully wrote" in result
        assert existing_file.read_text() == "overwritten content"

    @pytest.mark.asyncio
    async def test_write_missing_path(self, tool: WriteFileTool) -> None:
        """Test writing without path returns error."""
        result = await tool.execute(path=None, content="test")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_write_missing_content(self, tool: WriteFileTool, temp_workspace: Path) -> None:
        """Test writing without content returns error."""
        result = await tool.execute(path=str(temp_workspace / "file.txt"), content=None)
        assert "Error" in result


class TestEditFileTool:
    """Tests for EditFileTool."""

    @pytest.fixture
    def tool(self, temp_workspace: Path) -> EditFileTool:
        """Create an EditFileTool instance with workspace."""
        return EditFileTool(workspace=temp_workspace)

    def test_name(self, tool: EditFileTool) -> None:
        """Test tool has correct name."""
        assert tool.name == "edit_file"

    def test_parameters(self, tool: EditFileTool) -> None:
        """Test tool has correct parameters schema."""
        params = tool.parameters
        assert params["type"] == "object"
        assert "path" in params["required"]
        assert "old_text" in params["required"]
        assert "new_text" in params["required"]
        assert "replace_all" in params["properties"]

    @pytest.mark.asyncio
    async def test_edit_existing_content(self, tool: EditFileTool, temp_workspace: Path) -> None:
        """Test editing existing content in a file."""
        test_file = temp_workspace / "test_file.txt"
        result = await tool.execute(
            path=str(test_file),
            old_text="test content",
            new_text="modified content",
        )
        assert "Successfully edited" in result
        assert "modified content" in test_file.read_text()

    @pytest.mark.asyncio
    async def test_edit_nonexistent_file(self, tool: EditFileTool) -> None:
        """Test editing nonexistent file returns error."""
        result = await tool.execute(
            path="/nonexistent/file.txt",
            old_text="old",
            new_text="new",
        )
        assert "Error" in result or "not found" in result.lower()

    @pytest.mark.asyncio
    async def test_edit_not_found(self, tool: EditFileTool, temp_workspace: Path) -> None:
        """Test editing nonexistent content returns error."""
        test_file = temp_workspace / "test_file.txt"
        result = await tool.execute(
            path=str(test_file),
            old_text="nonexistent text",
            new_text="new text",
        )
        assert "not found" in result.lower()

    @pytest.mark.asyncio
    async def test_edit_replace_all(self, tool: EditFileTool, temp_workspace: Path) -> None:
        """Test replace_all flag replaces all occurrences."""
        test_file = temp_workspace / "multi.txt"
        test_file.write_text("line\nline\nline")
        result = await tool.execute(
            path=str(test_file),
            old_text="line",
            new_text="REPLACED",
            replace_all=True,
        )
        assert "Successfully edited" in result
        content = test_file.read_text()
        assert content.count("REPLACED") == 3


class TestListDirTool:
    """Tests for ListDirTool."""

    @pytest.fixture
    def tool(self, temp_workspace: Path) -> ListDirTool:
        """Create a ListDirTool instance with workspace."""
        return ListDirTool(workspace=temp_workspace)

    def test_name(self, tool: ListDirTool) -> None:
        """Test tool has correct name."""
        assert tool.name == "list_dir"

    def test_parameters(self, tool: ListDirTool) -> None:
        """Test tool has correct parameters schema."""
        params = tool.parameters
        assert params["type"] == "object"
        assert "path" in params["required"]
        assert "recursive" in params["properties"]
        assert "max_entries" in params["properties"]

    @pytest.mark.asyncio
    async def test_list_directory(self, tool: ListDirTool, temp_workspace: Path) -> None:
        """Test listing a directory."""
        result = await tool.execute(path=str(temp_workspace))
        assert "test_file.txt" in result
        assert "subdir" in result

    @pytest.mark.asyncio
    async def test_list_nonexistent_directory(self, tool: ListDirTool) -> None:
        """Test listing nonexistent directory returns error."""
        result = await tool.execute(path="/nonexistent/dir")
        assert "Error" in result or "not found" in result.lower()

    @pytest.mark.asyncio
    async def test_list_recursive(self, tool: ListDirTool, temp_workspace: Path) -> None:
        """Test recursive listing includes nested files."""
        result = await tool.execute(path=str(temp_workspace), recursive=True)
        assert "nested.txt" in result or "subdir" in result

    @pytest.mark.asyncio
    async def test_list_respects_max_entries(self, tool: ListDirTool, temp_workspace: Path) -> None:
        """Test max_entries limits results."""
        result = await tool.execute(path=str(temp_workspace), max_entries=1)
        assert "truncated" in result.lower() or result.count("\n") <= 2

    @pytest.mark.asyncio
    async def test_list_ignores_ignored_dirs(self, tool: ListDirTool, temp_workspace: Path) -> None:
        """Test listing ignores common ignored directories."""
        ignored_dir = temp_workspace / "__pycache__"
        ignored_dir.mkdir()
        (ignored_dir / "cached.pyc").write_text("cached")
        result = await tool.execute(path=str(temp_workspace), recursive=True)
        assert "__pycache__" not in result


class TestFunctionTool:
    """Tests for FunctionTool."""

    def test_function_tool_basic(self) -> None:
        """Test FunctionTool with basic function."""
        def multiply(a: int, b: int) -> int:
            return a * b

        tool = FunctionTool(
            name="multiply",
            description="Multiply two numbers",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
                "required": ["a", "b"],
            },
            func=multiply,
        )

        assert tool.name == "multiply"
        assert tool.description == "Multiply two numbers"

    def test_function_tool_cast_params(self) -> None:
        """Test FunctionTool casts parameters correctly."""
        def add(a: int, b: int) -> int:
            return a + b

        tool = FunctionTool(
            name="add",
            description="Add two numbers",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
                "required": ["a", "b"],
            },
            func=add,
        )

        # String values should be cast to int
        result = tool.cast_params({"a": "5", "b": "3"})
        assert result["a"] == 5
        assert result["b"] == 3

    def test_function_tool_validate_params(self) -> None:
        """Test FunctionTool validates parameters."""
        def divide(a: int, b: int) -> float:
            return a / b

        tool = FunctionTool(
            name="divide",
            description="Divide two numbers",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer", "minimum": 1},
                },
                "required": ["a", "b"],
            },
            func=divide,
        )

        errors = tool.validate_params({"a": 10, "b": 2})
        assert len(errors) == 0

        errors = tool.validate_params({"a": 10, "b": 0})
        assert len(errors) > 0

    def test_function_tool_to_schema(self) -> None:
        """Test FunctionTool generates correct schema."""
        def noop() -> None:
            pass

        tool = FunctionTool(
            name="noop",
            description="Do nothing",
            parameters={"type": "object", "properties": {}},
            func=noop,
        )

        schema = tool.to_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "noop"
        assert schema["function"]["description"] == "Do nothing"


class TestToolBaseClass:
    """Tests for Tool base class parameter handling."""

    def test_resolve_type_string(self) -> None:
        """Test _resolve_type with string input."""
        result = Tool._resolve_type("string")
        assert result == "string"

    def test_resolve_type_list_with_null(self) -> None:
        """Test _resolve_type with list containing null."""
        result = Tool._resolve_type(["string", "null"])
        assert result == "string"

    def test_resolve_type_non_string(self) -> None:
        """Test _resolve_type with non-string input returns None."""
        result = Tool._resolve_type(123)
        assert result is None
