"""Unit tests for intent_heuristic module."""

from __future__ import annotations

import pytest

from w_bot.agents.intent.intent_heuristic import (
    _contains_any,
    _looks_like_casual_chat,
    _looks_like_capability_question,
    _looks_like_project_inspection_request,
    _looks_like_file_read_request,
    _looks_like_file_edit_request,
    _looks_like_web_request,
    _looks_like_exec_request,
    _looks_like_spawn_request,
    _should_expose_run_skill,
    _should_enable_tools_for_text,
    _should_check_completion_for_turn,
)


class TestContainsAny:
    """Tests for _contains_any helper function."""

    def test_finds_matching_keyword(self) -> None:
        """Test returns True when keyword is found."""
        assert _contains_any("hello world", ("hello",)) is True
        assert _contains_any("hello world", ("world",)) is True
        assert _contains_any("hello world", ("foo", "bar", "hello")) is True

    def test_no_match(self) -> None:
        """Test returns False when no keyword matches."""
        assert _contains_any("hello world", ("foo", "bar")) is False

    def test_empty_text(self) -> None:
        """Test empty text returns False."""
        assert _contains_any("", ("hello",)) is False

    def test_empty_keywords(self) -> None:
        """Test empty keywords tuple returns False."""
        assert _contains_any("hello world", ()) is False


class TestLooksLikeCasualChat:
    """Tests for _looks_like_casual_chat function."""

    def test_short_smalltalk_hello(self) -> None:
        """Test 'hello' matches casual chat."""
        assert _looks_like_casual_chat("hello") is True

    def test_short_smalltalk_hi(self) -> None:
        """Test 'hi' matches casual chat."""
        assert _looks_like_casual_chat("hi") is True

    def test_short_smalltalk_chinese(self) -> None:
        """Test Chinese greetings match casual chat."""
        assert _looks_like_casual_chat("你好") is True
        assert _looks_like_casual_chat("早上好") is True
        assert _looks_like_casual_chat("晚上好") is True

    def test_short_smalltalk_thanks(self) -> None:
        """Test thanks matches casual chat."""
        assert _looks_like_casual_chat("谢谢") is True
        assert _looks_like_casual_chat("thanks") is True
        assert _looks_like_casual_chat("thank you") is True

    def test_bye(self) -> None:
        """Test 'bye' matches casual chat."""
        assert _looks_like_casual_chat("bye") is True
        assert _looks_like_casual_chat("再见") is True

    def test_empty_text(self) -> None:
        """Test empty text returns True (defaults to casual)."""
        assert _looks_like_casual_chat("") is True

    def test_longer_text_not_casual(self) -> None:
        """Test longer casual text doesn't match."""
        assert _looks_like_casual_chat("hello there, how are you today?") is False


class TestLooksLikeCapabilityQuestion:
    """Tests for _looks_like_capability_question function."""

    def test_chinese_capability_with_question(self) -> None:
        """Test Chinese capability questions with question mark match."""
        # "是否支持联网搜索?" - 有 "是否" 开头和问号
        assert _looks_like_capability_question("是否支持联网搜索?") is True
        # "do you support web search?" - 有 "do you" 开头和问号
        assert _looks_like_capability_question("do you support web search?") is True

    def test_chinese_capability_with_can(self) -> None:
        """Test Chinese capability starting with '能' or '可以' matches."""
        # "能处理这个任务吗?" - 以 "能" 开头，有问号
        assert _looks_like_capability_question("能处理这个任务吗?") is True

    def test_english_capability_question(self) -> None:
        """Test English capability questions match."""
        assert _looks_like_capability_question("can you read files?") is True
        assert _looks_like_capability_question("are you able to help me?") is True

    def test_explicit_execution_rejected(self) -> None:
        """Test explicit execution requests don't match."""
        assert _looks_like_capability_question("read this file") is False
        assert _looks_like_capability_question("/read file.txt") is False
        assert _looks_like_capability_question("帮我读文件") is False

    def test_empty_text(self) -> None:
        """Test empty text returns False."""
        assert _looks_like_capability_question("") is False


class TestLooksLikeProjectInspectionRequest:
    """Tests for _looks_like_project_inspection_request function."""

    def test_chinese_project_keywords(self) -> None:
        """Test Chinese project inspection keywords match."""
        assert _looks_like_project_inspection_request("当前项目结构") is True
        assert _looks_like_project_inspection_request("这个项目使用什么框架?") is True
        assert _looks_like_project_inspection_request("查看代码库") is True

    def test_english_project_keywords(self) -> None:
        """Test English project inspection keywords match."""
        assert _looks_like_project_inspection_request("project structure") is True
        assert _looks_like_project_inspection_request("inspect the codebase") is True
        assert _looks_like_project_inspection_request("explain the implementation") is True

    def test_why_questions_match(self) -> None:
        """Test 'why' questions about code match."""
        assert _looks_like_project_inspection_request("why is this implemented?") is True
        assert _looks_like_project_inspection_request("why does it fail?") is True


class TestLooksLikeFileReadRequest:
    """Tests for _looks_like_file_read_request function."""

    def test_chinese_read_keywords(self) -> None:
        """Test Chinese file read keywords match."""
        assert _looks_like_file_read_request("读取文件") is True
        assert _looks_like_file_read_request("打开这个文件") is True
        assert _looks_like_file_read_request("查看目录结构") is True

    def test_english_read_keywords(self) -> None:
        """Test English file read keywords match."""
        assert _looks_like_file_read_request("read file.txt") is True
        assert _looks_like_file_read_request("open config.json") is True
        assert _looks_like_file_read_request("inspect module.py") is True

    def test_file_extensions_match(self) -> None:
        """Test file extensions trigger read intent."""
        assert _looks_like_file_read_request("explain this .py file") is True
        assert _looks_like_file_read_request("readme.md content") is True


class TestLooksLikeFileEditRequest:
    """Tests for _looks_like_file_edit_request function."""

    def test_chinese_edit_keywords(self) -> None:
        """Test Chinese file edit keywords match."""
        assert _looks_like_file_edit_request("修改文件") is True
        assert _looks_like_file_edit_request("删除这行代码") is True
        assert _looks_like_file_edit_request("新增一个函数") is True

    def test_english_edit_keywords(self) -> None:
        """Test English file edit keywords match."""
        assert _looks_like_file_edit_request("edit this file") is True
        assert _looks_like_file_edit_request("patch the code") is True
        assert _looks_like_file_edit_request("update config.json") is True


class TestLooksLikeWebRequest:
    """Tests for _looks_like_web_request function."""

    def test_chinese_web_keywords(self) -> None:
        """Test Chinese web search keywords match."""
        assert _looks_like_web_request("搜索最新新闻") is True
        assert _looks_like_web_request("联网查询天气") is True
        assert _looks_like_web_request("打开这个网页") is True

    def test_english_web_keywords(self) -> None:
        """Test English web search keywords match."""
        assert _looks_like_web_request("web search for python") is True
        assert _looks_like_web_request("fetch the homepage") is True
        assert _looks_like_web_request("browse online documentation") is True


class TestLooksLikeExecRequest:
    """Tests for _looks_like_exec_request function."""

    def test_chinese_exec_keywords(self) -> None:
        """Test Chinese exec keywords match."""
        assert _looks_like_exec_request("运行这个命令") is True
        assert _looks_like_exec_request("执行shell脚本") is True
        assert _looks_like_exec_request("终端执行ls") is True

    def test_english_exec_keywords(self) -> None:
        """Test English exec keywords match."""
        assert _looks_like_exec_request("run command ls") is True
        assert _looks_like_exec_request("execute python script") is True
        assert _looks_like_exec_request("shell echo hello") is True


class TestLooksLikeSpawnRequest:
    """Tests for _looks_like_spawn_request function."""

    def test_spawn_keyword(self) -> None:
        """Test 'spawn' keyword matches."""
        assert _looks_like_spawn_request("spawn a subagent") is True

    def test_chinese_subagent_keywords(self) -> None:
        """Test Chinese subagent keywords match."""
        assert _looks_like_spawn_request("启动子 agent") is True
        assert _looks_like_spawn_request("子agent执行") is True

    def test_background_task_keyword(self) -> None:
        """Test '后台任务' matches."""
        assert _looks_like_spawn_request("后台任务处理") is True

    def test_parallel_keyword(self) -> None:
        """Test '并行' matches."""
        assert _looks_like_spawn_request("并行执行任务") is True


class TestShouldExposeRunSkill:
    """Tests for _should_expose_run_skill function."""

    def test_run_skill_keyword(self) -> None:
        """Test 'run_skill' triggers exposure."""
        assert _should_expose_run_skill("run_skill my_skill") is True

    def test_subagent_keyword(self) -> None:
        """Test 'subagent' triggers exposure."""
        assert _should_expose_run_skill("run a subagent") is True

    def test_background_keyword(self) -> None:
        """Test 'background' triggers exposure."""
        assert _should_expose_run_skill("run in background") is True

    def test_parallel_keyword(self) -> None:
        """Test 'parallel' triggers exposure."""
        assert _should_expose_run_skill("parallel execution") is True

    def test_empty_text(self) -> None:
        """Test empty text returns False."""
        assert _should_expose_run_skill("") is False


class TestShouldEnableToolsForText:
    """Tests for _should_enable_tools_for_text function."""

    def test_casual_chat_disables_tools(self) -> None:
        """Test casual chat should not enable tools."""
        assert _should_enable_tools_for_text("hello") is False
        assert _should_enable_tools_for_text("你好") is False

    def test_capability_question_disables_tools(self) -> None:
        """Test capability questions should not enable tools (introspection only)."""
        # "can you help me?" matches capability question
        assert _should_enable_tools_for_text("can you help me?") is False
        # "能处理任务吗?" - starts with "能" so it's a capability question
        assert _should_enable_tools_for_text("能处理任务吗?") is False

    def test_exec_request_enables_tools(self) -> None:
        """Test exec requests should enable tools."""
        assert _should_enable_tools_for_text("run ls command") is True

    def test_file_operations_enable_tools(self) -> None:
        """Test file operations should enable tools."""
        assert _should_enable_tools_for_text("read this file") is True
        assert _should_enable_tools_for_text("edit config") is True

    def test_web_request_enables_tools(self) -> None:
        """Test web requests should enable tools."""
        assert _should_enable_tools_for_text("search for information") is True


class TestShouldCheckCompletionForTurn:
    """Tests for _should_check_completion_for_turn function."""

    def test_capability_question_no_completion_check(self) -> None:
        """Test capability question should not check completion."""
        history = []
        assert _should_check_completion_for_turn("can you help me?", history) is False

    def test_exec_request_check_completion(self) -> None:
        """Test exec request should check completion."""
        history = []
        assert _should_check_completion_for_turn("run ls command", history) is True

    def test_empty_text_no_completion_check(self) -> None:
        """Test empty text should not check completion."""
        history = []
        assert _should_check_completion_for_turn("", history) is False
