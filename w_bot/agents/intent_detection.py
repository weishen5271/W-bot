"""Intent detection heuristics for classifying user message intents.

This module contains pure functions that analyze user text to determine
the intent behind their message, such as whether they want to execute code,
read files, edit files, spawn subagents, send messages, schedule cron jobs, etc.
"""

from langchain_core.messages import AnyMessage, HumanMessage, ToolMessage
from typing import Any


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


def _looks_like_casual_chat(text: str) -> bool:
    lowered = (text or "").strip().lower()
    if not lowered:
        return True
    short_smalltalk = {
        "你好",
        "hello",
        "hi",
        "hey",
        "谢谢",
        "thanks",
        "thank you",
        "早上好",
        "晚上好",
        "在吗",
        "bye",
        "再见",
    }
    if lowered in short_smalltalk:
        return True
    if len(lowered) <= 12 and any(token in lowered for token in ("你好", "hello", "hi", "thanks", "谢谢")):
        return True
    return False


def _looks_like_capability_question(text: str) -> bool:
    lowered = (text or "").strip().lower()
    if not lowered:
        return False

    question_markers = ("?", "？", "吗", "么", "呢")
    capability_markers = (
        "能不能",
        "能否",
        "可不可以",
        "可以不可以",
        "可否",
        "是否支持",
        "支不支持",
        "能不能读取",
        "能不能读",
        "能不能看",
        "能读取",
        "能读",
        "能看",
        "支持读取",
        "支持读",
        "支持看",
        "能处理",
        "会不会",
        "can you",
        "are you able to",
        "do you support",
        "whether you can",
    )
    explicit_execution_markers = (
        "/",
        "\\",
        ".pdf",
        ".doc",
        ".docx",
        ".txt",
        ".md",
        "这个文件",
        "这个pdf",
        "该文件",
        "帮我读",
        "帮我看",
        "请读取",
        "请帮我",
        "read this",
        "open this",
        "summarize this",
    )

    has_question_tone = _contains_any(lowered, question_markers)
    has_capability_tone = _contains_any(lowered, capability_markers)
    if not has_capability_tone:
        return False
    if _contains_any(lowered, explicit_execution_markers):
        return False
    return has_question_tone or lowered.startswith(("能", "可以", "是否", "do you", "can you"))


def _looks_like_project_inspection_request(text: str) -> bool:
    return _contains_any(
        text,
        (
            "当前项目",
            "这个项目",
            "仓库",
            "代码库",
            "langgraph",
            "流程",
            "实现",
            "源码",
            "agent.py",
            "函数",
            "方法",
            "模块",
            "why is",
            "why does",
            "project",
            "codebase",
            "workflow",
            "implementation",
        ),
    )


def _looks_like_file_read_request(text: str) -> bool:
    return _contains_any(
        text,
        (
            "读取",
            "查看",
            "分析文件",
            "看下",
            "目录",
            "文件",
            "read ",
            "open ",
            "inspect",
            "list ",
            ".py",
            ".md",
            ".json",
        ),
    )


def _looks_like_file_edit_request(text: str) -> bool:
    return _contains_any(
        text,
        (
            "修改",
            "删除",
            "新增",
            "重构",
            "修复",
            "patch",
            "edit",
            "update",
            "rewrite",
            "change",
        ),
    )


def _looks_like_web_request(text: str) -> bool:
    return _contains_any(
        text,
        (
            "搜索",
            "联网",
            "网页",
            "网站",
            "最新",
            "web",
            "search",
            "fetch",
            "browse",
            "internet",
            "online",
        ),
    )


def _looks_like_exec_request(text: str) -> bool:
    return _contains_any(
        text,
        (
            "执行",
            "运行",
            "命令",
            "终端",
            "shell",
            "powershell",
            "bash",
            "cmd",
            "python ",
            "exec",
            "run ",
            "command",
            "terminal",
        ),
    )


def _looks_like_spawn_request(text: str) -> bool:
    return _contains_any(text, ("spawn", "子 agent", "子agent", "后台任务", "并行"))


def _should_expose_run_skill(text: str) -> bool:
    normalized = (text or "").strip().lower()
    if not normalized:
        return False
    return _contains_any(
        normalized,
        (
            "run_skill",
            "子 agent",
            "子agent",
            "subagent",
            "spawn",
            "并行",
            "并发",
            "后台",
            "异步",
            "隔离",
            "委派",
            "delegate",
            "parallel",
            "background",
            "isolated",
        ),
    )


def _looks_like_message_request(text: str) -> bool:
    return _contains_any(text, ("发送消息", "通知", "message ", "send message"))


def _looks_like_cron_request(text: str) -> bool:
    return _contains_any(text, ("cron", "定时", "schedule", "scheduled"))


def _has_tool_messages_since_last_human(messages: list[AnyMessage]) -> bool:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return False
        if isinstance(message, ToolMessage):
            return True
    return False


def _response_looks_incomplete(text: str) -> bool:
    lowered = (text or "").strip().lower()
    if not lowered:
        return True
    markers = (
        "我先",
        "我会先",
        "接下来",
        "下一步",
        "然后我会",
        "我将",
        "稍后",
        "先检查",
        "先看看",
        "先分析",
        "正在",
        "请稍等",
        "如果你愿意",
        "如果需要我可以",
        "你可以再",
        "建议你",
        "i will",
        "i'll",
        "next,",
        "next step",
        "let me",
        "i'm going to",
        "please provide",
        "if you want, i can",
    )
    if any(marker in lowered for marker in markers):
        return True
    if lowered.endswith(("...", "…", "：", ":")):
        return True
    return False


def _continue_current_task_prompt() -> str:
    return (
        "继续执行当前任务，不要停在计划、说明或中间状态。"
        "如果还需要工具，直接继续调用工具；如果任务其实已经完成，只输出最终结果。"
        "不要重复刚才已经说过的计划或过程说明。"
        "只有在确实缺少必要信息、权限不足或存在真实阻塞时，才停止并明确说明阻塞点。"
    )


def _should_enable_tools_for_text(text: str) -> bool:
    lowered = (text or "").strip().lower()
    if not lowered:
        return False
    if _looks_like_casual_chat(lowered):
        return False
    if _looks_like_capability_question(lowered):
        return False
    if _looks_like_exec_request(lowered):
        return True
    if _looks_like_web_request(lowered):
        return True
    if _looks_like_file_edit_request(lowered):
        return True
    if _looks_like_file_read_request(lowered):
        return True
    if _looks_like_project_inspection_request(lowered):
        return True
    if _looks_like_spawn_request(lowered) or _looks_like_message_request(lowered) or _looks_like_cron_request(lowered):
        return True
    return True


def _should_check_completion_for_turn(user_text: str, history: list[AnyMessage]) -> bool:
    lowered = (user_text or "").strip().lower()
    if not lowered:
        return False
    if _looks_like_capability_question(lowered):
        return False
    if any(
        (
            _looks_like_exec_request(lowered),
            _looks_like_web_request(lowered),
            _looks_like_file_edit_request(lowered),
            _looks_like_file_read_request(lowered),
            _looks_like_project_inspection_request(lowered),
            _looks_like_spawn_request(lowered),
            _looks_like_message_request(lowered),
            _looks_like_cron_request(lowered),
        )
    ):
        return True
    return _has_tool_messages_since_last_human(history)
