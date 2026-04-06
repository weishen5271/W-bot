"""Heuristic intent classifier for W-bot.

This module contains pure functions that analyze user text to determine
the intent behind their message using keyword matching heuristics.
Used as the fast path in the two-stage intent classification.
"""

from langchain_core.messages import AnyMessage, HumanMessage, ToolMessage
from typing import Any

from .intent import IntentType, IntentDecision, IntentResult, ToolRecommendation


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


# === Two-stage classification helpers ===

# Confidence threshold for heuristic classification
HEURISTIC_CONFIDENCE_THRESHOLD = 0.85


def _build_intent_decision(intent_type: IntentType, confidence: float, reasoning: str = "") -> IntentDecision:
    """Build an IntentDecision with the given parameters."""
    return IntentDecision(intent=intent_type, confidence=confidence, reasoning=reasoning)


def heuristic_classify(text: str, history: list[AnyMessage] | None = None) -> IntentResult:
    """Classify user intent using heuristics - millisecond-level response.

    This is Stage 1 of the two-stage intent classification.
    It covers all _looks_like_* detection results but wraps them in IntentResult format.

    Args:
        text: User input text
        history: Optional message history for context

    Returns:
        IntentResult with classified intent and tool recommendations
    """
    lowered = (text or "").strip().lower()

    # === Fast path: high-confidence no-tool intents ===
    if _looks_like_casual_chat(lowered):
        return IntentResult.casual_chat()

    if _looks_like_capability_question(lowered):
        return IntentResult.capability_question()

    # === Tool intent detection ===
    detected_intents: list[IntentDecision] = []
    tools_to_recommend: list[ToolRecommendation] = []

    if _looks_like_file_read_request(lowered):
        detected_intents.append(_build_intent_decision(IntentType.FILE_READ, confidence=0.85))
        tools_to_recommend.append(ToolRecommendation(tool_name="filesystem", exposure_level=1.0, reason="file read request"))

    if _looks_like_file_edit_request(lowered):
        detected_intents.append(_build_intent_decision(IntentType.FILE_EDIT, confidence=0.85))
        tools_to_recommend.append(ToolRecommendation(tool_name="filesystem", exposure_level=1.0, reason="file edit request"))

    if _looks_like_exec_request(lowered):
        detected_intents.append(_build_intent_decision(IntentType.COMMAND_EXEC, confidence=0.90))
        tools_to_recommend.append(ToolRecommendation(tool_name="shell", exposure_level=1.0, reason="command execution request"))

    if _looks_like_web_request(lowered):
        detected_intents.append(_build_intent_decision(IntentType.WEB_SEARCH, confidence=0.85))
        tools_to_recommend.append(ToolRecommendation(tool_name="web", exposure_level=1.0, reason="web request"))

    if _looks_like_spawn_request(lowered):
        detected_intents.append(_build_intent_decision(IntentType.SPAWN_SUBAGENT, confidence=0.90))
        tools_to_recommend.append(ToolRecommendation(tool_name="spawn", exposure_level=1.0, reason="spawn subagent request"))

    if _looks_like_cron_request(lowered):
        detected_intents.append(_build_intent_decision(IntentType.CRON_SCHEDULE, confidence=0.95))
        tools_to_recommend.append(ToolRecommendation(tool_name="cron", exposure_level=1.0, reason="cron schedule request"))

    if _looks_like_message_request(lowered):
        detected_intents.append(_build_intent_decision(IntentType.MESSAGE_SEND, confidence=0.90))
        tools_to_recommend.append(ToolRecommendation(tool_name="message", exposure_level=1.0, reason="message send request"))

    if _looks_like_project_inspection_request(lowered):
        detected_intents.append(_build_intent_decision(IntentType.PROJECT_INSPECTION, confidence=0.80))
        # Project inspection may need multiple tools
        tools_to_recommend.append(ToolRecommendation(tool_name="filesystem", exposure_level=0.8, reason="project inspection"))
        tools_to_recommend.append(ToolRecommendation(tool_name="shell", exposure_level=0.3, reason="project inspection"))

    # run_skill exposure decision
    if _should_expose_run_skill(lowered):
        if not any(t.tool_name == "run_skill" for t in tools_to_recommend):
            tools_to_recommend.append(ToolRecommendation(tool_name="run_skill", exposure_level=0.9, reason="skill execution request"))

    # === Determine if LLM is needed ===
    if len(detected_intents) == 1 and detected_intents[0].confidence >= HEURISTIC_CONFIDENCE_THRESHOLD:
        # Single high-confidence intent, no LLM needed
        return IntentResult(
            primary_intent=detected_intents[0],
            should_enable_tools=True,
            recommended_tools=tuple(tools_to_recommend),
            requires_llm=False,
            metadata={"source": "heuristic"},
        )

    if len(detected_intents) == 0:
        # No match found, conservatively enable tools and let LLM decide
        return IntentResult(
            primary_intent=_build_intent_decision(IntentType.UNKNOWN, confidence=0.0),
            should_enable_tools=True,
            recommended_tools=tuple(tools_to_recommend),
            requires_llm=True,
            metadata={"source": "heuristic", "reason": "no_match"},
        )

    # Multiple intents or insufficient confidence, needs LLM
    return IntentResult(
        primary_intent=detected_intents[0] if detected_intents else _build_intent_decision(IntentType.UNKNOWN, confidence=0.0),
        secondary_intents=tuple(detected_intents[1:]),
        should_enable_tools=True,
        recommended_tools=tuple(tools_to_recommend),
        requires_llm=True,
        metadata={"source": "heuristic", "reason": "low_confidence_or_multi_intent"},
    )
