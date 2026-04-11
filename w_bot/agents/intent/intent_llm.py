"""LLM-based intent classifier for W-bot.

This module implements LLM-based intent classification as Stage 2
of the two-stage intent classification system.
"""

import asyncio
import json
from typing import Any

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage

from ..core.logging_config import get_logger
from .intent import IntentDecision, IntentResult, IntentType, ToolRecommendation

logger = get_logger(__name__)

LLM_INTENT_SYSTEM_PROMPT = """你是一个专业的意图识别专家。你的任务是根据用户输入判断其意图。

## 可识别的意图类型：
- CASUAL_CHAT: 闲聊问候（你好、谢谢、bye等）
- CAPABILITY_QUESTION: 询问bot能力（能做什么？）
- FILE_READ: 读取文件内容
- FILE_EDIT: 修改/编辑文件
- FILE_CREATE: 创建新文件
- COMMAND_EXEC: 执行命令/终端操作
- WEB_SEARCH: 搜索互联网
- WEB_FETCH: 获取网页内容
- SPAWN_SUBAGENT: 启动子agent
- RUN_SKILL: 运行技能
- CRON_SCHEDULE: 定时任务
- MESSAGE_SEND: 发送消息
- PROJECT_INSPECTION: 分析项目代码
- GENERAL_TASK: 一般性任务
- UNKNOWN: 无法判断

## 输出格式：
只输出一个JSON对象，包含以下字段：
{
  "intent": "意图类型",
  "confidence": 0.0-1.0,
  "reasoning": "简短推理",
  "recommended_tools": ["tool1", "tool2"] // 可选
}

confidence 评分标准：
- 0.95+: 非常确定
- 0.80-0.94: 较确定
- 0.60-0.79: 中等确定
- <0.60: 不确定，需要更多上下文

## 注意事项：
- 如果是闲聊或纯问题，不建议启用工具
- 如果是操作类请求，根据操作类型推荐工具
- 只推荐真正需要的工具，避免过度暴露
"""


def _build_intent_llm_messages(text: str, history: list[AnyMessage] | None = None) -> list[AnyMessage]:
    """Build messages for LLM intent classification."""
    history_context = ""
    if history:
        recent = history[-6:]  # Last 6 messages
        history_context = "\n\n对话历史（最近 6 条）：\n"
        for msg in recent:
            role = msg.__class__.__name__.replace("Message", "")
            content = getattr(msg, "content", "")
            history_context += f"- {role}: {content[:200]}\n"

    return [
        SystemMessage(content=LLM_INTENT_SYSTEM_PROMPT),
        HumanMessage(
            content=f"用户输入：\n{text}\n{history_context}\n\n请判断用户意图并输出 JSON。"
        ),
    ]


def _parse_llm_response(content: str) -> dict[str, Any] | None:
    """Parse LLM response and extract JSON."""
    try:
        # Handle code block format
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        return json.loads(content.strip())
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse LLM intent response as JSON: %s", exc)
        return None


def _convert_intent_string(intent_str: str) -> IntentType:
    """Convert intent string to IntentType enum."""
    # Try exact match first
    try:
        return IntentType[intent_str.upper()]
    except (KeyError, ValueError):
        pass

    # Try with underscores removed
    normalized = intent_str.upper().replace("_", "")
    for intent_type in IntentType:
        if intent_type.name.upper().replace("_", "") == normalized:
            return intent_type

    logger.warning("Unknown intent type: %s, defaulting to UNKNOWN", intent_str)
    return IntentType.UNKNOWN


async def _llm_classify_async(
    text: str,
    llm: Any,
    history: list[AnyMessage] | None = None,
    timeout: float = 10.0,
) -> IntentResult:
    """Perform LLM-based intent classification asynchronously.

    Args:
        text: User input text
        llm: LLM instance (must support ainvoke)
        history: Optional message history for context
        timeout: Timeout in seconds

    Returns:
        IntentResult with LLM classification
    """
    messages = _build_intent_llm_messages(text, history)

    try:
        # Use asyncio.wait_for to add timeout
        response = await asyncio.wait_for(llm.ainvoke(messages), timeout=timeout)
        content = getattr(response, "content", "").strip()

        if not content:
            raise ValueError("Empty LLM response")

        parsed = _parse_llm_response(content)
        if parsed is None:
            raise ValueError("Failed to parse LLM response")

        intent_str = parsed.get("intent", "UNKNOWN")
        confidence = float(parsed.get("confidence", 0.5))
        reasoning = str(parsed.get("reasoning", ""))
        recommended_tool_names = parsed.get("recommended_tools", [])

        intent_type = _convert_intent_string(intent_str)

        # Build tool recommendations
        tools = [
            ToolRecommendation(tool_name=tool, exposure_level=1.0, reason=f"recommended by LLM: {reasoning}")
            for tool in recommended_tool_names
        ]

        return IntentResult(
            primary_intent=IntentDecision(intent=intent_type, confidence=confidence, reasoning=reasoning),
            should_enable_tools=len(tools) > 0 or intent_type not in (IntentType.CASUAL_CHAT, IntentType.CAPABILITY_QUESTION),
            recommended_tools=tuple(tools),
            requires_llm=False,  # LLM decision is complete
            metadata={"source": "llm"},
        )

    except asyncio.TimeoutError:
        logger.warning("LLM intent classification timed out after %s seconds", timeout)
        return IntentResult.default_unknown()
    except Exception as exc:
        logger.warning("LLM intent classification failed: %s", exc)
        return IntentResult(
            primary_intent=IntentDecision(IntentType.UNKNOWN, confidence=0.0, reasoning=str(exc)),
            should_enable_tools=True,
            requires_llm=False,
            metadata={"source": "llm_fallback", "error": str(exc)},
        )


def llm_classify_sync(text: str, llm: Any, history: list[AnyMessage] | None = None) -> IntentResult:
    """Synchronous wrapper for LLM intent classification.

    This is a fallback for cases where async is not available.

    Args:
        text: User input text
        llm: LLM instance (must support invoke, not ainvoke)
        history: Optional message history

    Returns:
        IntentResult with LLM classification
    """
    messages = _build_intent_llm_messages(text, history)

    try:
        response = llm.invoke(messages)
        content = getattr(response, "content", "").strip()

        if not content:
            raise ValueError("Empty LLM response")

        parsed = _parse_llm_response(content)
        if parsed is None:
            raise ValueError("Failed to parse LLM response")

        intent_str = parsed.get("intent", "UNKNOWN")
        confidence = float(parsed.get("confidence", 0.5))
        reasoning = str(parsed.get("reasoning", ""))
        recommended_tool_names = parsed.get("recommended_tools", [])

        intent_type = _convert_intent_string(intent_str)

        # Build tool recommendations
        tools = [
            ToolRecommendation(tool_name=tool, exposure_level=1.0, reason=f"recommended by LLM: {reasoning}")
            for tool in recommended_tool_names
        ]

        return IntentResult(
            primary_intent=IntentDecision(intent=intent_type, confidence=confidence, reasoning=reasoning),
            should_enable_tools=len(tools) > 0 or intent_type not in (IntentType.CASUAL_CHAT, IntentType.CAPABILITY_QUESTION),
            recommended_tools=tuple(tools),
            requires_llm=False,
            metadata={"source": "llm"},
        )

    except Exception as exc:
        logger.warning("LLM intent classification failed: %s", exc)
        return IntentResult(
            primary_intent=IntentDecision(IntentType.UNKNOWN, confidence=0.0, reasoning=str(exc)),
            should_enable_tools=True,
            requires_llm=False,
            metadata={"source": "llm_fallback", "error": str(exc)},
        )
