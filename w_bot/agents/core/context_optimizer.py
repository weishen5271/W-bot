from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage

from .config import TokenOptimizationSettings
from .logging_config import get_logger
from .message_utils import (
    _apply_context_compaction_strategy,
    _build_summary_fallback,
    _determine_compaction_level,
    _messages_to_summary_text,
    _recent_window_start,
    _to_text_content,
)
from .token_tracker import TokenBudgetManager, token_count_with_estimation

logger = get_logger(__name__)


@dataclass(frozen=True)
class OptimizedContext:
    conversation_summary: str
    summarized_message_count: int
    recent_messages: list[AnyMessage]
    token_budget_state: dict[str, Any]
    context_compaction_level: str


class ContextOptimizer:
    """Prepare compact recent context and rolling summary for AgentRuntime."""

    def __init__(
        self,
        *,
        settings: TokenOptimizationSettings | None = None,
        llm: Any = None,
    ) -> None:
        self._settings = settings or _default_token_optimization_settings()
        self._llm = llm
        self._token_budget = TokenBudgetManager(
            context_window_tokens=self._settings.context_window_tokens,
            auto_compact_buffer_tokens=self._settings.auto_compact_buffer_tokens,
            warning_threshold_buffer_tokens=self._settings.warning_threshold_buffer_tokens,
            error_threshold_buffer_tokens=self._settings.error_threshold_buffer_tokens,
            blocking_buffer_tokens=self._settings.blocking_buffer_tokens,
        )

    def prepare(
        self,
        *,
        history: list[AnyMessage],
        conversation_summary: str = "",
        summarized_message_count: int = 0,
        defer_summary_update: bool = True,
    ) -> OptimizedContext:
        if not self._settings.enabled:
            return OptimizedContext(
                conversation_summary=conversation_summary,
                summarized_message_count=max(0, int(summarized_message_count or 0)),
                recent_messages=history,
                token_budget_state={},
                context_compaction_level="off",
            )

        summary = str(conversation_summary or "")
        summarized_count = max(0, int(summarized_message_count or 0))
        estimated_tokens = token_count_with_estimation(history)
        budget_state = self._token_budget.calculate_state(estimated_tokens)
        budget_dict = budget_state.to_dict()
        compaction_level = _determine_compaction_level(budget_dict)

        recent_turns = self._settings.max_recent_user_turns
        if budget_state.is_above_error_threshold:
            recent_turns = max(2, min(recent_turns, 4))
        if budget_state.is_at_blocking_limit:
            recent_turns = 1

        recent_start = _recent_window_start(history, max_user_turns=recent_turns)
        target_end = min(recent_start, len(history))
        unsummarized_count = max(0, target_end - summarized_count)
        should_force_summary = budget_state.is_above_auto_compact_threshold or budget_state.is_at_blocking_limit

        if unsummarized_count >= self._settings.summary_trigger_messages or (
            should_force_summary and target_end > summarized_count
        ):
            if defer_summary_update:
                recent_source = history[summarized_count:] or history
                return OptimizedContext(
                    conversation_summary=summary,
                    summarized_message_count=summarized_count,
                    recent_messages=_apply_context_compaction_strategy(
                        recent_source,
                        compaction_level=compaction_level,
                    ),
                    token_budget_state=budget_dict,
                    context_compaction_level=compaction_level,
                )

            transcript = _messages_to_summary_text(history[summarized_count:target_end])
            if transcript:
                summary = self._update_summary(
                    existing_summary=summary,
                    transcript=transcript,
                    max_chars=self._settings.summary_max_chars,
                )
                summarized_count = target_end

        return OptimizedContext(
            conversation_summary=summary,
            summarized_message_count=summarized_count,
            recent_messages=_apply_context_compaction_strategy(
                history[recent_start:],
                compaction_level=compaction_level,
            ),
            token_budget_state=budget_dict,
            context_compaction_level=compaction_level,
        )

    def _update_summary(
        self,
        *,
        existing_summary: str,
        transcript: str,
        max_chars: int,
    ) -> str:
        prompt = (
            "请维护一段会话滚动摘要，用于后续对话上下文压缩。"
            "请输出中文结构化纯文本，并严格使用以下小节："
            "【目标与约束】【关键决策】【已完成】【待处理】【风险与阻塞】。"
            "只保留后续继续任务最需要的信息，删除闲聊、重复描述和冗长工具输出。"
            f"总长度不超过 {max_chars} 字。"
        )
        payload = (
            f"已有摘要：\n{existing_summary or '无'}\n\n"
            f"新增对话片段：\n{transcript}\n\n"
            "请输出更新后的摘要："
        )
        text = ""
        if self._llm is not None:
            try:
                result = self._llm.invoke(
                    [
                        SystemMessage(content=prompt),
                        HumanMessage(content=payload),
                    ]
                )
                text = _to_text_content(result.content).strip()
            except Exception:
                logger.exception("Failed to update AgentRuntime rolling summary with LLM")
        if not text:
            text = _build_summary_fallback(existing_summary=existing_summary, transcript=transcript)
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rstrip()


def _default_token_optimization_settings() -> TokenOptimizationSettings:
    return TokenOptimizationSettings(
        enabled=True,
        max_recent_user_turns=6,
        summary_trigger_messages=12,
        summary_max_chars=1200,
        context_window_tokens=128000,
        auto_compact_buffer_tokens=13000,
        warning_threshold_buffer_tokens=20000,
        error_threshold_buffer_tokens=20000,
        blocking_buffer_tokens=3000,
        enable_dynamic_system_context=True,
        enable_git_status=True,
        git_status_max_chars=2000,
        enable_project_instruction_scan=True,
        project_instruction_files=("CLAUDE.md", "AGENTS.md", "WBOT.md"),
    )

