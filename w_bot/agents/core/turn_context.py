from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .context import ContextBuilder
from .message_utils import _base_system_prompt, _format_token_budget_snapshot


@dataclass(frozen=True)
class TurnPromptContext:
    system_prompt: str
    prepared_base_prompt: str


class TurnPromptBuilder:
    """Build the per-turn system prompt for explicit AgentRuntime."""

    def __init__(self, *, context_builder: ContextBuilder | None = None) -> None:
        self._context_builder = context_builder or ContextBuilder()

    def build(
        self,
        *,
        memory_context: str = "",
        conversation_summary: str = "",
        token_budget_state: dict[str, Any] | None = None,
        session_token_usage: Any = None,
        context_compaction_level: str = "",
    ) -> TurnPromptContext:
        budget_snapshot = _format_token_budget_snapshot(
            token_budget_state if isinstance(token_budget_state, dict) else {},
            session_token_usage,
        )
        prepared_base_prompt = self._context_builder.build_turn_system_prompt(
            base_prompt=_base_system_prompt(),
            budget_snapshot=budget_snapshot,
        )
        prompt_blocks = [
            prepared_base_prompt,
            f"已检索到的长期记忆:\n{memory_context.strip() or '无'}",
        ]
        if conversation_summary.strip():
            prompt_blocks.append(f"会话摘要（历史压缩）:\n{conversation_summary.strip()}")
        level = context_compaction_level.strip()
        if level:
            prompt_blocks.append(
                "上下文压缩等级:\n"
                f"- 当前等级: {level}\n"
                "- 等级越高，越应优先引用摘要、关键决策和最近消息，避免重复展开旧内容。"
            )
        return TurnPromptContext(
            system_prompt="\n\n".join(prompt_blocks),
            prepared_base_prompt=prepared_base_prompt,
        )

