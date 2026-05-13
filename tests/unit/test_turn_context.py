from __future__ import annotations

from w_bot.agents.core.turn_context import TurnPromptBuilder


def test_turn_prompt_builder_includes_memory_and_summary() -> None:
    builder = TurnPromptBuilder()

    result = builder.build(
        memory_context="记忆 A",
        conversation_summary="摘要 B",
        context_compaction_level="warning",
    )

    assert "已检索到的长期记忆" in result.system_prompt
    assert "记忆 A" in result.system_prompt
    assert "会话摘要" in result.system_prompt
    assert "摘要 B" in result.system_prompt
    assert "warning" in result.system_prompt
    assert result.prepared_base_prompt

