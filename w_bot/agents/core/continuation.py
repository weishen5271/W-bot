from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage

from ..intent.intent_detection import (
    _continue_current_task_prompt,
    _response_looks_incomplete,
    _should_check_completion_for_turn,
)
from .logging_config import get_logger
from .message_utils import _to_text_content

logger = get_logger(__name__)


@dataclass(frozen=True)
class ContinuationDecision:
    should_continue: bool
    prompt: str = ""
    reason: str = ""


class ContinuationController:
    """Decide whether a non-tool reply should continue the current turn."""

    def __init__(self, *, judge_llm: Any = None, max_attempts: int = 1) -> None:
        self._judge_llm = judge_llm
        self._max_attempts = max(0, int(max_attempts))

    @property
    def max_attempts(self) -> int:
        return self._max_attempts

    def decide(
        self,
        *,
        user_goal: str,
        history: list[AnyMessage],
        response: AIMessage,
        attempt: int = 0,
    ) -> ContinuationDecision:
        if attempt >= self._max_attempts:
            return ContinuationDecision(False)
        if not _should_check_completion_for_turn(user_goal, history):
            return ContinuationDecision(False)
        response_text = _to_text_content(response.content).strip()
        if not response_text:
            return ContinuationDecision(True, _continue_current_task_prompt(), "empty_response")
        if _response_looks_incomplete(response_text):
            return ContinuationDecision(True, _continue_current_task_prompt(), "incomplete_markers")
        if self._judge_llm is None:
            return ContinuationDecision(False)
        if not self._judge_reply_complete(user_goal=user_goal, response_text=response_text):
            return ContinuationDecision(True, _continue_current_task_prompt(), "judge_continue")
        return ContinuationDecision(False)

    def _judge_reply_complete(self, *, user_goal: str, response_text: str) -> bool:
        judge_prompt = (
            "你是一个严格的任务完成度裁判。"
            "你只判断 assistant 的最新回复是否已经真正完成了用户请求。"
            "如果回复仍停留在计划、说明、承诺稍后执行、阶段性汇报、要求用户做本可由 assistant 自己完成的动作，"
            "或者明显还缺少实现、检查、结果，则返回 CONTINUE。"
            "只有在用户请求已经被实际完成，或者 assistant 明确说明了无法继续且给出了真实阻塞原因时，才返回 COMPLETE。"
            "只能输出一个单词：COMPLETE 或 CONTINUE。"
        )
        try:
            decision = self._judge_llm.invoke(
                [
                    SystemMessage(content=judge_prompt),
                    HumanMessage(
                        content=(
                            f"用户请求：\n{user_goal.strip() or '(empty)'}\n\n"
                            f"assistant 最新回复：\n{response_text}\n\n"
                            "请判断是否已完成。"
                        )
                    ),
                ]
            )
        except Exception:
            logger.debug("Continuation completion judge failed", exc_info=True)
            return False
        decision_text = _to_text_content(getattr(decision, "content", "")).strip().upper()
        return decision_text.startswith("COMPLETE")

