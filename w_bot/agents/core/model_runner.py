from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage

from .logging_config import get_logger
from .message_utils import _is_messages_length_error
from .session_models import RuntimeConfig
from .streaming_utils import _invoke_llm_with_optional_stream
from .tool_analysis import _build_text_only_retry_messages

logger = get_logger(__name__)


@dataclass(frozen=True)
class ModelRunResult:
    message: AIMessage
    completed: bool = True
    error: str = ""


class ModelRunner:
    """Invoke the current chat model for one AgentRuntime step."""

    def __init__(self, *, llm: Any) -> None:
        self._llm = llm

    def invoke(
        self,
        *,
        messages: list[AnyMessage],
        config: RuntimeConfig,
        llm: Any = None,
        fallback_llm: Any = None,
        system_prompt: str = "",
    ) -> ModelRunResult:
        target_llm = llm or self._llm
        try:
            message = self._invoke(target_llm, messages, config)
        except Exception as exc:
            fallback = self._fallback_after_error(
                exc=exc,
                messages=messages,
                config=config,
                fallback_llm=fallback_llm,
                target_llm=target_llm,
                system_prompt=system_prompt,
            )
            if fallback is not None:
                return fallback
            logger.exception("ModelRunner invoke failed")
            return self._error_result(exc)
        if isinstance(message, AIMessage):
            return ModelRunResult(message=message)
        return ModelRunResult(message=AIMessage(content=str(getattr(message, "content", message) or "")))

    def _fallback_after_error(
        self,
        *,
        exc: Exception,
        messages: list[AnyMessage],
        config: RuntimeConfig,
        fallback_llm: Any,
        target_llm: Any,
        system_prompt: str,
    ) -> ModelRunResult | None:
        if fallback_llm is None:
            return None

        if _is_messages_length_error(exc):
            logger.warning("Provider rejected message payload; retry with text-only fallback: %s", exc)
            retry_messages = _build_text_only_retry_messages(
                system_prompt=system_prompt,
                history=messages,
            )
            try:
                return ModelRunResult(message=self._invoke(fallback_llm, retry_messages, config))
            except Exception as fallback_exc:
                logger.exception("Text-only compatibility fallback failed")
                return self._error_result(fallback_exc)

        if fallback_llm is target_llm:
            return None

        logger.warning("Route model failed; retry with text model: %s", exc)
        try:
            return ModelRunResult(message=self._invoke(fallback_llm, messages, config))
        except Exception as fallback_exc:
            logger.exception("Route fallback to text model failed")
            return self._error_result(fallback_exc)

    @staticmethod
    def _invoke(llm: Any, messages: list[AnyMessage], config: RuntimeConfig) -> AIMessage:
        return _invoke_llm_with_optional_stream(
            llm=llm,
            messages=messages,
            token_callback=config.stream_token_callback,
            debug_callback=config.debug_callback,
            tool_event_callback=config.tool_progress_callback,
        )

    @staticmethod
    def _error_result(exc: Exception) -> ModelRunResult:
        return ModelRunResult(
            message=AIMessage(content=f"模型调用失败：{type(exc).__name__}: {exc}"),
            completed=False,
            error=str(exc),
        )
