from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    @property
    def total(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
            "total_tokens": self.total,
        }

    def add(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_creation_input_tokens=self.cache_creation_input_tokens + other.cache_creation_input_tokens,
            cache_read_input_tokens=self.cache_read_input_tokens + other.cache_read_input_tokens,
        )


@dataclass(frozen=True)
class TokenBudgetState:
    used_tokens: int
    threshold_tokens: int
    percent_left: int
    is_above_warning_threshold: bool
    is_above_error_threshold: bool
    is_above_auto_compact_threshold: bool
    is_at_blocking_limit: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "used_tokens": self.used_tokens,
            "threshold_tokens": self.threshold_tokens,
            "percent_left": self.percent_left,
            "is_above_warning_threshold": self.is_above_warning_threshold,
            "is_above_error_threshold": self.is_above_error_threshold,
            "is_above_auto_compact_threshold": self.is_above_auto_compact_threshold,
            "is_at_blocking_limit": self.is_at_blocking_limit,
        }


class TokenBudgetManager:
    def __init__(
        self,
        *,
        context_window_tokens: int,
        auto_compact_buffer_tokens: int,
        warning_threshold_buffer_tokens: int,
        error_threshold_buffer_tokens: int,
        blocking_buffer_tokens: int,
    ) -> None:
        self._context_window_tokens = max(4096, int(context_window_tokens))
        self._auto_compact_buffer_tokens = max(0, int(auto_compact_buffer_tokens))
        self._warning_threshold_buffer_tokens = max(0, int(warning_threshold_buffer_tokens))
        self._error_threshold_buffer_tokens = max(0, int(error_threshold_buffer_tokens))
        self._blocking_buffer_tokens = max(0, int(blocking_buffer_tokens))

    @property
    def effective_context_window(self) -> int:
        return self._context_window_tokens

    @property
    def auto_compact_threshold(self) -> int:
        return max(1, self._context_window_tokens - self._auto_compact_buffer_tokens)

    def calculate_state(self, used_tokens: int) -> TokenBudgetState:
        threshold = self.auto_compact_threshold
        percent_left = max(0, round(((threshold - used_tokens) / threshold) * 100))
        warning_threshold = max(0, threshold - self._warning_threshold_buffer_tokens)
        error_threshold = max(0, threshold - self._error_threshold_buffer_tokens)
        blocking_limit = max(1, self._context_window_tokens - self._blocking_buffer_tokens)
        return TokenBudgetState(
            used_tokens=max(0, int(used_tokens)),
            threshold_tokens=threshold,
            percent_left=percent_left,
            is_above_warning_threshold=used_tokens >= warning_threshold,
            is_above_error_threshold=used_tokens >= error_threshold,
            is_above_auto_compact_threshold=used_tokens >= threshold,
            is_at_blocking_limit=used_tokens >= blocking_limit,
        )


def extract_token_usage(payload: Any) -> TokenUsage:
    candidates: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        candidates.append(payload)
        usage = payload.get("usage")
        if isinstance(usage, dict):
            candidates.append(usage)
    else:
        usage_metadata = getattr(payload, "usage_metadata", None)
        if isinstance(usage_metadata, dict):
            candidates.append(usage_metadata)
        response_metadata = getattr(payload, "response_metadata", None)
        if isinstance(response_metadata, dict):
            candidates.append(response_metadata)
            nested_usage = response_metadata.get("usage")
            if isinstance(nested_usage, dict):
                candidates.append(nested_usage)
        additional_kwargs = getattr(payload, "additional_kwargs", None)
        if isinstance(additional_kwargs, dict):
            candidates.append(additional_kwargs)
            nested_usage = additional_kwargs.get("usage")
            if isinstance(nested_usage, dict):
                candidates.append(nested_usage)

    for candidate in candidates:
        usage = _usage_from_mapping(candidate)
        if usage.total > 0:
            return usage
    return TokenUsage()


def token_count_with_estimation(messages: list[AnyMessage]) -> int:
    total = 0
    for message in messages:
        usage = extract_token_usage(message)
        if usage.total > 0:
            total += usage.total
            continue
        total += rough_message_token_estimation(message)
    return total


def rough_message_token_estimation(message: BaseMessage) -> int:
    role_overhead = 12
    if isinstance(message, SystemMessage):
        role_overhead = 24
    elif isinstance(message, ToolMessage):
        role_overhead = 18
    elif isinstance(message, AIMessage):
        role_overhead = 16
    elif isinstance(message, HumanMessage):
        role_overhead = 14
    return rough_token_count_estimation(_content_to_text(message.content)) + role_overhead


def rough_token_count_estimation(text: str) -> int:
    normalized = text.strip()
    if not normalized:
        return 0
    return max(1, len(normalized) // 4)


def _usage_from_mapping(data: dict[str, Any]) -> TokenUsage:
    return TokenUsage(
        input_tokens=_coerce_int(data.get("input_tokens")),
        output_tokens=_coerce_int(data.get("output_tokens")),
        cache_creation_input_tokens=_coerce_int(data.get("cache_creation_input_tokens")),
        cache_read_input_tokens=_coerce_int(data.get("cache_read_input_tokens")),
    )


def _coerce_int(value: Any) -> int:
    if value is None:
        return 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                item_type = str(item.get("type") or "")
                if item_type in {"text", "input_text", "output_text"}:
                    parts.append(str(item.get("text") or ""))
                elif "content" in item:
                    parts.append(str(item.get("content") or ""))
                else:
                    parts.append(str(item))
                continue
            parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content or "")
