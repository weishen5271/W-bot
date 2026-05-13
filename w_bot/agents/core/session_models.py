from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from langchain_core.messages import AnyMessage


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime options for a single explicit AgentRuntime turn."""

    recursion_limit: int = 20
    max_tool_steps_per_turn: int = 8
    max_same_tool_call_repeats: int = 3
    max_consecutive_tool_failures: int = 2
    defer_summary_update: bool = True
    status_callback: Callable[[str], None] | None = None
    stream_token_callback: Callable[[str], None] | None = None
    debug_callback: Callable[[str], None] | None = None
    tool_progress_callback: Callable[..., None] | None = None


@dataclass(frozen=True)
class AgentTurnResult:
    """Result returned by AgentRuntime after one user turn."""

    session_id: str
    final_response: str
    messages: list[AnyMessage] = field(default_factory=list)
    tool_calls: int = 0
    completed: bool = True
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
