from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from .message_utils import _resolve_thread_id


class RuntimeCompatAdapter:
    """Compatibility surface for tools that still expect a graph-like object."""

    def __init__(self, *, runtime: Any) -> None:
        self._runtime = runtime

    @property
    def session_search_llm(self) -> Any:
        return getattr(self._runtime, "llm", None)

    def get_state(self, config: dict[str, Any] | None = None, **kwargs: Any) -> Any:
        del kwargs
        thread_id = _resolve_thread_id(config)
        if thread_id == "-":
            messages = []
            summary = ""
            summarized_message_count = 0
        else:
            messages = self._runtime.get_session_messages(thread_id)
            summary, summarized_message_count = self._runtime.session_store.get_summary(thread_id)
        return SimpleNamespace(
            values={
                "messages": messages,
                "conversation_summary": summary,
                "summarized_message_count": summarized_message_count,
            }
        )

    async def aget_state(self, config: dict[str, Any] | None = None, **kwargs: Any) -> Any:
        return self.get_state(config, **kwargs)

    def list_subagents(self, *, status: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        del status, limit
        return []

    def wait_for_subagent(self, job_id: str, *, timeout_seconds: int = 60) -> dict[str, Any]:
        del timeout_seconds
        return {
            "success": False,
            "id": job_id,
            "status": "unsupported",
            "error": "Subagent runtime compatibility is not connected yet.",
        }

    def spawn_subagent(
        self,
        *,
        agent_type: str,
        task: str,
        label: str = "",
        context_messages: list[Any] | None = None,
        parent_thread_id: str = "-",
        status_callback: Any = None,
    ) -> dict[str, Any]:
        del agent_type, task, label, context_messages, parent_thread_id, status_callback
        return {
            "success": False,
            "status": "unsupported",
            "error": "Subagent runtime compatibility is not connected yet.",
        }

    async def run_skill_subagent(
        self,
        *,
        skill_name: str,
        task: str,
        arguments: dict[str, Any] | None = None,
        context_messages: list[Any] | None = None,
        thread_id: str = "-",
        status_callback: Any = None,
    ) -> dict[str, Any]:
        del skill_name, task, arguments, context_messages, thread_id, status_callback
        return {
            "success": False,
            "final_response": "",
            "error": "Skill subagent runtime compatibility is not connected yet.",
            "tool_calls": 0,
            "duration_seconds": 0.0,
        }

