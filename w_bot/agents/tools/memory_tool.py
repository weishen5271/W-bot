from __future__ import annotations

from typing import Any

from ..memory import LongTermMemoryStore
from .base import Tool


class SaveMemoryTool(Tool):
    def __init__(self, *, memory_store: LongTermMemoryStore, user_id: str) -> None:
        self._memory_store = memory_store
        self._user_id = user_id

    @property
    def name(self) -> str:
        return "save_memory"

    @property
    def description(self) -> str:
        return (
            "Save stable information into layered long-term memory. "
            "Use for durable user preferences, feedback, project decisions, constraints, and reusable references."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The durable memory to save. Keep it concise and factual.",
                    "minLength": 3,
                    "maxLength": 1000,
                },
                "memory_type": {
                    "type": "string",
                    "description": "Memory layer: user, feedback, project, reference, preference, decision, constraint, fact.",
                    "enum": [
                        "user",
                        "feedback",
                        "project",
                        "reference",
                        "preference",
                        "decision",
                        "constraint",
                        "fact",
                    ],
                },
                "priority": {
                    "type": "integer",
                    "description": "Importance from 0 to 3. Use 3 for corrections and hard constraints.",
                    "minimum": 0,
                    "maximum": 3,
                },
            },
            "required": ["text"],
        }

    async def execute(
        self,
        text: str,
        memory_type: str = "reference",
        priority: int = 2,
        **kwargs: Any,
    ) -> str:
        doc_id = self._memory_store.save(
            user_id=self._user_id,
            text=text,
            memory_type=memory_type,
            priority=priority,
        )
        if not doc_id:
            return "memory save skipped"
        return f"memory saved, id={doc_id}, type={memory_type}, priority={priority}"
