from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from .common import append_jsonl


def build_task_tools(*, workspace_root: Path) -> list[Any]:
    @tool
    def message(recipient: str, content: str) -> str:
        """Queue a message for a target recipient."""
        msg = {
            "id": uuid.uuid4().hex,
            "recipient": recipient,
            "content": content,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        append_jsonl(workspace_root / ".w_bot_messages.jsonl", msg)
        return f"Message queued: id={msg['id']} recipient={recipient}"

    @tool
    def spawn(task: str, context: str = "") -> str:
        """Record a background task request for later processing."""
        job = {
            "id": uuid.uuid4().hex,
            "task": task,
            "context": context,
            "status": "pending",
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        append_jsonl(workspace_root / ".w_bot_spawn_jobs.jsonl", job)
        return f"Spawned task: id={job['id']}"

    return [message, spawn]
