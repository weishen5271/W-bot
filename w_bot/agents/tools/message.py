"""Message tool for sending messages to users."""

import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from w_bot.agents.tools.base import Tool
from w_bot.agents.tools.common import append_jsonl


class MessageTool(Tool):
    """Tool to send messages to users on chat channels."""

    def __init__(self, workspace_root: Path):
        self._workspace_root = workspace_root

    @property
    def name(self) -> str:
        return "message"

    @property
    def description(self) -> str:
        return (
            "Send a message to the user, optionally with file attachments. "
            "This is the ONLY way to deliver files to the user."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "The message content to send"},
                "recipient": {"type": "string", "description": "Optional recipient id or channel target"},
                "media": {"type": "array", "items": {"type": "string"}, "description": "Optional list of file paths to attach"},
            },
            "required": ["content"],
        }

    async def execute(self, content: str, recipient: str | None = None, media: list[str] | None = None, **kwargs: Any) -> str:
        msg = {
            "id": uuid.uuid4().hex,
            "recipient": recipient or "default",
            "content": content,
            "media": media or [],
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        append_jsonl(self._workspace_root / ".w_bot_messages.jsonl", msg)
        media_info = f" with {len(msg['media'])} attachments" if msg["media"] else ""
        return f"Message queued: id={msg['id']} recipient={msg['recipient']}{media_info}"
