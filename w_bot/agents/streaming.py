from __future__ import annotations

from typing import Any, Callable

from langchain_core.messages import AIMessage


class StreamTextAssembler:
    def __init__(self) -> None:
        self._text = ""

    @property
    def text(self) -> str:
        return self._text

    def consume(self, incoming_text: str | None) -> str:
        payload = incoming_text or ""
        if not payload:
            return ""
        if not self._text:
            self._text = payload
            return payload
        if payload.startswith(self._text):
            delta = payload[len(self._text):]
            self._text = payload
            return delta
        if payload in self._text:
            return ""
        overlap_limit = min(len(self._text), len(payload))
        for size in range(overlap_limit, 0, -1):
            if self._text.endswith(payload[:size]):
                delta = payload[size:]
                self._text += delta
                return delta
        self._text += payload
        return payload


def normalize_display_text(text: str | None) -> str:
    payload = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    payload = "\n".join(line.rstrip() for line in payload.split("\n"))
    while "\n\n\n" in payload:
        payload = payload.replace("\n\n\n", "\n\n")
    return payload


def latest_non_tool_ai_reply(
    messages: list[Any],
    *,
    content_to_text: Callable[[Any], str],
) -> str:
    for message in reversed(messages):
        if not isinstance(message, AIMessage) or message.tool_calls:
            continue
        text = normalize_display_text(content_to_text(getattr(message, "content", ""))).strip()
        if text:
            return text
    return ""


def _message_to_text(content: Any) -> str:
    """Convert message content to displayable text.

    Handles str, list of blocks (dict with text), and dict formats.
    """
    if isinstance(content, str):
        return normalize_display_text(content)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            else:
                parts.append(str(item))
        return normalize_display_text("\n".join(part for part in parts if part))
    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            return normalize_display_text(text)
    return normalize_display_text(str(content))


def _latest_ai_reply_from_result(result: Any) -> str:
    """Extract the latest AI reply text from a graph invocation result."""
    values = result if isinstance(result, dict) else {}
    messages = values.get("messages", []) if isinstance(values.get("messages", []), list) else []
    return latest_non_tool_ai_reply(messages, content_to_text=_message_to_text)
