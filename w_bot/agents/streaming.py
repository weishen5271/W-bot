from __future__ import annotations

from typing import Any, Callable

from langchain_core.messages import AIMessage


def _normalize_control_chars(text: str | None) -> str:
    raw = str(text or "")
    if not raw:
        return ""
    # Keep CRLF as newline, then treat bare CR as "return to line start"
    # with in-place overwrite semantics, matching terminal behavior.
    raw = raw.replace("\r\n", "\n")
    lines: list[str] = []
    buffer: list[str] = []
    cursor = 0
    clear_on_write = False
    for ch in raw:
        if ch == "\r":
            cursor = 0
            clear_on_write = True
            continue
        if ch == "\n":
            lines.append("".join(buffer))
            buffer = []
            cursor = 0
            clear_on_write = False
            continue
        if clear_on_write:
            buffer = []
            cursor = 0
            clear_on_write = False
        if cursor < len(buffer):
            buffer[cursor] = ch
        else:
            buffer.append(ch)
        cursor += 1
    lines.append("".join(buffer))
    return "\n".join(line.rstrip() for line in lines)


class StreamTextAssembler:
    def __init__(self) -> None:
        self._text = ""

    @property
    def text(self) -> str:
        return self._text

    def consume(self, incoming_text: str | None) -> str:
        payload = _normalize_control_chars(incoming_text)
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
        # No overlap found - this might indicate a problem in streaming
        # Log for debugging
        import logging
        _logger = logging.getLogger(__name__)
        _logger.warning(
            "StreamTextAssembler: no overlap found. self._text=%r, payload=%r, payload_len=%d",
            self._text[:100] if self._text else "",
            payload[:100],
            len(payload)
        )
        self._text += payload
        return payload


def normalize_display_text(text: str | None) -> str:
    payload = _normalize_control_chars(text)
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
