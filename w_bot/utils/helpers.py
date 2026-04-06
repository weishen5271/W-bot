from __future__ import annotations

import base64
import json
from typing import Any


def detect_image_mime(raw: bytes) -> str | None:
    signatures = (
        (b"\xff\xd8\xff", "image/jpeg"),
        (b"\x89PNG\r\n\x1a\n", "image/png"),
        (b"GIF87a", "image/gif"),
        (b"GIF89a", "image/gif"),
        (b"RIFF", "image/webp"),
        (b"BM", "image/bmp"),
    )
    for prefix, mime in signatures:
        if raw.startswith(prefix):
            if mime == "image/webp" and len(raw) >= 12 and raw[8:12] != b"WEBP":
                continue
            return mime
    return None


def build_image_content_blocks(
    raw: bytes,
    mime: str,
    source: str,
    alt_text: str | None = None,
) -> list[dict[str, Any]]:
    encoded = base64.b64encode(raw).decode("ascii")
    return [
        {"type": "text", "text": alt_text or f"(Image: {source})"},
        {
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{encoded}"},
        },
    ]


def _pick(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Try keys in order, return first match or default."""
    for key in keys:
        if key in data:
            return data[key]
    return default


def _shorten_text(text: str, limit: int = 120) -> str:
    """Truncate text to limit, preserving word boundaries when possible."""
    compact = " ".join(str(text or "").split())
    if len(compact) <= limit:
        return compact
    return f"{compact[: max(0, limit - 3)]}..."


def _tool_result_to_text(result: Any) -> str:
    """Convert tool result to displayable string."""
    if isinstance(result, str):
        return result
    if isinstance(result, (int, float, bool)) or result is None:
        return str(result)
    try:
        return json.dumps(result, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        return str(result)
