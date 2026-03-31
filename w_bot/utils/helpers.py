from __future__ import annotations

import base64
import imghdr
from typing import Any


def detect_image_mime(raw: bytes) -> str | None:
    kind = imghdr.what(None, h=raw)
    if not kind:
        return None
    if kind == "jpeg":
        return "image/jpeg"
    return f"image/{kind}"


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
