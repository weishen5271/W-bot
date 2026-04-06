from __future__ import annotations

import unicodedata


def sanitize_user_text(raw: str) -> str:
    """清洗用户输入，避免控制字符导致下游解析或模型调用异常。"""
    text = str(raw or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")
    text = text.replace("\u200b", "")

    out: list[str] = []
    for ch in text:
        if ch in {"\n", "\t"}:
            out.append(ch)
            continue
        category = unicodedata.category(ch)
        if category in {"Cc", "Cs"}:
            continue
        out.append(ch)
    return "".join(out)

