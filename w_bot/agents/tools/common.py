from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from urllib import error as url_error
from urllib import request as url_request


def safe_completed_text(value: str | None) -> str:
    return (value or "").strip()


def resolve_workspace_path(path: str, *, workspace_root: Path) -> Path:
    candidate = Path(path)
    resolved = candidate.resolve() if candidate.is_absolute() else (workspace_root / candidate).resolve()
    if not is_relative_to(resolved, workspace_root):
        raise ValueError(f"Path escapes workspace: {path}")
    return resolved


def resolve_read_path(path: str, *, readonly_roots: list[Path]) -> Path:
    if not readonly_roots:
        raise ValueError("No readonly roots configured")
    candidate = Path(path)
    resolved = candidate.resolve() if candidate.is_absolute() else (readonly_roots[0] / candidate).resolve()
    for root in readonly_roots:
        if is_relative_to(resolved, root):
            return resolved
    raise ValueError(f"Path escapes readonly roots: {path}")


def is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def sanitize_tool_token(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_").lower()
    return cleaned or "tool"


def strip_html(raw: str) -> str:
    without_script = re.sub(r"<script[\s\S]*?</script>", " ", raw, flags=re.IGNORECASE)
    without_style = re.sub(r"<style[\s\S]*?</style>", " ", without_script, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", without_style)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False))
        f.write("\n")


def read_json_file(path: Path, *, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def http_get_json(
    *,
    url: str,
    headers: dict[str, Any] | None = None,
    timeout: int = 10,
) -> dict[str, Any] | str:
    req_headers = {"User-Agent": "W-bot/1.0"}
    if headers:
        req_headers.update({str(k): str(v) for k, v in headers.items()})
    req = url_request.Request(url, headers=req_headers)
    try:
        with url_request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8", errors="ignore"))
    except (url_error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
        return f"HTTP GET failed: {type(exc).__name__}: {exc}"


def http_post_json(
    *,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, Any] | None = None,
    timeout: int = 10,
) -> dict[str, Any] | str:
    body = json.dumps(payload).encode("utf-8")
    req_headers = {"Content-Type": "application/json", "User-Agent": "W-bot/1.0"}
    if headers:
        req_headers.update({str(k): str(v) for k, v in headers.items()})
    req = url_request.Request(url, data=body, headers=req_headers, method="POST")
    try:
        with url_request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8", errors="ignore"))
    except (url_error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
        return f"HTTP POST failed: {type(exc).__name__}: {exc}"
