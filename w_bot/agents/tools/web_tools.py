from __future__ import annotations

from typing import Any
from urllib import request as url_request

from langchain_core.tools import tool

from .common import http_post_json, strip_html


def build_web_tools(*, tavily_api_key: str) -> list[Any]:
    @tool
    def web_search(query: str, max_results: int = 5) -> str:
        """Search the web and return a short text summary of results."""
        if not tavily_api_key:
            return "TAVILY_API_KEY is not configured."

        payload = http_post_json(
            url="https://api.tavily.com/search",
            payload={
                "api_key": tavily_api_key,
                "query": query,
                "max_results": max(1, max_results),
            },
            timeout=20,
        )
        if isinstance(payload, str):
            return payload

        items = payload.get("results")
        if not isinstance(items, list) or not items:
            return "No results"

        lines: list[str] = []
        for item in items[: max(1, max_results)]:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "(no title)")
            url = str(item.get("url") or "")
            content = str(item.get("content") or "").strip()
            snippet = content[:220] + ("..." if len(content) > 220 else "")
            lines.append(f"- {title}\n  {url}\n  {snippet}")
        return "\n".join(lines) if lines else "No results"

    @tool
    def web_fetch(url: str, max_chars: int = 8000) -> str:
        """Fetch a web page and return extracted plain text."""
        req = url_request.Request(url, headers={"User-Agent": "W-bot/1.0"})
        try:
            with url_request.urlopen(req, timeout=20) as response:
                raw = response.read().decode("utf-8", errors="ignore")
        except Exception as exc:
            return f"Fetch failed: {type(exc).__name__}: {exc}"
        text = strip_html(raw)
        return text[: max(200, max_chars)]

    return [web_search, web_fetch]
