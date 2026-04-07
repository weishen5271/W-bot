"""MCP tool wrappers for HTTP-discovered tools."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from w_bot.agents.core.logging_config import get_logger
from w_bot.agents.tools.base import Tool
from w_bot.agents.tools.common import http_get_json, http_post_json, sanitize_tool_token

logger = get_logger(__name__)


class MCPToolWrapper(Tool):
    """MCP tool with retry, timeout, and error handling."""

    def __init__(
        self,
        *,
        full_name: str,
        description: str,
        base_url: str,
        invoke_path_template: str,
        remote_tool_name: str,
        headers: dict[str, Any],
        timeout: int = 20,
        retry_enabled: bool = True,
        max_attempts: int = 3,
        backoff_ms: int = 500,
    ):
        self._name = full_name
        self._description = description
        self._base_url = base_url
        self._invoke_path_template = invoke_path_template
        self._remote_tool_name = remote_tool_name
        self._headers = headers
        self._timeout = timeout
        self._retry_enabled = retry_enabled
        self._max_attempts = max_attempts
        self._backoff_ms = backoff_ms

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "arguments_json": {"type": "string", "description": "JSON string for remote tool arguments", "default": "{}"}
            },
        }

    async def execute(self, arguments_json: str = "{}", **kwargs: Any) -> str:
        attempts = self._max_attempts if self._retry_enabled else 1

        for attempt in range(attempts):
            try:
                return await self._execute_once(arguments_json)
            except Exception as e:
                if attempt < attempts - 1:
                    wait_ms = self._backoff_ms * (2 ** attempt)
                    logger.warning(
                        "MCP tool %s failed (attempt %d/%d), retrying in %dms: %s",
                        self._name, attempt + 1, attempts, wait_ms, e
                    )
                    await asyncio.sleep(wait_ms / 1000)
                else:
                    logger.error("MCP tool %s failed after %d attempts: %s", self._name, attempts, e)
                    return f"Error: {str(e)}"

        return "Error: Max retries exceeded"

    async def _execute_once(self, arguments_json: str) -> str:
        try:
            arguments = json.loads(arguments_json or "{}")
        except json.JSONDecodeError:
            return "arguments_json must be valid JSON"

        path = self._invoke_path_template.replace("{tool}", self._remote_tool_name)
        url = f"{self._base_url}{path}"

        # Support both sync and async HTTP
        if asyncio.get_event_loop().is_running():
            # Run in thread pool if already in async context
            raw = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: http_post_json(url=url, payload={"arguments": arguments}, headers=self._headers, timeout=self._timeout)
            )
        else:
            raw = http_post_json(url=url, payload={"arguments": arguments}, headers=self._headers, timeout=self._timeout)

        if isinstance(raw, str):
            return raw
        return json.dumps(raw, ensure_ascii=False)


def build_mcp_tools(mcp_servers: list[dict[str, Any]]) -> list[Tool]:
    tools: list[Tool] = []
    for server in mcp_servers:
        if not isinstance(server, dict):
            continue
        if server.get("enabled", True) is False:
            continue

        server_name = sanitize_tool_token(str(server.get("name") or "server"))
        base_url = str(server.get("baseUrl") or server.get("base_url") or "").strip().rstrip("/")
        if not base_url:
            continue

        discovery_path = str(server.get("discoveryPath") or server.get("discovery_path") or "/tools")
        invoke_path_template = str(server.get("invokePathTemplate") or server.get("invoke_path_template") or "/tools/{tool}")
        headers = server.get("headers") if isinstance(server.get("headers"), dict) else {}

        # Parse timeout
        timeout = server.get("timeout", 20)
        if not isinstance(timeout, int):
            timeout = 20

        # Parse retry config
        retry_config = server.get("retry", {})
        if not isinstance(retry_config, dict):
            retry_config = {}
        retry_enabled = retry_config.get("enabled", True) if retry_config else True
        max_attempts = retry_config.get("maxAttempts", retry_config.get("max_attempts", 3))
        backoff_ms = retry_config.get("backoffMs", retry_config.get("backoff_ms", 500))

        discovered = discover_mcp_tools(base_url=base_url, discovery_path=discovery_path, headers=headers)
        for item in discovered:
            tool_name = sanitize_tool_token(item.get("name", "tool"))
            tool_desc = item.get("description") or f"MCP tool {tool_name} from {server_name}"
            full_name = f"mcp_{server_name}_{tool_name}"
            tools.append(
                MCPToolWrapper(
                    full_name=full_name,
                    description=tool_desc,
                    base_url=base_url,
                    invoke_path_template=invoke_path_template,
                    remote_tool_name=item.get("name", tool_name),
                    headers=headers,
                    timeout=timeout,
                    retry_enabled=retry_enabled,
                    max_attempts=max_attempts,
                    backoff_ms=backoff_ms,
                )
            )
    return tools


def discover_mcp_tools(*, base_url: str, discovery_path: str, headers: dict[str, Any]) -> list[dict[str, str]]:
    url = f"{base_url}{discovery_path}"
    raw = http_get_json(url=url, headers=headers, timeout=10)
    if isinstance(raw, str):
        logger.warning("Failed to discover MCP tools from %s: %s", base_url, raw)
        return []

    tools = raw.get("tools") if isinstance(raw, dict) else raw
    if not isinstance(tools, list):
        return []

    result: list[dict[str, str]] = []
    for item in tools:
        if isinstance(item, str):
            result.append({"name": item, "description": ""})
            continue
        if isinstance(item, dict) and item.get("name"):
            result.append({"name": str(item["name"]), "description": str(item.get("description") or "")})
    return result
