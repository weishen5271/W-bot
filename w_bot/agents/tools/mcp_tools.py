from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import StructuredTool

from ..logging_config import get_logger
from .common import http_get_json, http_post_json, sanitize_tool_token

logger = get_logger(__name__)


def build_mcp_tools(mcp_servers: list[dict[str, Any]]) -> list[StructuredTool]:
    tools: list[StructuredTool] = []
    for server in mcp_servers:
        if not isinstance(server, dict):
            continue
        if server.get("enabled", True) is False:
            continue

        server_name = sanitize_tool_token(str(server.get("name") or "server"))
        base_url = str(server.get("base_url") or "").strip().rstrip("/")
        if not base_url:
            continue

        discovery_path = str(server.get("discovery_path") or "/tools")
        invoke_path_template = str(server.get("invoke_path_template") or "/tools/{tool}")
        headers = server.get("headers") if isinstance(server.get("headers"), dict) else {}

        discovered = discover_mcp_tools(base_url=base_url, discovery_path=discovery_path, headers=headers)
        for item in discovered:
            tool_name = sanitize_tool_token(item.get("name", "tool"))
            tool_desc = item.get("description") or f"MCP tool {tool_name} from {server_name}"
            full_name = f"mcp_{server_name}_{tool_name}"
            tools.append(
                make_mcp_tool(
                    full_name=full_name,
                    description=tool_desc,
                    base_url=base_url,
                    invoke_path_template=invoke_path_template,
                    remote_tool_name=item.get("name", tool_name),
                    headers=headers,
                )
            )
    return tools


def make_mcp_tool(
    *,
    full_name: str,
    description: str,
    base_url: str,
    invoke_path_template: str,
    remote_tool_name: str,
    headers: dict[str, Any],
) -> StructuredTool:
    def _call(arguments_json: str = "{}") -> str:
        try:
            arguments = json.loads(arguments_json or "{}")
        except json.JSONDecodeError:
            return "arguments_json must be valid JSON"

        path = invoke_path_template.replace("{tool}", remote_tool_name)
        url = f"{base_url}{path}"
        raw = http_post_json(url=url, payload={"arguments": arguments}, headers=headers, timeout=20)
        if isinstance(raw, str):
            return raw
        return json.dumps(raw, ensure_ascii=False)

    return StructuredTool.from_function(func=_call, name=full_name, description=description)


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
