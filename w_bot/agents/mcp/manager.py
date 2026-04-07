"""MCP server lifecycle and registry manager."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from ..core.logging_config import get_logger
from ..tools.common import http_get_json, http_post_json, sanitize_tool_token
from .protocol import (
    MCPTransport,
    HTTPTransport,
    SSETransport,
    StreamableHTTPTransport,
    StdioTransport,
    create_transport,
    MCPTool,
)

logger = get_logger(__name__)


class ServerStatus(Enum):
    UNKNOWN = "unknown"
    INITIALIZING = "initializing"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"


@dataclass
class MCPServerConfig:
    """MCP server configuration."""

    name: str
    transport: str = "http"
    base_url: str = ""
    command: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    headers: dict[str, str] = field(default_factory=dict)
    timeout: int = 20
    discovery_path: str = "/tools"
    invoke_path_template: str = "/tools/{tool}"
    health_check: dict[str, Any] = field(default_factory=dict)
    retry: dict[str, Any] = field(default_factory=dict)
    auth: dict[str, Any] = field(default_factory=dict)


@dataclass
class MCPServerState:
    """MCP server runtime state."""

    config: MCPServerConfig
    status: ServerStatus = ServerStatus.UNKNOWN
    last_health_check: datetime | None = None
    error_message: str | None = None
    tools: list[dict[str, str]] = field(default_factory=list)
    transport: MCPTransport | None = None


class MCPManager:
    """Manages MCP server lifecycle, tool discovery, and health checks."""

    def __init__(self, config: list[dict[str, Any]] | None = None):
        self._servers: dict[str, MCPServerState] = {}
        self._tools: dict[str, list[Any]] = {}
        if config:
            self.load_servers(config)

    def load_servers(self, config: list[dict[str, Any]]) -> None:
        """Load MCP server configurations."""
        for server_cfg in config:
            if not isinstance(server_cfg, dict):
                continue
            name = server_cfg.get("name", "unnamed")
            if not name:
                continue

            # Support both camelCase and snake_case keys
            base_url = str(
                server_cfg.get("baseUrl") or server_cfg.get("base_url", "")
            ).strip().rstrip("/")

            mcp_config = MCPServerConfig(
                name=name,
                transport=server_cfg.get("transport", "http"),
                base_url=base_url,
                command=server_cfg.get("command", ""),
                args=server_cfg.get("args", []),
                env=server_cfg.get("env", {}),
                enabled=server_cfg.get("enabled", True),
                headers=server_cfg.get("headers", {}) or {},
                timeout=server_cfg.get("timeout", 20),
                discovery_path=server_cfg.get("discoveryPath") or server_cfg.get("discovery_path", "/tools"),
                invoke_path_template=server_cfg.get("invokePathTemplate") or server_cfg.get("invoke_path_template", "/tools/{tool}"),
                health_check=server_cfg.get("healthCheck", {}) or {},
                retry=server_cfg.get("retry", {}) or {},
                auth=server_cfg.get("auth", {}) or {},
            )
            self._servers[name] = MCPServerState(config=mcp_config)

    def _create_transport(self, state: MCPServerState) -> MCPTransport:
        """Create transport instance based on server config."""
        config = state.config

        if config.transport == "stdio":
            return StdioTransport(
                command=config.command,
                args=config.args,
                env=config.env,
                timeout=config.timeout,
            )
        elif config.transport in ("sse", "server-sent-events"):
            return SSETransport(
                base_url=config.base_url,
                headers=config.headers,
                timeout=config.timeout,
            )
        elif config.transport == "streamable-http":
            return StreamableHTTPTransport(
                base_url=config.base_url,
                headers=config.headers,
                timeout=config.timeout,
            )
        else:
            # Default to HTTP
            return HTTPTransport(
                base_url=config.base_url,
                headers=config.headers,
                timeout=config.timeout,
            )

    async def start_server(self, name: str) -> bool:
        """Start an MCP server process (for stdio transport) or connect to HTTP server."""
        state = self._servers.get(name)
        if not state:
            raise ValueError(f"Unknown MCP server: {name}")

        try:
            state.status = ServerStatus.INITIALIZING

            # Create appropriate transport
            transport = self._create_transport(state)
            await transport.initialize()

            state.transport = transport
            state.status = ServerStatus.CONNECTED
            state.error_message = None

            logger.info("Started MCP server: %s (transport=%s)", name, state.config.transport)
            return True

        except Exception as e:
            logger.error("Failed to start MCP server %s: %s", name, e)
            state.status = ServerStatus.ERROR
            state.error_message = str(e)
            return False

    async def stop_server(self, name: str) -> None:
        """Stop an MCP server or close transport."""
        state = self._servers.get(name)
        if not state:
            return

        if state.transport:
            try:
                await state.transport.close()
            except Exception as e:
                logger.warning("Error closing transport for %s: %s", name, e)
            state.transport = None

        state.status = ServerStatus.DISCONNECTED
        logger.info("Stopped MCP server: %s", name)

    async def discover_tools(self, name: str) -> list[dict[str, str]]:
        """Discover tools from an MCP server using appropriate transport."""
        state = self._servers.get(name)
        if not state:
            logger.warning("MCP server not found: %s", name)
            return []

        # For stdio transport, we need to start the server first if not already started
        if state.config.transport == "stdio" and not state.transport:
            await self.start_server(name)

        # Fallback to HTTP discovery if transport not initialized
        if not state.transport:
            return await self._discover_tools_http(state)

        state.status = ServerStatus.INITIALIZING

        try:
            mcp_tools = await state.transport.list_tools()

            result: list[dict[str, str]] = []
            for tool in mcp_tools:
                if isinstance(tool, MCPTool):
                    result.append({
                        "name": tool.name,
                        "description": tool.description,
                    })
                elif isinstance(tool, dict):
                    result.append({
                        "name": tool.get("name", "unknown"),
                        "description": tool.get("description", ""),
                    })
                else:
                    result.append({
                        "name": str(tool),
                        "description": "",
                    })

            state.tools = result
            state.status = ServerStatus.CONNECTED
            state.last_health_check = datetime.now()
            state.error_message = None
            logger.info("Discovered %d tools from MCP server: %s", len(result), name)
            return result

        except Exception as e:
            logger.error("Error discovering tools from MCP server %s: %s", name, e)
            state.status = ServerStatus.ERROR
            state.error_message = str(e)
            return []

    async def _discover_tools_http(self, state: MCPServerState) -> list[dict[str, str]]:
        """Fallback HTTP-based tool discovery for servers without transport support."""
        if not state.config.base_url:
            logger.warning("No base_url configured for MCP server: %s", state.config.name)
            return []

        state.status = ServerStatus.INITIALIZING

        try:
            url = f"{state.config.base_url}{state.config.discovery_path}"
            raw = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: http_get_json(url=url, headers=state.config.headers, timeout=state.config.timeout)
            )

            if isinstance(raw, str):
                logger.warning("Failed to discover MCP tools from %s: %s", state.config.name, raw)
                state.status = ServerStatus.ERROR
                state.error_message = raw
                return []

            tools = raw.get("tools") if isinstance(raw, dict) else raw
            if not isinstance(tools, list):
                state.status = ServerStatus.ERROR
                state.error_message = "Invalid tools response format"
                return []

            result: list[dict[str, str]] = []
            for item in tools:
                if isinstance(item, str):
                    result.append({"name": item, "description": ""})
                    continue
                if isinstance(item, dict) and item.get("name"):
                    result.append({
                        "name": str(item["name"]),
                        "description": str(item.get("description") or "")
                    })

            state.tools = result
            state.status = ServerStatus.CONNECTED
            state.last_health_check = datetime.now()
            state.error_message = None
            logger.info("Discovered %d tools from MCP server (HTTP): %s", len(result), state.config.name)
            return result

        except Exception as e:
            logger.error("Error discovering tools from MCP server %s: %s", state.config.name, e)
            state.status = ServerStatus.ERROR
            state.error_message = str(e)
            return []

    async def call_tool(self, name: str, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Call a tool on an MCP server."""
        state = self._servers.get(name)
        if not state:
            raise ValueError(f"Unknown MCP server: {name}")

        if not state.transport:
            raise RuntimeError(f"MCP server {name} not connected. Call start_server first.")

        try:
            result = await state.transport.call_tool(tool_name, arguments)
            return result
        except Exception as e:
            logger.error("Error calling tool %s on MCP server %s: %s", tool_name, name, e)
            raise

    def get_tools(self, name: str) -> list[Any]:
        """Get cached tools for a server."""
        return self._tools.get(name, [])

    def list_servers(self) -> list[dict[str, Any]]:
        """List all servers with status."""
        return [
            {
                "name": name,
                "status": state.status.value,
                "transport": state.config.transport,
                "base_url": state.config.base_url,
                "command": state.config.command if state.config.transport == "stdio" else None,
                "tool_count": len(state.tools),
                "last_health_check": state.last_health_check.isoformat() if state.last_health_check else None,
                "error": state.error_message,
                "enabled": state.config.enabled,
            }
            for name, state in self._servers.items()
        ]

    async def health_check(self, name: str) -> bool:
        """Perform health check on a specific server."""
        state = self._servers.get(name)
        if not state:
            return False

        # For stdio, check if process is alive
        if state.config.transport == "stdio":
            if not state.transport:
                return False
            # TODO: Check process health
            state.last_health_check = datetime.now()
            return True

        if not state.config.base_url:
            return False

        health_check_cfg = state.config.health_check or {}
        health_path = health_check_cfg.get("path", "/health")
        timeout = health_check_cfg.get("timeout", 10)

        try:
            url = f"{state.config.base_url}{health_path}"
            raw = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: http_get_json(url=url, headers=state.config.headers, timeout=timeout)
            )

            if isinstance(raw, str):
                state.status = ServerStatus.ERROR
                state.error_message = f"Health check failed: {raw}"
                return False

            state.status = ServerStatus.CONNECTED
            state.last_health_check = datetime.now()
            state.error_message = None
            return True

        except Exception as e:
            state.status = ServerStatus.ERROR
            state.error_message = str(e)
            return False

    async def start_health_check_loop(self, interval_seconds: int = 30) -> None:
        """Start background health check loop."""
        while True:
            for name, state in self._servers.items():
                if not state.config.enabled:
                    continue
                if state.config.transport == "stdio" and not state.transport:
                    continue

                await self.health_check(name)

            await asyncio.sleep(interval_seconds)

    def get_server(self, name: str) -> MCPServerState | None:
        """Get server state by name."""
        return self._servers.get(name)

    @property
    def server_names(self) -> list[str]:
        """Get list of all server names."""
        return list(self._servers.keys())
