"""MCP protocol implementations for different transports."""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator


@dataclass
class MCPRequest:
    """MCP JSON-RPC request."""
    jsonrpc: str = "2.0"
    id: int | str | None = None
    method: str = ""
    params: dict[str, Any] | None = None


@dataclass
class MCPResponse:
    """MCP JSON-RPC response."""
    jsonrpc: str = "2.0"
    id: int | str | None = None
    result: Any = None
    error: dict[str, Any] | None = None


@dataclass
class MCPTool:
    """MCP tool definition."""
    name: str
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)


class MCPTransport(ABC):
    """Base class for MCP transports."""

    @abstractmethod
    async def initialize(self) -> dict[str, Any]:
        """Initialize the MCP server connection."""
        pass

    @abstractmethod
    async def list_tools(self) -> list[MCPTool]:
        """List available tools from the server."""
        pass

    @abstractmethod
    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Call a tool on the server."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the transport connection."""
        pass


class HTTPTransport(MCPTransport):
    """HTTP transport for MCP servers."""

    def __init__(
        self,
        base_url: str,
        headers: dict[str, str] | None = None,
        timeout: int = 20,
    ):
        self._base_url = base_url.rstrip("/")
        self._headers = headers or {}
        self._timeout = timeout
        self._request_id = 0

    async def initialize(self) -> dict[str, Any]:
        """Initialize the MCP server."""
        self._request_id += 1
        request = MCPRequest(
            id=self._request_id,
            method="initialize",
            params={"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "w-bot", "version": "1.0"}},
        )
        result = await self._send_request(request)
        return result

    async def list_tools(self) -> list[MCPTool]:
        """List tools via HTTP."""
        self._request_id += 1
        request = MCPRequest(id=self._request_id, method="tools/list")
        result = await self._send_request(request)
        tools = result.get("tools", []) if isinstance(result, dict) else []
        return [MCPTool(name=t["name"], description=t.get("description", ""), input_schema=t.get("inputSchema", {})) for t in tools]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Call tool via HTTP."""
        self._request_id += 1
        request = MCPRequest(
            id=self._request_id,
            method="tools/call",
            params={"name": name, "arguments": arguments},
        )
        return await self._send_request(request)

    async def _send_request(self, request: MCPRequest) -> Any:
        """Send JSON-RPC request via HTTP POST."""
        from ..tools.common import http_post_json

        url = f"{self._base_url}/rpc"
        payload = {
            "jsonrpc": request.jsonrpc,
            "id": request.id,
            "method": request.method,
            "params": request.params,
        }

        raw = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: http_post_json(url=url, payload=payload, headers=self._headers, timeout=self._timeout)
        )

        if isinstance(raw, str):
            raise RuntimeError(f"HTTP request failed: {raw}")

        if raw.get("error"):
            raise RuntimeError(f"JSON-RPC error: {raw['error']}")

        return raw.get("result")

    async def close(self) -> None:
        """Close HTTP transport (no-op)."""
        pass


class SSETransport(MCPTransport):
    """SSE (Server-Sent Events) transport for MCP servers."""

    def __init__(
        self,
        base_url: str,
        headers: dict[str, str] | None = None,
        timeout: int = 20,
    ):
        self._base_url = base_url.rstrip("/")
        self._headers = headers or {}
        self._timeout = timeout
        self._request_id = 0
        self._client: Any = None

    async def initialize(self) -> dict[str, Any]:
        """Initialize via SSE endpoint."""
        import httpx

        self._client = httpx.AsyncClient(timeout=self._timeout, headers=self._headers)
        self._request_id += 1

        request = MCPRequest(
            id=self._request_id,
            method="initialize",
            params={"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "w-bot", "version": "1.0"}},
        )

        # Send POST to SSE endpoint
        response = await self._client.post(
            f"{self._base_url}/sse",
            json=request.__dict__,
        )
        response.raise_for_status()

        # SSE responses come as events, we'll handle the initialization response
        return {"status": "initialized"}

    async def list_tools(self) -> list[MCPTool]:
        """List tools via SSE."""
        if not self._client:
            raise RuntimeError("Transport not initialized")

        self._request_id += 1
        request = MCPRequest(id=self._request_id, method="tools/list")

        response = await self._client.post(
            f"{self._base_url}/sse",
            json=request.__dict__,
        )
        response.raise_for_status()

        # Parse SSE events - in a real implementation, we'd read the event stream
        # For now, return empty as SSE requires async event handling
        return []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Call tool via SSE."""
        if not self._client:
            raise RuntimeError("Transport not initialized")

        self._request_id += 1
        request = MCPRequest(
            id=self._request_id,
            method="tools/call",
            params={"name": name, "arguments": arguments},
        )

        response = await self._client.post(
            f"{self._base_url}/sse",
            json=request.__dict__,
        )
        response.raise_for_status()
        return {"status": "called", "tool": name}

    async def close(self) -> None:
        """Close SSE client."""
        if self._client:
            await self._client.aclose()
            self._client = None


class StreamableHTTPTransport(MCPTransport):
    """StreamableHTTP transport for MCP servers (official MCP protocol)."""

    def __init__(
        self,
        base_url: str,
        headers: dict[str, str] | None = None,
        timeout: int = 20,
    ):
        self._base_url = base_url.rstrip("/")
        self._headers = headers or {}
        self._timeout = timeout
        self._request_id = 0
        self._session_id: str | None = None
        self._client: Any = None

    async def initialize(self) -> dict[str, Any]:
        """Initialize the MCP session via StreamableHTTP."""
        import httpx

        self._client = httpx.AsyncClient(timeout=self._timeout, headers=self._headers)
        self._request_id += 1

        request = MCPRequest(
            id=self._request_id,
            method="initialize",
            params={"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "w-bot", "version": "1.0"}},
        )

        headers = {**self._headers}
        if self._session_id:
            headers["MCP-Session-Id"] = self._session_id

        response = await self._client.post(
            f"{self._base_url}/mcp",
            json=request.__dict__,
            headers=headers,
        )

        if "mcp-session-id" in response.headers:
            self._session_id = response.headers["mcp-session-id"]

        if response.is_success:
            data = response.json()
            return data.get("result", {})
        else:
            raise RuntimeError(f"Initialize failed: {response.status_code}")

    async def list_tools(self) -> list[MCPTool]:
        """List tools via StreamableHTTP."""
        return await self._send_rpc_request("tools/list", {})

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Call tool via StreamableHTTP."""
        return await self._send_rpc_request("tools/call", {"name": name, "arguments": arguments})

    async def _send_rpc_request(self, method: str, params: dict[str, Any]) -> Any:
        """Send a JSON-RPC request via StreamableHTTP."""
        if not self._client:
            raise RuntimeError("Transport not initialized")

        self._request_id += 1
        request = MCPRequest(id=self._request_id, method=method, params=params)

        headers = {**self._headers}
        if self._session_id:
            headers["MCP-Session-Id"] = self._session_id

        response = await self._client.post(
            f"{self._base_url}/mcp",
            json=request.__dict__,
            headers=headers,
        )

        if "mcp-session-id" in response.headers:
            self._session_id = response.headers["mcp-session-id"]

        if response.is_success:
            data = response.json()
            if "error" in data:
                raise RuntimeError(f"RPC error: {data['error']}")
            return data.get("result", {})
        else:
            raise RuntimeError(f"Request failed: {response.status_code}")

    async def close(self) -> None:
        """Close StreamableHTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


class StdioTransport(MCPTransport):
    """Stdio transport for local MCP server processes (official MCP protocol)."""

    def __init__(
        self,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        timeout: int = 20,
    ):
        self._command = command
        self._args = args or []
        self._env = env or {}
        self._timeout = timeout
        self._request_id = 0
        self._process: asyncio.subprocess.Process | None = None
        self._stdout_reader: asyncio.StreamReader | None = None
        self._stdin_writer: asyncio.StreamWriter | None = None
        self._reader_task: asyncio.Task | None = None
        self._response_futures: dict[int, asyncio.Future] = {}
        self._initialized = False

    async def initialize(self) -> dict[str, Any]:
        """Start the MCP server process and initialize."""
        if self._process:
            return {"status": "already_initialized"}

        # Start the subprocess
        self._process = await asyncio.create_subprocess_exec(
            self._command,
            *self._args,
            env={**self._env},
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        self._stdout_reader = self._process.stdout
        self._stdin_writer = self._process.stdin

        # Start reading responses in background
        self._reader_task = asyncio.create_task(self._read_responses())

        # Send initialize request
        self._request_id += 1
        result = await self._send_request(
            "initialize",
            {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "w-bot", "version": "1.0"}}
        )

        self._initialized = True

        # Send initialized notification
        await self._send_notification("initialized", {})

        return result

    async def list_tools(self) -> list[MCPTool]:
        """List tools from the MCP server."""
        result = await self._send_request("tools/list", {})
        tools = result.get("tools", []) if isinstance(result, dict) else []
        return [
            MCPTool(
                name=t["name"] if isinstance(t, dict) else str(t),
                description=t.get("description", "") if isinstance(t, dict) else "",
                input_schema=t.get("inputSchema", {}) if isinstance(t, dict) else {},
            )
            for t in tools
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Call a tool on the MCP server."""
        return await self._send_request("tools/call", {"name": name, "arguments": arguments})

    async def _send_request(self, method: str, params: dict[str, Any]) -> Any:
        """Send a JSON-RPC request and wait for response."""
        if not self._stdin_writer or not self._stdout_reader:
            raise RuntimeError("Transport not initialized")

        request_id = self._request_id
        self._request_id += 1

        request = MCPRequest(id=request_id, method=method, params=params)
        request_json = json.dumps(request.__dict__, ensure_ascii=False) + "\n"

        # Create future for response
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._response_futures[request_id] = future

        try:
            self._stdin_writer.write(request_json.encode("utf-8"))
            await self._stdin_writer.drain()

            # Wait for response with timeout
            result = await asyncio.wait_for(future, timeout=self._timeout)
            return result
        except asyncio.TimeoutError:
            del self._response_futures[request_id]
            raise RuntimeError(f"Request {method} timed out after {self._timeout}s")

    async def _send_notification(self, method: str, params: dict[str, Any]) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        if not self._stdin_writer:
            return

        notification = MCPRequest(jsonrpc="2.0", method=method, params=params)
        notification_json = json.dumps(notification.__dict__, ensure_ascii=False) + "\n"
        self._stdin_writer.write(notification_json.encode("utf-8"))
        await self._stdin_writer.drain()

    async def _read_responses(self) -> None:
        """Background task to read and route responses."""
        if not self._stdout_reader:
            return

        try:
            while True:
                line = await self._stdout_reader.readline()
                if not line:
                    break

                try:
                    data = json.loads(line.decode("utf-8"))
                    request_id = data.get("id")

                    if request_id is not None and request_id in self._response_futures:
                        future = self._response_futures.pop(request_id)
                        if "error" in data:
                            future.set_exception(RuntimeError(f"RPC error: {data['error']}"))
                        else:
                            future.set_result(data.get("result", {}))
                    elif "method" in data and data.get("method", "").startswith("notifications/"):
                        # Handle notifications (e.g., logging)
                        logger.debug("Received notification: %s", data.get("method"))
                    else:
                        # Result without matching request ID
                        logger.debug("Received unexpected response: %s", data)
                except json.JSONDecodeError:
                    logger.warning("Failed to parse JSON from MCP server: %s", line)
        except Exception as e:
            logger.error("Error reading from MCP server: %s", e)

    async def close(self) -> None:
        """Stop the MCP server process."""
        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
            self._reader_task = None

        if self._process:
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._process.kill()
                await self._process.wait()
            except Exception as e:
                logger.warning("Error stopping MCP process: %s", e)
            self._process = None

        self._stdin_writer = None
        self._stdout_reader = None
        self._response_futures.clear()
        self._initialized = False


def create_transport(
    transport_type: str,
    **kwargs: Any,
) -> MCPTransport:
    """Factory function to create MCP transport by type."""
    transports = {
        "http": HTTPTransport,
        "https": HTTPTransport,
        "sse": SSETransport,
        "streamable-http": StreamableHTTPTransport,
        "stdio": StdioTransport,
    }

    transport_class = transports.get(transport_type.lower())
    if not transport_class:
        raise ValueError(f"Unknown transport type: {transport_type}. Supported: {list(transports.keys())}")

    return transport_class(**kwargs)


# Module-level logger
from ..core.logging_config import get_logger
logger = get_logger(__name__)
