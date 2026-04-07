"""MCP management CLI commands."""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

import click

from ..core.config import DEFAULT_APP_CONFIG_PATH, load_settings
from .manager import MCPManager


@click.group("mcp")
def mcp_group():
    """MCP server management."""
    pass


@mcp_group.command("list")
@click.option("--config", default=DEFAULT_APP_CONFIG_PATH, help="Config path")
def list_servers(config: str):
    """List all configured MCP servers and their status."""
    settings = load_settings(config_path=config)
    manager = MCPManager(settings.mcp_servers)

    servers = manager.list_servers()
    if not servers:
        click.echo("No MCP servers configured.")
        return

    click.echo(f"{'Name':<20} {'Status':<15} {'Transport':<10} {'Tools':<8} {'Last Health Check':<25}")
    click.echo("-" * 80)
    for srv in servers:
        status_display = srv["status"]
        if not srv["enabled"]:
            status_display += " (disabled)"
        click.echo(
            f"{srv['name']:<20} {status_display:<15} {srv['transport']:<10} "
            f"{srv['tool_count']:<8} {srv['last_health_check'] or 'N/A':<25}"
        )
        if srv.get("error"):
            click.echo(f"  Error: {srv['error']}")


@mcp_group.command("add")
@click.option("--config", default=DEFAULT_APP_CONFIG_PATH, help="Config path")
@click.option("--name", required=True, help="Server name")
@click.option("--url", required=True, help="Server base URL")
@click.option("--transport", default="http", type=click.Choice(["http", "sse", "streamable-http"]), help="Transport type")
@click.option("--discovery-path", default="/tools", help="Tool discovery path")
@click.option("--invoke-path", default="/tools/{tool}", help="Tool invoke path template")
def add_server(config: str, name: str, url: str, transport: str, discovery_path: str, invoke_path: str):
    """Add a new MCP server configuration."""
    cfg_path = _resolve_config_path(config)

    try:
        data = _load_config(cfg_path)
    except FileNotFoundError:
        click.echo(f"Config file not found: {cfg_path}")
        raise click.Abort()

    mcp_servers = data.get("agent", {}).get("mcpServers", [])
    # Check if server with same name exists
    for srv in mcp_servers:
        if srv.get("name") == name:
            click.echo(f"Server '{name}' already exists. Use 'remove' first to update.")
            return

    new_server = {
        "name": name,
        "enabled": True,
        "transport": transport,
        "baseUrl": url,
        "discoveryPath": discovery_path,
        "invokePathTemplate": invoke_path,
        "timeout": 20,
        "retry": {
            "enabled": True,
            "maxAttempts": 3,
            "backoffMs": 500,
        },
    }
    mcp_servers.append(new_server)

    if "agent" not in data:
        data["agent"] = {}
    data["agent"]["mcpServers"] = mcp_servers

    _save_config(cfg_path, data)
    click.echo(f"Added MCP server '{name}' -> {url}")
    click.echo(f"  Transport: {transport}")
    click.echo(f"  Discovery: {discovery_path}")


@mcp_group.command("remove")
@click.option("--config", default=DEFAULT_APP_CONFIG_PATH, help="Config path")
@click.option("--name", required=True, help="Server name")
def remove_server(config: str, name: str):
    """Remove an MCP server configuration."""
    cfg_path = _resolve_config_path(config)

    try:
        data = _load_config(cfg_path)
    except FileNotFoundError:
        click.echo(f"Config file not found: {cfg_path}")
        raise click.Abort()

    mcp_servers = data.get("agent", {}).get("mcpServers", [])
    before_count = len(mcp_servers)
    mcp_servers = [s for s in mcp_servers if s.get("name") != name]

    if len(mcp_servers) == before_count:
        click.echo(f"Server '{name}' not found.")
        return

    data["agent"]["mcpServers"] = mcp_servers
    _save_config(cfg_path, data)
    click.echo(f"Removed MCP server '{name}'")


@mcp_group.command("discover")
@click.option("--config", default=DEFAULT_APP_CONFIG_PATH, help="Config path")
@click.option("--name", default=None, help="Specific server to discover (default: all)")
@click.option("--json-output", is_flag=True, help="Output as JSON")
def discover_tools(config: str, name: str | None, json_output: bool):
    """Discover tools from MCP server(s)."""
    settings = load_settings(config_path=config)
    manager = MCPManager(settings.mcp_servers)

    async def _discover():
        if name:
            tools = await manager.discover_tools(name)
            return [{"name": name, "tools": tools}]
        else:
            servers = []
            for srv_name in settings.mcp_servers:
                srv_name_str = srv_name.get("name", "unnamed")
                tools = await manager.discover_tools(srv_name_str)
                servers.append({"name": srv_name_str, "tools": tools})
            return servers

    servers = asyncio.run(_discover())

    if json_output:
        click.echo(json.dumps(servers, ensure_ascii=False, indent=2))
    else:
        for srv in servers:
            click.echo(f"\n=== {srv['name']} ===")
            tools = srv["tools"]
            if not tools:
                click.echo("  (no tools discovered)")
            for tool in tools:
                desc = tool.get("description", "N/A")
                if desc:
                    click.echo(f"  - {tool['name']}: {desc}")
                else:
                    click.echo(f"  - {tool['name']}")


@mcp_group.command("test")
@click.option("--config", default=DEFAULT_APP_CONFIG_PATH, help="Config path")
@click.option("--name", required=True, help="Server name")
def test_server(config: str, name: str):
    """Test connection to an MCP server."""
    settings = load_settings(config_path=config)
    manager = MCPManager(settings.mcp_servers)

    async def _test():
        tools = await manager.discover_tools(name)
        if tools:
            click.echo(f"✓ Server '{name}' is healthy. Found {len(tools)} tools.")
            for tool in tools[:5]:  # Show first 5 tools
                click.echo(f"  - {tool['name']}")
            if len(tools) > 5:
                click.echo(f"  ... and {len(tools) - 5} more")
        else:
            click.echo(f"✗ Server '{name}' failed to discover tools.", err=True)
            state = manager.get_server(name)
            if state and state.error_message:
                click.echo(f"  Error: {state.error_message}", err=True)

    try:
        asyncio.run(_test())
    except Exception as e:
        click.echo(f"✗ Server '{name}' failed: {e}", err=True)


@mcp_group.command("health")
@click.option("--config", default=DEFAULT_APP_CONFIG_PATH, help="Config path")
@click.option("--name", default=None, help="Specific server to check (default: all)")
def health_check(config: str, name: str | None):
    """Perform health check on MCP server(s)."""
    settings = load_settings(config_path=config)
    manager = MCPManager(settings.mcp_servers)

    async def _health():
        if name:
            result = await manager.health_check(name)
            return [{"name": name, "healthy": result}]
        else:
            results = []
            for srv_name in settings.mcp_servers:
                srv_name_str = srv_name.get("name", "unnamed")
                result = await manager.health_check(srv_name_str)
                results.append({"name": srv_name_str, "healthy": result})
            return results

    results = asyncio.run(_health())

    for r in results:
        status_icon = "✓" if r["healthy"] else "✗"
        click.echo(f"{status_icon} {r['name']}: {'healthy' if r['healthy'] else 'unhealthy'}")


@mcp_group.command("healthd")
@click.option("--config", default=DEFAULT_APP_CONFIG_PATH, help="Config path")
@click.option("--interval", default=30, help="Health check interval in seconds")
def health_daemon(config: str, interval: int):
    """Run MCP health check daemon in background."""
    settings = load_settings(config_path=config)
    manager = MCPManager(settings.mcp_servers)

    async def _run_health_loop():
        click.echo(f"Starting MCP health check daemon (interval={interval}s)")
        click.echo("Press Ctrl+C to stop")
        try:
            await manager.start_health_check_loop(interval_seconds=interval)
        except KeyboardInterrupt:
            click.echo("\nHealth check daemon stopped.")

    asyncio.run(_run_health_loop())


@mcp_group.command("start")
@click.option("--config", default=DEFAULT_APP_CONFIG_PATH, help="Config path")
@click.option("--name", required=True, help="Server name")
def start_server(config: str, name: str):
    """Start an MCP server (for stdio transport or to initialize connection)."""
    settings = load_settings(config_path=config)
    manager = MCPManager(settings.mcp_servers)

    async def _start():
        success = await manager.start_server(name)
        if success:
            click.echo(f"✓ Started MCP server '{name}'")
        else:
            click.echo(f"✗ Failed to start MCP server '{name}'", err=True)
            state = manager.get_server(name)
            if state and state.error_message:
                click.echo(f"  Error: {state.error_message}", err=True)

    try:
        asyncio.run(_start())
    except Exception as e:
        click.echo(f"✗ Server '{name}' failed: {e}", err=True)


@mcp_group.command("stop")
@click.option("--config", default=DEFAULT_APP_CONFIG_PATH, help="Config path")
@click.option("--name", required=True, help="Server name")
def stop_server(config: str, name: str):
    """Stop an MCP server."""
    settings = load_settings(config_path=config)
    manager = MCPManager(settings.mcp_servers)

    async def _stop():
        await manager.stop_server(name)
        click.echo(f"✓ Stopped MCP server '{name}'")

    try:
        asyncio.run(_stop())
    except Exception as e:
        click.echo(f"✗ Failed to stop server '{name}': {e}", err=True)


@mcp_group.command("restart")
@click.option("--config", default=DEFAULT_APP_CONFIG_PATH, help="Config path")
@click.option("--name", required=True, help="Server name")
def restart_server(config: str, name: str):
    """Restart an MCP server."""
    settings = load_settings(config_path=config)
    manager = MCPManager(settings.mcp_servers)

    async def _restart():
        await manager.stop_server(name)
        success = await manager.start_server(name)
        if success:
            click.echo(f"✓ Restarted MCP server '{name}'")
        else:
            click.echo(f"✗ Failed to restart MCP server '{name}'", err=True)

    try:
        asyncio.run(_restart())
    except Exception as e:
        click.echo(f"✗ Restart failed for '{name}': {e}", err=True)


@mcp_group.command("add-stdio")
@click.option("--config", default=DEFAULT_APP_CONFIG_PATH, help="Config path")
@click.option("--name", required=True, help="Server name")
@click.option("--command", required=True, help="Command to run MCP server")
@click.option("--args", default="", help="Command arguments (comma-separated)")
@click.option("--env", default="", help="Environment variables (KEY=VALUE,comma-separated)")
def add_stdio_server(config: str, name: str, command: str, args: str, env: str):
    """Add a new stdio-based MCP server configuration."""
    cfg_path = _resolve_config_path(config)

    try:
        data = _load_config(cfg_path)
    except FileNotFoundError:
        click.echo(f"Config file not found: {cfg_path}")
        raise click.Abort()

    mcp_servers = data.get("agent", {}).get("mcpServers", [])
    # Check if server with same name exists
    for srv in mcp_servers:
        if srv.get("name") == name:
            click.echo(f"Server '{name}' already exists. Use 'remove' first to update.")
            return

    # Parse args and env
    parsed_args = [a.strip() for a in args.split(",") if a.strip()] if args else []
    parsed_env = {}
    if env:
        for pair in env.split(","):
            if "=" in pair:
                key, value = pair.split("=", 1)
                parsed_env[key.strip()] = value.strip()

    new_server = {
        "name": name,
        "enabled": True,
        "transport": "stdio",
        "command": command,
        "args": parsed_args,
        "env": parsed_env,
        "timeout": 60,
    }
    mcp_servers.append(new_server)

    if "agent" not in data:
        data["agent"] = {}
    data["agent"]["mcpServers"] = mcp_servers

    _save_config(cfg_path, data)
    click.echo(f"Added stdio MCP server '{name}'")
    click.echo(f"  Command: {command} {' '.join(parsed_args)}")


def _resolve_config_path(config: str) -> str:
    """Resolve config path to absolute path."""
    from pathlib import Path
    p = Path(config)
    if p.is_absolute():
        return config
    return str(Path.cwd() / config)


def _load_config(path: str) -> dict[str, Any]:
    """Load config from JSON file."""
    from pathlib import Path
    cfg_file = Path(path)
    if not cfg_file.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    return json.loads(cfg_file.read_text(encoding="utf-8"))


def _save_config(path: str, data: dict[str, Any]) -> None:
    """Save config to JSON file."""
    from pathlib import Path
    cfg_file = Path(path)
    cfg_file.parent.mkdir(parents=True, exist_ok=True)
    cfg_file.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def register_mcp_commands(parser: Any) -> None:
    """Register MCP commands to an argparse subparser."""
    mcp_parser = parser.add_parser("mcp", help="MCP server management")
    mcp_subparsers = mcp_parser.add_subparsers(dest="mcp_command")

    # list
    list_parser = mcp_subparsers.add_parser("list", help="List MCP servers")
    list_parser.add_argument("--config", default=DEFAULT_APP_CONFIG_PATH)

    # add
    add_parser = mcp_subparsers.add_parser("add", help="Add MCP server")
    add_parser.add_argument("--name", required=True)
    add_parser.add_argument("--url", required=True)
    add_parser.add_argument("--transport", default="http", choices=["http", "sse", "streamable-http"])
    add_parser.add_argument("--discovery-path", default="/tools")
    add_parser.add_argument("--invoke-path", default="/tools/{tool}")
    add_parser.add_argument("--config", default=DEFAULT_APP_CONFIG_PATH)

    # remove
    remove_parser = mcp_subparsers.add_parser("remove", help="Remove MCP server")
    remove_parser.add_argument("--name", required=True)
    remove_parser.add_argument("--config", default=DEFAULT_APP_CONFIG_PATH)

    # discover
    discover_parser = mcp_subparsers.add_parser("discover", help="Discover MCP tools")
    discover_parser.add_argument("--name", default=None)
    discover_parser.add_argument("--json-output", action="store_true")
    discover_parser.add_argument("--config", default=DEFAULT_APP_CONFIG_PATH)

    # test
    test_parser = mcp_subparsers.add_parser("test", help="Test MCP server")
    test_parser.add_argument("--name", required=True)
    test_parser.add_argument("--config", default=DEFAULT_APP_CONFIG_PATH)

    # health
    health_parser = mcp_subparsers.add_parser("health", help="Health check")
    health_parser.add_argument("--name", default=None)
    health_parser.add_argument("--config", default=DEFAULT_APP_CONFIG_PATH)


def run_mcp_command(args: Any) -> None:
    """Run MCP command based on parsed args."""
    if args.mcp_command == "list":
        list_servers.callback(config=args.config)
    elif args.mcp_command == "add":
        add_server.callback(
            config=args.config,
            name=args.name,
            url=args.url,
            transport=args.transport,
            discovery_path=args.discovery_path,
            invoke_path=args.invoke_path,
        )
    elif args.mcp_command == "remove":
        remove_server.callback(config=args.config, name=args.name)
    elif args.mcp_command == "discover":
        discover_tools.callback(config=args.config, name=args.name, json_output=args.json_output)
    elif args.mcp_command == "test":
        test_server.callback(config=args.config, name=args.name)
    elif args.mcp_command == "health":
        health_check.callback(config=args.config, name=args.name)
    else:
        click.echo(f"Unknown MCP command: {args.mcp_command}")
        sys.exit(1)
