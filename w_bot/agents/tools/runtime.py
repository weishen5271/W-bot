from __future__ import annotations

from pathlib import Path
from typing import Any

from ..logging_config import get_logger
from ..memory import LongTermMemoryStore
from .base import Tool
from .cron import CronTool
from .filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from .mcp import build_mcp_tools
from .message import MessageTool
from .registry import ToolRegistry
from .shell import ExecTool
from .spawn import SpawnTool
from .web import WebFetchTool, WebSearchTool

logger = get_logger(__name__)


def build_tools(
    *,
    memory_store: LongTermMemoryStore,
    user_id: str,
    tavily_api_key: str,
    enable_cron_service: bool,
    mcp_servers: list[dict[str, Any]] | None,
    extra_readonly_dirs: list[str] | None = None,
) -> list[Tool]:
    logger.info("Building tools for user_id=%s", user_id)
    _ = memory_store
    workspace_root = Path.cwd().resolve()
    readonly_roots = [workspace_root]
    for candidate in extra_readonly_dirs or []:
        try:
            readonly_roots.append(Path(candidate).resolve())
        except OSError:
            logger.warning("Skip invalid readonly root: %s", candidate)

    registry = ToolRegistry()
    registry.register(
        ReadFileTool(
            workspace=workspace_root,
            allowed_dir=workspace_root,
            extra_allowed_dirs=readonly_roots[1:],
        )
    )
    registry.register(WriteFileTool(workspace=workspace_root, allowed_dir=workspace_root))
    registry.register(EditFileTool(workspace=workspace_root, allowed_dir=workspace_root))
    registry.register(ListDirTool(workspace=workspace_root, allowed_dir=workspace_root))
    registry.register(WebSearchTool(provider="tavily" if tavily_api_key else "duckduckgo", api_key=tavily_api_key))
    registry.register(WebFetchTool())
    registry.register(MessageTool(workspace_root))
    registry.register(SpawnTool(workspace_root))
    registry.register(
        ExecTool(
            working_dir=str(workspace_root),
            restrict_to_workspace=True,
        )
    )

    if enable_cron_service:
        registry.register(CronTool(workspace_root))

    for tool in build_mcp_tools(mcp_servers or []):
        registry.register(tool)
    logger.info("Registered tools: %s", registry.tool_names)
    return registry.tools
