from __future__ import annotations

from pathlib import Path
from typing import Any

from ..logging_config import get_logger
from ..memory import LongTermMemoryStore
from .cron_tool import build_cron_tool
from .exec_tool import build_exec_tool
from .filesystem_tools import build_filesystem_tools
from .mcp_tools import build_mcp_tools
from .task_tools import build_task_tools
from .web_tools import build_web_tools

logger = get_logger(__name__)


def build_tools(
    *,
    memory_store: LongTermMemoryStore,
    user_id: str,
    tavily_api_key: str,
    enable_cron_service: bool,
    mcp_servers: list[dict[str, Any]] | None,
    extra_readonly_dirs: list[str] | None = None,
) -> list[Any]:
    logger.info("Building tools for user_id=%s", user_id)
    _ = memory_store
    workspace_root = Path.cwd().resolve()
    sandbox_root = workspace_root / ".sandbox"
    readonly_roots = [workspace_root]
    for candidate in extra_readonly_dirs or []:
        try:
            readonly_roots.append(Path(candidate).resolve())
        except OSError:
            logger.warning("Skip invalid readonly root: %s", candidate)

    tools: list[Any] = []
    tools.extend(build_filesystem_tools(workspace_root=workspace_root, readonly_roots=readonly_roots))
    tools.extend(build_web_tools(tavily_api_key=tavily_api_key))
    tools.extend(build_task_tools(workspace_root=workspace_root))
    tools.append(build_exec_tool(workspace_root=workspace_root, sandbox_root=sandbox_root))

    if enable_cron_service:
        tools.append(build_cron_tool(workspace_root=workspace_root))

    tools.extend(build_mcp_tools(mcp_servers or []))
    logger.info("Registered tools: %s", [getattr(t, "name", str(t)) for t in tools])
    return tools
