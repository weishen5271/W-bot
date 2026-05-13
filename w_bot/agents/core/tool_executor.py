from __future__ import annotations

import asyncio
import time
from typing import Any

from langchain_core.messages import AnyMessage, ToolMessage

from w_bot.utils.helpers import _tool_result_to_text

from .logging_config import get_logger
from .session_models import RuntimeConfig
from .tool_policy import ToolPolicy

logger = get_logger(__name__)


class ToolExecutor:
    """Execute model-requested tool calls with conservative ordering rules."""

    def __init__(
        self,
        *,
        tools_by_name: dict[str, Any],
        policy: ToolPolicy | None = None,
    ) -> None:
        self._tools_by_name = tools_by_name
        self._policy = policy or ToolPolicy()

    async def execute(
        self,
        *,
        tool_calls: list[dict[str, Any]],
        messages: list[AnyMessage],
        session_id: str,
        config: RuntimeConfig | None = None,
        runtime: Any = None,
    ) -> list[ToolMessage]:
        cfg = config or RuntimeConfig()
        if not tool_calls:
            return []
        if self._policy.can_parallelize(tool_calls):
            return await self._execute_parallel(
                tool_calls=tool_calls,
                messages=messages,
                session_id=session_id,
                config=cfg,
                runtime=runtime,
            )
        return [
            await self._execute_one(
                index=index,
                tool_call=tool_call,
                messages=messages,
                session_id=session_id,
                config=cfg,
                runtime=runtime,
            )
            for index, tool_call in enumerate(tool_calls)
        ]

    async def _execute_parallel(
        self,
        *,
        tool_calls: list[dict[str, Any]],
        messages: list[AnyMessage],
        session_id: str,
        config: RuntimeConfig,
        runtime: Any,
    ) -> list[ToolMessage]:
        results = await asyncio.gather(
            *(
                self._execute_one(
                    index=index,
                    tool_call=tool_call,
                    messages=messages,
                    session_id=session_id,
                    config=config,
                    runtime=runtime,
                )
                for index, tool_call in enumerate(tool_calls)
            ),
            return_exceptions=True,
        )
        tool_messages: list[ToolMessage] = []
        for index, result in enumerate(results):
            if isinstance(result, BaseException):
                logger.exception("Parallel tool execution failed", exc_info=result)
                tool_messages.append(
                    ToolMessage(
                        content=f"Tool execution failed: {type(result).__name__}: {result}",
                        tool_call_id=f"tool_call_{index}",
                        name="tool_error",
                    )
                )
                continue
            tool_messages.append(result)
        return tool_messages

    async def _execute_one(
        self,
        *,
        index: int,
        tool_call: dict[str, Any],
        messages: list[AnyMessage],
        session_id: str,
        config: RuntimeConfig,
        runtime: Any,
    ) -> ToolMessage:
        name = str(tool_call.get("name") or "").strip()
        tool_call_id = str(tool_call.get("id") or f"tool_call_{index}")
        tool = self._tools_by_name.get(name)
        if tool is None:
            return ToolMessage(content=f"Tool not found: {name}", tool_call_id=tool_call_id, name=name)
        args = tool_call.get("args")
        if args is None:
            args = tool_call.get("arguments")
        if not isinstance(args, dict):
            args = {}
        started_at = time.monotonic()
        self._emit_tool_progress(config, "tool.started", name, args)
        try:
            tool_context = {
                "runtime": runtime,
                "graph": getattr(runtime, "compat_adapter", runtime),
                "state_messages": list(messages),
                "config": config,
                "status_callback": config.status_callback,
                "tool_progress_callback": config.tool_progress_callback,
                "thread_id": session_id,
                "subagent_depth": 0,
            }
            effective_args = {**args, "_wbot_tool_context": tool_context}
            if hasattr(tool, "ainvoke") and callable(tool.ainvoke):
                raw_result = await tool.ainvoke(effective_args)
            elif hasattr(tool, "invoke") and callable(tool.invoke):
                raw_result = await asyncio.to_thread(tool.invoke, effective_args)
            else:
                raw_result = await asyncio.to_thread(tool, **effective_args)
            content = _tool_result_to_text(raw_result)
        except Exception as exc:
            logger.exception("Tool execution failed: %s", name)
            content = f"Tool execution failed: {type(exc).__name__}: {exc}"
        elapsed_seconds = time.monotonic() - started_at
        self._emit_tool_progress(config, "tool.completed", name, args, elapsed_seconds=elapsed_seconds)
        return ToolMessage(content=content, tool_call_id=tool_call_id, name=name)

    @staticmethod
    def _emit_tool_progress(
        config: RuntimeConfig,
        event_type: str,
        tool_name: str,
        args: dict[str, Any],
        *,
        elapsed_seconds: float | None = None,
    ) -> None:
        callback = config.tool_progress_callback
        if callback is None:
            return
        try:
            callback(
                event_type,
                tool_name,
                str(args.get("path") or args.get("query") or args.get("command") or tool_name),
                args,
                elapsed_seconds=elapsed_seconds,
                ok=True,
            )
        except Exception:
            logger.debug("Tool progress callback failed", exc_info=True)
