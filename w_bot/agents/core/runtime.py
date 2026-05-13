from __future__ import annotations

from typing import Any

import asyncio

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage

from .continuation import ContinuationController
from .context import ContextBuilder
from .context_optimizer import ContextOptimizer
from .config import TokenOptimizationSettings
from .logging_config import get_logger
from .memory_context import MemoryContextRetriever
from .message_utils import _extract_last_user_message, _to_text_content
from .model_routing import ModelRouter
from .model_runner import ModelRunner
from .post_turn import PostTurnProcessor
from .session_db import SessionStore
from .session_models import AgentTurnResult, RuntimeConfig
from .runtime_compat import RuntimeCompatAdapter
from .tool_analysis import (
    _extract_tool_failure_summary,
    _is_tool_failure_content,
    _same_tool_call_streak,
    _summarize_tool_calls,
)
from .tool_executor import ToolExecutor
from .turn_context import TurnPromptBuilder

logger = get_logger(__name__)


class AgentRuntime:
    """Explicit AgentLoop runtime.

    This implementation is intentionally still narrow: it owns the explicit
    model/tool loop and business session storage, while advanced context
    optimization and post-turn processors will be migrated in later steps. The
    channel entrypoints now use this runtime directly.
    """

    def __init__(
        self,
        *,
        llm: Any,
        session_store: SessionStore,
        user_id: str = "",
        source: str = "unknown",
        model_name: str = "",
        llm_image: Any = None,
        llm_audio: Any = None,
        memory_retriever: MemoryContextRetriever | None = None,
        context_builder: ContextBuilder | None = None,
        tools: list[Any] | None = None,
        tool_executor: ToolExecutor | None = None,
        post_turn_processor: PostTurnProcessor | None = None,
        session_search_db: Any = None,
        context_optimizer: ContextOptimizer | None = None,
        token_optimization_settings: TokenOptimizationSettings | None = None,
        continuation_controller: ContinuationController | None = None,
    ) -> None:
        self._llm = llm
        self._session_store = session_store
        self._user_id = user_id
        self._source = source.strip() or "unknown"
        self._model_name = model_name.strip()
        self._model_runner = ModelRunner(llm=llm)
        self._memory_retriever = memory_retriever
        self._turn_prompt_builder = TurnPromptBuilder(context_builder=context_builder)
        self._context_optimizer = context_optimizer or ContextOptimizer(
            settings=token_optimization_settings,
            llm=llm,
        )
        self._continuation_controller = continuation_controller or ContinuationController()
        self._compat_adapter = RuntimeCompatAdapter(runtime=self)
        self._post_turn_processor = post_turn_processor or PostTurnProcessor(
            session_store=session_store,
            session_search_db=session_search_db,
            context_optimizer=self._context_optimizer,
            source=self._source,
            user_id=self._user_id,
            model=self._model_name,
        )
        tools_by_name = {
            str(getattr(tool, "name", "")).strip(): tool
            for tool in (tools or [])
            if str(getattr(tool, "name", "")).strip()
        }
        self._model_router = ModelRouter(
            llm_text=llm,
            llm_image=llm_image,
            llm_audio=llm_audio,
            tools_by_name=tools_by_name,
        )
        self._tool_executor = tool_executor or ToolExecutor(tools_by_name=tools_by_name)

    @property
    def session_store(self) -> SessionStore:
        return self._session_store

    @property
    def llm(self) -> Any:
        return self._llm

    @property
    def compat_adapter(self) -> RuntimeCompatAdapter:
        return self._compat_adapter

    def get_session_messages(self, session_id: str, *, limit: int | None = None) -> list[AnyMessage]:
        return self._session_store.get_messages(session_id, limit=limit)

    def run_turn(
        self,
        *,
        session_id: str,
        inbound_messages: list[AnyMessage],
        config: RuntimeConfig | None = None,
    ) -> AgentTurnResult:
        cfg = config or RuntimeConfig()
        self._emit_status(cfg, "正在初始化显式 AgentRuntime 回合...")
        self._session_store.ensure_session(
            session_id=session_id,
            source=self._source,
            user_id=self._user_id,
            model=self._model_name,
        )
        if inbound_messages:
            self._session_store.append_messages(session_id, inbound_messages)

        memory_context = self._retrieve_memory_context(session_id=session_id, config=cfg)
        turn_messages: list[AnyMessage] = [*inbound_messages]
        total_tool_calls = 0
        consecutive_tool_failures = 0
        last_failed_tool_names = ""
        last_tool_error = ""
        error = ""
        for iteration in range(max(1, int(cfg.recursion_limit))):
            history = self._session_store.get_messages(session_id)
            prompt_messages = self._build_prompt_messages(
                history=history,
                memory_context=memory_context,
                session_id=session_id,
                config=cfg,
            )
            self._emit_status(cfg, "正在生成回复...")
            route_selection = self._model_router.select(history=history, messages=prompt_messages)
            self._emit_status(cfg, f"已选择模型路由：{route_selection.route}。")
            model_result = self._model_runner.invoke(
                messages=prompt_messages,
                config=cfg,
                llm=route_selection.llm,
                fallback_llm=self._llm,
                system_prompt=_to_text_content(getattr(prompt_messages[0], "content", "")),
            )
            response = model_result.message
            error = model_result.error
            self._session_store.append_message(session_id, response)
            turn_messages.append(response)

            tool_calls = list(getattr(response, "tool_calls", []) or [])
            if not tool_calls:
                continuation_result = self._maybe_continue_non_tool_reply(
                    session_id=session_id,
                    history=history,
                    memory_context=memory_context,
                    response=response,
                    config=cfg,
                    route_llm=route_selection.llm,
                    attempt=0,
                )
                if continuation_result is not None:
                    response = continuation_result
                    self._session_store.append_message(session_id, response)
                    turn_messages.append(response)
                    tool_calls = list(getattr(response, "tool_calls", []) or [])
                    if tool_calls:
                        # Let the normal tool branch handle the continuation result.
                        pass
                    else:
                        final_response = _to_text_content(response.content).strip()
                        self._after_turn(session_id=session_id, config=cfg)
                        self._emit_status(cfg, "回复已写入业务会话存储。")
                        return AgentTurnResult(
                            session_id=session_id,
                            final_response=final_response,
                            messages=turn_messages,
                            tool_calls=total_tool_calls,
                            completed=True,
                            error=error,
                            metadata={"iterations": iteration + 1, "continuation": True},
                        )
                if not tool_calls:
                    final_response = _to_text_content(response.content).strip()
                    self._after_turn(session_id=session_id, config=cfg)
                    self._emit_status(cfg, "回复已写入业务会话存储。")
                    return AgentTurnResult(
                        session_id=session_id,
                        final_response=final_response,
                        messages=turn_messages,
                        tool_calls=total_tool_calls,
                        completed=model_result.completed,
                        error=error,
                        metadata={"iterations": iteration + 1},
                    )

            guard_message = self._guard_tool_calls(
                history=[*history, response],
                config=cfg,
            )
            if guard_message is not None:
                self._session_store.append_message(session_id, guard_message)
                turn_messages.append(guard_message)
                final_response = _to_text_content(guard_message.content).strip()
                self._after_turn(session_id=session_id, config=cfg)
                return AgentTurnResult(
                    session_id=session_id,
                    final_response=final_response,
                    messages=turn_messages,
                    tool_calls=total_tool_calls,
                    completed=False,
                    error=error,
                    metadata={"iterations": iteration + 1, "stopped_by": "tool_guard"},
                )

            total_tool_calls += len(tool_calls)
            self._emit_status(cfg, f"准备执行工具调用：{_summarize_tool_calls(tool_calls)}")
            tool_messages = asyncio.run(
                self._tool_executor.execute(
                    tool_calls=tool_calls,
                    messages=[*history, response],
                    session_id=session_id,
                    config=cfg,
                    runtime=self,
                )
            )
            if tool_messages:
                self._session_store.append_messages(session_id, tool_messages)
                turn_messages.extend(tool_messages)
                failed_tool_names, failed_tool_errors = self._tool_failures(tool_messages)
                if failed_tool_errors:
                    consecutive_tool_failures += 1
                    last_failed_tool_names = ", ".join(failed_tool_names[:3])
                    last_tool_error = " | ".join(failed_tool_errors[:3])
                    self._emit_status(
                        cfg,
                        f"工具执行失败（连续 {consecutive_tool_failures} 次）：{last_failed_tool_names}",
                    )
                    if consecutive_tool_failures >= max(1, int(cfg.max_consecutive_tool_failures)):
                        recover = self._build_tool_failure_recover_message(
                            consecutive_failures=consecutive_tool_failures,
                            tool_name=last_failed_tool_names,
                            detail=last_tool_error,
                        )
                        self._session_store.append_message(session_id, recover)
                        turn_messages.append(recover)
                        final_response = _to_text_content(recover.content).strip()
                        self._after_turn(session_id=session_id, config=cfg)
                        return AgentTurnResult(
                            session_id=session_id,
                            final_response=final_response,
                            messages=turn_messages,
                            tool_calls=total_tool_calls,
                            completed=False,
                            error=error,
                            metadata={
                                "iterations": iteration + 1,
                                "stopped_by": "consecutive_tool_failures",
                                "consecutive_tool_failures": consecutive_tool_failures,
                            },
                        )
                else:
                    consecutive_tool_failures = 0
                    last_failed_tool_names = ""
                    last_tool_error = ""
                    self._emit_status(cfg, "工具执行完成，继续整理结果。")

        final_response = f"显式 AgentRuntime 已达到本轮最大迭代次数（{cfg.recursion_limit}），已停止继续调用。"
        fallback = AIMessage(content=final_response)
        self._session_store.append_message(session_id, fallback)
        turn_messages.append(fallback)
        self._after_turn(session_id=session_id, config=cfg)
        return AgentTurnResult(
            session_id=session_id,
            final_response=final_response,
            messages=turn_messages,
            tool_calls=total_tool_calls,
            completed=False,
            error=error,
            metadata={"iterations": max(1, int(cfg.recursion_limit)), "stopped_by": "recursion_limit"},
        )

    def _after_turn(self, *, session_id: str, config: RuntimeConfig) -> None:
        try:
            self._post_turn_processor.after_turn(session_id=session_id, config=config)
        except Exception:
            logger.warning("AgentRuntime post-turn processing failed: session_id=%s", session_id, exc_info=True)

    def _maybe_continue_non_tool_reply(
        self,
        *,
        session_id: str,
        history: list[AnyMessage],
        memory_context: str,
        response: AIMessage,
        config: RuntimeConfig,
        route_llm: Any,
        attempt: int,
    ) -> AIMessage | None:
        user_goal = _extract_last_user_message(history)
        decision = self._continuation_controller.decide(
            user_goal=user_goal,
            history=history,
            response=response,
            attempt=attempt,
        )
        if not decision.should_continue:
            return None
        self._emit_status(config, "检测到当前回复仍像中间状态，继续执行当前任务。")
        continuation_history = [
            *history,
            response,
            HumanMessage(content=decision.prompt),
        ]
        prompt_messages = self._build_prompt_messages(
            history=continuation_history,
            memory_context=memory_context,
            session_id=session_id,
            config=config,
        )
        result = self._model_runner.invoke(
            messages=prompt_messages,
            config=config,
            llm=route_llm,
            fallback_llm=self._llm,
            system_prompt=_to_text_content(getattr(prompt_messages[0], "content", "")),
        )
        return result.message

    def _guard_tool_calls(
        self,
        *,
        history: list[AnyMessage],
        config: RuntimeConfig,
    ) -> AIMessage | None:
        tool_steps = self._count_tool_steps_since_last_human(history)
        max_steps = max(1, int(config.max_tool_steps_per_turn))
        if tool_steps > max_steps:
            self._emit_status(config, f"工具调用已达上限（{max_steps}），本轮停止继续调用。")
            return AIMessage(
                content=(
                    f"本轮工具调用次数已达上限（{max_steps}）。"
                    "我已停止自动重试，建议你补充更明确的目标或约束后我再继续。"
                )
            )
        signature, repeat_count = _same_tool_call_streak(history)
        max_repeats = max(1, int(config.max_same_tool_call_repeats))
        if signature and repeat_count >= max_repeats:
            self._emit_status(config, f"检测到重复工具调用（连续 {repeat_count} 次），本轮停止继续调用。")
            return AIMessage(
                content=(
                    f"检测到同一工具调用连续重复 {repeat_count} 次。"
                    "我已停止自动重试，建议调整输入条件或改用其它策略后再继续。"
                )
            )
        return None

    @staticmethod
    def _count_tool_steps_since_last_human(messages: list[AnyMessage]) -> int:
        count = 0
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                break
            if isinstance(message, AIMessage) and message.tool_calls:
                count += 1
        return count

    @staticmethod
    def _tool_failures(tool_messages: list[AnyMessage]) -> tuple[list[str], list[str]]:
        failed_tool_names: list[str] = []
        failed_tool_errors: list[str] = []
        for index, message in enumerate(tool_messages):
            content = getattr(message, "content", "")
            if not _is_tool_failure_content(content):
                continue
            failed_tool_names.append(str(getattr(message, "name", "") or f"tool_call_{index}"))
            failed_tool_errors.append(_extract_tool_failure_summary(content))
        return failed_tool_names, failed_tool_errors

    @staticmethod
    def _build_tool_failure_recover_message(
        *,
        consecutive_failures: int,
        tool_name: str,
        detail: str,
    ) -> AIMessage:
        normalized_tool = tool_name.strip() or "工具"
        detail_suffix = f"\n最近一次错误：{detail}" if detail else ""
        return AIMessage(
            content=(
                f"{normalized_tool} 连续执行失败 {consecutive_failures} 次，我已停止自动重试，避免流程卡住。"
                "你可以调整输入参数、补充上下文，或让我改用别的策略继续处理。"
                f"{detail_suffix}"
            )
        )

    def _retrieve_memory_context(self, *, session_id: str, config: RuntimeConfig) -> str:
        if self._memory_retriever is None:
            return ""
        history = self._session_store.get_messages(session_id)
        query = _extract_last_user_message(history)
        self._emit_status(config, "正在检索长期记忆上下文...")
        memory_result = self._memory_retriever.retrieve_context(query)
        if memory_result.skipped:
            self._emit_status(config, "跳过长期记忆检索（当前回合无用户文本）。")
        elif memory_result.hit_count:
            self._emit_status(config, f"已加载 {memory_result.hit_count} 条长期记忆。")
        else:
            self._emit_status(config, "未命中长期记忆，继续直接回答。")
        return memory_result.context

    def _build_prompt_messages(
        self,
        *,
        history: list[AnyMessage],
        memory_context: str,
        session_id: str,
        config: RuntimeConfig,
    ) -> list[AnyMessage]:
        conversation_summary, summarized_message_count = self._session_store.get_summary(session_id)
        optimized = self._context_optimizer.prepare(
            history=history,
            conversation_summary=conversation_summary,
            summarized_message_count=summarized_message_count,
            defer_summary_update=config.defer_summary_update,
        )
        if (
            optimized.conversation_summary != conversation_summary
            or optimized.summarized_message_count != summarized_message_count
        ):
            self._session_store.update_summary(
                session_id,
                summary=optimized.conversation_summary,
                summarized_message_count=optimized.summarized_message_count,
            )
        turn_prompt = self._turn_prompt_builder.build(
            memory_context=memory_context,
            conversation_summary=optimized.conversation_summary,
            token_budget_state=optimized.token_budget_state,
            context_compaction_level=optimized.context_compaction_level,
        )
        return [
            SystemMessage(content=turn_prompt.system_prompt),
            *optimized.recent_messages,
        ]

    @staticmethod
    def _emit_status(config: RuntimeConfig, text: str) -> None:
        callback = config.status_callback
        if callback is None:
            return
        try:
            callback(text)
        except Exception:
            logger.debug("Runtime status callback failed", exc_info=True)
