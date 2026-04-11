from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from typing import Annotated, Any, Callable, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from w_bot.utils.helpers import _tool_result_to_text

from ..intent import IntentClassifier, ToolRegistry
from ..intent.intent_detection import (  # noqa: F401 - re-exported for tests
    _contains_any,
    _continue_current_task_prompt,
    _has_tool_messages_since_last_human,
    _looks_like_capability_question,
    _looks_like_casual_chat,
    _looks_like_cron_request,
    _looks_like_exec_request,
    _looks_like_file_edit_request,
    _looks_like_file_read_request,
    _looks_like_message_request,
    _looks_like_project_inspection_request,
    _looks_like_spawn_request,
    _looks_like_web_request,
    _response_looks_incomplete,
    _should_check_completion_for_turn,
    _should_enable_tools_for_text,
    _should_expose_run_skill,
)
from ..memory.memory import LongTermMemoryStore
from ..multimodal import MultimodalNormalizer, MultimodalRuntimeConfig
from ..providers import resolve_provider_capabilities
from ..skills.skills import SkillsLoader
from ..skills.subagent import SubagentManager
from .config import IntentClassifierSettings, MultimodalSettings, TokenOptimizationSettings
from .context import ContextBuilder
from .logging_config import get_logger
from .message_utils import (
    _apply_context_compaction_strategy,
    _base_system_prompt,
    _build_summary_fallback,
    _determine_compaction_level,
    _emit_status,
    _extract_last_user_message,
    _format_token_budget_snapshot,
    _has_native_image_blocks,
    _is_messages_length_error,
    _merge_token_usage_dicts,
    _messages_to_summary_text,
    _recent_window_start,
    _resolve_debug_callback,
    _resolve_status_callback,
    _resolve_stream_token_callback,
    _resolve_thread_id,
    _resolve_tool_progress_callback,
    _route_for_history,
    _should_defer_summary_update,
    _to_text_content,
    normalize_messages_for_llm,
    sanitize_messages_for_llm,
)
from .openclaw_profile import OpenClawProfileLoader
from .streaming_utils import (
    _invoke_llm_with_optional_stream,
)
from .token_tracker import TokenBudgetManager, extract_token_usage, token_count_with_estimation
from .tool_analysis import (
    _build_text_only_retry_messages,
    _count_tool_steps_since_last_human,
    _extract_tool_failure_summary,
    _is_tool_failure_content,
    _runtime_error_reply_text,
    _same_tool_call_streak,
    _summarize_tool_calls,
)

logger = get_logger(__name__)


_RUNTIME_TOKEN_CALLBACK: Callable[[str], None] | None = None
_RUNTIME_DEBUG_CALLBACK: Callable[[str], None] | None = None


def _tool_args_preview(tool_name: str, args: dict[str, Any]) -> str:
    candidates = [
        args.get("url"),
        args.get("query"),
        args.get("path"),
        args.get("command"),
        args.get("task"),
        args.get("id"),
        args.get("working_dir"),
    ]
    for item in candidates:
        text = str(item or "").strip()
        if text:
            compact = " ".join(text.split())
            return compact[:96] + ("..." if len(compact) > 96 else "")
    if not args:
        return tool_name
    try:
        raw = json.dumps(args, ensure_ascii=False, sort_keys=True)
    except Exception:
        raw = str(args)
    compact = " ".join(raw.split())
    return compact[:96] + ("..." if len(compact) > 96 else "")


def _tool_progress_emoji(tool_name: str) -> str:
    normalized = (tool_name or "").strip().lower()
    if any(token in normalized for token in ["browser", "navigate", "web"]):
        return "🌐"
    if any(token in normalized for token in ["search", "grep", "find"]):
        return "🔎"
    if any(token in normalized for token in ["read", "fetch", "load"]):
        return "📖"
    if any(token in normalized for token in ["write", "edit", "patch"]):
        return "✍"
    if any(token in normalized for token in ["exec", "shell", "command"]):
        return "⚙"
    if any(token in normalized for token in ["spawn", "subagent", "wait"]):
        return "🧩"
    return "⚡"


def _tool_progress_action(tool_name: str) -> str:
    normalized = (tool_name or "").strip().lower()
    for token, label in [
        ("navigate", "navigate"),
        ("search", "search"),
        ("fetch", "fetch"),
        ("read", "read"),
        ("write", "write"),
        ("edit", "edit"),
        ("exec", "exec"),
        ("shell", "exec"),
        ("spawn", "spawn"),
        ("wait", "wait"),
    ]:
        if token in normalized:
            return label
    compact = normalized.replace("mcp_", "").replace("_tool", "")
    return compact[:18] if compact else "run"


def _tool_progress_line(
    *,
    event_type: str,
    tool_name: str,
    preview: str,
    elapsed_seconds: float | None,
    ok: bool | None,
) -> str:
    if event_type == "tool.started":
        return f"  ┊ ⚡ preparing {tool_name}..."
    label = " ".join((preview or tool_name).split())
    if len(label) > 88:
        label = label[:85] + "..."
    duration = f"  {elapsed_seconds:.1f}s" if elapsed_seconds is not None else ""
    suffix = " [error]" if ok is False else ""
    return f"  ┊ {_tool_progress_emoji(tool_name)} {_tool_progress_action(tool_name)}  {label}{duration}{suffix}"


def _emit_tool_progress(
    config: RunnableConfig | None,
    *,
    event_type: str,
    tool_name: str,
    preview: str = "",
    elapsed_seconds: float | None = None,
    ok: bool | None = None,
    function_args: dict[str, Any] | None = None,
) -> None:
    callback = _resolve_tool_progress_callback(config)
    if callback is not None:
        try:
            callback(
                event_type,
                tool_name,
                preview,
                function_args or {},
                elapsed_seconds=elapsed_seconds,
                ok=ok,
            )
            return
        except Exception:
            logger.debug("Tool progress callback failed", exc_info=True)
    _emit_status(
        config,
        _tool_progress_line(
            event_type=event_type,
            tool_name=tool_name,
            preview=preview,
            elapsed_seconds=elapsed_seconds,
            ok=ok,
        ),
    )


def set_runtime_callbacks(
    *,
    token_callback: Callable[[str], None] | None = None,
    debug_callback: Callable[[str], None] | None = None,
) -> None:
    global _RUNTIME_TOKEN_CALLBACK
    global _RUNTIME_DEBUG_CALLBACK
    _RUNTIME_TOKEN_CALLBACK = token_callback
    _RUNTIME_DEBUG_CALLBACK = debug_callback


def clear_runtime_callbacks() -> None:
    set_runtime_callbacks(token_callback=None, debug_callback=None)


def _get_runtime_token_callback() -> Callable[[str], None] | None:
    return _RUNTIME_TOKEN_CALLBACK


def _get_runtime_debug_callback() -> Callable[[str], None] | None:
    return _RUNTIME_DEBUG_CALLBACK


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    long_term_context: str
    conversation_summary: str
    summarized_message_count: int
    prepared_system_prompt_base: str
    latest_token_usage: dict[str, int]
    session_token_usage: dict[str, int]
    token_budget_state: dict[str, Any]
    context_compaction_level: str
    last_tool_failed: bool
    consecutive_tool_failures: int
    last_tool_name: str
    last_tool_error: str


class ScheduledGraphApp:
    def __init__(self, graph: Any, owner: "WBotGraph") -> None:
        self._graph = graph
        self._owner = owner

    def invoke(self, inputs: Any, config: RunnableConfig | None = None, **kwargs: Any) -> Any:
        result = self._graph.invoke(inputs, config=config, **kwargs)
        self._owner.schedule_deferred_summary(config)
        return result

    async def ainvoke(self, inputs: Any, config: RunnableConfig | None = None, **kwargs: Any) -> Any:
        result = await self._graph.ainvoke(inputs, config=config, **kwargs)
        self._owner.schedule_deferred_summary(config)
        return result

    def get_state(self, config: RunnableConfig | None = None, **kwargs: Any) -> Any:
        return self._graph.get_state(config, **kwargs)

    async def aget_state(self, config: RunnableConfig | None = None, **kwargs: Any) -> Any:
        aget_state = getattr(self._graph, "aget_state", None)
        if callable(aget_state):
            return await aget_state(config, **kwargs)
        return await asyncio.to_thread(self._graph.get_state, config, **kwargs)

    def list_subagents(self, *, status: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        return self._owner.list_subagents(status=status, limit=limit)

    def wait_for_subagent(self, job_id: str, *, timeout_seconds: int = 60) -> dict[str, Any]:
        return self._owner.wait_for_subagent(job_id, timeout_seconds=timeout_seconds)

    def spawn_subagent(
        self,
        *,
        agent_type: str,
        task: str,
        label: str = "",
        context_messages: list[AnyMessage] | None = None,
        parent_thread_id: str = "-",
        status_callback: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        return self._owner.spawn_subagent(
            agent_type=agent_type,
            task=task,
            label=label,
            context_messages=context_messages,
            parent_thread_id=parent_thread_id,
            status_callback=status_callback,
        )

    def __getattr__(self, item: str) -> Any:
        if hasattr(self._owner, item):
            return getattr(self._owner, item)
        return getattr(self._graph, item)


class WBotGraph:
    def __init__(
        self,
        *,
        llm: ChatOpenAI,
        tools: list[Any],
        memory_store: LongTermMemoryStore,
        retrieve_top_k: int,
        user_id: str,
        checkpointer: Any,
        skills_loader: SkillsLoader | None = None,
        openclaw_profile_loader: OpenClawProfileLoader | None = None,
        multimodal_settings: MultimodalSettings | None = None,
        model_name: str = "",
        llm_image: ChatOpenAI | None = None,
        llm_audio: ChatOpenAI | None = None,
        image_model_name: str = "",
        audio_model_name: str = "",
        token_optimization_settings: TokenOptimizationSettings | None = None,
        max_tool_steps_per_turn: int = 8,
        max_same_tool_call_repeats: int = 3,
    ) -> None:
        """初始化对象并保存运行所需依赖。
        
        Args:
            llm: 大语言模型客户端实例。
            tools: 工具对象列表。
            memory_store: 长期记忆存储实例，用于检索与保存记忆。
            retrieve_top_k: 记忆检索返回的最大条数。
            user_id: 业务对象唯一标识。
            checkpointer: LangGraph 检查点对象，用于持久化状态。
            skills_loader: 技能加载器实例，用于读取 always 技能和技能摘要。
            openclaw_profile_loader: OpenClaw 档案加载器，用于注入人格与运行约束上下文。
            multimodal_settings: 多模态处理配置。
            model_name: 当前使用的模型名称。
            llm_image: 图像理解模型客户端实例。
            llm_audio: 音频理解模型客户端实例。
            image_model_name: 图像模型名称。
            audio_model_name: 音频模型名称。
            token_optimization_settings: Token 优化配置。
        """
        logger.info(
            "Initializing WBotGraph: user_id=%s, retrieve_top_k=%s",
            user_id,
            retrieve_top_k,
        )
        self._llm_plain = llm
        self._llm_text_base = llm
        self._llm_image_base = llm_image
        self._llm_audio_base = llm_audio
        self._memory_store = memory_store
        self._retrieve_top_k = retrieve_top_k
        self._user_id = user_id
        self._context_builder = ContextBuilder(
            skills_loader=skills_loader,
            openclaw_profile_loader=openclaw_profile_loader,
            token_optimization_settings=token_optimization_settings,
        )
        self._skills_loader = skills_loader
        self._prepared_system_prompt_base = self._context_builder.build_static_system_prompt(
            base_prompt=_base_system_prompt(),
        )
        self._multimodal_cfg = multimodal_settings
        self._text_model_name = model_name
        self._image_model_name = image_model_name
        self._audio_model_name = audio_model_name
        self._normalizer_by_model: dict[str, MultimodalNormalizer] = {}
        self._token_opt = token_optimization_settings or TokenOptimizationSettings(
            enabled=True,
            max_recent_user_turns=6,
            summary_trigger_messages=12,
            summary_max_chars=1200,
            context_window_tokens=128000,
            auto_compact_buffer_tokens=13000,
            warning_threshold_buffer_tokens=20000,
            error_threshold_buffer_tokens=20000,
            blocking_buffer_tokens=3000,
            enable_dynamic_system_context=True,
            enable_git_status=True,
            git_status_max_chars=2000,
            enable_project_instruction_scan=True,
            project_instruction_files=("CLAUDE.md", "AGENTS.md", "WBOT.md"),
        )
        self._token_budget = TokenBudgetManager(
            context_window_tokens=self._token_opt.context_window_tokens,
            auto_compact_buffer_tokens=self._token_opt.auto_compact_buffer_tokens,
            warning_threshold_buffer_tokens=self._token_opt.warning_threshold_buffer_tokens,
            error_threshold_buffer_tokens=self._token_opt.error_threshold_buffer_tokens,
            blocking_buffer_tokens=self._token_opt.blocking_buffer_tokens,
        )
        self._max_tool_steps_per_turn = max(1, int(max_tool_steps_per_turn))
        self._max_same_tool_call_repeats = max(1, int(max_same_tool_call_repeats))
        self._max_consecutive_tool_failures = 2
        self._tools_by_name = {
            str(getattr(tool, "name", "")).strip(): tool
            for tool in tools
            if str(getattr(tool, "name", "")).strip()
        }
        self._llm_tool_cache: dict[tuple[str, tuple[str, ...]], Any] = {}
        self._deferred_summary_lock = threading.Lock()
        self._deferred_summary_threads: dict[str, threading.Thread] = {}
        self._deferred_summary_rerun: dict[str, bool] = {}
        self._subagent_manager = SubagentManager(
            parent_graph=self,
            tools=list(self._tools_by_name.values()),
            workspace_root=Path.cwd().resolve(),
            skills_loader=skills_loader,
        )

        # Initialize intent classifier
        self._tools_registry = ToolRegistry(tools=list(self._tools_by_name.values()))
        self._intent_classifier: IntentClassifier | None = None
        if token_optimization_settings is not None:
            # Build intent classifier settings from token optimization settings
            # We create a default intent classifier settings here
            # In production, this should be passed as a separate parameter
            intent_cfg = IntentClassifierSettings(
                enabled=True,
                use_llm=True,
                llm_model_name="",
                llm_temperature=0.1,
                llm_timeout_seconds=10.0,
                confidence_threshold_heuristic=0.85,
                confidence_threshold_llm=0.70,
                enable_tool_exposure_control=True,
                max_tools_per_intent=4,
            )
            self._intent_classifier = IntentClassifier(
                llm=self._llm_plain,
                settings=intent_cfg,
                tools_registry=self._tools_registry,
            )

        graph_builder = StateGraph(AgentState)
        graph_builder.add_node("retrieve_memories", self._retrieve_memories)
        graph_builder.add_node("prepare_prompt_context", self._prepare_prompt_context)
        graph_builder.add_node("agent", self._agent)
        graph_builder.add_node("action", self._action)
        graph_builder.add_node("recover", self._recover_after_tool_failure)

        graph_builder.add_edge(START, "retrieve_memories")
        graph_builder.add_edge("retrieve_memories", "prepare_prompt_context")
        graph_builder.add_edge("prepare_prompt_context", "agent")
        graph_builder.add_conditional_edges(
            "agent",
            self._route_after_agent,
            {
                "action": "action",
                "end": END,
            },
        )
        graph_builder.add_conditional_edges(
            "action",
            self._route_after_action,
            {
                "agent": "agent",
                "recover": "recover",
            },
        )
        graph_builder.add_edge("recover", END)

        self._graph = graph_builder.compile(checkpointer=checkpointer)
        self.app = ScheduledGraphApp(self._graph, self)
        logger.info("LangGraph compiled successfully")

    def spawn_subagent(
        self,
        *,
        agent_type: str,
        task: str,
        label: str = "",
        context_messages: list[AnyMessage] | None = None,
        parent_thread_id: str = "-",
        status_callback: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        return self._subagent_manager.spawn(
            agent_type=agent_type,
            task=task,
            label=label,
            context_messages=context_messages,
            parent_thread_id=parent_thread_id,
            status_callback=status_callback,
        )

    def list_subagents(self, *, status: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        return self._subagent_manager.list_jobs(status=status, limit=limit)

    def wait_for_subagent(self, job_id: str, *, timeout_seconds: int = 60) -> dict[str, Any]:
        return self._subagent_manager.wait_for(job_id, timeout_seconds=timeout_seconds)

    async def run_skill_subagent(
        self,
        *,
        skill_name: str,
        task: str,
        arguments: dict[str, Any] | None = None,
        context_messages: list[AnyMessage] | None = None,
        thread_id: str = "-",
        status_callback: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        result = await self._subagent_manager.execute_skill(
            skill_name=skill_name,
            task=task,
            arguments=arguments,
            context_messages=context_messages,
            thread_id=thread_id,
            status_callback=status_callback,
        )
        return {
            "success": result.success,
            "final_response": result.final_response,
            "error": result.error,
            "tool_calls": result.tool_calls,
            "duration_seconds": result.duration_seconds,
        }

    def _retrieve_memories(
        self,
        state: AgentState,
        config: RunnableConfig | None = None,
    ) -> dict[str, str]:
        """检索并返回匹配结果。
        
        Args:
            state: Agent 当前状态字典。
            _: 占位参数，不参与业务逻辑。
        """
        query = _extract_last_user_message(state.get("messages", []))
        if not query:
            logger.debug("No user query found for memory retrieval")
            _emit_status(config, "跳过长期记忆检索（当前回合无用户文本）。")
            return {"long_term_context": ""}

        _emit_status(config, "正在检索长期记忆上下文...")
        logger.debug("Retrieving memories for current user query, query_len=%s", len(query))
        docs = self._memory_store.retrieve(
            user_id=self._user_id,
            query=query,
            k=self._retrieve_top_k,
        )
        if not docs:
            logger.debug("No keyword-matched long-term memories, fallback to recent memories")
            docs = self._memory_store.retrieve_recent(
                user_id=self._user_id,
                k=self._retrieve_top_k,
            )
            if not docs:
                logger.debug("No long-term memories found")
                _emit_status(config, "未命中长期记忆，继续直接回答。")
                return {"long_term_context": ""}

        rendered = self._memory_store.render_context(docs)
        logger.debug("Retrieved %s long-term memory items", len(docs))
        _emit_status(config, f"已加载 {len(docs)} 条长期记忆。")
        return {"long_term_context": rendered}

    def _agent(
        self,
        state: AgentState,
        config: RunnableConfig | None = None,
    ) -> dict[str, Any]:
        """处理agent相关逻辑并返回结果。
        
        Args:
            state: Agent 当前状态字典。
            _: 占位参数，不参与业务逻辑。
        """
        memory_context = state.get("long_term_context") or "无"
        history = state.get("messages", [])
        sanitized_history = sanitize_messages_for_llm(history)
        if len(sanitized_history) != len(history):
            logger.warning(
                "Sanitized message history before LLM invoke: original=%s, sanitized=%s",
                len(history),
                len(sanitized_history),
            )

        optimized = self._prepare_optimized_context(
            state=state,
            history=sanitized_history,
            config=config,
        )
        _emit_status(config, "正在整理对话上下文...")
        prepared_base_prompt = str(
            state.get("prepared_system_prompt_base")
            or self._prepared_system_prompt_base
        )
        prompt_blocks = [
            prepared_base_prompt,
            f"已检索到的长期记忆:\n{memory_context or '无'}",
        ]
        conversation_summary = optimized["conversation_summary"].strip()
        if conversation_summary:
            prompt_blocks.append(f"会话摘要（历史压缩）:\n{conversation_summary}")
        compaction_level = str(optimized.get("context_compaction_level") or "").strip()
        if compaction_level:
            prompt_blocks.append(
                "上下文压缩等级:\n"
                f"- 当前等级: {compaction_level}\n"
                "- 等级越高，越应优先引用摘要、关键决策和最近消息，避免重复展开旧内容。"
            )
        full_system_prompt = "\n\n".join(prompt_blocks)
        messages: list[AnyMessage] = [
            SystemMessage(content=full_system_prompt),
            *normalize_messages_for_llm(
                optimized["recent_messages"],
                normalizer=self._normalizer_for_current_turn(optimized["recent_messages"]),
            ),
        ]
        logger.debug("Invoking LLM with %s messages", len(messages))
        llm, selected_route, selected_tools = self._llm_for_current_turn(
            optimized["recent_messages"],
            messages=messages,
        )
        logger.info(
            "Model routing selected route=%s tools=%s",
            selected_route,
            ",".join(selected_tools) or "-",
        )
        _emit_status(config, f"已选择模型路由：{selected_route}。")
        _emit_status(config, "正在生成回复...")
        token_callback = _resolve_stream_token_callback(config) or _get_runtime_token_callback()
        debug_callback = _resolve_debug_callback(config) or _get_runtime_debug_callback()
        user_goal = _extract_last_user_message(sanitized_history)
        try:
            response = _invoke_llm_with_optional_stream(
                llm=llm,
                messages=messages,
                token_callback=token_callback,
                debug_callback=debug_callback,
            )
        except Exception as exc:
            if _is_messages_length_error(exc):
                logger.warning(
                    "Provider rejected message payload; retry with strict text-only fallback: %s",
                    exc,
                )
                _emit_status(config, "触发兼容回退：改用纯文本上下文重试。")
                fallback_messages = _build_text_only_retry_messages(
                    system_prompt=full_system_prompt,
                    history=messages,
                )
                try:
                    response = _invoke_llm_with_optional_stream(
                        llm=self._bind_tools_for_route("text", selected_tools),
                        messages=fallback_messages,
                        token_callback=token_callback,
                        debug_callback=debug_callback,
                    )
                except Exception as fallback_exc:
                    logger.exception("Text-only compatibility fallback failed")
                    _emit_status(config, "模型调用失败，已返回兜底提示。")
                    response = AIMessage(content=_runtime_error_reply_text(fallback_exc))
            elif selected_route != "text":
                logger.warning(
                    "Route=%s model failed; retry with text model, error=%s",
                    selected_route,
                    exc,
                )
                _emit_status(config, f"{selected_route} 模型调用失败，正在回退到 text 模型重试。")
                try:
                    response = _invoke_llm_with_optional_stream(
                        llm=self._bind_tools_for_route("text", selected_tools),
                        messages=messages,
                        token_callback=token_callback,
                        debug_callback=debug_callback,
                    )
                except Exception as fallback_exc:
                    logger.exception("Route fallback to text model failed")
                    _emit_status(config, "模型调用失败，已返回兜底提示。")
                    response = AIMessage(content=_runtime_error_reply_text(fallback_exc))
            else:
                logger.exception("Text model invoke failed")
                _emit_status(config, "模型调用失败，已返回兜底提示。")
                response = AIMessage(content=_runtime_error_reply_text(exc))
        continuation_attempts = 0
        while (
            not response.tool_calls
            and continuation_attempts < 1
            and self._should_continue_after_non_tool_reply(
                user_goal=user_goal,
                history=sanitized_history,
                response=response,
            )
        ):
            continuation_attempts += 1
            logger.info("Detected incomplete non-tool reply; continuing current turn")
            _emit_status(config, "检测到当前回复仍像中间状态，继续执行当前任务。")
            continuation_messages = [
                *messages,
                response,
                HumanMessage(content=_continue_current_task_prompt()),
            ]
            try:
                response = _invoke_llm_with_optional_stream(
                    llm=llm,
                    messages=continuation_messages,
                    token_callback=token_callback,
                    debug_callback=debug_callback,
                )
            except Exception as continuation_exc:
                logger.exception("Continuation invoke failed")
                _emit_status(config, "继续执行当前任务时发生异常，本轮先输出当前结果。")
                response = AIMessage(content=_runtime_error_reply_text(continuation_exc))
                break
        logger.debug("LLM response received, has_tool_calls=%s", bool(response.tool_calls))
        if response.tool_calls:
            projected_history = [*sanitized_history, response]
            tool_steps = _count_tool_steps_since_last_human(projected_history)
            if tool_steps > self._max_tool_steps_per_turn:
                logger.warning(
                    "Tool loop guard triggered: max_tool_steps_per_turn=%s, tool_steps=%s",
                    self._max_tool_steps_per_turn,
                    tool_steps,
                )
                _emit_status(
                    config,
                    f"工具调用已达上限（{self._max_tool_steps_per_turn}），本轮停止继续调用。",
                )
                response = AIMessage(
                    content=(
                        f"本轮工具调用次数已达上限（{self._max_tool_steps_per_turn}）。"
                        "我已停止自动重试，建议你补充更明确的目标或约束后我再继续。"
                    )
                )
            else:
                signature, repeat_count = _same_tool_call_streak(projected_history)
                if repeat_count >= self._max_same_tool_call_repeats:
                    logger.warning(
                        "Tool loop guard triggered: repeated_call=%s, repeat_count=%s",
                        signature,
                        repeat_count,
                    )
                    _emit_status(
                        config,
                        f"检测到重复工具调用（连续 {repeat_count} 次），本轮停止继续调用。",
                    )
                    response = AIMessage(
                        content=(
                            f"检测到同一工具调用连续重复 {repeat_count} 次。"
                            "我已停止自动重试，建议调整输入条件或改用其它策略后再继续。"
                        )
                    )

        if response.tool_calls:
            _emit_status(config, f"准备执行工具调用：{_summarize_tool_calls(response.tool_calls)}")
        else:
            _emit_status(config, "回复已生成。")
        latest_usage = extract_token_usage(response)
        session_usage = _merge_token_usage_dicts(
            state.get("session_token_usage"),
            latest_usage.to_dict(),
        )
        return {
            "messages": [response],
            "conversation_summary": optimized["conversation_summary"],
            "summarized_message_count": optimized["summarized_message_count"],
            "latest_token_usage": latest_usage.to_dict(),
            "session_token_usage": session_usage,
            "token_budget_state": optimized["token_budget_state"],
            "context_compaction_level": optimized["context_compaction_level"],
        }

    def _prepare_prompt_context(
        self,
        state: AgentState,
        config: RunnableConfig | None = None,
    ) -> dict[str, str]:
        """预先构建当前回合内可复用的系统提示词固定部分。"""
        _emit_status(config, "正在准备回合级提示词上下文...")
        budget_state = state.get("token_budget_state")
        budget_snapshot = _format_token_budget_snapshot(
            budget_state if isinstance(budget_state, dict) else {},
            state.get("session_token_usage"),
        )
        prepared_prompt = self._context_builder.build_turn_system_prompt(
            base_prompt=_base_system_prompt(),
            budget_snapshot=budget_snapshot,
        )
        return {"prepared_system_prompt_base": prepared_prompt}

    def _action(
        self,
        state: AgentState,
        config: RunnableConfig | None = None,
    ) -> dict[str, Any]:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._action_async(state, config))
        return loop.run_until_complete(self._action_async(state, config))

    async def _action_async(
        self,
        state: AgentState,
        config: RunnableConfig | None = None,
    ) -> dict[str, Any]:
        messages = state.get("messages", [])
        if not messages:
            return {
                "messages": [],
                "last_tool_failed": False,
                "consecutive_tool_failures": 0,
                "last_tool_name": "",
                "last_tool_error": "",
            }
        last = messages[-1]
        if not isinstance(last, AIMessage) or not last.tool_calls:
            return {
                "messages": [],
                "last_tool_failed": False,
                "consecutive_tool_failures": 0,
                "last_tool_name": "",
                "last_tool_error": "",
            }

        async def _execute_tool(index: int, tool_call: dict[str, Any]) -> ToolMessage:
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
            preview = _tool_args_preview(name, args)
            _emit_tool_progress(
                config,
                event_type="tool.started",
                tool_name=name,
                preview=preview,
                function_args=args,
            )
            started_at = time.monotonic()
            try:
                tool_context = {
                    "graph": self,
                    "state_messages": list(messages),
                    "config": config,
                    "status_callback": _resolve_status_callback(config),
                    "tool_progress_callback": _resolve_tool_progress_callback(config),
                    "thread_id": _resolve_thread_id(config),
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
            _emit_tool_progress(
                config,
                event_type="tool.completed",
                tool_name=name,
                preview=preview,
                elapsed_seconds=elapsed_seconds,
                ok=not _is_tool_failure_content(content),
                function_args=args,
            )
            return ToolMessage(content=content, tool_call_id=tool_call_id, name=name)

        _emit_status(config, f"工具并发执行中：{_summarize_tool_calls(last.tool_calls)}")
        results = await asyncio.gather(
            *(_execute_tool(index, tool_call) for index, tool_call in enumerate(last.tool_calls)),
            return_exceptions=True,
        )
        tool_messages: list[ToolMessage] = []
        failed_tool_names: list[str] = []
        failed_tool_errors: list[str] = []
        for index, result in enumerate(results):
            if isinstance(result, BaseException):
                logger.exception("Concurrent tool execution failed", exc_info=result)
                failed_tool_names.append(f"tool_call_{index}")
                failed_tool_errors.append(f"{type(result).__name__}: {result}")
                tool_messages.append(
                    ToolMessage(
                        content=f"Tool execution failed: {type(result).__name__}: {result}",
                        tool_call_id=f"tool_call_{index}",
                        name="tool_error",
                    )
                )
                continue
            if _is_tool_failure_content(result.content):
                failed_tool_names.append(str(result.name or f"tool_call_{index}"))
                failed_tool_errors.append(_extract_tool_failure_summary(result.content))
            tool_messages.append(result)
        last_tool_failed = bool(failed_tool_errors)
        previous_failures = int(state.get("consecutive_tool_failures", 0) or 0)
        consecutive_tool_failures = previous_failures + 1 if last_tool_failed else 0
        if last_tool_failed:
            _emit_status(
                config,
                f"工具执行失败（连续 {consecutive_tool_failures} 次）：{', '.join(failed_tool_names)}",
            )
        else:
            _emit_status(config, "工具执行完成，继续整理结果。")
        return {
            "messages": tool_messages,
            "last_tool_failed": last_tool_failed,
            "consecutive_tool_failures": consecutive_tool_failures,
            "last_tool_name": ", ".join(failed_tool_names[:3]),
            "last_tool_error": " | ".join(failed_tool_errors[:3]),
        }

    def _recover_after_tool_failure(
        self,
        state: AgentState,
        config: RunnableConfig | None = None,
    ) -> dict[str, Any]:
        consecutive_failures = int(state.get("consecutive_tool_failures", 0) or 0)
        tool_name = str(state.get("last_tool_name") or "工具").strip()
        detail = str(state.get("last_tool_error") or "").strip()
        _emit_status(config, "工具连续失败，已停止自动重试并生成兜底回复。")
        detail_suffix = f"\n最近一次错误：{detail}" if detail else ""
        return {
            "messages": [
                AIMessage(
                    content=(
                        f"{tool_name} 连续执行失败 {consecutive_failures} 次，我已停止自动重试，避免流程卡住。"
                        "你可以调整输入参数、补充上下文，或让我改用别的策略继续处理。"
                        f"{detail_suffix}"
                    )
                )
            ],
            "last_tool_failed": False,
            "consecutive_tool_failures": 0,
            "last_tool_name": "",
            "last_tool_error": "",
        }

    @staticmethod
    def _route_after_agent(state: AgentState) -> str:
        """处理route/after/agent相关逻辑并返回结果。
        
        Args:
            state: Agent 当前状态字典。
        """
        messages = state.get("messages", [])
        if not messages:
            logger.debug("Route decision: no messages -> end")
            return "end"
        last = messages[-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            logger.debug("Route decision: tool call detected -> action")
            return "action"
        logger.debug("Route decision: no tool call -> end")
        return "end"

    def _route_after_action(self, state: AgentState) -> str:
        consecutive_failures = int(state.get("consecutive_tool_failures", 0) or 0)
        if consecutive_failures >= self._max_consecutive_tool_failures:
            logger.warning(
                "Route decision after action: consecutive tool failures reached limit=%s",
                consecutive_failures,
            )
            return "recover"
        logger.debug("Route decision after action: continue agent")
        return "agent"

    def _should_continue_after_non_tool_reply(
        self,
        *,
        user_goal: str,
        history: list[AnyMessage],
        response: AIMessage,
    ) -> bool:
        if not _should_check_completion_for_turn(user_goal, history):
            return False
        response_text = _to_text_content(response.content).strip()
        if not response_text:
            return True
        if self._skill_creation_looks_done(user_goal=user_goal, history=history):
            return False
        if _response_looks_incomplete(response_text):
            return True
        return not self._judge_reply_complete(user_goal=user_goal, response_text=response_text)

    def _skill_creation_looks_done(self, *, user_goal: str, history: list[AnyMessage]) -> bool:
        normalized_goal = (user_goal or "").strip().lower()
        if not normalized_goal:
            return False
        if "skill" not in normalized_goal and "技能" not in user_goal:
            return False
        if self._skills_loader is None:
            return False
        skills_root = str(self._skills_loader.workspace_skills_dir).lower()
        for message in reversed(history):
            if isinstance(message, HumanMessage):
                break
            if not isinstance(message, ToolMessage):
                continue
            content = _to_text_content(getattr(message, "content", "")).strip()
            lowered = content.lower()
            if "error" in lowered:
                continue
            if "skill.md" not in lowered or skills_root not in lowered:
                continue
            if "successfully wrote" in lowered or "successfully edited" in lowered:
                return True
        return False

    def _judge_reply_complete(self, *, user_goal: str, response_text: str) -> bool:
        judge_prompt = (
            "你是一个严格的任务完成度裁判。"
            "你只判断 assistant 的最新回复是否已经真正完成了用户请求。"
            "如果回复仍停留在计划、说明、承诺稍后执行、阶段性汇报、要求用户做本可由 assistant 自己完成的动作，"
            "或者明显还缺少实现、检查、结果，则返回 CONTINUE。"
            "只有在用户请求已经被实际完成，或者 assistant 明确说明了无法继续且给出了真实阻塞原因时，才返回 COMPLETE。"
            "只能输出一个单词：COMPLETE 或 CONTINUE。"
        )
        try:
            decision = self._llm_plain.invoke(
                [
                    SystemMessage(content=judge_prompt),
                    HumanMessage(
                        content=(
                            f"用户请求：\n{user_goal.strip() or '(empty)'}\n\n"
                            f"assistant 最新回复：\n{response_text}\n\n"
                            "请判断是否已完成。"
                        )
                    ),
                ]
            )
        except Exception:
            logger.debug("Completion judge failed", exc_info=True)
            return False
        decision_text = _to_text_content(getattr(decision, "content", "")).strip().upper()
        return decision_text.startswith("COMPLETE")

    def _llm_for_current_turn(
        self,
        history: list[AnyMessage],
        *,
        messages: list[AnyMessage],
    ) -> tuple[Any, str, tuple[str, ...]]:
        """处理llm/for/current/turn相关逻辑并返回结果。
        
        Args:
            history: 历史消息列表。
            messages: 消息列表，通常按时间顺序排列。
        """
        route = _route_for_history(history)
        has_native_image = _has_native_image_blocks(messages)
        tool_names = self._select_tool_names_for_current_turn(history, messages=messages)
        if route == "image" and has_native_image and self._llm_image_base is not None:
            return self._llm_for_route_with_tools("image", tool_names), "image", tool_names
        if route == "audio" and self._llm_audio_base is not None:
            return self._llm_for_route_with_tools("audio", tool_names), "audio", tool_names
        return self._llm_for_route_with_tools("text", tool_names), "text", tool_names

    def _llm_for_route_with_tools(self, route: str, tool_names: tuple[str, ...]) -> Any:
        if not tool_names:
            if route == "image" and self._llm_image_base is not None:
                return self._llm_image_base
            if route == "audio" and self._llm_audio_base is not None:
                return self._llm_audio_base
            return self._llm_plain
        return self._bind_tools_for_route(route, tool_names)

    def _bind_tools_for_route(self, route: str, tool_names: tuple[str, ...]) -> Any:
        cache_key = (route, tool_names)
        cached = self._llm_tool_cache.get(cache_key)
        if cached is not None:
            return cached

        base_llm = self._llm_text_base
        if route == "image" and self._llm_image_base is not None:
            base_llm = self._llm_image_base
        elif route == "audio" and self._llm_audio_base is not None:
            base_llm = self._llm_audio_base

        selected_tools = [self._tools_by_name[name] for name in tool_names if name in self._tools_by_name]
        if not selected_tools:
            return base_llm

        bindable_tools = [
            tool.to_schema() if hasattr(tool, "to_schema") and callable(tool.to_schema) else tool
            for tool in selected_tools
        ]
        bound = base_llm.bind_tools(bindable_tools)
        self._llm_tool_cache[cache_key] = bound
        return bound

    def _select_tool_names_for_current_turn(
        self,
        history: list[AnyMessage],
        *,
        messages: list[AnyMessage],
    ) -> tuple[str, ...]:
        latest_user_text = _extract_last_user_message(messages or history)

        # Use new IntentClassifier if available
        if self._intent_classifier is not None:
            result = self._intent_classifier.classify_sync(latest_user_text, history)
            return self._intent_classifier.select_tools_for_intent(result)

        # Fallback: backward compatible logic
        tool_names = set(self._tools_by_name)
        if "run_skill" in tool_names:
            if not _should_expose_run_skill(latest_user_text.lower()):
                tool_names.discard("run_skill")
        return tuple(sorted(tool_names))

    def _normalizer_for_current_turn(self, history: list[AnyMessage]) -> MultimodalNormalizer | None:
        """将输入标准化为统一结构。
        
        Args:
            history: 历史消息列表。
        """
        if self._multimodal_cfg is None:
            return None

        route = _route_for_history(history)
        model_name = self._text_model_name
        if route == "image" and self._image_model_name:
            model_name = self._image_model_name
        if route == "audio" and self._audio_model_name:
            model_name = self._audio_model_name

        if model_name in self._normalizer_by_model:
            return self._normalizer_by_model[model_name]

        capabilities = resolve_provider_capabilities(model_name=model_name)
        normalizer = MultimodalNormalizer(
            cfg=MultimodalRuntimeConfig(
                enabled=self._multimodal_cfg.enabled,
                max_file_bytes=self._multimodal_cfg.max_file_bytes,
                max_total_bytes_per_turn=self._multimodal_cfg.max_total_bytes_per_turn,
                max_files_per_turn=self._multimodal_cfg.max_files_per_turn,
                audio_mode=self._multimodal_cfg.audio_mode,
                video_keyframe_interval_sec=self._multimodal_cfg.video_keyframe_interval_sec,
                video_max_frames=self._multimodal_cfg.video_max_frames,
                document_max_chars=self._multimodal_cfg.document_max_chars,
            ),
            capabilities=capabilities,
        )
        self._normalizer_by_model[model_name] = normalizer
        return normalizer

    def _prepare_optimized_context(
        self,
        *,
        state: AgentState,
        history: list[AnyMessage],
        config: RunnableConfig | None = None,
    ) -> dict[str, Any]:
        """处理prepare/optimized/context相关逻辑并返回结果。
        
        Args:
            state: Agent 当前状态字典。
            history: 历史消息列表。
        """
        if not self._token_opt.enabled:
            return {
                "conversation_summary": state.get("conversation_summary", ""),
                "summarized_message_count": int(state.get("summarized_message_count", 0) or 0),
                "recent_messages": history,
                "token_budget_state": {},
                "context_compaction_level": "off",
            }

        summary = str(state.get("conversation_summary", "") or "")
        summarized_count = max(0, int(state.get("summarized_message_count", 0) or 0))
        estimated_tokens = token_count_with_estimation(history)
        budget_state = self._token_budget.calculate_state(estimated_tokens)
        compaction_level = _determine_compaction_level(budget_state.to_dict())
        recent_turns = self._token_opt.max_recent_user_turns
        if budget_state.is_above_error_threshold:
            recent_turns = max(2, min(recent_turns, 4))
        if budget_state.is_at_blocking_limit:
            recent_turns = 1

        recent_start = _recent_window_start(history, max_user_turns=recent_turns)
        target_end = min(recent_start, len(history))
        unsummarized_count = max(0, target_end - summarized_count)
        should_force_summary = budget_state.is_above_auto_compact_threshold or budget_state.is_at_blocking_limit

        if unsummarized_count >= self._token_opt.summary_trigger_messages or (
            should_force_summary and target_end > summarized_count
        ):
            if _should_defer_summary_update(config):
                return {
                    "conversation_summary": summary,
                    "summarized_message_count": summarized_count,
                    "recent_messages": _apply_context_compaction_strategy(
                        history[summarized_count:] or history,
                        compaction_level=compaction_level,
                    ),
                    "token_budget_state": budget_state.to_dict(),
                    "context_compaction_level": compaction_level,
                }
            delta = history[summarized_count:target_end]
            transcript = _messages_to_summary_text(delta)
            if transcript:
                summary = self._update_summary(
                    existing_summary=summary,
                    transcript=transcript,
                    max_chars=self._token_opt.summary_max_chars,
                )
                summarized_count = target_end
                logger.debug(
                    "Updated rolling summary: summarized_count=%s, summary_chars=%s",
                    summarized_count,
                    len(summary),
                )

        recent_messages = _apply_context_compaction_strategy(
            history[recent_start:],
            compaction_level=compaction_level,
        )
        return {
            "conversation_summary": summary,
            "summarized_message_count": summarized_count,
            "recent_messages": recent_messages,
            "token_budget_state": budget_state.to_dict(),
            "context_compaction_level": compaction_level,
        }

    def _update_summary(
        self,
        *,
        existing_summary: str,
        transcript: str,
        max_chars: int,
    ) -> str:
        """更新内部状态或中间结果。
        
        Args:
            existing_summary: 已有对话摘要文本。
            transcript: 待压缩的对话转写文本。
            max_chars: 返回文本片段允许的最大字符数。
        """
        prompt = (
            "请维护一段会话滚动摘要，用于后续对话上下文压缩。"
            "请输出中文结构化纯文本，并严格使用以下小节："
            "【目标与约束】【关键决策】【已完成】【待处理】【风险与阻塞】。"
            "只保留后续继续任务最需要的信息，删除闲聊、重复描述和冗长工具输出。"
            f"总长度不超过 {max_chars} 字。"
        )
        payload = (
            f"已有摘要：\n{existing_summary or '无'}\n\n"
            f"新增对话片段：\n{transcript}\n\n"
            "请输出更新后的摘要："
        )
        try:
            result = self._llm_plain.invoke(
                [
                    SystemMessage(content=prompt),
                    HumanMessage(content=payload),
                ]
            )
            text = _to_text_content(result.content).strip()
        except Exception:
            logger.exception("Failed to update rolling summary with LLM")
            text = _build_summary_fallback(existing_summary=existing_summary, transcript=transcript)

        if not text:
            text = _build_summary_fallback(existing_summary=existing_summary, transcript=transcript)
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rstrip()

    def schedule_deferred_summary(self, config: RunnableConfig | None) -> None:
        if not _should_defer_summary_update(config):
            return
        thread_id = _resolve_thread_id(config)
        if thread_id == "-":
            return
        with self._deferred_summary_lock:
            worker = self._deferred_summary_threads.get(thread_id)
            if worker is not None and worker.is_alive():
                self._deferred_summary_rerun[thread_id] = True
                return
            self._deferred_summary_rerun[thread_id] = False
            worker = threading.Thread(
                target=self._run_deferred_summary_worker,
                args=(thread_id,),
                name=f"summary-{thread_id}",
                daemon=True,
            )
            self._deferred_summary_threads[thread_id] = worker
            worker.start()

    def _run_deferred_summary_worker(self, thread_id: str) -> None:
        while True:
            try:
                self._refresh_summary_for_thread(thread_id)
            except Exception:
                logger.exception("Deferred summary refresh failed: thread_id=%s", thread_id)

            with self._deferred_summary_lock:
                should_rerun = self._deferred_summary_rerun.get(thread_id, False)
                if should_rerun:
                    self._deferred_summary_rerun[thread_id] = False
                    continue
                self._deferred_summary_threads.pop(thread_id, None)
                self._deferred_summary_rerun.pop(thread_id, None)
                break

    def _refresh_summary_for_thread(self, thread_id: str) -> None:
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        snapshot = self._graph.get_state(config)
        values = getattr(snapshot, "values", None) or {}
        messages = values.get("messages", [])
        if not isinstance(messages, list) or not messages:
            return

        state: AgentState = {
            "messages": messages,
            "long_term_context": str(values.get("long_term_context") or ""),
            "conversation_summary": str(values.get("conversation_summary") or ""),
            "summarized_message_count": int(values.get("summarized_message_count") or 0),
            "prepared_system_prompt_base": str(
                values.get("prepared_system_prompt_base") or self._prepared_system_prompt_base
            ),
            "latest_token_usage": values.get("latest_token_usage") or {},
            "session_token_usage": values.get("session_token_usage") or {},
            "token_budget_state": values.get("token_budget_state") or {},
            "context_compaction_level": str(values.get("context_compaction_level") or ""),
        }
        optimized = self._prepare_optimized_context(state=state, history=sanitize_messages_for_llm(messages), config=None)
        if (
            optimized["conversation_summary"] == state["conversation_summary"]
            and optimized["summarized_message_count"] == state["summarized_message_count"]
        ):
            return

        update_state = getattr(self._graph, "update_state", None)
        if not callable(update_state):
            logger.warning("Compiled graph does not support update_state; skip deferred summary refresh")
            return
        update_state(
            config,
            {
                "conversation_summary": optimized["conversation_summary"],
                "summarized_message_count": optimized["summarized_message_count"],
            },
        )
        logger.info(
            "Deferred summary refreshed: thread_id=%s summarized_message_count=%s",
            thread_id,
            optimized["summarized_message_count"],
        )
