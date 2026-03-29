from __future__ import annotations

import json
from typing import Annotated, Any, Callable, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from .config import MultimodalSettings, TokenOptimizationSettings
from .context import ContextBuilder
from .logging_config import get_logger
from .memory import LongTermMemoryStore
from .multimodal import MultimodalNormalizer, MultimodalRuntimeConfig, parse_human_payload
from .openclaw_profile import OpenClawProfileLoader
from .providers import resolve_provider_capabilities
from .skills import SkillsLoader

logger = get_logger(__name__)


_RUNTIME_TOKEN_CALLBACK: Callable[[str], None] | None = None
_RUNTIME_DEBUG_CALLBACK: Callable[[str], None] | None = None


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
        self._llm_text = llm.bind_tools(tools)
        self._llm_image = llm_image.bind_tools(tools) if llm_image is not None else None
        self._llm_audio = llm_audio.bind_tools(tools) if llm_audio is not None else None
        self._memory_store = memory_store
        self._retrieve_top_k = retrieve_top_k
        self._user_id = user_id
        self._context_builder = ContextBuilder(
            skills_loader=skills_loader,
            openclaw_profile_loader=openclaw_profile_loader,
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
        )
        self._max_tool_steps_per_turn = max(1, int(max_tool_steps_per_turn))
        self._max_same_tool_call_repeats = max(1, int(max_same_tool_call_repeats))

        graph_builder = StateGraph(AgentState)
        graph_builder.add_node("retrieve_memories", self._retrieve_memories)
        graph_builder.add_node("agent", self._agent)
        graph_builder.add_node("action", ToolNode(tools))

        graph_builder.add_edge(START, "retrieve_memories")
        graph_builder.add_edge("retrieve_memories", "agent")
        graph_builder.add_conditional_edges(
            "agent",
            self._route_after_agent,
            {
                "action": "action",
                "end": END,
            },
        )
        graph_builder.add_edge("action", "agent")

        self.app = graph_builder.compile(checkpointer=checkpointer)
        logger.info("LangGraph compiled successfully")

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

        lines = [f"- {doc.page_content}" for doc in docs]
        logger.debug("Retrieved %s long-term memory items", len(lines))
        _emit_status(config, f"已加载 {len(lines)} 条长期记忆。")
        return {"long_term_context": "\n".join(lines)}

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
        system_prompt = (
            "你是 W-bot CLI Agent。"
            "优先给出清晰、可执行的答案。"
            "用户点名 skill 时优先使用；否则按意图匹配最小必要 skill，命中后先读 SKILL.md 再执行。"
            "未使用 skill 时简要说明原因。"
            "需要精确计算、脚本验证或数据处理时使用 execute_python。"
            "当用户偏好、长期事实或关键经验值得保留时调用 save_memory。"
            "工具调用参数必须严格匹配 schema。"
        )
        memory_context = state.get("long_term_context") or "无"
        history = state.get("messages", [])
        sanitized_history = sanitize_messages_for_llm(history)
        if len(sanitized_history) != len(history):
            logger.warning(
                "Sanitized message history before LLM invoke: original=%s, sanitized=%s",
                len(history),
                len(sanitized_history),
            )

        optimized = self._prepare_optimized_context(state=state, history=sanitized_history)
        _emit_status(config, "正在整理对话上下文...")
        full_system_prompt = self._context_builder.build_system_prompt(
            base_prompt=system_prompt,
            memory_context=memory_context,
            conversation_summary=optimized["conversation_summary"],
        )
        messages: list[AnyMessage] = [
            SystemMessage(content=full_system_prompt),
            *normalize_messages_for_llm(
                optimized["recent_messages"],
                normalizer=self._normalizer_for_current_turn(optimized["recent_messages"]),
            ),
        ]
        logger.debug("Invoking LLM with %s messages", len(messages))
        llm, selected_route = self._llm_for_current_turn(optimized["recent_messages"], messages=messages)
        logger.info("Model routing selected route=%s", selected_route)
        _emit_status(config, f"已选择模型路由：{selected_route}。")
        _emit_status(config, "正在生成回复...")
        token_callback = _resolve_stream_token_callback(config) or _get_runtime_token_callback()
        debug_callback = _resolve_debug_callback(config) or _get_runtime_debug_callback()
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
                        llm=self._llm_text,
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
                        llm=self._llm_text,
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
        return {
            "messages": [response],
            "conversation_summary": optimized["conversation_summary"],
            "summarized_message_count": optimized["summarized_message_count"],
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

    def _llm_for_current_turn(
        self,
        history: list[AnyMessage],
        *,
        messages: list[AnyMessage],
    ) -> tuple[Any, str]:
        """处理llm/for/current/turn相关逻辑并返回结果。
        
        Args:
            history: 历史消息列表。
            messages: 消息列表，通常按时间顺序排列。
        """
        route = _route_for_history(history)
        has_native_image = _has_native_image_blocks(messages)
        if route == "image" and has_native_image and self._llm_image is not None:
            return self._llm_image, "image"
        if route == "audio" and self._llm_audio is not None:
            return self._llm_audio, "audio"
        return self._llm_text, "text"

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
            }

        summary = str(state.get("conversation_summary", "") or "")
        summarized_count = max(0, int(state.get("summarized_message_count", 0) or 0))
        recent_start = _recent_window_start(history, max_user_turns=self._token_opt.max_recent_user_turns)
        target_end = min(recent_start, len(history))
        unsummarized_count = max(0, target_end - summarized_count)

        if unsummarized_count >= self._token_opt.summary_trigger_messages:
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

        recent_messages = history[recent_start:]
        return {
            "conversation_summary": summary,
            "summarized_message_count": summarized_count,
            "recent_messages": recent_messages,
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
            "要求：保留目标、约束、已完成、待办、关键偏好和重要结论；"
            "删除闲聊与重复；输出中文纯文本。"
            f"长度不超过 {max_chars} 字。"
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
            text = existing_summary

        if not text:
            return existing_summary
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rstrip()


def _extract_last_user_message(messages: list[AnyMessage]) -> str:
    """从输入中提取所需信息。
    
    Args:
        messages: 消息列表，通常按时间顺序排列。
    """
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            content, _, is_blocks = parse_human_payload(
                message.content,
                additional_kwargs=getattr(message, "additional_kwargs", None),
            )
            if is_blocks:
                return _human_blocks_to_text(message.content)
            return content
    return ""


def message_kind(message: AnyMessage) -> str:
    """处理message/kind相关逻辑并返回结果。
    
    Args:
        message: 单条消息对象。
    """
    if isinstance(message, HumanMessage):
        return "human"
    if isinstance(message, ToolMessage):
        return "tool"
    if isinstance(message, AIMessage) and message.tool_calls:
        return "action"
    if isinstance(message, AIMessage):
        return "thought"
    return "other"


def sanitize_messages_for_llm(messages: list[AnyMessage]) -> list[AnyMessage]:
    """处理sanitize/messages/for/llm相关逻辑并返回结果。
    
    Args:
        messages: 消息列表，通常按时间顺序排列。
    """

    sanitized: list[AnyMessage] = []
    idx = 0

    while idx < len(messages):
        message = messages[idx]

        if isinstance(message, AIMessage) and message.tool_calls:
            expected_ids = _extract_tool_call_ids(message)
            tool_block: list[ToolMessage] = []
            matched_ids: set[str] = set()

            lookahead = idx + 1
            while lookahead < len(messages):
                candidate = messages[lookahead]
                if not isinstance(candidate, ToolMessage):
                    break
                tool_block.append(candidate)
                tool_call_id = getattr(candidate, "tool_call_id", None)
                if isinstance(tool_call_id, str):
                    matched_ids.add(tool_call_id)
                lookahead += 1

            if expected_ids and expected_ids.issubset(matched_ids):
                sanitized.append(message)
                sanitized.extend(tool_block)
            idx = lookahead
            continue

        if isinstance(message, ToolMessage):
            idx += 1
            continue

        sanitized.append(message)
        idx += 1

    return sanitized


def _extract_tool_call_ids(message: AIMessage) -> set[str]:
    """从输入中提取所需信息。
    
    Args:
        message: 单条消息对象。
    """
    ids: set[str] = set()
    for tool_call in message.tool_calls:
        if isinstance(tool_call, dict):
            call_id = tool_call.get("id")
            if isinstance(call_id, str):
                ids.add(call_id)
    return ids


def normalize_messages_for_llm(
    messages: list[AnyMessage],
    *,
    normalizer: MultimodalNormalizer | None,
) -> list[AnyMessage]:
    """将输入标准化为统一结构。
    
    Args:
        messages: 消息列表，通常按时间顺序排列。
        normalizer: 多模态归一化器实例。
    """
    if normalizer is None:
        return messages

    last_human_idx = -1
    for idx, msg in enumerate(messages):
        if isinstance(msg, HumanMessage):
            last_human_idx = idx

    if last_human_idx < 0:
        return messages

    out: list[AnyMessage] = []
    for idx, msg in enumerate(messages):
        if not isinstance(msg, HumanMessage):
            out.append(msg)
            continue

        text, media, is_blocks = parse_human_payload(
            msg.content,
            additional_kwargs=getattr(msg, "additional_kwargs", None),
        )
        if is_blocks:
            out.append(HumanMessage(content=_human_blocks_to_text(msg.content)))
            continue

        normalized = normalizer.normalize(
            text=text,
            media=media,
            compact_media=idx != last_human_idx,
        )
        if not normalized.blocks:
            out.append(HumanMessage(content=text))
            continue

        all_text_blocks = bool(normalized.blocks) and all(
            isinstance(block, dict) and block.get("type") == "text"
            for block in normalized.blocks
        )
        if all_text_blocks:
            merged_text = "\n".join(
                str(block.get("text") or "").strip()
                for block in normalized.blocks
                if isinstance(block, dict)
            ).strip()
            out.append(HumanMessage(content=merged_text))
            continue

        out.append(HumanMessage(content=normalized.blocks))
    return out


def _human_blocks_to_text(content: Any) -> str:
    """处理human/blocks/to/text相关逻辑并返回结果。
    
    Args:
        content: 消息内容主体。
    """
    if not isinstance(content, list):
        return str(content)
    lines: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str) and text.strip():
                lines.append(text.strip())
    return "\n".join(lines)


def _route_for_history(messages: list[AnyMessage]) -> str:
    """处理route/for/history相关逻辑并返回结果。
    
    Args:
        messages: 消息列表，通常按时间顺序排列。
    """
    for message in reversed(messages):
        if not isinstance(message, HumanMessage):
            continue
        _, media, _ = parse_human_payload(
            message.content,
            additional_kwargs=getattr(message, "additional_kwargs", None),
        )
        kinds = {item.kind for item in media}
        if "image" in kinds:
            return "image"
        if "audio" in kinds:
            return "audio"
        return "text"
    return "text"


def _is_messages_length_error(exc: Exception) -> bool:
    """判断条件是否满足。
    
    Args:
        exc: 捕获到的异常对象。
    """
    text = str(exc).lower()
    return "messages parameter length invalid" in text


def _resolve_stream_token_callback(config: RunnableConfig | None) -> Callable[[str], None] | None:
    if config is None:
        return None
    if not hasattr(config, "get"):
        return None
    configurable = config.get("configurable")  # type: ignore[call-arg]
    if not isinstance(configurable, dict):
        return None
    callback = configurable.get("token_callback")
    if callable(callback):
        return callback
    return None


def _resolve_status_callback(config: RunnableConfig | None) -> Callable[[str], None] | None:
    if config is None:
        return None
    if not hasattr(config, "get"):
        return None
    configurable = config.get("configurable")  # type: ignore[call-arg]
    if not isinstance(configurable, dict):
        return None
    callback = configurable.get("status_callback")
    if callable(callback):
        return callback
    return None


def _resolve_debug_callback(config: RunnableConfig | None) -> Callable[[str], None] | None:
    if config is None:
        return None
    if not hasattr(config, "get"):
        return None
    configurable = config.get("configurable")  # type: ignore[call-arg]
    if not isinstance(configurable, dict):
        return None
    callback = configurable.get("debug_callback")
    if callable(callback):
        return callback
    return None


def _emit_status(config: RunnableConfig | None, text: str) -> None:
    normalized = text.strip()
    if not normalized:
        return
    logger.info("Step status: thread_id=%s status=%s", _resolve_thread_id(config), normalized)
    callback = _resolve_status_callback(config)
    if callback is None:
        return
    try:
        callback(normalized)
    except Exception:
        logger.debug("Status callback failed", exc_info=True)


def _resolve_thread_id(config: RunnableConfig | None) -> str:
    if config is None:
        return "-"
    if not hasattr(config, "get"):
        return "-"
    configurable = config.get("configurable")  # type: ignore[call-arg]
    if not isinstance(configurable, dict):
        return "-"
    thread_id = configurable.get("thread_id")
    if isinstance(thread_id, str) and thread_id.strip():
        return thread_id.strip()
    return "-"


def _invoke_llm_with_optional_stream(
    *,
    llm: Any,
    messages: list[AnyMessage],
    token_callback: Callable[[str], None] | None,
    debug_callback: Callable[[str], None] | None = None,
) -> AIMessage:
    if token_callback is None:
        return llm.invoke(messages)

    if debug_callback is not None:
        debug_callback("completion_start")
    merged: AIMessage | None = None
    chunk_count = 0
    chunk_text_count = 0
    merged_emitted_text = ""
    emitted_any = False
    for chunk in llm.stream(messages):
        chunk_count += 1
        merged = chunk if merged is None else (merged + chunk)
        text = _extract_stream_chunk_text(chunk)
        if not text:
            # Nanobot-style behavior: recover incremental text from merged
            # stream state when provider-specific chunk payload has no direct
            # text field.
            merged_text = _to_stream_text_content(getattr(merged, "content", ""))
            if merged_text and merged_text.startswith(merged_emitted_text):
                text = merged_text[len(merged_emitted_text):]
                merged_emitted_text = merged_text
        if text:
            token_callback(text)
            chunk_text_count += 1
            if debug_callback is not None and not emitted_any:
                debug_callback("first_token_emitted")
            emitted_any = True
        merged_text = _to_stream_text_content(getattr(merged, "content", ""))
        if merged_text and merged_text.startswith(merged_emitted_text):
            merged_emitted_text = merged_text

    if merged is None:
        return AIMessage(content="")
    if not emitted_any:
        # Non-streaming provider behavior: emit the final answer right after
        # this completion call returns, instead of waiting for the full turn.
        final_text = _to_stream_text_content(getattr(merged, "content", "")).strip()
        if final_text:
            token_callback(final_text)
            if debug_callback is not None:
                debug_callback("fallback_emit_final_text")
    if debug_callback is not None:
        debug_callback(
            f"completion_end chunks={chunk_count} chunks_with_text={chunk_text_count} emitted_any={emitted_any}"
        )
    logger.info(
        "LLM stream stats: chunks=%s, chunks_with_text=%s, emitted_any=%s",
        chunk_count,
        chunk_text_count,
        emitted_any,
    )
    return merged


def _extract_stream_chunk_text(chunk: Any) -> str:
    # Preferred path: LangChain message chunks usually implement text().
    try:
        text_method = getattr(chunk, "text", None)
        if callable(text_method):
            value = text_method()
            if isinstance(value, str) and value:
                return value
    except Exception:
        logger.debug("chunk.text() extraction failed", exc_info=True)

    content = getattr(chunk, "content", "")
    text = _to_stream_text_content(content)
    if text:
        return text

    additional = getattr(chunk, "additional_kwargs", None)
    if additional is not None:
        text = _to_stream_text_content(additional)
        if text:
            return text
    return ""


def _to_stream_text_content(content: Any) -> str:
    """Extract only text-like stream payload fields, avoiding metadata noise."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        lines: list[str] = []
        for item in content:
            extracted = _to_stream_text_content(item)
            if extracted:
                lines.append(extracted)
        return "\n".join(lines)
    if isinstance(content, dict):
        block_type = str(content.get("type") or "").lower()
        if block_type in {"text", "text_delta", "output_text"}:
            block_text = content.get("text")
            if isinstance(block_text, str):
                return block_text
        text = content.get("text")
        if isinstance(text, str):
            return text
        delta = content.get("delta")
        if isinstance(delta, str):
            return delta
        if isinstance(delta, (dict, list)):
            delta_text = _to_stream_text_content(delta)
            if delta_text:
                return delta_text
        output_text = content.get("output_text")
        if isinstance(output_text, str):
            return output_text
        choices = content.get("choices")
        if isinstance(choices, list):
            lines: list[str] = []
            for choice in choices:
                extracted = _to_stream_text_content(choice)
                if extracted:
                    lines.append(extracted)
            if lines:
                return "\n".join(lines)
        message = content.get("message")
        if isinstance(message, (dict, list, str)):
            message_text = _to_stream_text_content(message)
            if message_text:
                return message_text
        nested = content.get("content")
        if nested is not None and nested is not content:
            nested_text = _to_stream_text_content(nested)
            if nested_text:
                return nested_text
        return ""
    return ""


def _summarize_tool_calls(tool_calls: list[dict[str, Any]]) -> str:
    names: list[str] = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        name = str(tool_call.get("name") or "").strip()
        if name:
            names.append(name)
    if not names:
        return f"{len(tool_calls)} 个工具"
    return ", ".join(names)


def _count_tool_steps_since_last_human(messages: list[AnyMessage]) -> int:
    count = 0
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            break
        if isinstance(message, AIMessage) and message.tool_calls:
            count += 1
    return count


def _same_tool_call_streak(messages: list[AnyMessage]) -> tuple[str, int]:
    signatures: list[str] = []
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            break
        if isinstance(message, ToolMessage):
            continue
        if isinstance(message, AIMessage) and message.tool_calls:
            signature = _tool_call_signature(message.tool_calls)
            if signature:
                signatures.append(signature)
            continue
        break

    if not signatures:
        return "", 0

    latest = signatures[0]
    repeat_count = 0
    for signature in signatures:
        if signature != latest:
            break
        repeat_count += 1
    return latest, repeat_count


def _tool_call_signature(tool_calls: list[dict[str, Any]]) -> str:
    normalized: list[str] = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        name = str(tool_call.get("name") or "").strip()
        args = tool_call.get("args")
        if args is None:
            args = tool_call.get("arguments")
        try:
            args_text = json.dumps(args, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            args_text = str(args)
        normalized.append(f"{name}:{args_text}")
    return "|".join(normalized)


def _build_text_only_retry_messages(
    *,
    system_prompt: str,
    history: list[AnyMessage],
) -> list[AnyMessage]:
    """构建并返回目标对象。
    
    Args:
        system_prompt: 系统提示词文本。
        history: 历史消息列表。
    """
    user_text = ""
    for message in reversed(history):
        if not isinstance(message, HumanMessage):
            continue
        user_text = _to_text_content(message.content).strip()
        if user_text:
            break

    if not user_text:
        user_text = "用户发送了一条消息，请给出可执行回复。"

    return [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_text),
    ]


def _runtime_error_reply_text(exc: BaseException | None = None) -> str:
    detail = _format_exception_brief(exc)
    suffix = f"\n异常详情：{detail}" if detail else ""
    return (
        "这次处理出现了临时异常，但服务没有中断。"
        "你可以继续对话，或调整一下问题后再试。"
        f"{suffix}"
    )


def _format_exception_brief(exc: BaseException | None) -> str:
    if exc is None:
        return ""
    message = str(exc).strip()
    if message:
        return f"{type(exc).__name__}: {message}"
    return type(exc).__name__


def _to_text_content(content: Any) -> str:
    """将输入转换为目标格式。
    
    Args:
        content: 消息内容主体。
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        lines: list[str] = []
        for item in content:
            extracted = _to_text_content(item)
            if extracted:
                lines.append(extracted)
        return "\n".join(lines)
    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            return text
        delta = content.get("delta")
        if isinstance(delta, str):
            return delta
        output_text = content.get("output_text")
        if isinstance(output_text, str):
            return output_text
        reasoning = content.get("reasoning_content")
        if isinstance(reasoning, str):
            return reasoning
        completion = content.get("completion")
        if isinstance(completion, str):
            return completion
        message = content.get("message")
        if isinstance(message, str):
            return message
        arguments = content.get("arguments")
        if isinstance(arguments, str):
            return arguments
        nested = content.get("content")
        if nested is not None and nested is not content:
            nested_text = _to_text_content(nested)
            if nested_text:
                return nested_text
        # Fallback: recursively scan common text-carrying fields.
        lines: list[str] = []
        for key, value in content.items():
            if key in {"text", "delta", "output_text", "content"}:
                continue
            extracted = _to_text_content(value)
            if extracted:
                lines.append(extracted)
        if lines:
            return "\n".join(lines)
        return ""
    return str(content)


def _has_native_image_blocks(messages: list[AnyMessage]) -> bool:
    """判断是否包含目标内容。
    
    Args:
        messages: 消息列表，通常按时间顺序排列。
    """
    for message in messages:
        if not isinstance(message, HumanMessage):
            continue
        content = message.content
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = str(block.get("type") or "").lower()
            if block_type in {"image_url", "image"}:
                return True
    return False


def _recent_window_start(messages: list[AnyMessage], *, max_user_turns: int) -> int:
    """处理recent/window/start相关逻辑并返回结果。
    
    Args:
        messages: 消息列表，通常按时间顺序排列。
        max_user_turns: 数值限制参数，用于控制处理规模。
    """
    if max_user_turns <= 0:
        return 0
    seen = 0
    for idx in range(len(messages) - 1, -1, -1):
        if isinstance(messages[idx], HumanMessage):
            seen += 1
            if seen == max_user_turns:
                return idx
    return 0


def _messages_to_summary_text(messages: list[AnyMessage]) -> str:
    """处理messages/to/summary/text相关逻辑并返回结果。
    
    Args:
        messages: 消息列表，通常按时间顺序排列。
    """
    lines: list[str] = []
    for message in messages:
        if isinstance(message, HumanMessage):
            role = "用户"
        elif isinstance(message, ToolMessage):
            role = "工具"
        elif isinstance(message, AIMessage) and message.tool_calls:
            role = "助手(工具调用)"
        elif isinstance(message, AIMessage):
            role = "助手"
        else:
            role = "消息"

        text = _to_text_content(message.content).strip()
        if not text:
            continue
        lines.append(f"{role}: {text}")
    return "\n".join(lines)
