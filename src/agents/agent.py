from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from .config import MultimodalSettings
from .logging_config import get_logger
from .context import ContextBuilder
from .memory import LongTermMemoryStore
from .multimodal import MultimodalNormalizer, MultimodalRuntimeConfig, parse_human_payload
from .providers import resolve_provider_capabilities
from .skills import SkillsLoader

logger = get_logger(__name__)


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    long_term_context: str


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
        multimodal_settings: MultimodalSettings | None = None,
        model_name: str = "",
        llm_image: ChatOpenAI | None = None,
        llm_audio: ChatOpenAI | None = None,
        image_model_name: str = "",
        audio_model_name: str = "",
    ) -> None:
        logger.info(
            "Initializing WBotGraph: user_id=%s, retrieve_top_k=%s",
            user_id,
            retrieve_top_k,
        )
        self._llm_text = llm.bind_tools(tools)
        self._llm_image = llm_image.bind_tools(tools) if llm_image is not None else None
        self._llm_audio = llm_audio.bind_tools(tools) if llm_audio is not None else None
        self._memory_store = memory_store
        self._retrieve_top_k = retrieve_top_k
        self._user_id = user_id
        self._context_builder = ContextBuilder(skills_loader=skills_loader)
        self._multimodal_cfg = multimodal_settings
        self._text_model_name = model_name
        self._image_model_name = image_model_name
        self._audio_model_name = audio_model_name
        self._normalizer_by_model: dict[str, MultimodalNormalizer] = {}

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
        _: RunnableConfig | None = None,
    ) -> dict[str, str]:
        query = _extract_last_user_message(state.get("messages", []))
        if not query:
            logger.debug("No user query found for memory retrieval")
            return {"long_term_context": ""}

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
                return {"long_term_context": ""}

        lines = [f"- {doc.page_content}" for doc in docs]
        logger.debug("Retrieved %s long-term memory items", len(lines))
        return {"long_term_context": "\n".join(lines)}

    def _agent(
        self,
        state: AgentState,
        _: RunnableConfig | None = None,
    ) -> dict[str, list[AIMessage]]:
        system_prompt = (
            "你是 W-bot CLI Agent。"
            "你需要优先给出清晰、可执行的答案。"
            "当任务需要精确计算、脚本验证或数据处理时，使用 execute_python 工具。"
            "当用户偏好、长期事实或关键经验值得保留时，调用 save_memory。"
            "工具调用参数必须严格匹配 schema。"
        )
        memory_context = state.get("long_term_context") or "无"
        full_system_prompt = self._context_builder.build_system_prompt(
            base_prompt=system_prompt,
            memory_context=memory_context,
        )

        history = state.get("messages", [])
        sanitized_history = sanitize_messages_for_llm(history)
        if len(sanitized_history) != len(history):
            logger.warning(
                "Sanitized message history before LLM invoke: original=%s, sanitized=%s",
                len(history),
                len(sanitized_history),
            )

        messages: list[AnyMessage] = [
            SystemMessage(content=full_system_prompt),
            *normalize_messages_for_llm(
                sanitized_history,
                normalizer=self._normalizer_for_current_turn(sanitized_history),
            ),
        ]
        logger.debug("Invoking LLM with %s messages", len(messages))
        llm, selected_route = self._llm_for_current_turn(sanitized_history, messages=messages)
        logger.info("Model routing selected route=%s", selected_route)
        try:
            response = llm.invoke(messages)
        except Exception as exc:
            if not _is_messages_length_error(exc):
                raise
            logger.warning(
                "Provider rejected message payload; retry with strict text-only fallback: %s",
                exc,
            )
            fallback_messages = _build_text_only_retry_messages(
                system_prompt=full_system_prompt,
                history=messages,
            )
            response = self._llm_text.invoke(fallback_messages)
        logger.debug("LLM response received, has_tool_calls=%s", bool(response.tool_calls))
        return {"messages": [response]}

    @staticmethod
    def _route_after_agent(state: AgentState) -> str:
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
        route = _route_for_history(history)
        has_native_image = _has_native_image_blocks(messages)
        if route == "image" and has_native_image and self._llm_image is not None:
            return self._llm_image, "image"
        if route == "audio" and self._llm_audio is not None:
            return self._llm_audio, "audio"
        return self._llm_text, "text"

    def _normalizer_for_current_turn(self, history: list[AnyMessage]) -> MultimodalNormalizer | None:
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


def _extract_last_user_message(messages: list[AnyMessage]) -> str:
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
    """Drop malformed tool-call segments so providers don't reject the request."""

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
    text = str(exc).lower()
    return "messages parameter length invalid" in text


def _build_text_only_retry_messages(
    *,
    system_prompt: str,
    history: list[AnyMessage],
) -> list[AnyMessage]:
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


def _to_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return _human_blocks_to_text(content)
    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            return text
        return str(content)
    return str(content)


def _has_native_image_blocks(messages: list[AnyMessage]) -> bool:
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
