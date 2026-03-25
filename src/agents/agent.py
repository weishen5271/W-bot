from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from .logging_config import get_logger
from .memory import LongTermMemoryStore

logger = get_logger(__name__)


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    long_term_context: str


class CyberCoreGraph:
    def __init__(
        self,
        *,
        llm: ChatOpenAI,
        tools: list[Any],
        memory_store: LongTermMemoryStore,
        retrieve_top_k: int,
        user_id: str,
        checkpointer: Any,
    ) -> None:
        logger.info(
            "Initializing CyberCoreGraph: user_id=%s, retrieve_top_k=%s",
            user_id,
            retrieve_top_k,
        )
        self._llm = llm.bind_tools(tools)
        self._memory_store = memory_store
        self._retrieve_top_k = retrieve_top_k
        self._user_id = user_id

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
            logger.debug("No relevant long-term memories found")
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
            "你是 CyberCore CLI Agent。"
            "你需要优先给出清晰、可执行的答案。"
            "当任务需要精确计算、脚本验证或数据处理时，使用 execute_python 工具。"
            "当用户偏好、长期事实或关键经验值得保留时，调用 save_memory。"
            "工具调用参数必须严格匹配 schema。"
        )
        memory_context = state.get("long_term_context") or "无"
        memory_block = f"已检索到的长期记忆:\n{memory_context}"

        history = state.get("messages", [])
        sanitized_history = sanitize_messages_for_llm(history)
        if len(sanitized_history) != len(history):
            logger.warning(
                "Sanitized message history before LLM invoke: original=%s, sanitized=%s",
                len(history),
                len(sanitized_history),
            )

        messages: list[AnyMessage] = [
            SystemMessage(content=system_prompt),
            SystemMessage(content=memory_block),
            *sanitized_history,
        ]
        logger.debug("Invoking LLM with %s messages", len(messages))
        response = self._llm.invoke(messages)
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


def _extract_last_user_message(messages: list[AnyMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            content = message.content
            if isinstance(content, str):
                return content
            return str(content)
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
