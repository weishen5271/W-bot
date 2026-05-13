from __future__ import annotations

from typing import Any

from .context_optimizer import ContextOptimizer
from .logging_config import get_logger
from .session_db import SessionStore
from .session_models import RuntimeConfig

logger = get_logger(__name__)


class PostTurnProcessor:
    """Run business post-processing after an AgentRuntime turn."""

    def __init__(
        self,
        *,
        session_store: SessionStore,
        session_search_db: Any = None,
        context_optimizer: ContextOptimizer | None = None,
        source: str = "unknown",
        user_id: str = "",
        model: str = "",
    ) -> None:
        self._session_store = session_store
        self._session_search_db = session_search_db
        self._context_optimizer = context_optimizer
        self._source = source.strip() or "unknown"
        self._user_id = user_id
        self._model = model

    def after_turn(self, *, session_id: str, config: RuntimeConfig | None = None) -> None:
        if config is None or config.defer_summary_update:
            self.refresh_summary(session_id=session_id)
        self.flush_session_search_index(session_id=session_id)

    def refresh_summary(self, *, session_id: str) -> None:
        if self._context_optimizer is None:
            return
        if not session_id or session_id == "-":
            return
        try:
            history = self._session_store.get_messages(session_id)
            if not history:
                return
            conversation_summary, summarized_message_count = self._session_store.get_summary(session_id)
            optimized = self._context_optimizer.prepare(
                history=history,
                conversation_summary=conversation_summary,
                summarized_message_count=summarized_message_count,
                defer_summary_update=False,
            )
            if (
                optimized.conversation_summary == conversation_summary
                and optimized.summarized_message_count == summarized_message_count
            ):
                return
            self._session_store.update_summary(
                session_id,
                summary=optimized.conversation_summary,
                summarized_message_count=optimized.summarized_message_count,
            )
        except Exception:
            logger.warning("Failed to refresh AgentRuntime rolling summary: session_id=%s", session_id, exc_info=True)

    def flush_session_search_index(self, *, session_id: str) -> None:
        if self._session_search_db is None:
            return
        if not session_id or session_id == "-":
            return
        try:
            messages = self._session_store.get_messages(session_id)
            if not messages:
                return
            self._session_search_db.sync_langchain_messages(
                session_id=session_id,
                messages=messages,
                source=self._source,
                user_id=self._user_id,
                model=self._model,
            )
        except Exception:
            logger.warning("Failed to flush AgentRuntime session search index: session_id=%s", session_id, exc_info=True)
