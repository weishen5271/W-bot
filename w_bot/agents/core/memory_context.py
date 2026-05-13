from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class MemoryRetrievalResult:
    context: str = ""
    hit_count: int = 0
    used_recent_fallback: bool = False
    skipped: bool = False


class MemoryContextRetriever:
    """Retrieve long-term memory context for the current turn."""

    def __init__(
        self,
        *,
        memory_store: Any,
        user_id: str,
        retrieve_top_k: int,
    ) -> None:
        self._memory_store = memory_store
        self._user_id = user_id
        self._retrieve_top_k = max(1, int(retrieve_top_k))

    def retrieve_context(self, query: str) -> MemoryRetrievalResult:
        normalized_query = str(query or "").strip()
        if not normalized_query:
            logger.debug("Skip long-term memory retrieval: empty query")
            return MemoryRetrievalResult(skipped=True)

        logger.debug(
            "Retrieving long-term memory context: user_id=%s query_len=%s top_k=%s",
            self._user_id,
            len(normalized_query),
            self._retrieve_top_k,
        )
        docs = self._memory_store.retrieve(
            user_id=self._user_id,
            query=normalized_query,
            k=self._retrieve_top_k,
        )
        used_recent_fallback = False
        if not docs:
            docs = self._memory_store.retrieve_recent(
                user_id=self._user_id,
                k=self._retrieve_top_k,
            )
            used_recent_fallback = bool(docs)
        if not docs:
            return MemoryRetrievalResult()

        rendered = str(self._memory_store.render_context(docs) or "").strip()
        return MemoryRetrievalResult(
            context=rendered,
            hit_count=len(docs),
            used_recent_fallback=used_recent_fallback,
        )

