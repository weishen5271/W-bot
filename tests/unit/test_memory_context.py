from __future__ import annotations

from langchain_core.documents import Document

from w_bot.agents.core.memory_context import MemoryContextRetriever


class FakeMemoryStore:
    def __init__(self, docs: list[Document], recent_docs: list[Document] | None = None) -> None:
        self.docs = docs
        self.recent_docs = recent_docs or []
        self.retrieve_calls = 0
        self.recent_calls = 0

    def retrieve(self, *, user_id: str, query: str, k: int) -> list[Document]:
        del user_id, query, k
        self.retrieve_calls += 1
        return self.docs

    def retrieve_recent(self, *, user_id: str, k: int) -> list[Document]:
        del user_id, k
        self.recent_calls += 1
        return self.recent_docs

    def render_context(self, docs: list[Document]) -> str:
        return "\n".join(doc.page_content for doc in docs)


def test_memory_context_retriever_skips_empty_query() -> None:
    store = FakeMemoryStore([])
    retriever = MemoryContextRetriever(memory_store=store, user_id="u1", retrieve_top_k=4)

    result = retriever.retrieve_context("")

    assert result.skipped is True
    assert store.retrieve_calls == 0


def test_memory_context_retriever_uses_recent_fallback() -> None:
    store = FakeMemoryStore([], [Document(page_content="recent memory")])
    retriever = MemoryContextRetriever(memory_store=store, user_id="u1", retrieve_top_k=4)

    result = retriever.retrieve_context("query")

    assert result.context == "recent memory"
    assert result.hit_count == 1
    assert result.used_recent_fallback is True
    assert store.retrieve_calls == 1
    assert store.recent_calls == 1

