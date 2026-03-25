from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any

from langchain_core.documents import Document
from langchain_milvus import Milvus
from .logging_config import get_logger

try:
    # Common path in langchain-community
    from langchain_community.embeddings import FastEmbedEmbeddings
except ImportError:  # pragma: no cover - compatibility fallback
    # Some versions expose class from the module path
    from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

logger = get_logger(__name__)


class LongTermMemoryStore:
    def __init__(
        self,
        milvus_uri: str,
        collection_name: str,
        consistency_level: str = "Session",
    ) -> None:
        logger.info(
            "Initializing Milvus memory store: uri=%s, collection=%s, consistency=%s",
            milvus_uri,
            collection_name,
            consistency_level,
        )
        self._milvus_uri = milvus_uri
        self._collection_name = collection_name
        self._consistency_level = consistency_level
        self._embedding = FastEmbedEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
        self._local = threading.local()
        self._disabled = False
        self._init_error_logged = False

    def retrieve(self, user_id: str, query: str, k: int = 4) -> list[Document]:
        vectorstore = self._get_vectorstore()
        if vectorstore is None:
            return []
        logger.debug("Retrieving long-term memories: user_id=%s, k=%s", user_id, k)
        try:
            docs = vectorstore.similarity_search(
                query=query,
                k=k,
                expr=f'user_id == "{user_id}"',
            )
            logger.debug("Retrieved %s memories from Milvus", len(docs))
            return docs
        except Exception:
            logger.exception("Failed retrieving memories from Milvus, returning empty result")
            self._reset_local_vectorstore()
            return []

    def save(self, user_id: str, text: str, memory_type: str = "experience") -> str:
        vectorstore = self._get_vectorstore()
        if vectorstore is None:
            return ""
        logger.info(
            "Saving long-term memory: user_id=%s, type=%s, text_len=%s",
            user_id,
            memory_type,
            len(text),
        )
        timestamp = datetime.now(tz=timezone.utc).isoformat()
        metadata: dict[str, Any] = {
            "user_id": user_id,
            "type": memory_type,
            "timestamp": timestamp,
        }
        try:
            ids = vectorstore.add_documents(
                documents=[Document(page_content=text, metadata=metadata)]
            )
            logger.debug("Saved long-term memory ids=%s", ids)
            return ids[0] if ids else ""
        except Exception:
            logger.exception("Failed saving memory to Milvus")
            self._reset_local_vectorstore()
            return ""

    def _get_vectorstore(self) -> Milvus | None:
        if self._disabled:
            return None

        vectorstore = getattr(self._local, "vectorstore", None)
        if vectorstore is not None:
            return vectorstore

        try:
            vectorstore = Milvus(
                embedding_function=self._embedding,
                collection_name=self._collection_name,
                connection_args={"uri": self._milvus_uri},
                consistency_level=self._consistency_level,
                auto_id=True,
            )
            self._local.vectorstore = vectorstore
            return vectorstore
        except Exception as exc:
            if not self._init_error_logged:
                logger.warning(
                    "Milvus unavailable, long-term memory disabled for this run: %s",
                    exc,
                )
                self._init_error_logged = True
            else:
                logger.debug("Milvus still unavailable", exc_info=True)
            self._disabled = True
            return None

    def _reset_local_vectorstore(self) -> None:
        if hasattr(self._local, "vectorstore"):
            del self._local.vectorstore
