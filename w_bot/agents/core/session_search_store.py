from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage

from .logging_config import get_logger
from .message_utils import _to_text_content

logger = get_logger(__name__)


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS session_search_sessions (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    user_id TEXT,
    model TEXT,
    parent_session_id TEXT,
    started_at REAL NOT NULL,
    ended_at REAL,
    message_count INTEGER DEFAULT 0,
    tool_call_count INTEGER DEFAULT 0,
    title TEXT
);

CREATE TABLE IF NOT EXISTS session_search_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES session_search_sessions(id),
    ordinal INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT,
    tool_call_id TEXT,
    tool_calls TEXT,
    tool_name TEXT,
    timestamp REAL NOT NULL,
    UNIQUE(session_id, ordinal)
);

CREATE INDEX IF NOT EXISTS idx_session_search_sessions_source
    ON session_search_sessions(source);
CREATE INDEX IF NOT EXISTS idx_session_search_sessions_parent
    ON session_search_sessions(parent_session_id);
CREATE INDEX IF NOT EXISTS idx_session_search_sessions_started
    ON session_search_sessions(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_session_search_messages_session
    ON session_search_messages(session_id, ordinal);
"""

FTS_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS session_search_messages_fts USING fts5(
    content,
    content=session_search_messages,
    content_rowid=id
);

CREATE TRIGGER IF NOT EXISTS session_search_messages_fts_insert
AFTER INSERT ON session_search_messages BEGIN
    INSERT INTO session_search_messages_fts(rowid, content)
    VALUES (new.id, new.content);
END;

CREATE TRIGGER IF NOT EXISTS session_search_messages_fts_delete
AFTER DELETE ON session_search_messages BEGIN
    INSERT INTO session_search_messages_fts(session_search_messages_fts, rowid, content)
    VALUES('delete', old.id, old.content);
END;

CREATE TRIGGER IF NOT EXISTS session_search_messages_fts_update
AFTER UPDATE ON session_search_messages BEGIN
    INSERT INTO session_search_messages_fts(session_search_messages_fts, rowid, content)
    VALUES('delete', old.id, old.content);
    INSERT INTO session_search_messages_fts(rowid, content)
    VALUES (new.id, new.content);
END;
"""


class SessionSearchDB:
    """SQLite-backed searchable transcript index for W-bot sessions."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path).expanduser()
        if not self.db_path.is_absolute():
            self.db_path = (Path.cwd() / self.db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.setup()

    def setup(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA_SQL)
            conn.executescript(FTS_SQL)

    def ensure_session(
        self,
        session_id: str,
        *,
        source: str = "unknown",
        user_id: str = "",
        model: str = "",
        parent_session_id: str | None = None,
        title: str = "",
    ) -> None:
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO session_search_sessions (
                    id, source, user_id, model, parent_session_id, started_at, title
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    source = COALESCE(NULLIF(excluded.source, ''), session_search_sessions.source),
                    user_id = COALESCE(NULLIF(excluded.user_id, ''), session_search_sessions.user_id),
                    model = COALESCE(NULLIF(excluded.model, ''), session_search_sessions.model),
                    parent_session_id = COALESCE(excluded.parent_session_id, session_search_sessions.parent_session_id),
                    title = COALESCE(NULLIF(excluded.title, ''), session_search_sessions.title)
                """,
                (session_id, source, user_id or None, model or None, parent_session_id, now, title or None),
            )

    def upsert_message(
        self,
        *,
        session_id: str,
        ordinal: int,
        role: str,
        content: str,
        tool_call_id: str | None = None,
        tool_calls: Any = None,
        tool_name: str | None = None,
    ) -> None:
        tool_calls_json = _json_dumps_or_none(tool_calls)
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO session_search_messages (
                    session_id, ordinal, role, content, tool_call_id, tool_calls, tool_name, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id, ordinal) DO UPDATE SET
                    role = excluded.role,
                    content = excluded.content,
                    tool_call_id = excluded.tool_call_id,
                    tool_calls = excluded.tool_calls,
                    tool_name = excluded.tool_name
                """,
                (
                    session_id,
                    max(0, int(ordinal)),
                    role,
                    content,
                    tool_call_id,
                    tool_calls_json,
                    tool_name,
                    time.time(),
                ),
            )
            self._refresh_session_counts(conn, session_id)

    def sync_langchain_messages(
        self,
        *,
        session_id: str,
        messages: list[AnyMessage],
        source: str,
        user_id: str,
        model: str,
        title: str = "",
    ) -> None:
        if not session_id or session_id == "-":
            return
        self.ensure_session(
            session_id,
            source=source,
            user_id=user_id,
            model=model,
            title=title or _first_user_title(messages),
        )
        for ordinal, message in enumerate(messages):
            record = langchain_message_to_record(message)
            if record is None:
                continue
            self.upsert_message(session_id=session_id, ordinal=ordinal, **record)

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM session_search_sessions WHERE id = ? LIMIT 1",
                (session_id,),
            ).fetchone()
        return dict(row) if row is not None else None

    def list_sessions_rich(
        self,
        *,
        source: str | None = None,
        exclude_sources: list[str] | None = None,
        limit: int = 20,
        offset: int = 0,
        include_children: bool = False,
    ) -> list[dict[str, Any]]:
        where: list[str] = []
        params: list[Any] = []
        if not include_children:
            where.append("s.parent_session_id IS NULL")
        if source:
            where.append("s.source = ?")
            params.append(source)
        if exclude_sources:
            where.append(f"s.source NOT IN ({','.join('?' for _ in exclude_sources)})")
            params.extend(exclude_sources)
        where_sql = f"WHERE {' AND '.join(where)}" if where else ""
        params.extend([max(1, int(limit)), max(0, int(offset))])
        sql = f"""
            SELECT s.*,
                COALESCE((
                    SELECT SUBSTR(REPLACE(REPLACE(m.content, X'0A', ' '), X'0D', ' '), 1, 63)
                    FROM session_search_messages m
                    WHERE m.session_id = s.id AND m.role = 'user' AND m.content IS NOT NULL
                    ORDER BY m.ordinal LIMIT 1
                ), '') AS _preview_raw,
                COALESCE((
                    SELECT MAX(m2.timestamp)
                    FROM session_search_messages m2
                    WHERE m2.session_id = s.id
                ), s.started_at) AS last_active
            FROM session_search_sessions s
            {where_sql}
            ORDER BY last_active DESC
            LIMIT ? OFFSET ?
        """
        with self._lock, self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        sessions: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            raw = str(item.pop("_preview_raw", "") or "").strip()
            item["preview"] = raw[:60] + ("..." if len(raw) > 60 else "")
            sessions.append(item)
        return sessions

    def get_messages_as_conversation(self, session_id: str) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT role, content, tool_call_id, tool_calls, tool_name
                FROM session_search_messages
                WHERE session_id = ?
                ORDER BY ordinal
                """,
                (session_id,),
            ).fetchall()
        messages: list[dict[str, Any]] = []
        for row in rows:
            msg = {"role": row["role"], "content": row["content"] or ""}
            if row["tool_call_id"]:
                msg["tool_call_id"] = row["tool_call_id"]
            if row["tool_name"]:
                msg["tool_name"] = row["tool_name"]
            if row["tool_calls"]:
                try:
                    msg["tool_calls"] = json.loads(row["tool_calls"])
                except (TypeError, json.JSONDecodeError):
                    pass
            messages.append(msg)
        return messages

    def search_messages(
        self,
        *,
        query: str,
        source_filter: list[str] | None = None,
        exclude_sources: list[str] | None = None,
        role_filter: list[str] | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        sanitized = self._sanitize_fts5_query(query)
        if not sanitized:
            return []

        where = ["session_search_messages_fts MATCH ?"]
        params: list[Any] = [sanitized]
        if source_filter:
            where.append(f"s.source IN ({','.join('?' for _ in source_filter)})")
            params.extend(source_filter)
        if exclude_sources:
            where.append(f"s.source NOT IN ({','.join('?' for _ in exclude_sources)})")
            params.extend(exclude_sources)
        if role_filter:
            where.append(f"m.role IN ({','.join('?' for _ in role_filter)})")
            params.extend(role_filter)
        params.extend([max(1, int(limit)), max(0, int(offset))])

        sql = f"""
            SELECT
                m.id,
                m.session_id,
                m.role,
                snippet(session_search_messages_fts, 0, '>>>', '<<<', '...', 40) AS snippet,
                m.timestamp,
                m.tool_name,
                s.source,
                s.model,
                s.started_at AS session_started
            FROM session_search_messages_fts
            JOIN session_search_messages m ON m.id = session_search_messages_fts.rowid
            JOIN session_search_sessions s ON s.id = m.session_id
            WHERE {' AND '.join(where)}
            ORDER BY rank
            LIMIT ? OFFSET ?
        """
        with self._lock, self._connect() as conn:
            try:
                rows = conn.execute(sql, params).fetchall()
            except sqlite3.OperationalError:
                logger.debug("Invalid FTS5 session search query: %s", query, exc_info=True)
                return []
            matches = [dict(row) for row in rows]

            for match in matches:
                ctx_rows = conn.execute(
                    """
                    SELECT role, content
                    FROM session_search_messages
                    WHERE session_id = ?
                      AND ordinal >= (
                          SELECT ordinal FROM session_search_messages WHERE id = ?
                      ) - 1
                      AND ordinal <= (
                          SELECT ordinal FROM session_search_messages WHERE id = ?
                      ) + 1
                    ORDER BY ordinal
                    """,
                    (match["session_id"], match["id"], match["id"]),
                ).fetchall()
                match["context"] = [
                    {"role": row["role"], "content": (row["content"] or "")[:200]}
                    for row in ctx_rows
                ]
        return matches

    @staticmethod
    def _sanitize_fts5_query(query: str) -> str:
        text = (query or "").strip()
        if not text:
            return ""
        quoted_parts: list[str] = []

        def _preserve(match: re.Match[str]) -> str:
            quoted_parts.append(match.group(0))
            return f"\x00Q{len(quoted_parts) - 1}\x00"

        text = re.sub(r'"[^"]*"', _preserve, text)
        text = re.sub(r'[+{}()"^]', " ", text)
        text = re.sub(r"\*+", "*", text)
        text = re.sub(r"(^|\s)\*", r"\1", text)
        text = re.sub(r"(?i)^(AND|OR|NOT)\b\s*", "", text.strip())
        text = re.sub(r"(?i)\s+(AND|OR|NOT)\s*$", "", text.strip())
        text = re.sub(r"\b(\w+(?:[.-]\w+)+)\b", r'"\1"', text)
        for index, quoted in enumerate(quoted_parts):
            text = text.replace(f"\x00Q{index}\x00", quoted)
        return text.strip()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0, isolation_level=None, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    @staticmethod
    def _refresh_session_counts(conn: sqlite3.Connection, session_id: str) -> None:
        row = conn.execute(
            """
            SELECT COUNT(*) AS message_count,
                   COALESCE(SUM(CASE WHEN role = 'tool' OR tool_calls IS NOT NULL THEN 1 ELSE 0 END), 0) AS tool_count
            FROM session_search_messages
            WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()
        conn.execute(
            """
            UPDATE session_search_sessions
            SET message_count = ?, tool_call_count = ?
            WHERE id = ?
            """,
            (int(row["message_count"] or 0), int(row["tool_count"] or 0), session_id),
        )


def resolve_session_search_db_path(short_term_memory_path: str) -> str:
    target = Path(short_term_memory_path).expanduser()
    if not target.is_absolute():
        target = (Path.cwd() / target).resolve()
    if target.suffix:
        return str(target.with_name(f"{target.stem}_session_search.sqlite"))
    return str(target / "session_search.sqlite")


def langchain_message_to_record(message: AnyMessage) -> dict[str, Any] | None:
    if isinstance(message, SystemMessage):
        role = "system"
    elif isinstance(message, HumanMessage):
        role = "user"
    elif isinstance(message, ToolMessage):
        role = "tool"
    elif isinstance(message, AIMessage):
        role = "assistant"
    else:
        return None

    text = _to_text_content(getattr(message, "content", "")).strip()
    tool_calls = getattr(message, "tool_calls", None) if isinstance(message, AIMessage) else None
    return {
        "role": role,
        "content": text,
        "tool_call_id": getattr(message, "tool_call_id", None),
        "tool_calls": tool_calls,
        "tool_name": getattr(message, "name", None) if isinstance(message, ToolMessage) else None,
    }


def _first_user_title(messages: list[AnyMessage]) -> str:
    for message in messages:
        if not isinstance(message, HumanMessage):
            continue
        text = _to_text_content(message.content).strip()
        if text:
            return text[:80]
    return ""


def _json_dumps_or_none(value: Any) -> str | None:
    if not value:
        return None
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return json.dumps(str(value), ensure_ascii=False)
