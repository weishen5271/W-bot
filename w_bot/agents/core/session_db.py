from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage

from .logging_config import get_logger
from .token_tracker import extract_token_usage

logger = get_logger(__name__)


SCHEMA_VERSION = 1


class SessionStore:
    """Business-level SQLite session store for explicit AgentRuntime turns."""

    def __init__(self, file_path: str) -> None:
        target = Path(file_path).expanduser()
        if not target.is_absolute():
            target = (Path.cwd() / target).resolve()
        self._file_path = target
        self._lock = threading.RLock()
        self.setup()

    @property
    def file_path(self) -> Path:
        return self._file_path

    def setup(self) -> None:
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            self._initialize_schema(conn)

    def ensure_session(
        self,
        *,
        session_id: str,
        source: str = "unknown",
        user_id: str = "",
        model: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        normalized = _require_session_id(session_id)
        now = _utc_now()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions (
                    id, source, user_id, model, created_at, updated_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    source = excluded.source,
                    user_id = COALESCE(NULLIF(excluded.user_id, ''), sessions.user_id),
                    model = COALESCE(NULLIF(excluded.model, ''), sessions.model),
                    updated_at = excluded.updated_at,
                    metadata_json = COALESCE(excluded.metadata_json, sessions.metadata_json)
                """,
                (
                    normalized,
                    source.strip() or "unknown",
                    user_id.strip(),
                    model.strip(),
                    now,
                    now,
                    _json_dumps(metadata) if metadata else None,
                ),
            )

    def append_message(self, session_id: str, message: AnyMessage) -> None:
        self.append_messages(session_id, [message])

    def append_messages(self, session_id: str, messages: list[AnyMessage]) -> None:
        normalized = _require_session_id(session_id)
        if not messages:
            return
        now = _utc_now()
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                self._ensure_session_unlocked(conn, session_id=normalized, now=now)
                tool_call_count = 0
                token_usage_total: dict[str, int] = {}
                for message in messages:
                    row = _message_to_row(message, created_at=now)
                    if row["tool_calls_json"]:
                        try:
                            calls = json.loads(row["tool_calls_json"])
                            if isinstance(calls, list):
                                tool_call_count += len(calls)
                        except Exception:
                            tool_call_count += 1
                    usage = extract_token_usage(row["token_usage_json"] and json.loads(row["token_usage_json"]))
                    token_usage_total = _merge_usage(token_usage_total, usage.to_dict())
                    conn.execute(
                        """
                        INSERT INTO messages (
                            session_id, role, content, tool_call_id, tool_calls_json,
                            tool_name, additional_kwargs_json, response_metadata_json,
                            token_usage_json, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            normalized,
                            row["role"],
                            row["content"],
                            row["tool_call_id"],
                            row["tool_calls_json"],
                            row["tool_name"],
                            row["additional_kwargs_json"],
                            row["response_metadata_json"],
                            row["token_usage_json"],
                            row["created_at"],
                        ),
                    )
                conn.execute(
                    """
                    UPDATE sessions
                    SET updated_at = ?,
                        message_count = message_count + ?,
                        tool_call_count = tool_call_count + ?,
                        input_tokens = input_tokens + ?,
                        output_tokens = output_tokens + ?
                    WHERE id = ?
                    """,
                    (
                        now,
                        len(messages),
                        tool_call_count,
                        int(token_usage_total.get("input_tokens", 0) or 0),
                        int(token_usage_total.get("output_tokens", 0) or 0),
                        normalized,
                    ),
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def get_messages(self, session_id: str, *, limit: int | None = None) -> list[AnyMessage]:
        normalized = _require_session_id(session_id)
        params: list[Any] = [normalized]
        limit_clause = ""
        if limit is not None:
            limit_clause = "LIMIT ?"
            params.append(max(0, int(limit)))
        query = f"""
            SELECT role, content, tool_call_id, tool_calls_json, tool_name,
                   additional_kwargs_json, response_metadata_json, token_usage_json
            FROM messages
            WHERE session_id = ?
            ORDER BY id ASC
            {limit_clause}
        """
        with self._lock, self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [_row_to_message(row) for row in rows]

    def get_summary(self, session_id: str) -> tuple[str, int]:
        normalized = _require_session_id(session_id)
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT summary, summarized_message_count FROM sessions WHERE id = ?",
                (normalized,),
            ).fetchone()
        if row is None:
            return "", 0
        return str(row["summary"] or ""), int(row["summarized_message_count"] or 0)

    def update_summary(
        self,
        session_id: str,
        *,
        summary: str,
        summarized_message_count: int,
    ) -> None:
        normalized = _require_session_id(session_id)
        now = _utc_now()
        with self._lock, self._connect() as conn:
            self._ensure_session_unlocked(conn, session_id=normalized, now=now)
            conn.execute(
                """
                UPDATE sessions
                SET summary = ?, summarized_message_count = ?, updated_at = ?
                WHERE id = ?
                """,
                (summary, max(0, int(summarized_message_count)), now, normalized),
            )

    def get_token_usage(self, session_id: str) -> dict[str, int]:
        normalized = _require_session_id(session_id)
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT input_tokens, output_tokens
                FROM sessions
                WHERE id = ?
                """,
                (normalized,),
            ).fetchone()
        if row is None:
            return {}
        input_tokens = int(row["input_tokens"] or 0)
        output_tokens = int(row["output_tokens"] or 0)
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._file_path, timeout=30.0, isolation_level=None, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.OperationalError as exc:
            logger.warning("SessionStore WAL unavailable for %s: %s; fallback to DELETE", self._file_path, exc)
            conn.execute("PRAGMA journal_mode=DELETE")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _initialize_schema(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS session_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                user_id TEXT,
                model TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                ended_at TEXT,
                message_count INTEGER NOT NULL DEFAULT 0,
                tool_call_count INTEGER NOT NULL DEFAULT 0,
                input_tokens INTEGER NOT NULL DEFAULT 0,
                output_tokens INTEGER NOT NULL DEFAULT 0,
                summary TEXT,
                summarized_message_count INTEGER NOT NULL DEFAULT 0,
                metadata_json TEXT
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                tool_call_id TEXT,
                tool_calls_json TEXT,
                tool_name TEXT,
                additional_kwargs_json TEXT,
                response_metadata_json TEXT,
                token_usage_json TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            CREATE INDEX IF NOT EXISTS idx_messages_session_id_id
                ON messages(session_id, id);
            """
        )
        conn.execute(
            "INSERT OR REPLACE INTO session_meta (key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )

    @staticmethod
    def _ensure_session_unlocked(conn: sqlite3.Connection, *, session_id: str, now: str) -> None:
        conn.execute(
            """
            INSERT OR IGNORE INTO sessions (id, source, created_at, updated_at)
            VALUES (?, 'unknown', ?, ?)
            """,
            (session_id, now, now),
        )


def resolve_session_store_path(configured_path: str) -> str:
    target = Path(configured_path).expanduser()
    if not target.is_absolute():
        target = (Path.cwd() / target).resolve()
    if target.suffix.lower() in {".pkl", ".pickle"}:
        target = target.with_name(f"{target.stem}_sessions.sqlite")
    elif target.suffix.lower() != ".sqlite":
        target = target.with_suffix(".sqlite")
    return str(target)


def _message_to_row(message: AnyMessage, *, created_at: str) -> dict[str, str | None]:
    additional_kwargs = dict(getattr(message, "additional_kwargs", {}) or {})
    response_metadata = dict(getattr(message, "response_metadata", {}) or {})
    usage = extract_token_usage(message).to_dict()
    tool_calls = getattr(message, "tool_calls", None)
    return {
        "role": _message_role(message),
        "content": _json_dumps(getattr(message, "content", "")),
        "tool_call_id": str(getattr(message, "tool_call_id", "") or "") or None,
        "tool_calls_json": _json_dumps(tool_calls) if tool_calls else None,
        "tool_name": str(getattr(message, "name", "") or "") or None,
        "additional_kwargs_json": _json_dumps(additional_kwargs) if additional_kwargs else None,
        "response_metadata_json": _json_dumps(response_metadata) if response_metadata else None,
        "token_usage_json": _json_dumps(usage) if any(int(v or 0) for v in usage.values()) else None,
        "created_at": created_at,
    }


def _row_to_message(row: sqlite3.Row) -> AnyMessage:
    role = str(row["role"] or "").strip().lower()
    content = _json_loads(row["content"], default="")
    additional_kwargs = _json_loads(row["additional_kwargs_json"], default={})
    response_metadata = _json_loads(row["response_metadata_json"], default={})
    if not isinstance(additional_kwargs, dict):
        additional_kwargs = {}
    if not isinstance(response_metadata, dict):
        response_metadata = {}

    if role == "human":
        return HumanMessage(content=content, additional_kwargs=additional_kwargs, response_metadata=response_metadata)
    if role == "system":
        return SystemMessage(content=content, additional_kwargs=additional_kwargs, response_metadata=response_metadata)
    if role == "tool":
        return ToolMessage(
            content=content,
            tool_call_id=str(row["tool_call_id"] or ""),
            name=str(row["tool_name"] or "") or None,
            additional_kwargs=additional_kwargs,
            response_metadata=response_metadata,
        )
    if role == "ai":
        tool_calls = _json_loads(row["tool_calls_json"], default=[])
        if not isinstance(tool_calls, list):
            tool_calls = []
        return AIMessage(
            content=content,
            tool_calls=tool_calls,
            additional_kwargs=additional_kwargs,
            response_metadata=response_metadata,
        )
    return HumanMessage(content=content, additional_kwargs=additional_kwargs, response_metadata=response_metadata)


def _message_role(message: AnyMessage) -> str:
    if isinstance(message, HumanMessage):
        return "human"
    if isinstance(message, AIMessage):
        return "ai"
    if isinstance(message, ToolMessage):
        return "tool"
    if isinstance(message, SystemMessage):
        return "system"
    return str(getattr(message, "type", "") or "message")


def _require_session_id(session_id: str) -> str:
    normalized = str(session_id or "").strip()
    if not normalized:
        raise ValueError("session_id is required")
    return normalized


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _json_loads(value: Any, *, default: Any) -> Any:
    if value is None or value == "":
        return default
    try:
        return json.loads(str(value))
    except Exception:
        return default


def _merge_usage(previous: dict[str, int], latest: dict[str, int]) -> dict[str, int]:
    merged = dict(previous)
    for key, value in latest.items():
        try:
            merged[key] = int(merged.get(key, 0) or 0) + int(value or 0)
        except Exception:
            continue
    return merged

