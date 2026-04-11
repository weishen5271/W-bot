from __future__ import annotations

import pickle
import sqlite3
import threading
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata, CheckpointTuple, get_checkpoint_id, get_checkpoint_metadata
from langgraph.checkpoint.memory import WRITES_IDX_MAP, InMemorySaver

from .logging_config import get_logger

logger = get_logger(__name__)


class WorkspaceFileCheckpointer(InMemorySaver):
    """Use a workspace-local SQLite database for LangGraph short-term memory."""

    def __init__(self, file_path: str) -> None:
        super().__init__()
        self._file_path = Path(file_path).expanduser()
        if not self._file_path.is_absolute():
            self._file_path = (Path.cwd() / self._file_path).resolve()
        self._legacy_pickle_path = self._infer_legacy_pickle_path(self._file_path)
        self._lock = threading.RLock()
        self.setup()

    @property
    def file_path(self) -> Path:
        return self._file_path

    def setup(self) -> None:
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            self._initialize_schema(conn)
            self._migrate_schema_if_needed(conn)
            self._maybe_migrate_legacy_pickle(conn)

    def close(self) -> None:
        return

    def __enter__(self) -> "WorkspaceFileCheckpointer":
        self.setup()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        thread_id = str(config["configurable"]["thread_id"])
        checkpoint_ns = str(config["configurable"].get("checkpoint_ns", ""))
        requested_checkpoint_id = get_checkpoint_id(config)

        with self._lock, self._connect() as conn:
            row = self._fetch_checkpoint_row(
                conn,
                thread_id=thread_id,
                checkpoint_ns=checkpoint_ns,
                checkpoint_id=requested_checkpoint_id,
            )
            if row is None:
                return None
            return self._row_to_checkpoint_tuple(conn, row)

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: dict[str, Any],
    ) -> RunnableConfig:
        checkpoint_copy = checkpoint.copy()
        thread_id = str(config["configurable"]["thread_id"])
        checkpoint_ns = str(config["configurable"].get("checkpoint_ns", ""))
        checkpoint_id = str(checkpoint["id"])
        parent_checkpoint_id = config["configurable"].get("checkpoint_id")
        channel_values = dict(checkpoint_copy.pop("channel_values", {}))
        metadata_payload = get_checkpoint_metadata(config, metadata)

        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                for channel_name, version in new_versions.items():
                    if channel_name in channel_values:
                        value_type, value_blob = self.serde.dumps_typed(channel_values[channel_name])
                    else:
                        value_type, value_blob = ("empty", b"")
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO checkpoint_blobs (
                            thread_id, checkpoint_ns, channel_name, version, value_type, value_blob
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            thread_id,
                            checkpoint_ns,
                            str(channel_name),
                            self._version_key(version),
                            value_type,
                            sqlite3.Binary(value_blob),
                        ),
                    )

                checkpoint_type, checkpoint_blob = self.serde.dumps_typed(checkpoint_copy)
                metadata_type, metadata_blob = self.serde.dumps_typed(metadata_payload)
                conn.execute(
                    """
                    INSERT OR REPLACE INTO checkpoints (
                        thread_id,
                        checkpoint_ns,
                        checkpoint_id,
                        checkpoint_type,
                        checkpoint_blob,
                        metadata_type,
                        metadata_blob,
                        parent_checkpoint_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        thread_id,
                        checkpoint_ns,
                        checkpoint_id,
                        checkpoint_type,
                        sqlite3.Binary(checkpoint_blob),
                        metadata_type,
                        sqlite3.Binary(metadata_blob),
                        str(parent_checkpoint_id) if parent_checkpoint_id else None,
                    ),
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        thread_id = str(config["configurable"]["thread_id"])
        checkpoint_ns = str(config["configurable"].get("checkpoint_ns", ""))
        checkpoint_id = str(config["configurable"]["checkpoint_id"])

        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                for idx, (channel_name, value) in enumerate(writes):
                    write_idx = WRITES_IDX_MAP.get(channel_name, idx)
                    if write_idx >= 0:
                        existing = conn.execute(
                            """
                            SELECT 1
                            FROM checkpoint_writes
                            WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?
                              AND task_id = ? AND write_idx = ?
                            LIMIT 1
                            """,
                            (thread_id, checkpoint_ns, checkpoint_id, task_id, write_idx),
                        ).fetchone()
                        if existing is not None:
                            continue

                    value_type, value_blob = self.serde.dumps_typed(value)
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO checkpoint_writes (
                            thread_id,
                            checkpoint_ns,
                            checkpoint_id,
                            task_id,
                            write_idx,
                            channel_name,
                            value_type,
                            value_blob,
                            task_path
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            thread_id,
                            checkpoint_ns,
                            checkpoint_id,
                            task_id,
                            write_idx,
                            channel_name,
                            value_type,
                            sqlite3.Binary(value_blob),
                            task_path,
                        ),
                    )
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: dict[str, Any],
    ) -> RunnableConfig:
        return self.put(config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        self.put_writes(config, writes, task_id, task_path)

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        configurable = (config or {}).get("configurable", {})
        thread_id = configurable.get("thread_id")
        checkpoint_ns = configurable.get("checkpoint_ns")
        checkpoint_id = get_checkpoint_id(config) if config else None
        before_checkpoint_id = get_checkpoint_id(before) if before else None

        clauses = []
        params: list[Any] = []
        if thread_id is not None:
            clauses.append("thread_id = ?")
            params.append(str(thread_id))
        if checkpoint_ns is not None:
            clauses.append("checkpoint_ns = ?")
            params.append(str(checkpoint_ns))
        if checkpoint_id is not None:
            clauses.append("checkpoint_id = ?")
            params.append(str(checkpoint_id))
        if before_checkpoint_id is not None:
            clauses.append("checkpoint_id < ?")
            params.append(str(before_checkpoint_id))

        where_clause = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        limit_clause = ""
        if limit is not None:
            limit_clause = "LIMIT ?"
            params.append(max(0, int(limit)))

        query = f"""
            SELECT
                thread_id,
                checkpoint_ns,
                checkpoint_id,
                checkpoint_type,
                checkpoint_blob,
                metadata_type,
                metadata_blob,
                parent_checkpoint_id
            FROM checkpoints
            {where_clause}
            ORDER BY thread_id, checkpoint_ns, checkpoint_id DESC
            {limit_clause}
        """

        with self._lock, self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
            for row in rows:
                checkpoint_tuple = self._row_to_checkpoint_tuple(conn, row)
                if checkpoint_tuple is None:
                    continue
                if filter and not all(
                    query_value == checkpoint_tuple.metadata.get(query_key)
                    for query_key, query_value in filter.items()
                ):
                    continue
                yield checkpoint_tuple

    def delete_thread(self, thread_id: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute("DELETE FROM checkpoint_writes WHERE thread_id = ?", (thread_id,))
                conn.execute("DELETE FROM checkpoint_blobs WHERE thread_id = ?", (thread_id,))
                conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    async def adelete_thread(self, thread_id: str) -> None:
        self.delete_thread(thread_id)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._file_path, timeout=30.0, isolation_level=None, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _initialize_schema(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS checkpoint_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS checkpoints (
                thread_id TEXT NOT NULL,
                checkpoint_ns TEXT NOT NULL,
                checkpoint_id TEXT NOT NULL,
                checkpoint_type TEXT NOT NULL,
                checkpoint_blob BLOB NOT NULL,
                metadata_type TEXT NOT NULL,
                metadata_blob BLOB NOT NULL,
                parent_checkpoint_id TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
            );

            CREATE TABLE IF NOT EXISTS checkpoint_writes (
                thread_id TEXT NOT NULL,
                checkpoint_ns TEXT NOT NULL,
                checkpoint_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                write_idx INTEGER NOT NULL,
                channel_name TEXT NOT NULL,
                value_type TEXT NOT NULL,
                value_blob BLOB NOT NULL,
                task_path TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, write_idx)
            );

            CREATE TABLE IF NOT EXISTS checkpoint_blobs (
                thread_id TEXT NOT NULL,
                checkpoint_ns TEXT NOT NULL,
                channel_name TEXT NOT NULL,
                version TEXT NOT NULL,
                value_type TEXT NOT NULL,
                value_blob BLOB NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (thread_id, checkpoint_ns, channel_name, version)
            );

            CREATE INDEX IF NOT EXISTS idx_checkpoints_thread_ns
                ON checkpoints(thread_id, checkpoint_ns, checkpoint_id DESC);

            CREATE INDEX IF NOT EXISTS idx_checkpoint_writes_lookup
                ON checkpoint_writes(thread_id, checkpoint_ns, checkpoint_id);
            """
        )
    def _migrate_schema_if_needed(self, conn: sqlite3.Connection) -> None:
        row = conn.execute(
            """
            SELECT value
            FROM checkpoint_meta
            WHERE key = 'schema_version'
            LIMIT 1
            """
        ).fetchone()
        if row is not None and str(row["value"]) == "2":
            return

        foreign_keys = conn.execute("PRAGMA foreign_key_list(checkpoint_writes)").fetchall()
        if not foreign_keys:
            conn.execute(
                """
                INSERT OR REPLACE INTO checkpoint_meta (key, value)
                VALUES ('schema_version', '2')
                """
            )
            return

        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.executescript(
                """
                ALTER TABLE checkpoint_writes RENAME TO checkpoint_writes_old;

                CREATE TABLE checkpoint_writes (
                    thread_id TEXT NOT NULL,
                    checkpoint_ns TEXT NOT NULL,
                    checkpoint_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    write_idx INTEGER NOT NULL,
                    channel_name TEXT NOT NULL,
                    value_type TEXT NOT NULL,
                    value_blob BLOB NOT NULL,
                    task_path TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, write_idx)
                );

                INSERT OR REPLACE INTO checkpoint_writes (
                    thread_id,
                    checkpoint_ns,
                    checkpoint_id,
                    task_id,
                    write_idx,
                    channel_name,
                    value_type,
                    value_blob,
                    task_path,
                    created_at
                )
                SELECT
                    thread_id,
                    checkpoint_ns,
                    checkpoint_id,
                    task_id,
                    write_idx,
                    channel_name,
                    value_type,
                    value_blob,
                    task_path,
                    created_at
                FROM checkpoint_writes_old;

                DROP TABLE checkpoint_writes_old;

                CREATE INDEX IF NOT EXISTS idx_checkpoint_writes_lookup
                    ON checkpoint_writes(thread_id, checkpoint_ns, checkpoint_id);
                """
            )
            conn.execute(
                """
                INSERT OR REPLACE INTO checkpoint_meta (key, value)
                VALUES ('schema_version', '2')
                """
            )
            conn.commit()
            logger.info("Migrated checkpoint_writes schema to remove foreign key: %s", self._file_path)
        except Exception:
            conn.rollback()
            raise

    def _fetch_checkpoint_row(
        self,
        conn: sqlite3.Connection,
        *,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str | None,
    ) -> sqlite3.Row | None:
        if checkpoint_id:
            return conn.execute(
                """
                SELECT
                    thread_id,
                    checkpoint_ns,
                    checkpoint_id,
                    checkpoint_type,
                    checkpoint_blob,
                    metadata_type,
                    metadata_blob,
                    parent_checkpoint_id
                FROM checkpoints
                WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?
                LIMIT 1
                """,
                (thread_id, checkpoint_ns, checkpoint_id),
            ).fetchone()
        return conn.execute(
            """
            SELECT
                thread_id,
                checkpoint_ns,
                checkpoint_id,
                checkpoint_type,
                checkpoint_blob,
                metadata_type,
                metadata_blob,
                parent_checkpoint_id
            FROM checkpoints
            WHERE thread_id = ? AND checkpoint_ns = ?
            ORDER BY checkpoint_id DESC
            LIMIT 1
            """,
            (thread_id, checkpoint_ns),
        ).fetchone()

    def _row_to_checkpoint_tuple(
        self,
        conn: sqlite3.Connection,
        row: sqlite3.Row,
    ) -> CheckpointTuple | None:
        checkpoint = self.serde.loads_typed((row["checkpoint_type"], bytes(row["checkpoint_blob"])))
        metadata = self.serde.loads_typed((row["metadata_type"], bytes(row["metadata_blob"])))
        if not isinstance(checkpoint, dict):
            logger.warning("Ignored invalid checkpoint payload: thread=%s checkpoint=%s", row["thread_id"], row["checkpoint_id"])
            return None
        channel_versions = checkpoint.get("channel_versions", {}) or {}
        pending_writes = self._load_pending_writes(
            conn,
            thread_id=row["thread_id"],
            checkpoint_ns=row["checkpoint_ns"],
            checkpoint_id=row["checkpoint_id"],
        )
        return CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": row["thread_id"],
                    "checkpoint_ns": row["checkpoint_ns"],
                    "checkpoint_id": row["checkpoint_id"],
                }
            },
            checkpoint={
                **checkpoint,
                "channel_values": self._load_blob_values(
                    conn,
                    thread_id=row["thread_id"],
                    checkpoint_ns=row["checkpoint_ns"],
                    channel_versions=channel_versions,
                ),
            },
            metadata=metadata,
            parent_config=(
                {
                    "configurable": {
                        "thread_id": row["thread_id"],
                        "checkpoint_ns": row["checkpoint_ns"],
                        "checkpoint_id": row["parent_checkpoint_id"],
                    }
                }
                if row["parent_checkpoint_id"]
                else None
            ),
            pending_writes=pending_writes,
        )

    def _load_pending_writes(
        self,
        conn: sqlite3.Connection,
        *,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
    ) -> list[tuple[str, str, Any]]:
        rows = conn.execute(
            """
            SELECT task_id, channel_name, value_type, value_blob
            FROM checkpoint_writes
            WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?
            ORDER BY task_id, write_idx
            """,
            (thread_id, checkpoint_ns, checkpoint_id),
        ).fetchall()
        return [
            (
                str(row["task_id"]),
                str(row["channel_name"]),
                self.serde.loads_typed((row["value_type"], bytes(row["value_blob"]))),
            )
            for row in rows
        ]

    def _load_blob_values(
        self,
        conn: sqlite3.Connection,
        *,
        thread_id: str,
        checkpoint_ns: str,
        channel_versions: dict[str, Any],
    ) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for channel_name, version in channel_versions.items():
            row = conn.execute(
                """
                SELECT value_type, value_blob
                FROM checkpoint_blobs
                WHERE thread_id = ? AND checkpoint_ns = ? AND channel_name = ? AND version = ?
                LIMIT 1
                """,
                (thread_id, checkpoint_ns, str(channel_name), self._version_key(version)),
            ).fetchone()
            if row is None:
                continue
            if row["value_type"] == "empty":
                continue
            values[str(channel_name)] = self.serde.loads_typed((row["value_type"], bytes(row["value_blob"])))
        return values

    def _maybe_migrate_legacy_pickle(self, conn: sqlite3.Connection) -> None:
        if not self._legacy_pickle_path.exists():
            return
        existing = conn.execute("SELECT 1 FROM checkpoints LIMIT 1").fetchone()
        if existing is not None:
            return

        try:
            payload = pickle.loads(self._legacy_pickle_path.read_bytes())
        except Exception:
            logger.exception("Failed to load legacy short-term memory file: %s", self._legacy_pickle_path)
            return
        if not isinstance(payload, dict):
            logger.warning("Ignored invalid legacy short-term memory payload: %s", self._legacy_pickle_path)
            return

        storage = payload.get("storage", {})
        writes = payload.get("writes", {})
        blobs = payload.get("blobs", {})

        conn.execute("BEGIN IMMEDIATE")
        try:
            for (thread_id, checkpoint_ns, channel_name, version), typed_value in blobs.items():
                if not (
                    isinstance(thread_id, str)
                    and isinstance(checkpoint_ns, str)
                    and isinstance(channel_name, str)
                    and isinstance(typed_value, tuple)
                    and len(typed_value) == 2
                ):
                    continue
                value_type, value_blob = typed_value
                conn.execute(
                    """
                    INSERT OR REPLACE INTO checkpoint_blobs (
                        thread_id, checkpoint_ns, channel_name, version, value_type, value_blob
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        thread_id,
                        checkpoint_ns,
                        channel_name,
                        self._version_key(version),
                        str(value_type),
                        sqlite3.Binary(bytes(value_blob)),
                    ),
                )

            if isinstance(storage, dict):
                for thread_id, ns_map in storage.items():
                    if not isinstance(thread_id, str) or not isinstance(ns_map, dict):
                        continue
                    for checkpoint_ns, checkpoint_map in ns_map.items():
                        if not isinstance(checkpoint_ns, str) or not isinstance(checkpoint_map, dict):
                            continue
                        for checkpoint_id, saved in checkpoint_map.items():
                            if not isinstance(saved, tuple) or len(saved) != 3:
                                continue
                            checkpoint_payload, metadata_payload, parent_checkpoint_id = saved
                            if not (
                                isinstance(checkpoint_payload, tuple)
                                and len(checkpoint_payload) == 2
                                and isinstance(metadata_payload, tuple)
                                and len(metadata_payload) == 2
                            ):
                                continue
                            conn.execute(
                                """
                                INSERT OR REPLACE INTO checkpoints (
                                    thread_id,
                                    checkpoint_ns,
                                    checkpoint_id,
                                    checkpoint_type,
                                    checkpoint_blob,
                                    metadata_type,
                                    metadata_blob,
                                    parent_checkpoint_id
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                """,
                                (
                                    thread_id,
                                    checkpoint_ns,
                                    str(checkpoint_id),
                                    str(checkpoint_payload[0]),
                                    sqlite3.Binary(bytes(checkpoint_payload[1])),
                                    str(metadata_payload[0]),
                                    sqlite3.Binary(bytes(metadata_payload[1])),
                                    str(parent_checkpoint_id) if parent_checkpoint_id else None,
                                ),
                            )

            if isinstance(writes, dict):
                for outer_key, inner_map in writes.items():
                    if not (
                        isinstance(outer_key, tuple)
                        and len(outer_key) == 3
                        and isinstance(inner_map, dict)
                    ):
                        continue
                    thread_id, checkpoint_ns, checkpoint_id = outer_key
                    if not (
                        isinstance(thread_id, str)
                        and isinstance(checkpoint_ns, str)
                        and isinstance(checkpoint_id, str)
                    ):
                        continue
                    for inner_key, value in inner_map.items():
                        if not (
                            isinstance(inner_key, tuple)
                            and len(inner_key) == 2
                            and isinstance(value, tuple)
                            and len(value) == 4
                        ):
                            continue
                        task_id, write_idx = inner_key
                        stored_task_id, channel_name, typed_value, task_path = value
                        if not (
                            isinstance(task_id, str)
                            and isinstance(write_idx, int)
                            and isinstance(stored_task_id, str)
                            and isinstance(channel_name, str)
                            and isinstance(typed_value, tuple)
                            and len(typed_value) == 2
                        ):
                            continue
                        conn.execute(
                            """
                            INSERT OR REPLACE INTO checkpoint_writes (
                                thread_id,
                                checkpoint_ns,
                                checkpoint_id,
                                task_id,
                                write_idx,
                                channel_name,
                                value_type,
                                value_blob,
                                task_path
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                thread_id,
                                checkpoint_ns,
                                checkpoint_id,
                                stored_task_id,
                                write_idx,
                                channel_name,
                                str(typed_value[0]),
                                sqlite3.Binary(bytes(typed_value[1])),
                                str(task_path or ""),
                            ),
                        )

            conn.commit()
            logger.info(
                "Migrated legacy short-term memory from %s to SQLite %s",
                self._legacy_pickle_path,
                self._file_path,
            )
        except Exception:
            conn.rollback()
            logger.exception("Failed to migrate legacy short-term memory into SQLite: %s", self._file_path)

    @staticmethod
    def _infer_legacy_pickle_path(sqlite_path: Path) -> Path:
        if sqlite_path.suffix.lower() == ".sqlite":
            return sqlite_path.with_suffix(".pkl")
        return sqlite_path

    @staticmethod
    def _version_key(version: Any) -> str:
        return str(version)


def resolve_short_term_memory_path(configured_path: str) -> str:
    target = Path(configured_path).expanduser()
    if not target.is_absolute():
        target = (Path.cwd() / target).resolve()
    if target.suffix.lower() == ".pkl":
        target = target.with_suffix(".sqlite")
    elif not target.suffix:
        target = target.with_suffix(".sqlite")
    return str(target)
