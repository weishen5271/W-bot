from __future__ import annotations

import gzip
import hashlib
import json
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import psycopg2
from psycopg2.extras import execute_values

from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ShortTermMemoryOptimizationSettings:
    enabled: bool
    run_on_startup: bool
    interval_minutes: int
    keep_recent_checkpoints: int
    summary_batch_size: int
    max_threads_per_run: int
    max_checkpoints_per_thread: int
    archive_before_delete: bool
    compress_level: int


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS checkpoint_rolling_summaries (
    id BIGSERIAL PRIMARY KEY,
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    summary_hash TEXT NOT NULL UNIQUE,
    summary_text TEXT NOT NULL,
    source_count INTEGER NOT NULL,
    source_checkpoint_ids TEXT[] NOT NULL,
    first_checkpoint_id TEXT NOT NULL,
    last_checkpoint_id TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS checkpoint_rolling_summaries_thread_ns_idx
ON checkpoint_rolling_summaries (thread_id, checkpoint_ns, created_at DESC);

CREATE TABLE IF NOT EXISTS checkpoint_blob_store (
    blob_hash TEXT PRIMARY KEY,
    codec TEXT NOT NULL,
    compressed_blob BYTEA NOT NULL,
    original_bytes INTEGER NOT NULL,
    compressed_bytes INTEGER NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS checkpoint_cold_archive_entries (
    id BIGSERIAL PRIMARY KEY,
    source_table TEXT NOT NULL,
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL DEFAULT '',
    task_id TEXT NOT NULL DEFAULT '',
    idx INTEGER NOT NULL DEFAULT -1,
    channel TEXT NOT NULL DEFAULT '',
    version TEXT NOT NULL DEFAULT '',
    type TEXT NOT NULL DEFAULT '',
    blob_hash TEXT NOT NULL REFERENCES checkpoint_blob_store(blob_hash),
    archived_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (
        source_table,
        thread_id,
        checkpoint_ns,
        checkpoint_id,
        task_id,
        idx,
        channel,
        version
    )
);

CREATE INDEX IF NOT EXISTS checkpoint_cold_archive_entries_thread_ns_idx
ON checkpoint_cold_archive_entries (thread_id, checkpoint_ns, archived_at DESC);
"""


def start_short_memory_optimizer_worker(
    *,
    postgres_dsn: str,
    settings: ShortTermMemoryOptimizationSettings,
) -> tuple[threading.Thread | None, threading.Event | None]:
    """启动短期记忆优化后台任务。

    Args:
        postgres_dsn: PostgreSQL 连接串。
        settings: 短期记忆优化配置。
    """
    if not settings.enabled:
        logger.info("Short-term memory optimizer disabled")
        return None, None

    stop_event = threading.Event()
    thread = threading.Thread(
        target=_optimizer_loop,
        kwargs={
            "postgres_dsn": postgres_dsn,
            "settings": settings,
            "stop_event": stop_event,
        },
        daemon=True,
        name="short-memory-optimizer",
    )
    thread.start()
    return thread, stop_event


def _optimizer_loop(
    *,
    postgres_dsn: str,
    settings: ShortTermMemoryOptimizationSettings,
    stop_event: threading.Event,
) -> None:
    """处理优化循环并执行周期任务。"""
    interval_seconds = max(60, settings.interval_minutes * 60)

    if settings.run_on_startup:
        _run_once_safe(postgres_dsn=postgres_dsn, settings=settings)

    while not stop_event.wait(interval_seconds):
        _run_once_safe(postgres_dsn=postgres_dsn, settings=settings)


def _run_once_safe(*, postgres_dsn: str, settings: ShortTermMemoryOptimizationSettings) -> None:
    """执行单轮优化并捕获异常。"""
    try:
        optimize_short_term_memory(postgres_dsn=postgres_dsn, settings=settings)
    except Exception:
        logger.exception("Short-term memory optimization run failed")


def optimize_short_term_memory(
    *,
    postgres_dsn: str,
    settings: ShortTermMemoryOptimizationSettings,
) -> None:
    """执行一次短期记忆优化流程。"""
    conn = psycopg2.connect(postgres_dsn)
    conn.autocommit = False
    try:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_SQL)
        conn.commit()

        targets = _select_compaction_targets(conn=conn, settings=settings)
        if not targets:
            logger.debug("No short-memory compaction targets found")
            return

        for thread_id, checkpoint_ns in targets:
            _compact_thread_ns(
                conn=conn,
                settings=settings,
                thread_id=thread_id,
                checkpoint_ns=checkpoint_ns,
            )
    finally:
        conn.close()


def _select_compaction_targets(
    *,
    conn: Any,
    settings: ShortTermMemoryOptimizationSettings,
) -> list[tuple[str, str]]:
    """选择需要压缩的线程分组。"""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT thread_id, checkpoint_ns
            FROM checkpoints
            GROUP BY thread_id, checkpoint_ns
            HAVING COUNT(*) > %s
            ORDER BY COUNT(*) DESC
            LIMIT %s
            """,
            (settings.keep_recent_checkpoints, settings.max_threads_per_run),
        )
        rows = cur.fetchall()
    return [(str(row[0]), str(row[1] or "")) for row in rows]


def _compact_thread_ns(
    *,
    conn: Any,
    settings: ShortTermMemoryOptimizationSettings,
    thread_id: str,
    checkpoint_ns: str,
) -> None:
    """压缩单个 thread_id + checkpoint_ns 分区。"""
    stale_ids = _select_stale_checkpoint_ids(
        conn=conn,
        thread_id=thread_id,
        checkpoint_ns=checkpoint_ns,
        keep_recent=settings.keep_recent_checkpoints,
        max_rows=settings.max_checkpoints_per_thread,
    )
    if not stale_ids:
        return

    _save_rolling_summaries(
        conn=conn,
        thread_id=thread_id,
        checkpoint_ns=checkpoint_ns,
        checkpoint_ids=stale_ids,
        batch_size=settings.summary_batch_size,
    )

    if settings.archive_before_delete:
        _archive_checkpoint_writes(
            conn=conn,
            thread_id=thread_id,
            checkpoint_ns=checkpoint_ns,
            checkpoint_ids=stale_ids,
            compress_level=settings.compress_level,
        )

    with conn.cursor() as cur:
        cur.execute(
            """
            DELETE FROM checkpoint_writes
            WHERE thread_id = %s AND checkpoint_ns = %s AND checkpoint_id = ANY(%s)
            """,
            (thread_id, checkpoint_ns, stale_ids),
        )
        deleted_writes = cur.rowcount
        cur.execute(
            """
            DELETE FROM checkpoints
            WHERE thread_id = %s AND checkpoint_ns = %s AND checkpoint_id = ANY(%s)
            """,
            (thread_id, checkpoint_ns, stale_ids),
        )
        deleted_checkpoints = cur.rowcount
    conn.commit()

    orphan_rows = _find_orphan_blobs(
        conn=conn,
        thread_id=thread_id,
        checkpoint_ns=checkpoint_ns,
    )
    if settings.archive_before_delete and orphan_rows:
        _archive_orphan_blobs(
            conn=conn,
            thread_id=thread_id,
            checkpoint_ns=checkpoint_ns,
            rows=orphan_rows,
            compress_level=settings.compress_level,
        )
    if orphan_rows:
        _delete_orphan_blobs(conn=conn, thread_id=thread_id, checkpoint_ns=checkpoint_ns)

    logger.info(
        "Compacted short-memory: thread_id=%s checkpoint_ns=%s stale=%s deleted_checkpoints=%s deleted_writes=%s orphan_blobs=%s",
        thread_id,
        checkpoint_ns,
        len(stale_ids),
        deleted_checkpoints,
        deleted_writes,
        len(orphan_rows),
    )


def _select_stale_checkpoint_ids(
    *,
    conn: Any,
    thread_id: str,
    checkpoint_ns: str,
    keep_recent: int,
    max_rows: int,
) -> list[str]:
    """查询超过保留阈值的历史 checkpoint ID。"""
    offset = max(keep_recent, 0)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT checkpoint_id
            FROM checkpoints
            WHERE thread_id = %s AND checkpoint_ns = %s
            ORDER BY checkpoint_id DESC
            OFFSET %s
            LIMIT %s
            """,
            (thread_id, checkpoint_ns, offset, max_rows),
        )
        rows = cur.fetchall()
    return [str(row[0]) for row in rows]


def _save_rolling_summaries(
    *,
    conn: Any,
    thread_id: str,
    checkpoint_ns: str,
    checkpoint_ids: list[str],
    batch_size: int,
) -> None:
    """按批次将历史 checkpoint 写为滚动摘要。"""
    safe_batch_size = max(1, batch_size)
    for idx in range(0, len(checkpoint_ids), safe_batch_size):
        chunk = checkpoint_ids[idx : idx + safe_batch_size]
        summary_text = _build_summary_text(
            thread_id=thread_id,
            checkpoint_ns=checkpoint_ns,
            checkpoint_ids=chunk,
        )
        summary_hash = hashlib.sha256(
            "|".join([thread_id, checkpoint_ns, ",".join(chunk)]).encode("utf-8")
        ).hexdigest()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO checkpoint_rolling_summaries (
                    thread_id,
                    checkpoint_ns,
                    summary_hash,
                    summary_text,
                    source_count,
                    source_checkpoint_ids,
                    first_checkpoint_id,
                    last_checkpoint_id
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (summary_hash) DO NOTHING
                """,
                (
                    thread_id,
                    checkpoint_ns,
                    summary_hash,
                    summary_text,
                    len(chunk),
                    chunk,
                    chunk[-1],
                    chunk[0],
                ),
            )
    conn.commit()


def _build_summary_text(
    *,
    thread_id: str,
    checkpoint_ns: str,
    checkpoint_ids: list[str],
) -> str:
    """构造摘要文本，替代历史原文。"""
    now = datetime.now(timezone.utc).isoformat()
    brief = (
        f"Rolling summary for thread={thread_id}, ns={checkpoint_ns}, "
        f"checkpoints={len(checkpoint_ids)}, range={checkpoint_ids[-1]}..{checkpoint_ids[0]}"
    )
    payload = {
        "generated_at": now,
        "thread_id": thread_id,
        "checkpoint_ns": checkpoint_ns,
        "source_count": len(checkpoint_ids),
        "first_checkpoint_id": checkpoint_ids[-1],
        "last_checkpoint_id": checkpoint_ids[0],
        "source_checkpoint_ids": checkpoint_ids,
        "brief": brief,
    }
    return json.dumps(payload, ensure_ascii=False)


def _archive_checkpoint_writes(
    *,
    conn: Any,
    thread_id: str,
    checkpoint_ns: str,
    checkpoint_ids: list[str],
    compress_level: int,
) -> None:
    """归档 checkpoint_writes，并进行去重压缩。"""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT checkpoint_id, task_id, idx, channel, type, blob
            FROM checkpoint_writes
            WHERE thread_id = %s AND checkpoint_ns = %s AND checkpoint_id = ANY(%s)
            """,
            (thread_id, checkpoint_ns, checkpoint_ids),
        )
        rows = cur.fetchall()
    if not rows:
        return

    store_rows: list[tuple[str, str, bytes, int, int]] = []
    archive_rows: list[tuple[str, str, str, str, str, int, str, str, str, str]] = []
    for checkpoint_id, task_id, idx, channel, type_value, blob in rows:
        blob_bytes = bytes(blob or b"")
        blob_hash, compressed, original_size, compressed_size = _compress_blob(
            blob_bytes=blob_bytes,
            compress_level=compress_level,
        )
        store_rows.append((blob_hash, "gzip", compressed, original_size, compressed_size))
        archive_rows.append(
            (
                "checkpoint_writes",
                thread_id,
                checkpoint_ns,
                str(checkpoint_id),
                str(task_id),
                int(idx),
                str(channel),
                "",
                str(type_value) if type_value is not None else "",
                blob_hash,
            )
        )

    _upsert_blob_store(conn=conn, rows=store_rows)
    _insert_archive_entries(conn=conn, rows=archive_rows)
    conn.commit()


def _find_orphan_blobs(
    *,
    conn: Any,
    thread_id: str,
    checkpoint_ns: str,
) -> list[tuple[str, str | None, bytes | None]]:
    """查找不再被 checkpoint 引用的 blob。"""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT bl.channel, bl.version, bl.blob
            FROM checkpoint_blobs bl
            WHERE bl.thread_id = %s
              AND bl.checkpoint_ns = %s
              AND NOT EXISTS (
                  SELECT 1
                  FROM checkpoints c
                  CROSS JOIN LATERAL jsonb_each_text(
                      COALESCE(c.checkpoint -> 'channel_versions', '{}'::jsonb)
                  ) cv
                  WHERE c.thread_id = bl.thread_id
                    AND c.checkpoint_ns = bl.checkpoint_ns
                    AND cv.key = bl.channel
                    AND cv.value = bl.version
              )
            """,
            (thread_id, checkpoint_ns),
        )
        rows = cur.fetchall()
    return [(str(row[0]), str(row[1]) if row[1] is not None else None, row[2]) for row in rows]


def _archive_orphan_blobs(
    *,
    conn: Any,
    thread_id: str,
    checkpoint_ns: str,
    rows: list[tuple[str, str | None, bytes | None]],
    compress_level: int,
) -> None:
    """归档 orphan checkpoint_blobs，并进行去重压缩。"""
    store_rows: list[tuple[str, str, bytes, int, int]] = []
    archive_rows: list[tuple[str, str, str, str, str, int, str, str, str, str]] = []
    for channel, version, blob in rows:
        blob_bytes = bytes(blob or b"")
        blob_hash, compressed, original_size, compressed_size = _compress_blob(
            blob_bytes=blob_bytes,
            compress_level=compress_level,
        )
        store_rows.append((blob_hash, "gzip", compressed, original_size, compressed_size))
        archive_rows.append(
            (
                "checkpoint_blobs",
                thread_id,
                checkpoint_ns,
                "",
                "",
                -1,
                channel,
                version or "",
                "",
                blob_hash,
            )
        )

    _upsert_blob_store(conn=conn, rows=store_rows)
    _insert_archive_entries(conn=conn, rows=archive_rows)
    conn.commit()


def _delete_orphan_blobs(*, conn: Any, thread_id: str, checkpoint_ns: str) -> None:
    """删除 orphan checkpoint_blobs。"""
    with conn.cursor() as cur:
        cur.execute(
            """
            DELETE FROM checkpoint_blobs bl
            WHERE bl.thread_id = %s
              AND bl.checkpoint_ns = %s
              AND NOT EXISTS (
                  SELECT 1
                  FROM checkpoints c
                  CROSS JOIN LATERAL jsonb_each_text(
                      COALESCE(c.checkpoint -> 'channel_versions', '{}'::jsonb)
                  ) cv
                  WHERE c.thread_id = bl.thread_id
                    AND c.checkpoint_ns = bl.checkpoint_ns
                    AND cv.key = bl.channel
                    AND cv.value = bl.version
              )
            """,
            (thread_id, checkpoint_ns),
        )
    conn.commit()


def _compress_blob(*, blob_bytes: bytes, compress_level: int) -> tuple[str, bytes, int, int]:
    """对 blob 执行哈希与压缩。"""
    blob_hash = hashlib.sha256(blob_bytes).hexdigest()
    safe_level = max(1, min(9, compress_level))
    compressed = gzip.compress(blob_bytes, compresslevel=safe_level)
    return blob_hash, compressed, len(blob_bytes), len(compressed)


def _upsert_blob_store(*, conn: Any, rows: list[tuple[str, str, bytes, int, int]]) -> None:
    """批量写入去重后 blob 存储。"""
    if not rows:
        return
    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO checkpoint_blob_store (
                blob_hash,
                codec,
                compressed_blob,
                original_bytes,
                compressed_bytes
            )
            VALUES %s
            ON CONFLICT (blob_hash) DO NOTHING
            """,
            rows,
            page_size=1000,
        )


def _insert_archive_entries(
    *,
    conn: Any,
    rows: list[tuple[str, str, str, str, str, int, str, str, str, str]],
) -> None:
    """批量写入归档引用记录。"""
    if not rows:
        return
    normalized: list[tuple[Any, ...]] = []
    for (
        source_table,
        thread_id,
        checkpoint_ns,
        checkpoint_id,
        task_id,
        idx,
        channel,
        version,
        type_value,
        blob_hash,
    ) in rows:
        normalized.append(
            (
                source_table,
                thread_id,
                checkpoint_ns,
                checkpoint_id or "",
                task_id or "",
                int(idx if idx is not None else -1),
                channel or "",
                version or "",
                type_value or "",
                blob_hash,
            )
        )
    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO checkpoint_cold_archive_entries (
                source_table,
                thread_id,
                checkpoint_ns,
                checkpoint_id,
                task_id,
                idx,
                channel,
                version,
                type,
                blob_hash
            )
            VALUES %s
            ON CONFLICT (
                source_table,
                thread_id,
                checkpoint_ns,
                checkpoint_id,
                task_id,
                idx,
                channel,
                version
            ) DO NOTHING
            """,
            normalized,
            page_size=1000,
        )
