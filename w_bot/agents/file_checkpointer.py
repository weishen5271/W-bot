from __future__ import annotations

import pickle
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Sequence

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver

from .logging_config import get_logger

logger = get_logger(__name__)

_LOCK_RETRY_DELAY_SEC = 0.1
_LOCK_TIMEOUT_SEC = 5.0
_REPLACE_RETRY_DELAY_SEC = 0.2
_REPLACE_RETRY_COUNT = 5

try:
    import msvcrt
except ImportError:  # pragma: no cover - Windows first, but keep module importable elsewhere
    msvcrt = None


class WorkspaceFileCheckpointer(InMemorySaver):
    """将 LangGraph 短期记忆持久化到工作区本地文件。"""

    def __init__(self, file_path: str) -> None:
        super().__init__()
        self._file_path = Path(file_path).expanduser()
        if not self._file_path.is_absolute():
            self._file_path = (Path.cwd() / self._file_path).resolve()
        self._lock = threading.RLock()
        self._load_from_disk()

    @property
    def file_path(self) -> Path:
        return self._file_path

    def setup(self) -> None:
        self._file_path.parent.mkdir(parents=True, exist_ok=True)

    def close(self) -> None:
        with self._lock:
            self._flush_to_disk()

    def __enter__(self) -> "WorkspaceFileCheckpointer":
        self.setup()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Any,
        metadata: Any,
        new_versions: Any,
    ) -> RunnableConfig:
        with self._lock:
            result = super().put(config, checkpoint, metadata, new_versions)
            self._flush_to_disk()
            return result

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        with self._lock:
            super().put_writes(config, writes, task_id, task_path)
            self._flush_to_disk()

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Any,
        metadata: Any,
        new_versions: Any,
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

    def delete_thread(self, thread_id: str) -> None:
        with self._lock:
            super().delete_thread(thread_id)
            self._flush_to_disk()

    async def adelete_thread(self, thread_id: str) -> None:
        self.delete_thread(thread_id)

    def _load_from_disk(self) -> None:
        if not self._file_path.exists():
            return

        try:
            with self._file_path.open("rb") as f:
                payload = pickle.load(f)
        except Exception:
            logger.exception("Failed to load workspace short-term memory file: %s", self._file_path)
            return

        if not isinstance(payload, dict):
            logger.warning("Ignored invalid short-term memory payload: %s", self._file_path)
            return

        storage = defaultdict(lambda: defaultdict(dict))
        raw_storage = payload.get("storage", {})
        if isinstance(raw_storage, dict):
            for thread_id, ns_payload in raw_storage.items():
                if not isinstance(ns_payload, dict):
                    continue
                ns_store = defaultdict(dict)
                for checkpoint_ns, checkpoints in ns_payload.items():
                    if isinstance(checkpoints, dict):
                        ns_store[str(checkpoint_ns)] = dict(checkpoints)
                storage[str(thread_id)] = ns_store
        self.storage = storage

        writes = defaultdict(dict)
        raw_writes = payload.get("writes", {})
        if isinstance(raw_writes, dict):
            for key, value in raw_writes.items():
                if isinstance(key, tuple) and isinstance(value, dict):
                    writes[key] = dict(value)
        self.writes = writes

        blobs = {}
        raw_blobs = payload.get("blobs", {})
        if isinstance(raw_blobs, dict):
            for key, value in raw_blobs.items():
                if isinstance(key, tuple):
                    blobs[key] = value
        self.blobs = blobs
        logger.info("Loaded workspace short-term memory from %s", self._file_path)

    def _flush_to_disk(self) -> None:
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "storage": {
                thread_id: {checkpoint_ns: dict(checkpoints) for checkpoint_ns, checkpoints in ns_map.items()}
                for thread_id, ns_map in self.storage.items()
            },
            "writes": dict(self.writes),
            "blobs": dict(self.blobs),
        }
        with self._interprocess_lock():
            temp_path = self._write_temp_payload(payload)
            try:
                self._replace_with_retry(temp_path)
            finally:
                if temp_path.exists():
                    temp_path.unlink(missing_ok=True)

    def _write_temp_payload(self, payload: dict[str, Any]) -> Path:
        with NamedTemporaryFile(
            mode="wb",
            suffix=".tmp",
            prefix=f"{self._file_path.name}.",
            dir=self._file_path.parent,
            delete=False,
        ) as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            return Path(f.name)

    def _replace_with_retry(self, temp_path: Path) -> None:
        last_error: PermissionError | None = None
        for attempt in range(1, _REPLACE_RETRY_COUNT + 1):
            try:
                temp_path.replace(self._file_path)
                if attempt > 1:
                    logger.warning(
                        "Retried replacing short-term memory file successfully on attempt %s: %s",
                        attempt,
                        self._file_path,
                    )
                return
            except PermissionError as exc:
                last_error = exc
                if attempt == _REPLACE_RETRY_COUNT:
                    break
                logger.warning(
                    "Retrying short-term memory file replace after PermissionError "
                    "(attempt %s/%s): %s",
                    attempt,
                    _REPLACE_RETRY_COUNT,
                    self._file_path,
                )
                time.sleep(_REPLACE_RETRY_DELAY_SEC)
        assert last_error is not None
        raise last_error

    @contextmanager
    def _interprocess_lock(self) -> Any:
        lock_path = self._file_path.with_suffix(self._file_path.suffix + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+b") as lock_file:
            self._acquire_lock(lock_file)
            try:
                yield
            finally:
                self._release_lock(lock_file)

    def _acquire_lock(self, lock_file: Any) -> None:
        if msvcrt is None:
            return

        deadline = time.monotonic() + _LOCK_TIMEOUT_SEC
        while True:
            try:
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                return
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out acquiring short-term memory lock: {self._file_path}")
                time.sleep(_LOCK_RETRY_DELAY_SEC)

    def _release_lock(self, lock_file: Any) -> None:
        if msvcrt is None:
            return
        try:
            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            logger.warning("Failed to release short-term memory lock cleanly: %s", self._file_path)


def resolve_short_term_memory_path(configured_path: str) -> str:
    target = Path(configured_path).expanduser()
    if not target.is_absolute():
        target = (Path.cwd() / target).resolve()
    return str(target)
