from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime

from .logging_config import get_logger

logger = get_logger(__name__)

RECENT_SESSIONS_LIMIT = 20


@dataclass
class SessionRecord:
    session_id: str
    updated_at: str
    title: str = ""
    workspace_root: str = ""
    last_phase: str = ""
    last_action: str = ""
    last_error: str = ""
    task_count: int = 0


def upsert_session_record(
    records: list[SessionRecord],
    session_id: str,
    *,
    title: str = "",
    workspace_root: str = "",
    last_phase: str = "",
    last_action: str = "",
    last_error: str = "",
    task_count: int = 0,
) -> list[SessionRecord]:
    now = datetime.now().isoformat(timespec="seconds")
    deduped = [record for record in records if record.session_id != session_id]
    deduped.insert(
        0,
        SessionRecord(
            session_id=session_id,
            updated_at=now,
            title=title,
            workspace_root=workspace_root,
            last_phase=last_phase,
            last_action=last_action,
            last_error=last_error,
            task_count=max(0, int(task_count or 0)),
        ),
    )
    return deduped[:RECENT_SESSIONS_LIMIT]


class SessionStateStore:
    def __init__(self, file_path: str) -> None:
        self._file_path = file_path

    def load(self) -> str | None:
        if not os.path.exists(self._file_path):
            return None

        try:
            with open(self._file_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            logger.exception("Failed to load session state file: %s", self._file_path)
            return None

        current_session_id = None
        if isinstance(payload, dict):
            current_session_id = payload.get("current_session_id") or payload.get("session_id")
        if isinstance(current_session_id, str) and current_session_id.strip():
            return current_session_id.strip()
        return None

    def save(
        self,
        session_id: str,
        *,
        title: str = "",
        workspace_root: str = "",
        last_phase: str = "",
        last_action: str = "",
        last_error: str = "",
        task_count: int = 0,
    ) -> None:
        folder = os.path.dirname(os.path.abspath(self._file_path))
        if folder and not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

        recent = upsert_session_record(
            self.list_recent(),
            session_id,
            title=title,
            workspace_root=workspace_root,
            last_phase=last_phase,
            last_action=last_action,
            last_error=last_error,
            task_count=task_count,
        )
        payload = {
            "current_session_id": session_id,
            "recent_sessions": [
                {
                    "session_id": record.session_id,
                    "updated_at": record.updated_at,
                    "title": record.title,
                    "workspace_root": record.workspace_root,
                    "last_phase": record.last_phase,
                    "last_action": record.last_action,
                    "last_error": record.last_error,
                    "task_count": record.task_count,
                }
                for record in recent[:RECENT_SESSIONS_LIMIT]
            ],
        }
        with open(self._file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

    def list_recent(self) -> list[SessionRecord]:
        if not os.path.exists(self._file_path):
            return []

        try:
            with open(self._file_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            logger.exception("Failed to list session state file: %s", self._file_path)
            return []

        if not isinstance(payload, dict):
            return []

        raw_recent = payload.get("recent_sessions")
        records: list[SessionRecord] = []
        if isinstance(raw_recent, list):
            for item in raw_recent:
                if not isinstance(item, dict):
                    continue
                session_id = str(item.get("session_id") or "").strip()
                updated_at = str(item.get("updated_at") or "").strip()
                title = str(item.get("title") or "").strip()
                workspace_root = str(item.get("workspace_root") or "").strip()
                last_phase = str(item.get("last_phase") or "").strip()
                last_action = str(item.get("last_action") or "").strip()
                last_error = str(item.get("last_error") or "").strip()
                task_count = item.get("task_count") or 0
                if session_id:
                    records.append(
                        SessionRecord(
                            session_id=session_id,
                            updated_at=updated_at or datetime.now().isoformat(timespec="seconds"),
                            title=title,
                            workspace_root=workspace_root,
                            last_phase=last_phase,
                            last_action=last_action,
                            last_error=last_error,
                            task_count=max(0, int(task_count or 0)),
                        )
                    )

        legacy_session_id = payload.get("session_id")
        current_session_id = payload.get("current_session_id") or legacy_session_id
        if isinstance(current_session_id, str) and current_session_id.strip():
            current_session_id = current_session_id.strip()
            if not any(record.session_id == current_session_id for record in records):
                records.insert(
                    0,
                    SessionRecord(
                        session_id=current_session_id,
                        updated_at=datetime.now().isoformat(timespec="seconds"),
                    ),
                )
        return records[:RECENT_SESSIONS_LIMIT]
