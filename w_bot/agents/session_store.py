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


def upsert_session_record(records: list[SessionRecord], session_id: str) -> list[SessionRecord]:
    now = datetime.now().isoformat(timespec="seconds")
    deduped = [record for record in records if record.session_id != session_id]
    deduped.insert(0, SessionRecord(session_id=session_id, updated_at=now))
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

    def save(self, session_id: str) -> None:
        folder = os.path.dirname(os.path.abspath(self._file_path))
        if folder and not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

        recent = upsert_session_record(self.list_recent(), session_id)
        payload = {
            "current_session_id": session_id,
            "recent_sessions": [
                {"session_id": record.session_id, "updated_at": record.updated_at}
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
                if session_id:
                    records.append(
                        SessionRecord(
                            session_id=session_id,
                            updated_at=updated_at or datetime.now().isoformat(timespec="seconds"),
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
