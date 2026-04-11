from __future__ import annotations

import json
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .logging_config import get_logger

logger = get_logger(__name__)


def _render_escalation_request(request: EscalationRequest) -> str:
    """Format an escalation request as a human-readable string with English labels."""
    lines = [
        f"[bold cyan]Escalation[/bold cyan]: {request.id}",
        f"- status: {request.status}",
        f"- risk_type: {request.risk_type}",
        f"- working_dir: {request.working_dir}",
        f"- command: {request.command}",
    ]
    if request.justification:
        lines.append(f"- justification: {request.justification}")
    if request.prefix_rule:
        lines.append(f"- prefix_rule: {' '.join(request.prefix_rule)}")
    if request.denial_reason:
        lines.append(f"- denial_reason: {request.denial_reason}")
    return "\n".join(lines)


def _render_escalation_request_simple(request: EscalationRequest) -> str:
    """Format an escalation request as a human-readable string with Chinese labels."""
    lines = [
        f"提权请求: {request.id}",
        f"状态: {request.status}",
        f"风险类型: {request.risk_type}",
        f"工作目录: {request.working_dir}",
        f"命令: {request.command}",
    ]
    if request.justification:
        lines.append(f"用途说明: {request.justification}")
    if request.prefix_rule:
        lines.append(f"授权前缀: {' '.join(request.prefix_rule)}")
    if request.denial_reason:
        lines.append(f"拒绝原因: {request.denial_reason}")
    return "\n".join(lines)


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _normalize_command(command: str) -> str:
    return " ".join(str(command or "").strip().split())


@dataclass
class EscalationRequest:
    id: str
    session_id: str
    command: str
    working_dir: str
    justification: str
    prefix_rule: list[str]
    risk_type: str
    status: str
    created_at: str
    updated_at: str
    approved_at: str = ""
    approved_by: str = ""
    approval_reason: str = ""
    denied_at: str = ""
    denied_by: str = ""
    denial_reason: str = ""


class EscalationManager:
    def __init__(self, file_path: str) -> None:
        self._file_path = Path(file_path).expanduser()
        if not self._file_path.is_absolute():
            self._file_path = Path.cwd() / self._file_path
        self._lock = threading.Lock()

    def create_request(
        self,
        *,
        session_id: str,
        command: str,
        working_dir: str,
        justification: str,
        prefix_rule: list[str] | None,
        risk_type: str,
    ) -> EscalationRequest:
        normalized_command = _normalize_command(command)
        normalized_dir = str(Path(working_dir).expanduser())
        normalized_justification = str(justification or "").strip()
        normalized_prefix = [str(item).strip() for item in (prefix_rule or []) if str(item).strip()]
        session_key = str(session_id or "-").strip() or "-"

        with self._lock:
            payload = self._load_unlocked()
            requests = payload.setdefault("requests", [])
            for item in requests:
                if not isinstance(item, dict):
                    continue
                if (
                    str(item.get("session_id") or "").strip() == session_key
                    and str(item.get("status") or "").strip() == "pending"
                    and _normalize_command(str(item.get("command") or "")) == normalized_command
                    and str(item.get("working_dir") or "").strip() == normalized_dir
                ):
                    return self._deserialize_request(item)

            now = _now_iso()
            request = EscalationRequest(
                id=uuid.uuid4().hex[:12],
                session_id=session_key,
                command=normalized_command,
                working_dir=normalized_dir,
                justification=normalized_justification,
                prefix_rule=normalized_prefix,
                risk_type=str(risk_type or "permission").strip() or "permission",
                status="pending",
                created_at=now,
                updated_at=now,
            )
            requests.append(asdict(request))
            self._save_unlocked(payload)
            return request

    def list_requests(
        self,
        *,
        session_id: str | None = None,
        status: str | None = None,
        limit: int = 20,
    ) -> list[EscalationRequest]:
        with self._lock:
            payload = self._load_unlocked()
        requests = payload.get("requests", [])
        if not isinstance(requests, list):
            return []
        items: list[EscalationRequest] = []
        target_session = str(session_id or "").strip()
        target_status = str(status or "").strip().lower()
        for item in reversed(requests):
            if not isinstance(item, dict):
                continue
            request = self._deserialize_request(item)
            if target_session and request.session_id != target_session:
                continue
            if target_status and request.status.lower() != target_status:
                continue
            items.append(request)
            if len(items) >= max(1, int(limit or 20)):
                break
        return items

    def get_request(self, request_id: str) -> EscalationRequest | None:
        target = str(request_id or "").strip()
        if not target:
            return None
        with self._lock:
            payload = self._load_unlocked()
            requests = payload.get("requests", [])
            if not isinstance(requests, list):
                return None
            for item in requests:
                if isinstance(item, dict) and str(item.get("id") or "").strip() == target:
                    return self._deserialize_request(item)
        return None

    def approve_request(
        self,
        *,
        request_id: str,
        approved_by: str = "",
        reason: str = "",
    ) -> EscalationRequest | None:
        target = str(request_id or "").strip()
        if not target:
            return None
        with self._lock:
            payload = self._load_unlocked()
            requests = payload.get("requests", [])
            if not isinstance(requests, list):
                return None
            for index, item in enumerate(requests):
                if not isinstance(item, dict) or str(item.get("id") or "").strip() != target:
                    continue
                updated = dict(item)
                now = _now_iso()
                updated["status"] = "approved"
                updated["approved_at"] = now
                updated["approved_by"] = str(approved_by or "").strip()
                updated["approval_reason"] = str(reason or "").strip()
                updated["updated_at"] = now
                requests[index] = updated
                self._save_unlocked(payload)
                return self._deserialize_request(updated)
        return None

    def deny_request(
        self,
        *,
        request_id: str,
        reason: str = "",
        denied_by: str = "",
    ) -> EscalationRequest | None:
        target = str(request_id or "").strip()
        if not target:
            return None
        with self._lock:
            payload = self._load_unlocked()
            requests = payload.get("requests", [])
            if not isinstance(requests, list):
                return None
            for index, item in enumerate(requests):
                if not isinstance(item, dict) or str(item.get("id") or "").strip() != target:
                    continue
                updated = dict(item)
                now = _now_iso()
                updated["status"] = "denied"
                updated["denied_at"] = now
                updated["denied_by"] = str(denied_by or "").strip()
                updated["updated_at"] = now
                updated["denial_reason"] = str(reason or "").strip()
                requests[index] = updated
                self._save_unlocked(payload)
                return self._deserialize_request(updated)
        return None

    def is_command_approved(self, *, session_id: str, command: str) -> bool:
        session_key = str(session_id or "").strip() or "-"
        normalized_command = _normalize_command(command)
        if not normalized_command:
            return False
        with self._lock:
            payload = self._load_unlocked()
        requests = payload.get("requests", [])
        if not isinstance(requests, list):
            return False
        for item in requests:
            if not isinstance(item, dict):
                continue
            if str(item.get("session_id") or "").strip() != session_key:
                continue
            if str(item.get("status") or "").strip() != "approved":
                continue
            approved_command = _normalize_command(str(item.get("command") or ""))
            if approved_command and approved_command == normalized_command:
                return True
            prefix_rule = item.get("prefix_rule")
            if self._matches_prefix_rule(normalized_command, prefix_rule):
                return True
        return False

    def render_request_summary(self, request: EscalationRequest) -> str:
        lines = [
            f"提权请求ID: {request.id}",
            f"状态: {request.status}",
            f"风险类型: {request.risk_type}",
            f"工作目录: {request.working_dir}",
            f"命令: {request.command}",
        ]
        if request.justification:
            lines.append(f"用途说明: {request.justification}")
        if request.prefix_rule:
            lines.append(f"授权前缀: {' '.join(request.prefix_rule)}")
        return "\n".join(lines)

    @staticmethod
    def _matches_prefix_rule(command: str, prefix_rule: Any) -> bool:
        if not isinstance(prefix_rule, list) or not prefix_rule:
            return False
        try:
            import shlex

            command_tokens = shlex.split(command, posix=False)
        except ValueError:
            command_tokens = command.split()
        prefix_tokens = [str(item).strip() for item in prefix_rule if str(item).strip()]
        if not prefix_tokens or len(command_tokens) < len(prefix_tokens):
            return False
        for index, prefix_token in enumerate(prefix_tokens):
            if command_tokens[index].lower() != prefix_token.lower():
                return False
        return True

    def _load_unlocked(self) -> dict[str, Any]:
        if not self._file_path.exists():
            return {"requests": []}
        try:
            payload = json.loads(self._file_path.read_text(encoding="utf-8"))
        except Exception:
            logger.exception("Failed to load escalation state: %s", self._file_path)
            return {"requests": []}
        return payload if isinstance(payload, dict) else {"requests": []}

    def _save_unlocked(self, payload: dict[str, Any]) -> None:
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        self._file_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _deserialize_request(item: dict[str, Any]) -> EscalationRequest:
        prefix_rule = item.get("prefix_rule")
        return EscalationRequest(
            id=str(item.get("id") or "").strip(),
            session_id=str(item.get("session_id") or "").strip(),
            command=_normalize_command(str(item.get("command") or "")),
            working_dir=str(item.get("working_dir") or "").strip(),
            justification=str(item.get("justification") or "").strip(),
            prefix_rule=[str(part).strip() for part in prefix_rule] if isinstance(prefix_rule, list) else [],
            risk_type=str(item.get("risk_type") or "permission").strip(),
            status=str(item.get("status") or "pending").strip(),
            created_at=str(item.get("created_at") or "").strip(),
            updated_at=str(item.get("updated_at") or "").strip(),
            approved_at=str(item.get("approved_at") or "").strip(),
            approved_by=str(item.get("approved_by") or "").strip(),
            approval_reason=str(item.get("approval_reason") or "").strip(),
            denied_at=str(item.get("denied_at") or "").strip(),
            denied_by=str(item.get("denied_by") or "").strip(),
            denial_reason=str(item.get("denial_reason") or "").strip(),
        )
