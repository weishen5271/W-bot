"""Cron tool for scheduling reminders and tasks."""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from w_bot.agents.tools.base import Tool
from w_bot.agents.tools.common import read_json_file


class CronTool(Tool):
    """Tool to schedule reminders and recurring tasks."""

    def __init__(self, workspace_root: Path):
        self._jobs_file = workspace_root / ".w_bot_cron_jobs.json"

    @property
    def name(self) -> str:
        return "cron"

    @property
    def description(self) -> str:
        return "Schedule reminders and recurring tasks. Actions: add, list, remove."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["add", "list", "remove"], "description": "Action to perform"},
                "message": {"type": "string", "description": "Reminder message (for add)"},
                "every_seconds": {"type": "integer", "description": "Interval in seconds"},
                "cron_expr": {"type": "string", "description": "Cron expression"},
                "at": {"type": "string", "description": "ISO datetime for one-time execution"},
                "job_id": {"type": "string", "description": "Job ID (for remove)"},
            },
            "required": ["action"],
        }

    async def execute(
        self,
        action: str,
        message: str = "",
        every_seconds: int | None = None,
        cron_expr: str | None = None,
        at: str | None = None,
        job_id: str | None = None,
        **kwargs: Any,
    ) -> str:
        jobs = read_json_file(self._jobs_file, default=[])
        if not isinstance(jobs, list):
            jobs = []

        if action == "list":
            return json.dumps(jobs, ensure_ascii=False, indent=2)
        if action == "add":
            if not message:
                return "Error: message is required for add"
            if not any([every_seconds, cron_expr, at]):
                return "Error: either every_seconds, cron_expr, or at is required"
            job = {
                "id": uuid.uuid4().hex,
                "message": message,
                "every_seconds": every_seconds,
                "cron_expr": cron_expr,
                "at": at,
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }
            jobs.append(job)
            self._jobs_file.write_text(json.dumps(jobs, ensure_ascii=False, indent=2), encoding="utf-8")
            return f"Created job '{message[:30]}' (id: {job['id']})"
        if action == "remove":
            if not job_id:
                return "Error: job_id is required for remove"
            new_jobs = [job for job in jobs if job.get("id") != job_id]
            self._jobs_file.write_text(json.dumps(new_jobs, ensure_ascii=False, indent=2), encoding="utf-8")
            if len(new_jobs) == len(jobs):
                return f"Job {job_id} not found"
            return f"Removed job {job_id}"
        return f"Unknown action: {action}"
