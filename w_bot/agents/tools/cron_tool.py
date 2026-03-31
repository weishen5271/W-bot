from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path

from langchain_core.tools import StructuredTool

from .common import read_json_file


def build_cron_tool(*, workspace_root: Path) -> StructuredTool:
    def _cron(action: str, task_name: str, schedule: str = "", payload: str = "") -> str:
        jobs_file = workspace_root / ".w_bot_cron_jobs.json"
        jobs = read_json_file(jobs_file, default=[])
        if not isinstance(jobs, list):
            jobs = []

        if action == "list":
            return json.dumps(jobs, ensure_ascii=False, indent=2)

        if action == "create":
            if not schedule:
                return "schedule is required for create"
            job = {
                "id": uuid.uuid4().hex,
                "task_name": task_name,
                "schedule": schedule,
                "payload": payload,
                "enabled": True,
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }
            jobs.append(job)
            jobs_file.write_text(json.dumps(jobs, ensure_ascii=False, indent=2), encoding="utf-8")
            return f"Cron job created: id={job['id']}"

        if action == "delete":
            new_jobs = [j for j in jobs if j.get("task_name") != task_name]
            jobs_file.write_text(json.dumps(new_jobs, ensure_ascii=False, indent=2), encoding="utf-8")
            return f"Cron jobs removed: {len(jobs) - len(new_jobs)}"

        return "Unsupported action. Use list/create/delete."

    return StructuredTool.from_function(
        func=_cron,
        name="cron",
        description="Create/list/delete cron jobs when cron service is enabled.",
    )
