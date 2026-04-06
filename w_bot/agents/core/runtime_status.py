from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


def _now_ms() -> int:
    return int(time.time() * 1000)


@dataclass
class TaskBoardStatus:
    running: int = 0
    pending: int = 0
    completed: int = 0
    failed: int = 0
    highlighted_tasks: list[str] = field(default_factory=list)


@dataclass
class RuntimeStatusSnapshot:
    session_id: str
    phase: str = "idle"
    phase_label: str = "空闲"
    recent_action: str = ""
    waiting_reason: str = ""
    phase_since_ms: int = field(default_factory=_now_ms)
    last_error: str = ""
    last_error_phase: str = ""
    retry_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_cost: float = 0.0
    tasks: TaskBoardStatus = field(default_factory=TaskBoardStatus)

    def set_session(self, session_id: str) -> None:
        self.session_id = session_id

    def set_phase(self, phase: str, label: str, *, recent_action: str | None = None) -> None:
        self.phase = phase.strip() or "running"
        self.phase_label = label.strip() or "处理中"
        self.phase_since_ms = _now_ms()
        if recent_action is not None:
            self.recent_action = recent_action.strip()
        if phase != "failed":
            self.last_error = ""
            self.last_error_phase = ""

    def set_recent_action(self, recent_action: str) -> None:
        self.recent_action = (recent_action or "").strip()

    def set_waiting(self, reason: str, *, action: str = "") -> None:
        label = "等待中"
        if reason == "permission":
            label = "等待授权中"
        elif reason == "subagent":
            label = "等待子任务中"
        elif reason == "user":
            label = "等待你的输入"
        elif reason == "command":
            label = "等待命令完成"
        self.waiting_reason = reason
        self.set_phase(f"waiting_{reason}" if reason else "waiting", label, recent_action=action or self.recent_action)

    def mark_failed(self, message: str, *, phase: str | None = None) -> None:
        phase_name = phase or self.phase
        self.last_error = (message or "").strip()
        self.last_error_phase = phase_name
        self.set_phase("failed", "执行失败", recent_action=self.recent_action)

    def record_status_message(self, text: str) -> None:
        message = (text or "").strip()
        if not message:
            return
        self.recent_action = message
        phase, label = infer_phase_from_text(message)
        if phase != self.phase or label != self.phase_label:
            self.phase_since_ms = _now_ms()
        if phase.startswith("waiting_"):
            self.waiting_reason = phase.removeprefix("waiting_")
        self.phase = phase
        self.phase_label = label

    def update_usage(self, *, input_tokens: int = 0, output_tokens: int = 0, total_cost: float | None = None) -> None:
        self.input_tokens = max(0, int(input_tokens))
        self.output_tokens = max(0, int(output_tokens))
        if total_cost is not None:
            self.total_cost = max(0.0, float(total_cost))

    def refresh_tasks(self, jobs: list[dict[str, Any]] | None) -> None:
        summary = TaskBoardStatus()
        for job in jobs or []:
            status = str(job.get("status") or "").strip().lower()
            label = str(job.get("label") or job.get("agent_type") or job.get("id") or "-").strip()
            task_id = str(job.get("id") or "").strip()
            item = f"{label}#{task_id[:8]}" if task_id else label
            if status in {"running"}:
                summary.running += 1
                if len(summary.highlighted_tasks) < 2:
                    summary.highlighted_tasks.append(f"{item} 运行中")
            elif status in {"pending"}:
                summary.pending += 1
            elif status in {"completed"}:
                summary.completed += 1
            elif status in {"failed", "timeout"}:
                summary.failed += 1
                if len(summary.highlighted_tasks) < 2:
                    summary.highlighted_tasks.append(f"{item} {status}")
        self.tasks = summary

    def spinner_text(self) -> str:
        elapsed = max(0, (_now_ms() - self.phase_since_ms) // 1000)
        parts = [self.phase_label]
        if elapsed:
            parts.append(f"{elapsed}s")
        if self.recent_action:
            parts.append(f"最近动作: {self.recent_action}")
        if self.tasks.running or self.tasks.pending:
            task_summary = f"后台任务: {self.tasks.running} running"
            if self.tasks.pending:
                task_summary += f", {self.tasks.pending} pending"
            parts.append(task_summary)
        return " | ".join(parts)


def infer_phase_from_text(text: str) -> tuple[str, str]:
    normalized = (text or "").strip().lower()
    if not normalized:
        return "running", "处理中"
    if "wait" in normalized or "等待" in text:
        if "授权" in text or "permission" in normalized:
            return "waiting_permission", "等待授权中"
        if "子任务" in text or "subagent" in normalized:
            return "waiting_subagent", "等待子任务中"
        if "输入" in text or "user" in normalized:
            return "waiting_user", "等待你的输入"
        return "waiting", "等待中"
    if any(token in normalized for token in ["tool", "exec", "command", "shell"]) or "工具" in text or "执行" in text:
        return "executing", "执行命令中"
    if any(token in normalized for token in ["search", "grep", "find"]) or "搜索" in text or "检索" in text:
        return "searching", "搜索代码中"
    if any(token in normalized for token in ["read", "load", "file"]) or "读取" in text:
        return "reading", "读取文件中"
    if any(token in normalized for token in ["summary", "render", "final"]) or "总结" in text or "整理" in text:
        return "summarizing", "整理结果中"
    if any(token in normalized for token in ["memory", "prompt", "model", "context"]) or "记忆" in text or "模型" in text or "上下文" in text:
        return "analyzing", "分析上下文中"
    if "失败" in text or "error" in normalized or "failed" in normalized:
        return "failed", "执行失败"
    return "running", "处理中"
