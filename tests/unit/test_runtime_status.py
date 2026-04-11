"""Unit tests for runtime_status module."""

from __future__ import annotations

import pytest

from w_bot.agents.core.runtime_status import (
    RuntimeStatusSnapshot,
    TaskBoardStatus,
    infer_phase_from_text,
)


class TestTaskBoardStatus:
    """Tests for TaskBoardStatus dataclass."""

    def test_default_values(self) -> None:
        """Test TaskBoardStatus initializes with correct defaults."""
        status = TaskBoardStatus()
        assert status.running == 0
        assert status.pending == 0
        assert status.completed == 0
        assert status.failed == 0
        assert status.highlighted_tasks == []

    def test_with_values(self) -> None:
        """Test TaskBoardStatus with custom values."""
        status = TaskBoardStatus(
            running=2,
            pending=1,
            completed=5,
            failed=1,
            highlighted_tasks=["task1", "task2"],
        )
        assert status.running == 2
        assert status.pending == 1
        assert status.completed == 5
        assert status.failed == 1
        assert len(status.highlighted_tasks) == 2


class TestRuntimeStatusSnapshot:
    """Tests for RuntimeStatusSnapshot dataclass."""

    def test_initialization(self) -> None:
        """Test RuntimeStatusSnapshot initializes correctly."""
        snapshot = RuntimeStatusSnapshot(session_id="test-session")
        assert snapshot.session_id == "test-session"
        assert snapshot.phase == "idle"
        assert snapshot.phase_label == "空闲"
        assert snapshot.input_tokens == 0
        assert snapshot.output_tokens == 0
        assert snapshot.total_cost == 0.0

    def test_set_session(self) -> None:
        """Test set_session updates session_id."""
        snapshot = RuntimeStatusSnapshot(session_id="old-session")
        snapshot.set_session("new-session")
        assert snapshot.session_id == "new-session"

    def test_set_phase(self) -> None:
        """Test set_phase updates phase and label."""
        snapshot = RuntimeStatusSnapshot(session_id="test")
        snapshot.set_phase("running", "处理中", recent_action="执行任务")
        assert snapshot.phase == "running"
        assert snapshot.phase_label == "处理中"
        assert snapshot.recent_action == "执行任务"

    def test_set_phase_clears_error(self) -> None:
        """Test set_phase clears last error when phase is not failed."""
        snapshot = RuntimeStatusSnapshot(session_id="test")
        snapshot.mark_failed("some error", phase="running")
        assert snapshot.last_error == "some error"
        snapshot.set_phase("running", "处理中")
        assert snapshot.last_error == ""

    def test_begin_turn(self) -> None:
        """Test begin_turn resets state for new turn."""
        snapshot = RuntimeStatusSnapshot(session_id="test")
        snapshot.set_phase("failed", "失败")
        snapshot.last_error = "previous error"
        snapshot.begin_turn(recent_action="starting new turn")
        assert snapshot.phase == "running"
        assert snapshot.recent_action == "starting new turn"
        assert snapshot.last_error == ""

    def test_set_waiting_permission(self) -> None:
        """Test set_waiting with permission reason."""
        snapshot = RuntimeStatusSnapshot(session_id="test")
        snapshot.set_waiting("permission", action="等待用户审批")
        assert snapshot.waiting_reason == "permission"
        assert snapshot.phase == "waiting_permission"
        assert snapshot.phase_label == "等待授权中"

    def test_set_waiting_subagent(self) -> None:
        """Test set_waiting with subagent reason."""
        snapshot = RuntimeStatusSnapshot(session_id="test")
        snapshot.set_waiting("subagent")
        assert snapshot.waiting_reason == "subagent"
        assert snapshot.phase == "waiting_subagent"

    def test_set_waiting_user(self) -> None:
        """Test set_waiting with user reason."""
        snapshot = RuntimeStatusSnapshot(session_id="test")
        snapshot.set_waiting("user")
        assert snapshot.waiting_reason == "user"
        assert snapshot.phase == "waiting_user"

    def test_mark_failed(self) -> None:
        """Test mark_failed sets error state."""
        snapshot = RuntimeStatusSnapshot(session_id="test")
        snapshot.mark_failed("tool execution failed", phase="executing")
        assert snapshot.last_error == "tool execution failed"
        assert snapshot.last_error_phase == "executing"
        assert snapshot.phase == "failed"

    def test_update_usage(self) -> None:
        """Test update_usage updates token counts and cost."""
        snapshot = RuntimeStatusSnapshot(session_id="test")
        snapshot.update_usage(input_tokens=100, output_tokens=50, total_cost=0.05)
        assert snapshot.input_tokens == 100
        assert snapshot.output_tokens == 50
        assert snapshot.total_cost == 0.05

    def test_update_usage_negative_values(self) -> None:
        """Test update_usage ignores negative values (clamps to 0)."""
        snapshot = RuntimeStatusSnapshot(session_id="test")
        snapshot.update_usage(input_tokens=100, output_tokens=50, total_cost=0.05)
        snapshot.update_usage(input_tokens=-10, output_tokens=-5, total_cost=-0.01)
        # Negative values are clamped to 0, not ignored
        assert snapshot.input_tokens == 0
        assert snapshot.output_tokens == 0
        assert snapshot.total_cost == 0.0

    def test_refresh_tasks_empty(self) -> None:
        """Test refresh_tasks with empty list."""
        snapshot = RuntimeStatusSnapshot(session_id="test")
        snapshot.refresh_tasks([])
        assert snapshot.tasks.running == 0
        assert snapshot.tasks.pending == 0
        assert snapshot.tasks.completed == 0
        assert snapshot.tasks.failed == 0

    def test_refresh_tasks_running(self) -> None:
        """Test refresh_tasks counts running jobs."""
        snapshot = RuntimeStatusSnapshot(session_id="test")
        jobs = [
            {"id": "job1", "status": "running", "label": "worker"},
            {"id": "job2", "status": "running", "agent_type": "explore"},
        ]
        snapshot.refresh_tasks(jobs)
        assert snapshot.tasks.running == 2
        assert len(snapshot.tasks.highlighted_tasks) == 2

    def test_refresh_tasks_pending(self) -> None:
        """Test refresh_tasks counts pending jobs."""
        snapshot = RuntimeStatusSnapshot(session_id="test")
        jobs = [
            {"id": "job1", "status": "pending"},
            {"id": "job2", "status": "pending"},
        ]
        snapshot.refresh_tasks(jobs)
        assert snapshot.tasks.pending == 2

    def test_refresh_tasks_completed(self) -> None:
        """Test refresh_tasks counts completed jobs."""
        snapshot = RuntimeStatusSnapshot(session_id="test")
        jobs = [
            {"id": "job1", "status": "completed"},
        ]
        snapshot.refresh_tasks(jobs)
        assert snapshot.tasks.completed == 1

    def test_refresh_tasks_failed(self) -> None:
        """Test refresh_tasks counts failed jobs."""
        snapshot = RuntimeStatusSnapshot(session_id="test")
        jobs = [
            {"id": "job1", "status": "failed"},
            {"id": "job2", "status": "timeout"},
        ]
        snapshot.refresh_tasks(jobs)
        assert snapshot.tasks.failed == 2

    def test_refresh_tasks_highlight_limit(self) -> None:
        """Test refresh_tasks limits highlighted tasks to 2."""
        snapshot = RuntimeStatusSnapshot(session_id="test")
        jobs = [
            {"id": f"job{i}", "status": "running", "label": f"task{i}"}
            for i in range(5)
        ]
        snapshot.refresh_tasks(jobs)
        assert len(snapshot.tasks.highlighted_tasks) == 2

    def test_spinner_text_idle(self) -> None:
        """Test spinner_text with idle phase."""
        snapshot = RuntimeStatusSnapshot(session_id="test")
        text = snapshot.spinner_text()
        assert "空闲" in text

    def test_spinner_text_with_phase_label(self) -> None:
        """Test spinner_text includes phase label."""
        snapshot = RuntimeStatusSnapshot(session_id="test")
        snapshot.set_phase("running", "处理中")
        text = snapshot.spinner_text()
        assert "处理中" in text

    def test_progress_lines(self) -> None:
        """Test progress_lines returns list of status lines."""
        snapshot = RuntimeStatusSnapshot(session_id="test")
        snapshot.set_phase("running", "处理中")
        lines = snapshot.progress_lines()
        assert isinstance(lines, list)
        assert len(lines) >= 1


class TestInferPhaseFromText:
    """Tests for infer_phase_from_text function."""

    @pytest.mark.parametrize("text,expected_phase", [
        ("执行失败", "failed"),
        ("error occurred", "failed"),
        ("failed to execute", "failed"),
    ])
    def test_failure_phases(self, text: str, expected_phase: str) -> None:
        """Test inference of failure phases."""
        phase, _ = infer_phase_from_text(text)
        assert phase == expected_phase

    @pytest.mark.parametrize("text,expected_phase", [
        ("等待审批中", "waiting_permission"),
        ("approval required", "waiting_permission"),  # has "approval" and "wait"
    ])
    def test_permission_phases(self, text: str, expected_phase: str) -> None:
        """Test inference of permission waiting phases."""
        phase, _ = infer_phase_from_text(text)
        assert phase == expected_phase

    def test_waiting_authorization_returns_waiting(self) -> None:
        """Test that '等待授权' without 'wait' returns generic waiting."""
        # "等待授权" doesn't have "审批" or "提权", and "wait" is not in it
        phase, _ = infer_phase_from_text("等待授权")
        assert phase == "waiting"

    @pytest.mark.parametrize("text,expected_phase", [
        ("等待子任务返回", "waiting_subagent"),
        ("waiting for subagent", "waiting_subagent"),
    ])
    def test_subagent_phases(self, text: str, expected_phase: str) -> None:
        """Test inference of subagent waiting phases."""
        phase, _ = infer_phase_from_text(text)
        assert phase == expected_phase

    def test_background_task_without_wait_returns_running(self) -> None:
        """Test that '后台任务进行中' without 'wait' returns running."""
        # "后台任务进行中" has no "wait" so it doesn't enter the waiting branch
        phase, _ = infer_phase_from_text("后台任务进行中")
        assert phase == "running"

    @pytest.mark.parametrize("text,expected_phase", [
        ("搜索代码中", "searching"),
        ("grepping files", "searching"),
        ("find in files", "searching"),
    ])
    def test_searching_phase(self, text: str, expected_phase: str) -> None:
        """Test inference of searching phase."""
        phase, _ = infer_phase_from_text(text)
        assert phase == expected_phase

    @pytest.mark.parametrize("text,expected_phase", [
        ("读取文件中", "reading"),
        ("loading file", "reading"),
    ])
    def test_reading_phase(self, text: str, expected_phase: str) -> None:
        """Test inference of reading phase."""
        phase, _ = infer_phase_from_text(text)
        assert phase == expected_phase

    @pytest.mark.parametrize("text,expected_phase", [
        ("调用工具中", "executing"),
        ("executing command", "executing"),
        ("执行shell命令", "executing"),
    ])
    def test_executing_phase(self, text: str, expected_phase: str) -> None:
        """Test inference of executing phase."""
        phase, _ = infer_phase_from_text(text)
        assert phase == expected_phase

    @pytest.mark.parametrize("text,expected_phase", [
        ("整理结果中", "summarizing"),
        ("rendering response", "summarizing"),
    ])
    def test_summarizing_phase(self, text: str, expected_phase: str) -> None:
        """Test inference of summarizing phase."""
        phase, _ = infer_phase_from_text(text)
        assert phase == expected_phase

    @pytest.mark.parametrize("text,expected_phase", [
        ("分析上下文中", "analyzing"),
        ("inspecting requirements", "analyzing"),
        ("加载记忆", "reading"),  # "load" matches "read" group first
    ])
    def test_analyzing_phase(self, text: str, expected_phase: str) -> None:
        """Test inference of analyzing phase."""
        phase, _ = infer_phase_from_text(text)
        assert phase == expected_phase

    def test_empty_text_returns_running(self) -> None:
        """Test empty text defaults to running phase."""
        phase, label = infer_phase_from_text("")
        assert phase == "running"
        assert label == "处理中"

    def test_unknown_text_returns_running(self) -> None:
        """Test unknown text defaults to running phase."""
        phase, label = infer_phase_from_text("random unknown text")
        assert phase == "running"
        assert label == "处理中"
