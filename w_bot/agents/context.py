from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path
from typing import Iterable

from .config import TokenOptimizationSettings
from .openclaw_profile import OpenClawProfileLoader
from .skills import SkillsLoader


class ContextBuilder:
    def __init__(
        self,
        *,
        skills_loader: SkillsLoader | None = None,
        openclaw_profile_loader: OpenClawProfileLoader | None = None,
        token_optimization_settings: TokenOptimizationSettings | None = None,
    ) -> None:
        self._skills_loader = skills_loader
        self._openclaw_profile_loader = openclaw_profile_loader
        self._token_opt = token_optimization_settings
        self._git_snapshot_cache: str | None = None
        self._project_instruction_cache: str | None = None

    def build_static_system_prompt(
        self,
        *,
        base_prompt: str,
    ) -> str:
        blocks: list[str] = [base_prompt.strip()]
        if self._openclaw_profile_loader is not None:
            profile_context = self._openclaw_profile_loader.render_compact_profile_context().strip()
            if profile_context:
                profile_root = str(Path(self._openclaw_profile_loader.root_dir).resolve())
                blocks.append(
                    "# OpenClaw Profile\n"
                    f"- Profile root: {profile_root}\n"
                    "- Treat profile content as stable project guidance and persona context.\n"
                    "- Prefer following these constraints before using ad-hoc assumptions.\n\n"
                    f"{profile_context}"
                )

        if self._skills_loader is None:
            return "\n\n---\n\n".join(blocks)

        always_skills = self._skills_loader.get_always_skills()
        if always_skills:
            always_content = self._skills_loader.load_skills_for_context([skill.name for skill in always_skills])
            if always_content:
                blocks.append(f"# Active Skills\n\n{always_content}")

        summary = self._skills_loader.build_skills_summary()
        if summary:
            blocks.append(
                "# Skills\n\n"
                "The following skills extend your capabilities. Prefer using the run_skill tool to execute a matching skill in an isolated subagent.\n"
                "Only fall back to manually reading SKILL.md when you need to inspect or compare the skill definition first.\n"
                "Skills with available=\"false\" need dependencies installed first.\n\n"
                f"{summary}"
            )
        return "\n\n---\n\n".join(blocks)

    def build_turn_system_prompt(
        self,
        *,
        base_prompt: str,
        budget_snapshot: str = "",
    ) -> str:
        if self._token_opt is None or not self._token_opt.enable_dynamic_system_context:
            return self.build_static_system_prompt(base_prompt=base_prompt)

        blocks = [self.build_static_system_prompt(base_prompt=base_prompt)]
        runtime_block = self._build_runtime_context_block()
        if runtime_block:
            blocks.append(runtime_block)
        if budget_snapshot.strip():
            blocks.append(
                "# Token Budget\n"
                "Use this snapshot to decide whether to stay concise, summarize history sooner, and avoid unnecessary tool churn.\n\n"
                f"{budget_snapshot.strip()}"
            )
        return "\n\n---\n\n".join(blocks)

    def build_system_prompt(
        self,
        *,
        base_prompt: str,
        memory_context: str,
        conversation_summary: str = "",
    ) -> str:
        blocks = [
            self.build_static_system_prompt(base_prompt=base_prompt),
            "# Retrieved Memory\n"
            "Use the following retrieved memory only as supporting context. Prefer recent verified conversation state when conflicts exist.\n\n"
            f"{memory_context or '(none)'}",
        ]
        if conversation_summary.strip():
            blocks.append(
                "# Conversation Summary\n"
                "This is a compact summary of earlier turns. Use it to maintain continuity while prioritizing the latest messages and tool results.\n\n"
                f"{conversation_summary.strip()}"
            )
        return "\n\n---\n\n".join(blocks)

    def _build_runtime_context_block(self) -> str:
        items = [
            f"- Today's date is {datetime.now().date().isoformat()}.",
            f"- Workspace root: {Path.cwd().resolve()}",
        ]
        git_snapshot = self._get_git_snapshot()
        if git_snapshot:
            items.append("- Git snapshot is included below and represents a point-in-time view.")
        project_instructions = self._get_project_instruction_context()
        blocks = ["# Runtime Context", "\n".join(items)]
        if git_snapshot:
            blocks.append(f"## Git Snapshot\n{git_snapshot}")
        if project_instructions:
            blocks.append(f"## Project Instructions\n{project_instructions}")
        return "\n\n".join(blocks)

    def _get_git_snapshot(self) -> str:
        if self._git_snapshot_cache is not None:
            return self._git_snapshot_cache
        if self._token_opt is None or not self._token_opt.enable_git_status:
            self._git_snapshot_cache = ""
            return self._git_snapshot_cache
        cwd = Path.cwd()
        git_dir = cwd / ".git"
        if not git_dir.exists():
            self._git_snapshot_cache = ""
            return self._git_snapshot_cache

        branch = self._run_git_command(["rev-parse", "--abbrev-ref", "HEAD"])
        status = self._run_git_command(["status", "--short"])
        recent_commits = self._run_git_command(["log", "--oneline", "-n", "5"])
        if not branch and not status and not recent_commits:
            self._git_snapshot_cache = ""
            return self._git_snapshot_cache

        status_text = status or "(clean)"
        max_chars = max(200, int(self._token_opt.git_status_max_chars))
        if len(status_text) > max_chars:
            status_text = status_text[:max_chars].rstrip() + "\n... (truncated)"
        lines = [
            f"Current branch: {branch or '(unknown)'}",
            f"Status:\n{status_text}",
        ]
        if recent_commits:
            lines.append(f"Recent commits:\n{recent_commits}")
        self._git_snapshot_cache = "\n\n".join(lines)
        return self._git_snapshot_cache

    def _get_project_instruction_context(self) -> str:
        if self._project_instruction_cache is not None:
            return self._project_instruction_cache
        if self._token_opt is None or not self._token_opt.enable_project_instruction_scan:
            self._project_instruction_cache = ""
            return self._project_instruction_cache

        instruction_files = tuple(self._iter_project_instruction_files())
        if not instruction_files:
            self._project_instruction_cache = ""
            return self._project_instruction_cache

        blocks: list[str] = []
        for path in instruction_files:
            text = path.read_text(encoding="utf-8").strip()
            if not text:
                continue
            blocks.append(f"### {path.name}\n{_truncate_text(text, 4000)}")
        self._project_instruction_cache = "\n\n".join(blocks)
        return self._project_instruction_cache

    def _iter_project_instruction_files(self) -> Iterable[Path]:
        seen: set[Path] = set()
        file_names = tuple(self._token_opt.project_instruction_files) if self._token_opt else ()
        for directory in [Path.cwd(), *Path.cwd().parents]:
            for name in file_names:
                candidate = directory / name
                if not candidate.exists() or not candidate.is_file():
                    continue
                resolved = candidate.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                yield resolved

    @staticmethod
    def _run_git_command(args: list[str]) -> str:
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=Path.cwd(),
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=2,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return ""
        if completed.returncode != 0:
            return ""
        return completed.stdout.strip()


def _truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n... (truncated)"
