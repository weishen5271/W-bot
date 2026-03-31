from __future__ import annotations

from pathlib import Path

from .openclaw_profile import OpenClawProfileLoader
from .skills import SkillsLoader


class ContextBuilder:
    def __init__(
        self,
        *,
        skills_loader: SkillsLoader | None = None,
        openclaw_profile_loader: OpenClawProfileLoader | None = None,
    ) -> None:
        self._skills_loader = skills_loader
        self._openclaw_profile_loader = openclaw_profile_loader

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
                "The following skills extend your capabilities. To use a skill, read its SKILL.md file using the read_file tool.\n"
                "Skills with available=\"false\" need dependencies installed first.\n\n"
                f"{summary}"
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
