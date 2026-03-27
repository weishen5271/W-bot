from __future__ import annotations

from .skills import SkillsLoader


class ContextBuilder:
    def __init__(self, *, skills_loader: SkillsLoader | None = None) -> None:
        self._skills_loader = skills_loader

    def build_system_prompt(
        self,
        *,
        base_prompt: str,
        memory_context: str,
    ) -> str:
        blocks: list[str] = [base_prompt.strip(), f"已检索到的长期记忆:\n{memory_context or '无'}"]

        if self._skills_loader is None:
            return "\n\n".join(blocks)

        always_skills = self._skills_loader.get_always_skills()
        if always_skills:
            always_lines = [
                "以下是 always=true 且当前可用的 Skill 全文（已去除 frontmatter）：",
            ]
            for skill in always_skills:
                always_lines.append(f"\n[SKILL: {skill.name}]")
                always_lines.append(skill.content)
            blocks.append("\n".join(always_lines))

        summary = self._skills_loader.build_skills_summary()
        blocks.append(
            "可用 Skill 摘要如下。"
            "当任务需要某个 skill 时，先使用 read_file 读取对应 SKILL.md 全文，再按其步骤执行。\n"
            f"{summary}"
        )
        return "\n\n".join(blocks)
