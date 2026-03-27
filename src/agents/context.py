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
            "执行规则："
            "A) 先检查是否有与用户请求匹配的可用 skill；"
            "B) 若用户显式点名 skill，且该 skill 可用，必须优先使用；"
            "C) 命中后先使用 read_file 读取对应 SKILL.md 全文，再按其步骤执行；"
            "D) 若多个 skill 同时命中，选择最小必要集合，不做无关 skill 扩展；"
            "E) 若未命中或不可用，在最终答复里说明未使用原因。\n"
            f"{summary}"
        )
        return "\n\n".join(blocks)
