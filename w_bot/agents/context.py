from __future__ import annotations

from .openclaw_profile import OpenClawProfileLoader
from .skills import SkillsLoader


class ContextBuilder:
    def __init__(
        self,
        *,
        skills_loader: SkillsLoader | None = None,
        openclaw_profile_loader: OpenClawProfileLoader | None = None,
    ) -> None:
        """初始化对象并保存运行所需依赖。
        
        Args:
            skills_loader: 技能加载器实例，用于读取 always 技能和技能摘要。
            openclaw_profile_loader: OpenClaw 档案加载器，用于注入人格与操作约束上下文。
        """
        self._skills_loader = skills_loader
        self._openclaw_profile_loader = openclaw_profile_loader

    def build_static_system_prompt(
        self,
        *,
        base_prompt: str,
    ) -> str:
        """组装回合内可复用的系统提示词固定部分。
        
        Args:
            base_prompt: 基础系统提示词模板。
        """
        blocks: list[str] = [base_prompt.strip()]
        if self._openclaw_profile_loader is not None:
            profile_context = self._openclaw_profile_loader.render_profile_context().strip()
            if profile_context:
                blocks.append(f"OpenClaw 档案上下文:\n{profile_context}")

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
            "规则：优先用户点名 skill；否则按意图匹配最小必要 skill；"
            "命中后先 read_file 读取 SKILL.md 全文再执行；"
            "未使用 skill 时在答复中给一句原因。\n"
            f"{summary}"
        )
        return "\n\n".join(blocks)

    def build_system_prompt(
        self,
        *,
        base_prompt: str,
        memory_context: str,
        conversation_summary: str = "",
    ) -> str:
        """组装完整系统提示词，合并固定部分与动态上下文。"""
        blocks = [
            self.build_static_system_prompt(base_prompt=base_prompt),
            f"已检索到的长期记忆:\n{memory_context or '无'}",
        ]
        if conversation_summary.strip():
            blocks.append(f"会话摘要（历史压缩）:\n{conversation_summary.strip()}")
        return "\n\n".join(blocks)
