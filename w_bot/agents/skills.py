from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .logging_config import get_logger

logger = get_logger(__name__)

BUILTIN_SKILLS_DIR = Path(__file__).resolve().parent / "skills_catalog"


@dataclass(frozen=True)
class SkillRequirementCheck:
    available: bool
    missing_bins: tuple[str, ...]
    missing_env: tuple[str, ...]


@dataclass(frozen=True)
class SkillSpec:
    name: str
    description: str
    path: Path
    source: str
    always: bool
    requires_bins: tuple[str, ...]
    requires_env: tuple[str, ...]
    content: str


class SkillsLoader:
    def __init__(
        self,
        *,
        workspace_skills_dir: str = "skills",
        builtin_skills_dir: str | None = None,
    ) -> None:
        """初始化对象并保存运行所需依赖。
        
        Args:
            workspace_skills_dir: 工作区技能目录路径。
            builtin_skills_dir: 内置技能目录路径。
        """
        self.workspace_skills_dir = _resolve_to_abs(Path(workspace_skills_dir))
        self.builtin_skills_dir = _resolve_to_abs(Path(builtin_skills_dir)) if builtin_skills_dir else BUILTIN_SKILLS_DIR
        self.workspace_skills_dir.mkdir(parents=True, exist_ok=True)

    def list_skills(self, *, filter_unavailable: bool = False) -> list[SkillSpec]:
        """处理list/skills相关逻辑并返回结果。
        
        Args:
            filter_unavailable: 是否过滤不可用技能。
        """
        merged: dict[str, SkillSpec] = {}
        for item in self._scan_dir(self.builtin_skills_dir, source="builtin"):
            merged[item.name] = item
        for item in self._scan_dir(self.workspace_skills_dir, source="workspace"):
            merged[item.name] = item

        ordered = sorted(merged.values(), key=lambda s: s.name)
        if not filter_unavailable:
            return ordered
        return [skill for skill in ordered if self.check_requirements(skill).available]

    def get_always_skills(self) -> list[SkillSpec]:
        """处理get/always/skills相关逻辑并返回结果。
        """
        always_skills: list[SkillSpec] = []
        for skill in self.list_skills(filter_unavailable=False):
            check = self.check_requirements(skill)
            if skill.always and check.available:
                always_skills.append(skill)
        return always_skills

    def build_skills_summary(self) -> str:
        """构建并返回目标对象。
        """
        skills = self.list_skills(filter_unavailable=False)
        if not skills:
            return "<skills>\n(no skills found)\n</skills>"

        lines = ["<skills>"]
        for skill in skills:
            check = self.check_requirements(skill)
            line = (
                f"- {skill.name}: {skill.description or '(no description)'}; "
                f"source={skill.source}; available={'true' if check.available else 'false'}"
            )
            if not check.available:
                missing: list[str] = []
                if check.missing_bins:
                    missing.append(f"missing_bins={','.join(check.missing_bins)}")
                if check.missing_env:
                    missing.append(f"missing_env={','.join(check.missing_env)}")
                line += f"; requires={'/'.join(missing)}"
            lines.append(line)
        lines.append("</skills>")
        return "\n".join(lines)

    def check_requirements(self, skill: SkillSpec) -> SkillRequirementCheck:
        """检查条件并返回判断结果。
        
        Args:
            skill: 技能对象。
        """
        missing_bins = tuple(bin_name for bin_name in skill.requires_bins if shutil.which(bin_name) is None)
        missing_env = tuple(env_name for env_name in skill.requires_env if not os.getenv(env_name))
        available = not missing_bins and not missing_env
        return SkillRequirementCheck(
            available=available,
            missing_bins=missing_bins,
            missing_env=missing_env,
        )

    def _scan_dir(self, skills_dir: Path, *, source: str) -> list[SkillSpec]:
        """处理scan/dir相关逻辑并返回结果。
        
        Args:
            skills_dir: 技能目录路径。
            source: 来源文本或来源标识。
        """
        if not skills_dir.exists() or not skills_dir.is_dir():
            return []

        discovered: list[SkillSpec] = []
        for item in sorted(skills_dir.iterdir()):
            if not item.is_dir():
                continue
            skill_file = item / "SKILL.md"
            if not skill_file.exists() or not skill_file.is_file():
                continue
            parsed = self._parse_skill_file(skill_file=skill_file, fallback_name=item.name, source=source)
            if parsed is not None:
                discovered.append(parsed)
        return discovered

    def _parse_skill_file(self, *, skill_file: Path, fallback_name: str, source: str) -> SkillSpec | None:
        """解析输入并返回结构化结果。
        
        Args:
            skill_file: 技能文件路径。
            fallback_name: 名称参数，用于标识目标对象。
            source: 来源文本或来源标识。
        """
        try:
            raw = skill_file.read_text(encoding="utf-8")
        except OSError:
            logger.exception("Failed to read skill file: %s", skill_file)
            return None

        meta, body = _parse_frontmatter(raw)
        name = str(meta.get("name") or fallback_name).strip() or fallback_name
        description = str(meta.get("description") or "").strip()

        metadata = _parse_metadata(meta.get("metadata"))
        requires = metadata.get("requires") if isinstance(metadata.get("requires"), dict) else {}
        bins = _coerce_str_tuple(requires.get("bins"))
        env = _coerce_str_tuple(requires.get("env"))
        always = _coerce_bool(meta.get("always")) or _coerce_bool(metadata.get("always"))

        return SkillSpec(
            name=name,
            description=description,
            path=skill_file.resolve(),
            source=source,
            always=always,
            requires_bins=bins,
            requires_env=env,
            content=body.strip(),
        )


def _parse_frontmatter(raw: str) -> tuple[dict[str, Any], str]:
    """解析输入并返回结构化结果。
    
    Args:
        raw: 原始输入内容。
    """
    lines = raw.splitlines()
    if len(lines) < 3 or lines[0].strip() != "---":
        return {}, raw

    end_idx = None
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            end_idx = idx
            break
    if end_idx is None:
        return {}, raw

    meta_lines = lines[1:end_idx]
    body = "\n".join(lines[end_idx + 1 :])
    payload: dict[str, Any] = {}
    for line in meta_lines:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        payload[key.strip()] = _parse_scalar(value.strip())
    return payload, body


def _parse_scalar(raw: str) -> Any:
    """解析输入并返回结构化结果。
    
    Args:
        raw: 原始输入内容。
    """
    if not raw:
        return ""
    if raw.startswith(("'", '"')) and raw.endswith(("'", '"')) and len(raw) >= 2:
        return raw[1:-1]
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    return raw


def _parse_metadata(value: Any) -> dict[str, Any]:
    """解析输入并返回结构化结果。
    
    Args:
        value: 待转换或校验的值。
    """
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        candidate = value.strip()
        attempts = [candidate]
        if '\\"' in candidate:
            attempts.append(candidate.replace('\\"', '"'))
        for raw in attempts:
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, str):
                try:
                    nested = json.loads(parsed)
                except json.JSONDecodeError:
                    continue
                if isinstance(nested, dict):
                    return nested
                continue
            if isinstance(parsed, dict):
                return parsed
    return {}


def _coerce_str_tuple(value: Any) -> tuple[str, ...]:
    """处理coerce/str/tuple相关逻辑并返回结果。
    
    Args:
        value: 待转换或校验的值。
    """
    if not isinstance(value, list):
        return ()
    result = [str(item).strip() for item in value if str(item).strip()]
    return tuple(result)


def _coerce_bool(value: Any) -> bool:
    """处理coerce/bool相关逻辑并返回结果。
    
    Args:
        value: 待转换或校验的值。
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _resolve_to_abs(path: Path) -> Path:
    """处理resolve/to/abs相关逻辑并返回结果。
    
    Args:
        path: 文件路径。
    """
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()
