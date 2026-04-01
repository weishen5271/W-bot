from __future__ import annotations

import json
import os
import re
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
        self.workspace_skills_dir = _resolve_to_abs(Path(workspace_skills_dir))
        self.builtin_skills_dir = _resolve_to_abs(Path(builtin_skills_dir)) if builtin_skills_dir else BUILTIN_SKILLS_DIR
        self.workspace_skills_dir.mkdir(parents=True, exist_ok=True)

    def list_skills(self, *, filter_unavailable: bool = False) -> list[SkillSpec]:
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
        always_skills: list[SkillSpec] = []
        for skill in self.list_skills(filter_unavailable=False):
            check = self.check_requirements(skill)
            if skill.always and check.available:
                always_skills.append(skill)
        return always_skills

    def load_skill(self, name: str) -> str | None:
        for skill in self.list_skills(filter_unavailable=False):
            if skill.name == name:
                return skill.path.read_text(encoding="utf-8")
        return None

    def get_skill(self, name: str) -> SkillSpec | None:
        normalized = name.strip()
        if not normalized:
            return None
        for skill in self.list_skills(filter_unavailable=False):
            if skill.name == normalized:
                return skill
        return None

    def load_skills_for_context(self, skill_names: list[str]) -> str:
        parts: list[str] = []
        for name in skill_names:
            content = self.load_skill(name)
            if not content:
                continue
            stripped = _strip_frontmatter(content)
            if stripped:
                parts.append(f"### Skill: {name}\n\n{stripped}")
        return "\n\n---\n\n".join(parts)

    def build_skills_summary(self) -> str:
        skills = self.list_skills(filter_unavailable=False)
        if not skills:
            return ""

        lines = ["<skills>"]
        for skill in skills:
            check = self.check_requirements(skill)
            lines.append(f'  <skill available="{"true" if check.available else "false"}">')
            lines.append(f"    <name>{_escape_xml(skill.name)}</name>")
            lines.append(f"    <description>{_escape_xml(skill.description or skill.name)}</description>")
            lines.append(f"    <location>{_escape_xml(str(skill.path))}</location>")
            if not check.available:
                missing: list[str] = []
                if check.missing_bins:
                    missing.extend(f"CLI: {bin_name}" for bin_name in check.missing_bins)
                if check.missing_env:
                    missing.extend(f"ENV: {env_name}" for env_name in check.missing_env)
                if missing:
                    lines.append(f"    <requires>{_escape_xml(', '.join(missing))}</requires>")
            lines.append("  </skill>")
        lines.append("</skills>")
        return "\n".join(lines)

    def check_requirements(self, skill: SkillSpec) -> SkillRequirementCheck:
        missing_bins = tuple(bin_name for bin_name in skill.requires_bins if shutil.which(bin_name) is None)
        missing_env = tuple(env_name for env_name in skill.requires_env if not os.getenv(env_name))
        available = not missing_bins and not missing_env
        return SkillRequirementCheck(
            available=available,
            missing_bins=missing_bins,
            missing_env=missing_env,
        )

    def _scan_dir(self, skills_dir: Path, *, source: str) -> list[SkillSpec]:
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
    if not raw:
        return ""
    if raw.startswith(("'", '"')) and raw.endswith(("'", '"')) and len(raw) >= 2:
        return raw[1:-1]
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    return raw


def _parse_metadata(value: Any) -> dict[str, Any]:
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
    if not isinstance(value, list):
        return ()
    result = [str(item).strip() for item in value if str(item).strip()]
    return tuple(result)


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _resolve_to_abs(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def _strip_frontmatter(content: str) -> str:
    if content.startswith("---"):
        match = re.match(r"^---\n.*?\n---\n?", content, re.DOTALL)
        if match:
            return content[match.end():].strip()
    return content.strip()


def _escape_xml(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
