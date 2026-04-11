from __future__ import annotations

import shutil
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class OpenClawFileSpec:
    path: str
    title: str
    placeholder: str


PROFILE_FILES: tuple[OpenClawFileSpec, ...] = (
    OpenClawFileSpec("AGENTS.md", "操作手册", "# AGENTS\n\n在这里定义执行原则、流程和约束。\n"),
    OpenClawFileSpec("SOUL.md", "灵魂", "# SOUL\n\n在这里定义风格、价值观和沟通语气。\n"),
    OpenClawFileSpec("IDENTITY.md", "身份", "# IDENTITY\n\n在这里定义名称、定位和能力边界。\n"),
    OpenClawFileSpec("USER.md", "用户", "# USER\n\n在这里记录你服务对象的背景和偏好。\n"),
    OpenClawFileSpec("TOOLS.md", "工具", "# TOOLS\n\n在这里记录工具使用规范和限制。\n"),
    OpenClawFileSpec("memory/MEMORY.md", "长期记忆", "# MEMORY\n\n在这里沉淀长期稳定事实。\n"),
    OpenClawFileSpec("memory/HISTORY.md", "历史日志", "# HISTORY\n\n在这里沉淀会话历史。\n"),
    OpenClawFileSpec("BOOTSTRAP.md", "出生仪式", ""),
    OpenClawFileSpec("BOOT.md", "重启清单", "# BOOT\n\n每次启动都应执行的初始化步骤。\n"),
    OpenClawFileSpec("HEARTBEAT.md", "心跳待办", "# HEARTBEAT\n\n定期检查项与维护节奏。\n"),
)

PROFILE_INCLUDE_ORDER: tuple[str, ...] = (
    "IDENTITY.md",
    "SOUL.md",
    "AGENTS.md",
    "USER.md",
    "TOOLS.md",
    "BOOT.md",
    "HEARTBEAT.md",
)


class OpenClawProfileLoader:
    def __init__(
        self,
        *,
        root_dir: str = ".",
        enabled: bool = True,
        auto_init: bool = True,
    ) -> None:
        self._enabled = enabled
        self._auto_init = auto_init
        self._root = Path(root_dir).expanduser()
        if not self._root.is_absolute():
            self._root = (Path.cwd() / self._root).resolve()
        self._startup_bootstrap_note = ""

    @property
    def root_dir(self) -> Path:
        return self._root

    def prepare_startup(self) -> None:
        if not self._enabled:
            return

        if self._auto_init:
            created = self._ensure_scaffold()
            if created:
                logger.info("OpenClaw profile scaffold created: %s", created)

        self._startup_bootstrap_note = self._consume_bootstrap_once()

    def resolve_memory_file_path(self, configured_path: str) -> str:
        configured = Path(configured_path).expanduser()
        if configured.is_absolute():
            return str(configured)

        configured_clean = configured_path.strip()
        if configured_clean and configured_clean.upper() not in {"MEMORY.MD", "MEMORY/MEMORY.MD"}:
            return str((Path.cwd() / configured).resolve())

        preferred = self._root / "memory" / "MEMORY.md"
        legacy_candidates = (
            self._root / "memory" / "MEMORY.MD",
            self._root / "MEMORY.md",
            self._root / "MEMORY.MD",
        )
        if preferred.exists():
            return str(preferred)
        for legacy in legacy_candidates:
            if legacy.exists():
                return str(legacy)
        return str(preferred)

    def render_profile_context(self) -> str:
        if not self._enabled:
            return ""

        blocks: list[str] = []
        for filename in PROFILE_INCLUDE_ORDER:
            content = self._read_non_empty(filename)
            if not content:
                continue
            blocks.append(f"[{filename}]\n{content}")

        bootstrap = self._startup_bootstrap_note.strip()
        if bootstrap:
            blocks.append(
                "[BOOTSTRAP_CONSUMED]\n以下内容来自本次启动前的 BOOTSTRAP.md（已消费并删除）：\n"
                f"{bootstrap}"
            )

        if not blocks:
            return ""
        return "\n\n".join(blocks)

    def render_compact_profile_context(self) -> str:
        """渲染轻量版档案上下文，保留核心规则并压缩次级档案。"""
        if not self._enabled:
            return ""

        summary_lines: list[str] = []
        for filename in PROFILE_INCLUDE_ORDER:
            content = self._read_non_empty(filename)
            if not content:
                continue
            max_chars = 220 if filename in {"IDENTITY.md", "SOUL.md", "AGENTS.md", "TOOLS.md"} else 140
            condensed = self._condense_markdown(content, max_chars=max_chars)
            if condensed:
                summary_lines.append(f"- {filename}: {condensed}")
        if summary_lines:
            blocks = ["[PROFILE_SUMMARY]\n" + "\n".join(summary_lines)]
        else:
            blocks = []

        bootstrap = self._startup_bootstrap_note.strip()
        if bootstrap:
            blocks.append(
                "[BOOTSTRAP_CONSUMED]\n以下内容来自本次启动前的 BOOTSTRAP.md（已消费并删除）：\n"
                f"{self._condense_markdown(bootstrap, max_chars=240)}"
            )

        if not blocks:
            return ""
        return "\n\n".join(blocks)

    def onboard(self) -> list[str]:
        """初始化工作区档案模板，只补齐缺失文件。"""
        return self._ensure_scaffold()

    def _ensure_scaffold(self) -> list[str]:
        created: list[str] = []
        self._root.mkdir(parents=True, exist_ok=True)
        created.extend(self._copy_template_tree())

        skills_dir = self._root / "skills"
        if not skills_dir.exists():
            skills_dir.mkdir(parents=True, exist_ok=True)
            created.append(str(skills_dir))

        for spec in PROFILE_FILES:
            target = self._root / spec.path
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(spec.placeholder, encoding="utf-8")
            created.append(str(target))
        return created

    def _copy_template_tree(self) -> list[str]:
        created: list[str] = []
        template_root = self._resolve_template_root()
        if template_root is None or not template_root.exists():
            return created

        for source in sorted(template_root.rglob("*")):
            if source.name == "__pycache__":
                continue
            relative = source.relative_to(template_root)
            target = self._root / relative
            if source.is_dir():
                if not target.exists():
                    target.mkdir(parents=True, exist_ok=True)
                    created.append(str(target))
                continue
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            created.append(str(target))
        return created

    def _resolve_template_root(self) -> Path | None:
        try:
            package_root = Path(str(resources.files("w_bot"))) / "template"
        except (ModuleNotFoundError, FileNotFoundError):
            package_root = None

        if package_root is not None and package_root.exists():
            return package_root

        local_root = Path(__file__).resolve().parent.parent / "template"
        if local_root.exists():
            return local_root
        return None

    def _consume_bootstrap_once(self) -> str:
        bootstrap_path = self._root / "BOOTSTRAP.md"
        if not bootstrap_path.exists() or not bootstrap_path.is_file():
            return ""

        content = bootstrap_path.read_text(encoding="utf-8").strip()
        if not content:
            return ""

        try:
            bootstrap_path.unlink()
            logger.info("Consumed and removed BOOTSTRAP.md: %s", bootstrap_path)
        except OSError:
            logger.exception("Failed to remove BOOTSTRAP.md after consume: %s", bootstrap_path)
        return content

    def _read_non_empty(self, filename: str) -> str:
        target = self._root / filename
        if not target.exists() or not target.is_file():
            return ""
        content = target.read_text(encoding="utf-8").strip()
        return content

    @staticmethod
    def _condense_markdown(content: str, *, max_chars: int = 320) -> str:
        lines: list[str] = []
        for raw in content.splitlines():
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                continue
            if line.startswith(("- ", "* ")):
                line = line[2:].strip()
            elif len(line) > 2 and line[0].isdigit() and line[1] == ".":
                line = line[2:].strip()
            lines.append(line)

        merged = "；".join(part for part in lines if part)
        if len(merged) <= max_chars:
            return merged
        return merged[: max_chars - 1].rstrip() + "…"
