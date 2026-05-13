from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from w_bot.agents.skills import SkillsLoader
from w_bot.agents.tools.runtime import build_tools
from w_bot.agents.tools.skill import SkillViewTool, SkillsListTool


def _write_skill(root: Path, name: str = "weather") -> Path:
    skill_dir = root / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "references").mkdir()
    (skill_dir / "references" / "api.md").write_text("Use wttr.in for quick checks.", encoding="utf-8")
    (skill_dir / "SKILL.md").write_text(
        "\n".join(
            [
                "---",
                f"name: {name}",
                "description: Weather workflow",
                "---",
                "",
                "# Weather",
                "",
                "Prefer checking the city with wttr.in.",
            ]
        ),
        encoding="utf-8",
    )
    return skill_dir


def test_build_tools_registers_skill_discovery_tools(tmp_path) -> None:
    _write_skill(tmp_path / "skills")
    loader = SkillsLoader(
        workspace_skills_dir=str(tmp_path / "skills"),
        builtin_skills_dir=str(tmp_path / "builtin"),
    )

    tools = build_tools(
        memory_store=MagicMock(),
        user_id="tester",
        tavily_api_key="",
        enable_cron_service=False,
        mcp_servers=[],
        skills_loader=loader,
    )

    names = [tool.name for tool in tools]
    assert "skills_list" in names
    assert "skill_view" in names
    assert "run_skill" in names
    assert names.index("skills_list") < names.index("run_skill")
    assert names.index("skill_view") < names.index("run_skill")


@pytest.mark.asyncio
async def test_skill_view_loads_skill_and_linked_files(tmp_path) -> None:
    _write_skill(tmp_path / "skills")
    loader = SkillsLoader(
        workspace_skills_dir=str(tmp_path / "skills"),
        builtin_skills_dir=str(tmp_path / "builtin"),
    )

    listed = json.loads(await SkillsListTool(skills_loader=loader).execute())
    viewed = json.loads(await SkillViewTool(skills_loader=loader).execute(name="weather"))

    assert listed["skills"][0]["name"] == "weather"
    assert viewed["name"] == "weather"
    assert "# Weather" in viewed["content"]
    assert "references/api.md" in viewed["linked_files"]


@pytest.mark.asyncio
async def test_skill_view_rejects_paths_outside_skill_dir(tmp_path) -> None:
    _write_skill(tmp_path / "skills")
    secret = tmp_path / "secret.txt"
    secret.write_text("should not be readable", encoding="utf-8")
    loader = SkillsLoader(
        workspace_skills_dir=str(tmp_path / "skills"),
        builtin_skills_dir=str(tmp_path / "builtin"),
    )

    result = json.loads(
        await SkillViewTool(skills_loader=loader).execute(
            name="weather",
            file_path="../secret.txt",
        )
    )

    assert "outside skill" in result["error"]
    assert "should not be readable" not in json.dumps(result, ensure_ascii=False)
