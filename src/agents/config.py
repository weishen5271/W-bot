from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_APP_CONFIG_PATH = "configs/app.json"


@dataclass(frozen=True)
class Settings:
    dashscope_api_key: str
    bailian_base_url: str
    bailian_model_name: str
    tavily_api_key: str
    postgres_dsn: str
    memory_file_path: str
    user_id: str
    session_id: str
    session_state_file_path: str
    retrieve_top_k: int
    enable_exec_tool: bool
    enable_cron_service: bool
    mcp_servers: list[dict[str, Any]]
    enable_skills: bool
    skills_workspace_dir: str
    skills_builtin_dir: str


def load_settings(
    *,
    config_path: str = DEFAULT_APP_CONFIG_PATH,
    overrides: dict[str, Any] | None = None,
) -> Settings:
    payload = _load_or_create_app_config(config_path)
    agent_cfg = payload.get("agent") if isinstance(payload.get("agent"), dict) else {}

    merged = dict(agent_cfg)
    if overrides:
        merged.update(overrides)

    settings = Settings(
        dashscope_api_key=_must_value(merged, "dashscopeApiKey", "dashscope_api_key"),
        bailian_base_url=_string_value(
            merged,
            "bailianBaseUrl",
            "bailian_base_url",
            default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
        bailian_model_name=_string_value(
            merged,
            "bailianModelName",
            "bailian_model_name",
            default="qwen-plus",
        ),
        tavily_api_key=_string_value(merged, "tavilyApiKey", "tavily_api_key", default=""),
        postgres_dsn=_must_value(merged, "postgresDsn", "postgres_dsn"),
        memory_file_path=_string_value(
            merged,
            "memoryFilePath",
            "memory_file_path",
            default="MEMORY.MD",
        ),
        user_id=_string_value(merged, "userId", "user_id", default="cli_user"),
        session_id=_string_value(
            merged,
            "sessionId",
            "session_id",
            default=datetime.now().strftime("cli_session_%Y%m%d_%H%M%S"),
        ),
        session_state_file_path=_string_value(
            merged,
            "sessionStateFilePath",
            "session_state_file_path",
            default=".cybercore_session.json",
        ),
        retrieve_top_k=_int_value(merged, "retrieveTopK", "retrieve_top_k", default=4),
        enable_exec_tool=_bool_value(merged, "enableExecTool", "enable_exec_tool", default=False),
        enable_cron_service=_bool_value(
            merged,
            "enableCronService",
            "enable_cron_service",
            default=False,
        ),
        mcp_servers=_list_value(merged, "mcpServers", "mcp_servers", default=[]),
        enable_skills=_bool_value(merged, "enableSkills", "enable_skills", default=True),
        skills_workspace_dir=_string_value(
            merged,
            "skillsWorkspaceDir",
            "skills_workspace_dir",
            default="skills",
        ),
        skills_builtin_dir=_string_value(
            merged,
            "skillsBuiltinDir",
            "skills_builtin_dir",
            default="",
        ),
    )
    logger.info(
        "Settings loaded from %s: model=%s, session_id=%s, user_id=%s, memory_file=%s, top_k=%s",
        config_path,
        settings.bailian_model_name,
        settings.session_id,
        settings.user_id,
        settings.memory_file_path,
        settings.retrieve_top_k,
    )
    return settings


def _load_or_create_app_config(config_path: str) -> dict[str, Any]:
    target = Path(config_path)
    if not target.is_absolute():
        target = Path.cwd() / target

    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(default_app_config(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise FileNotFoundError(
            f"Config not found. A template has been generated at: {target}. "
            "Please fill required fields and retry."
        )

    raw = target.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON config: {target}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Invalid config root, expected object: {target}")
    return data


def default_app_config() -> dict[str, Any]:
    return {
        "agent": {
            "dashscopeApiKey": "",
            "bailianBaseUrl": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "bailianModelName": "qwen-plus",
            "tavilyApiKey": "",
            "postgresDsn": "",
            "milvusUri": "http://<host>:19530",
            "memoryCollection": "cybercore_long_term_memory_cli",
            "memoryFilePath": "MEMORY.MD",
            "userId": "feishu_bot",
            "sessionId": "",
            "sessionStateFilePath": ".cybercore_session.json",
            "retrieveTopK": 4,
            "enableExecTool": False,
            "enableCronService": False,
            "mcpServers": [],
            "enableSkills": True,
            "skillsWorkspaceDir": "skills",
            "skillsBuiltinDir": "",
        },
        "channels": {
            "feishu": {
                "enabled": True,
                "appId": "cli_xxx",
                "appSecret": "xxx",
                "encryptKey": "",
                "verificationToken": "",
                "allowFrom": ["*"],
                "groupPolicy": "mention",
                "replyToMessage": True,
                "reactEmoji": "THUMBSUP",
            }
        },
        "threadPrefix": "feishu",
    }


def _pick(data: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return None


def _must_value(data: dict[str, Any], *keys: str) -> str:
    value = _pick(data, *keys)
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise ValueError(f"Missing required config: {'/'.join(keys)}")


def _string_value(data: dict[str, Any], *keys: str, default: str) -> str:
    value = _pick(data, *keys)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default


def _int_value(data: dict[str, Any], *keys: str, default: int) -> int:
    value = _pick(data, *keys)
    if value is None:
        return default
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        logger.warning("Invalid int config for %s, fallback=%s", "/".join(keys), default)
        return default


def _bool_value(data: dict[str, Any], *keys: str, default: bool) -> bool:
    value = _pick(data, *keys)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


def _list_value(data: dict[str, Any], *keys: str, default: list[Any]) -> list[Any]:
    value = _pick(data, *keys)
    if isinstance(value, list):
        return value
    return list(default)
