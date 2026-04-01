from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_APP_CONFIG_PATH = "configs/app.json"
DEFAULT_OPENCLAW_PROFILE_ROOT_DIR = "~/.wbot"


@dataclass(frozen=True)
class Settings:
    model_provider: str
    llm_api_key: str
    llm_base_url: str
    llm_extra_headers: dict[str, str]
    llm_temperature: float
    dashscope_api_key: str
    bailian_base_url: str
    bailian_model_name: str
    tavily_api_key: str
    memory_file_path: str
    short_term_memory_path: str
    user_id: str
    session_id: str
    session_state_file_path: str
    retrieve_top_k: int
    enable_cron_service: bool
    mcp_servers: list[dict[str, Any]]
    enable_skills: bool
    skills_workspace_dir: str
    skills_builtin_dir: str
    model_routing: "ModelRoutingSettings"
    multimodal: "MultimodalSettings"
    token_optimization: "TokenOptimizationSettings"
    short_term_memory_optimization: "ShortTermMemoryOptimizationSettings"
    expose_step_logs: bool
    enable_openclaw_profile: bool
    openclaw_profile_root_dir: str
    openclaw_auto_init: bool
    loop_guard: "LoopGuardSettings"
    enable_streaming: bool
    enable_console_logs: bool


@dataclass(frozen=True)
class MultimodalSettings:
    enabled: bool
    max_file_bytes: int
    max_total_bytes_per_turn: int
    max_files_per_turn: int
    audio_mode: str
    video_keyframe_interval_sec: int
    video_max_frames: int
    document_max_chars: int
    temp_ttl_hours: int
    media_root_dir: str


@dataclass(frozen=True)
class ModelRoutingSettings:
    text_model_name: str
    image_model_name: str
    audio_model_name: str


@dataclass(frozen=True)
class TokenOptimizationSettings:
    enabled: bool
    max_recent_user_turns: int
    summary_trigger_messages: int
    summary_max_chars: int
    context_window_tokens: int
    auto_compact_buffer_tokens: int
    warning_threshold_buffer_tokens: int
    error_threshold_buffer_tokens: int
    blocking_buffer_tokens: int
    enable_dynamic_system_context: bool
    enable_git_status: bool
    git_status_max_chars: int
    enable_project_instruction_scan: bool
    project_instruction_files: tuple[str, ...]


@dataclass(frozen=True)
class ShortTermMemoryOptimizationSettings:
    enabled: bool
    run_on_startup: bool
    interval_minutes: int
    keep_recent_checkpoints: int
    summary_batch_size: int
    max_threads_per_run: int
    max_checkpoints_per_thread: int
    archive_before_delete: bool
    compress_level: int


@dataclass(frozen=True)
class LoopGuardSettings:
    recursion_limit: int
    max_tool_steps_per_turn: int
    max_same_tool_call_repeats: int


def load_settings(
    *,
    config_path: str = DEFAULT_APP_CONFIG_PATH,
    overrides: dict[str, Any] | None = None,
) -> Settings:
    """加载应用配置并转换为结构化设置对象。
    
    Args:
        config_path: 目标路径参数，用于定位文件或目录。
        overrides: 覆盖配置项字典，用于覆盖文件中的默认配置。
    """
    payload = _load_or_create_app_config(config_path)
    agent_cfg = payload.get("agent") if isinstance(payload.get("agent"), dict) else {}
    agents_cfg = payload.get("agents") if isinstance(payload.get("agents"), dict) else {}
    agents_defaults = (
        agents_cfg.get("defaults") if isinstance(agents_cfg.get("defaults"), dict) else {}
    )
    providers_payload = payload.get("providers") if isinstance(payload.get("providers"), dict) else {}

    merged = dict(agent_cfg)
    if overrides:
        merged.update(overrides)
    model_provider = _string_value(agents_defaults, "provider", default="dashscope")
    model_name_from_defaults = _string_value(agents_defaults, "model", default="qwen-plus")
    temperature = _float_value(agents_defaults, "temperature", default=0.2)
    provider_payload = _dict_value(providers_payload, model_provider)
    active_provider = provider_payload
    model_routing_payload = (
        merged.get("modelRouting") if isinstance(merged.get("modelRouting"), dict) else {}
    )
    multimodal_payload = merged.get("multimodal") if isinstance(merged.get("multimodal"), dict) else {}
    token_opt_payload = (
        merged.get("tokenOptimization") if isinstance(merged.get("tokenOptimization"), dict) else {}
    )
    short_mem_opt_payload = (
        merged.get("shortTermMemoryOptimization")
        if isinstance(merged.get("shortTermMemoryOptimization"), dict)
        else {}
    )
    loop_guard_payload = merged.get("loopGuard") if isinstance(merged.get("loopGuard"), dict) else {}

    settings = Settings(
        model_provider=model_provider,
        llm_api_key=_must_value(active_provider, "apiKey", "api_key"),
        llm_base_url=_string_value(
            active_provider,
            "apiBase",
            "api_base",
            default="",
        ),
        llm_extra_headers=_header_dict_value(active_provider, "extraHeaders", "extra_headers", default={}),
        llm_temperature=temperature,
        dashscope_api_key=_must_value(active_provider, "apiKey", "api_key"),
        bailian_base_url=_string_value(
            active_provider,
            "apiBase",
            "api_base",
            default="",
        ),
        bailian_model_name=_string_value(
            agents_defaults,
            "model",
            default=model_name_from_defaults,
        ),
        tavily_api_key=_string_value(merged, "tavilyApiKey", "tavily_api_key", default=""),
        memory_file_path=_string_value(
            merged,
            "memoryFilePath",
            "memory_file_path",
            default="memory/MEMORY.md",
        ),
        short_term_memory_path=_string_value(
            merged,
            "shortTermMemoryPath",
            "short_term_memory_path",
            default="memory/short_term_memory.pkl",
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
            default=".w_bot_session.json",
        ),
        retrieve_top_k=_int_value(merged, "retrieveTopK", "retrieve_top_k", default=4),
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
        model_routing=ModelRoutingSettings(
            text_model_name=_string_value(
                model_routing_payload,
                "textModelName",
                "text_model_name",
                default=_string_value(
                    agents_defaults,
                    "model",
                    default=model_name_from_defaults,
                ),
            ),
            image_model_name=_string_value(
                model_routing_payload,
                "imageModelName",
                "image_model_name",
                default="",
            ),
            audio_model_name=_string_value(
                model_routing_payload,
                "audioModelName",
                "audio_model_name",
                default="",
            ),
        ),
        multimodal=MultimodalSettings(
            enabled=_bool_value(multimodal_payload, "enabled", default=False),
            max_file_bytes=_int_value(multimodal_payload, "maxFileBytes", "max_file_bytes", default=20 * 1024 * 1024),
            max_total_bytes_per_turn=_int_value(
                multimodal_payload,
                "maxTotalBytesPerTurn",
                "max_total_bytes_per_turn",
                default=50 * 1024 * 1024,
            ),
            max_files_per_turn=_int_value(
                multimodal_payload,
                "maxFilesPerTurn",
                "max_files_per_turn",
                default=10,
            ),
            audio_mode=_string_value(multimodal_payload, "audioMode", "audio_mode", default="auto").lower(),
            video_keyframe_interval_sec=_int_value(
                multimodal_payload,
                "videoKeyframeIntervalSec",
                "video_keyframe_interval_sec",
                default=3,
            ),
            video_max_frames=_int_value(multimodal_payload, "videoMaxFrames", "video_max_frames", default=12),
            document_max_chars=_int_value(
                multimodal_payload,
                "documentMaxChars",
                "document_max_chars",
                default=120000,
            ),
            temp_ttl_hours=_int_value(multimodal_payload, "tempTtlHours", "temp_ttl_hours", default=24),
            media_root_dir=_string_value(multimodal_payload, "mediaRootDir", "media_root_dir", default="media"),
        ),
        token_optimization=TokenOptimizationSettings(
            enabled=_bool_value(token_opt_payload, "enabled", default=True),
            max_recent_user_turns=_int_value(
                token_opt_payload,
                "maxRecentUserTurns",
                "max_recent_user_turns",
                default=6,
            ),
            summary_trigger_messages=_int_value(
                token_opt_payload,
                "summaryTriggerMessages",
                "summary_trigger_messages",
                default=12,
            ),
            summary_max_chars=_int_value(
                token_opt_payload,
                "summaryMaxChars",
                "summary_max_chars",
                default=1200,
            ),
            context_window_tokens=_int_value(
                token_opt_payload,
                "contextWindowTokens",
                "context_window_tokens",
                default=128000,
            ),
            auto_compact_buffer_tokens=_int_value(
                token_opt_payload,
                "autoCompactBufferTokens",
                "auto_compact_buffer_tokens",
                default=13000,
            ),
            warning_threshold_buffer_tokens=_int_value(
                token_opt_payload,
                "warningThresholdBufferTokens",
                "warning_threshold_buffer_tokens",
                default=20000,
            ),
            error_threshold_buffer_tokens=_int_value(
                token_opt_payload,
                "errorThresholdBufferTokens",
                "error_threshold_buffer_tokens",
                default=20000,
            ),
            blocking_buffer_tokens=_int_value(
                token_opt_payload,
                "blockingBufferTokens",
                "blocking_buffer_tokens",
                default=3000,
            ),
            enable_dynamic_system_context=_bool_value(
                token_opt_payload,
                "enableDynamicSystemContext",
                "enable_dynamic_system_context",
                default=True,
            ),
            enable_git_status=_bool_value(
                token_opt_payload,
                "enableGitStatus",
                "enable_git_status",
                default=True,
            ),
            git_status_max_chars=_int_value(
                token_opt_payload,
                "gitStatusMaxChars",
                "git_status_max_chars",
                default=2000,
            ),
            enable_project_instruction_scan=_bool_value(
                token_opt_payload,
                "enableProjectInstructionScan",
                "enable_project_instruction_scan",
                default=True,
            ),
            project_instruction_files=tuple(
                str(item).strip()
                for item in _list_value(
                    token_opt_payload,
                    "projectInstructionFiles",
                    "project_instruction_files",
                    default=["CLAUDE.md", "AGENTS.md", "WBOT.md"],
                )
                if str(item).strip()
            ),
        ),
        short_term_memory_optimization=ShortTermMemoryOptimizationSettings(
            enabled=_bool_value(short_mem_opt_payload, "enabled", default=True),
            run_on_startup=_bool_value(short_mem_opt_payload, "runOnStartup", "run_on_startup", default=True),
            interval_minutes=_int_value(short_mem_opt_payload, "intervalMinutes", "interval_minutes", default=15),
            keep_recent_checkpoints=_int_value(
                short_mem_opt_payload,
                "keepRecentCheckpoints",
                "keep_recent_checkpoints",
                default=120,
            ),
            summary_batch_size=_int_value(
                short_mem_opt_payload,
                "summaryBatchSize",
                "summary_batch_size",
                default=20,
            ),
            max_threads_per_run=_int_value(
                short_mem_opt_payload,
                "maxThreadsPerRun",
                "max_threads_per_run",
                default=20,
            ),
            max_checkpoints_per_thread=_int_value(
                short_mem_opt_payload,
                "maxCheckpointsPerThread",
                "max_checkpoints_per_thread",
                default=400,
            ),
            archive_before_delete=_bool_value(
                short_mem_opt_payload,
                "archiveBeforeDelete",
                "archive_before_delete",
                default=True,
            ),
            compress_level=_int_value(short_mem_opt_payload, "compressLevel", "compress_level", default=6),
        ),
        expose_step_logs=_bool_value(
            merged,
            "exposeStepLogs",
            "expose_step_logs",
            default=True,
        ),
        enable_openclaw_profile=_bool_value(
            merged,
            "enableOpenClawProfile",
            "enable_openclaw_profile",
            default=True,
        ),
        openclaw_profile_root_dir=normalize_openclaw_profile_root_dir(
            _string_value(
                merged,
                "openClawProfileRootDir",
                "openclaw_profile_root_dir",
                default=DEFAULT_OPENCLAW_PROFILE_ROOT_DIR,
            )
        ),
        openclaw_auto_init=_bool_value(
            merged,
            "openClawAutoInit",
            "openclaw_auto_init",
            default=True,
        ),
        enable_streaming=_bool_value(
            merged,
            "enableStreaming",
            "enable_streaming",
            default=False,
        ),
        enable_console_logs=_bool_value(
            merged,
            "enableConsoleLogs",
            "enable_console_logs",
            default=True,
        ),
        loop_guard=LoopGuardSettings(
            recursion_limit=_int_value(
                loop_guard_payload,
                "recursionLimit",
                "recursion_limit",
                default=20,
            ),
            max_tool_steps_per_turn=_int_value(
                loop_guard_payload,
                "maxToolStepsPerTurn",
                "max_tool_steps_per_turn",
                default=8,
            ),
            max_same_tool_call_repeats=_int_value(
                loop_guard_payload,
                "maxSameToolCallRepeats",
                "max_same_tool_call_repeats",
                default=3,
            ),
        ),
    )
    logger.info(
        "Settings loaded from %s: provider=%s, model=%s, session_id=%s, user_id=%s, memory_file=%s, top_k=%s",
        config_path,
        settings.model_provider,
        settings.bailian_model_name,
        settings.session_id,
        settings.user_id,
        settings.memory_file_path,
        settings.retrieve_top_k,
    )
    return settings


def _load_or_create_app_config(config_path: str) -> dict[str, Any]:
    """加载目标配置或数据并返回。
    
    Args:
        config_path: 目标路径参数，用于定位文件或目录。
    """
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
    """处理default/app/config相关逻辑并返回结果。
    """
    return {
        "agent": {
            "tavilyApiKey": "",
            "milvusUri": "http://<host>:19530",
            "memoryCollection": "w_bot_long_term_memory_cli",
            "memoryFilePath": "memory/MEMORY.md",
            "shortTermMemoryPath": "memory/short_term_memory.pkl",
            "userId": "feishu_bot",
            "sessionId": "",
            "sessionStateFilePath": ".w_bot_session.json",
            "retrieveTopK": 4,
            "enableCronService": False,
            "mcpServers": [],
            "enableSkills": True,
            "skillsWorkspaceDir": "skills",
            "skillsBuiltinDir": "",
            "modelRouting": {
                "textModelName": "qwen-plus",
                "imageModelName": "",
                "audioModelName": "",
            },
            "multimodal": {
                "enabled": True,
                "maxFileBytes": 20971520,
                "maxTotalBytesPerTurn": 52428800,
                "maxFilesPerTurn": 10,
                "audioMode": "auto",
                "videoKeyframeIntervalSec": 3,
                "videoMaxFrames": 12,
                "documentMaxChars": 120000,
                "tempTtlHours": 24,
                "mediaRootDir": "media",
            },
            "tokenOptimization": {
                "enabled": True,
                "maxRecentUserTurns": 6,
                "summaryTriggerMessages": 12,
                "summaryMaxChars": 1200,
                "contextWindowTokens": 128000,
                "autoCompactBufferTokens": 13000,
                "warningThresholdBufferTokens": 20000,
                "errorThresholdBufferTokens": 20000,
                "blockingBufferTokens": 3000,
                "enableDynamicSystemContext": True,
                "enableGitStatus": True,
                "gitStatusMaxChars": 2000,
                "enableProjectInstructionScan": True,
                "projectInstructionFiles": ["CLAUDE.md", "AGENTS.md", "WBOT.md"],
            },
            "shortTermMemoryOptimization": {
                "enabled": True,
                "runOnStartup": True,
                "intervalMinutes": 15,
                "keepRecentCheckpoints": 120,
                "summaryBatchSize": 20,
                "maxThreadsPerRun": 20,
                "maxCheckpointsPerThread": 400,
                "archiveBeforeDelete": True,
                "compressLevel": 6,
            },
            "exposeStepLogs": True,
            "enableOpenClawProfile": True,
            "openClawProfileRootDir": DEFAULT_OPENCLAW_PROFILE_ROOT_DIR,
            "openClawAutoInit": True,
            "loopGuard": {
                "recursionLimit": 20,
                "maxToolStepsPerTurn": 8,
                "maxSameToolCallRepeats": 3,
            },
            "enableConsoleLogs": True,
        },
        "agents": {
            "defaults": {
                "provider": "dashscope",
                "model": "qwen-plus",
                "temperature": 0.2,
            }
        },
        "providers": _default_provider_configs(),
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
            },
            "web": {
                "enabled": True,
                "host": "127.0.0.1",
                "port": 8000,
            },
        },
        "threadPrefix": "feishu",
    }


def _default_provider_configs() -> dict[str, dict[str, Any]]:
    return {
        "custom": {"apiKey": "", "apiBase": "", "extraHeaders": None},
        "azureOpenai": {"apiKey": "", "apiBase": "", "extraHeaders": None},
        "anthropic": {"apiKey": "", "apiBase": "", "extraHeaders": None},
        "openai": {"apiKey": "", "apiBase": "", "extraHeaders": None},
        "openrouter": {"apiKey": "", "apiBase": "", "extraHeaders": None},
        "deepseek": {"apiKey": "", "apiBase": "", "extraHeaders": None},
        "groq": {"apiKey": "", "apiBase": "", "extraHeaders": None},
        "zhipu": {"apiKey": "", "apiBase": "", "extraHeaders": None},
        "dashscope": {
            "apiKey": "",
            "apiBase": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "extraHeaders": None,
        },
        "vllm": {"apiKey": "", "apiBase": "", "extraHeaders": None},
        "ollama": {"apiKey": "", "apiBase": "", "extraHeaders": None},
        "ovms": {"apiKey": "", "apiBase": "", "extraHeaders": None},
        "gemini": {"apiKey": "", "apiBase": "", "extraHeaders": None},
        "moonshot": {"apiKey": "", "apiBase": "", "extraHeaders": None},
        "minimax": {"apiKey": "", "apiBase": "", "extraHeaders": None},
        "mistral": {"apiKey": "", "apiBase": "", "extraHeaders": None},
        "stepfun": {"apiKey": "", "apiBase": "", "extraHeaders": None},
        "aihubmix": {"apiKey": "", "apiBase": "", "extraHeaders": None},
        "siliconflow": {"apiKey": "", "apiBase": "", "extraHeaders": None},
        "volcengine": {"apiKey": "", "apiBase": "", "extraHeaders": None},
        "volcengineCodingPlan": {"apiKey": "", "apiBase": "", "extraHeaders": None},
        "byteplus": {"apiKey": "", "apiBase": "", "extraHeaders": None},
        "byteplusCodingPlan": {"apiKey": "", "apiBase": "", "extraHeaders": None},
    }


def normalize_openclaw_profile_root_dir(value: str) -> str:
    normalized = (value or "").strip()
    if normalized in {"", "."}:
        return DEFAULT_OPENCLAW_PROFILE_ROOT_DIR
    return normalized


def _pick(data: dict[str, Any], *keys: str) -> Any:
    """处理pick相关逻辑并返回结果。
    
    Args:
        data: 输入字典对象，用于按键名读取配置值。
        keys: 候选键名列表，按顺序尝试读取。
    """
    for key in keys:
        if key in data:
            return data[key]
    return None


def _dict_value(data: dict[str, Any], *keys: str) -> dict[str, Any]:
    value = _pick(data, *keys)
    if isinstance(value, dict):
        return value
    return {}


def _must_value(data: dict[str, Any], *keys: str) -> str:
    """处理must/value相关逻辑并返回结果。
    
    Args:
        data: 输入字典对象，用于读取必填配置。
        keys: 候选键名列表，按顺序尝试读取。
    """
    value = _pick(data, *keys)
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise ValueError(f"Missing required config: {'/'.join(keys)}")


def _string_value(data: dict[str, Any], *keys: str, default: str) -> str:
    """处理string/value相关逻辑并返回结果。
    
    Args:
        data: 输入字典对象，用于读取字符串配置。
        keys: 候选键名列表，按顺序尝试读取。
        default: 缺失配置时使用的默认值。
    """
    value = _pick(data, *keys)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default


def _int_value(data: dict[str, Any], *keys: str, default: int) -> int:
    """处理int/value相关逻辑并返回结果。
    
    Args:
        data: 输入字典对象，用于读取整数配置。
        keys: 候选键名列表，按顺序尝试读取。
        default: 缺失配置时使用的默认值。
    """
    value = _pick(data, *keys)
    if value is None:
        return default
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        logger.warning("Invalid int config for %s, fallback=%s", "/".join(keys), default)
        return default


def _bool_value(data: dict[str, Any], *keys: str, default: bool) -> bool:
    """处理bool/value相关逻辑并返回结果。
    
    Args:
        data: 输入字典对象，用于读取布尔配置。
        keys: 候选键名列表，按顺序尝试读取。
        default: 缺失配置时使用的默认值。
    """
    value = _pick(data, *keys)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


def _list_value(data: dict[str, Any], *keys: str, default: list[Any]) -> list[Any]:
    """处理list/value相关逻辑并返回结果。
    
    Args:
        data: 输入字典对象，用于读取列表配置。
        keys: 候选键名列表，按顺序尝试读取。
        default: 缺失配置时使用的默认值。
    """
    value = _pick(data, *keys)
    if isinstance(value, list):
        return value
    return list(default)


def _float_value(data: dict[str, Any], *keys: str, default: float) -> float:
    value = _pick(data, *keys)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        logger.warning("Invalid float config for %s, fallback=%s", "/".join(keys), default)
        return default


def _header_dict_value(data: dict[str, Any], *keys: str, default: dict[str, str]) -> dict[str, str]:
    value = _pick(data, *keys)
    if not isinstance(value, dict):
        return dict(default)
    out: dict[str, str] = {}
    for key, item in value.items():
        k = str(key).strip()
        v = str(item).strip()
        if k and v:
            out[k] = v
    return out
