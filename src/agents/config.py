from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime

from dotenv import load_dotenv
from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class Settings:
    dashscope_api_key: str
    bailian_base_url: str
    bailian_model_name: str
    e2b_api_key: str
    postgres_dsn: str
    memory_file_path: str
    user_id: str
    session_id: str
    session_state_file_path: str
    retrieve_top_k: int


def load_settings() -> Settings:
    load_dotenv()
    logger.info("Environment variables loaded from .env (if present)")

    settings = Settings(
        dashscope_api_key=_must_env("DASHSCOPE_API_KEY"),
        bailian_base_url=os.getenv(
            "BAILIAN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ),
        bailian_model_name=os.getenv("BAILIAN_MODEL_NAME", "qwen-plus"),
        e2b_api_key=_must_env("E2B_API_KEY"),
        postgres_dsn=_must_env("POSTGRES_DSN"),
        memory_file_path=os.getenv("CYBERCORE_MEMORY_FILE", "MEMORY.MD"),
        user_id=os.getenv("CYBERCORE_USER_ID", "cli_user"),
        session_id=os.getenv(
            "CYBERCORE_SESSION_ID",
            datetime.now().strftime("cli_session_%Y%m%d_%H%M%S"),
        ),
        session_state_file_path=os.getenv(
            "CYBERCORE_SESSION_STATE_FILE",
            ".cybercore_session.json",
        ),
        retrieve_top_k=int(os.getenv("CYBERCORE_RETRIEVE_TOP_K", "4")),
    )
    logger.info(
        "Settings loaded: model=%s, session_id=%s, user_id=%s, memory_file=%s, session_state_file=%s, top_k=%s",
        settings.bailian_model_name,
        settings.session_id,
        settings.user_id,
        settings.memory_file_path,
        settings.session_state_file_path,
        settings.retrieve_top_k,
    )
    return settings


def _must_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        logger.error("Missing required environment variable: %s", name)
        raise ValueError(f"Missing required environment variable: {name}")
    return value
