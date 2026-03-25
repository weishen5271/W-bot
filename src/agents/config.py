from __future__ import annotations

import os
from dataclasses import dataclass

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
    milvus_uri: str
    memory_collection: str
    user_id: str
    thread_id: str
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
        milvus_uri=os.getenv("MILVUS_URI", "http://localhost:19530"),
        memory_collection=os.getenv(
            "CYBERCORE_MEMORY_COLLECTION", "cybercore_long_term_memory_cli"
        ),
        user_id=os.getenv("CYBERCORE_USER_ID", "cli_user"),
        thread_id=os.getenv("CYBERCORE_THREAD_ID", "cli_thread_main"),
        retrieve_top_k=int(os.getenv("CYBERCORE_RETRIEVE_TOP_K", "4")),
    )
    logger.info(
        "Settings loaded: model=%s, thread_id=%s, user_id=%s, milvus_uri=%s, top_k=%s",
        settings.bailian_model_name,
        settings.thread_id,
        settings.user_id,
        settings.milvus_uri,
        settings.retrieve_top_k,
    )
    return settings


def _must_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        logger.error("Missing required environment variable: %s", name)
        raise ValueError(f"Missing required environment variable: {name}")
    return value
