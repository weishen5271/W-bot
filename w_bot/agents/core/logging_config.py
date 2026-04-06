from __future__ import annotations

import logging
import os


def setup_logging(*, enable_console_logs: bool | None = None) -> None:
    """处理setup/logging相关逻辑并返回结果。
    """
    if enable_console_logs is None:
        enable_console_logs = _bool_env("CYBERCORE_ENABLE_CONSOLE_LOGS", default=True)
    level_name = os.getenv("CYBERCORE_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    if not enable_console_logs:
        logging.basicConfig(level=logging.CRITICAL, handlers=[logging.NullHandler()], force=True)
        return

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )
    # Suppress verbose third-party HTTP request logs in long-running sessions.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """处理get/logger相关逻辑并返回结果。
    
    Args:
        name: 名称参数，用于标识目标对象。
    """
    return logging.getLogger(name)


def _bool_env(key: str, *, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}
