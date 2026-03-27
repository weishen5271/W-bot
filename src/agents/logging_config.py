from __future__ import annotations

import logging
import os


def setup_logging() -> None:
    """处理setup/logging相关逻辑并返回结果。
    """
    level_name = os.getenv("CYBERCORE_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def get_logger(name: str) -> logging.Logger:
    """处理get/logger相关逻辑并返回结果。
    
    Args:
        name: 名称参数，用于标识目标对象。
    """
    return logging.getLogger(name)
