from __future__ import annotations

import os
import sys

_SRC_DIR = os.path.join(os.path.dirname(__file__), "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from agents.logging_config import get_logger, setup_logging
from agents.cli import run_cli

setup_logging()
logger = get_logger(__name__)


if __name__ == "__main__":
    logger.info("Starting CyberCore CLI from main entrypoint")
    run_cli()
