from __future__ import annotations

import argparse
import os
import sys

_SRC_DIR = os.path.join(os.path.dirname(__file__), "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from agents.logging_config import get_logger, setup_logging
from agents.cli import run_cli
from channels.feishu.gateway import run_feishu_gateway

setup_logging()
logger = get_logger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CyberCore entrypoint")
    subparsers = parser.add_subparsers(dest="mode")

    cli_parser = subparsers.add_parser("cli", help="Run CLI mode")
    cli_parser.add_argument(
        "--config",
        default="configs/app.json",
        help="Path to app config JSON for CLI mode",
    )
    feishu_parser = subparsers.add_parser("feishu", help="Run Feishu gateway mode")
    feishu_parser.add_argument(
        "--config",
        default="configs/app.json",
        help="Path to config JSON for Feishu gateway",
    )

    args = parser.parse_args()
    mode = args.mode or "cli"
    if mode == "feishu":
        logger.info("Starting CyberCore Feishu gateway from main entrypoint")
        run_feishu_gateway(config_path=args.config)
    else:
        logger.info("Starting CyberCore CLI from main entrypoint")
        run_cli(config_path=getattr(args, "config", "configs/app.json"))
