from __future__ import annotations

import argparse

from w_bot.agents.cli import run_cli
from w_bot.agents.logging_config import get_logger, setup_logging
from w_bot.channels.feishu.gateway import run_feishu_gateway


def main() -> None:
    setup_logging()
    logger = get_logger(__name__)

    parser = argparse.ArgumentParser(description="W-bot entrypoint")
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
        logger.info("Starting W-bot Feishu gateway from main entrypoint")
        run_feishu_gateway(config_path=args.config)
    else:
        logger.info("Starting W-bot CLI from main entrypoint")
        run_cli(config_path=getattr(args, "config", "configs/app.json"))


if __name__ == "__main__":
    main()
