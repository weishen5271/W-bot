from __future__ import annotations

import argparse
import json
from pathlib import Path

from w_bot.agents.cli import run_cli
from w_bot.agents.config import (
    DEFAULT_OPENCLAW_PROFILE_ROOT_DIR,
    normalize_openclaw_profile_root_dir,
)
from w_bot.agents.openclaw_profile import OpenClawProfileLoader
from w_bot.channels.feishu.gateway import run_feishu_gateway


def run_onboard(*, config_path: str = "configs/app.json", root_dir: str | None = None) -> None:
    target_root = root_dir
    if target_root is None:
        target_root = _resolve_profile_root_from_config(config_path) or DEFAULT_OPENCLAW_PROFILE_ROOT_DIR

    loader = OpenClawProfileLoader(root_dir=target_root, enabled=True, auto_init=True)
    created = loader.onboard()

    if created:
        print(f"Initialized W-bot profile at: {loader.root_dir}")
        for path in created:
            print(f"- {path}")
    else:
        print(f"W-bot profile already initialized: {loader.root_dir}")


def _resolve_profile_root_from_config(config_path: str) -> str | None:
    config_file = Path(config_path).expanduser()
    if not config_file.is_absolute():
        config_file = Path.cwd() / config_file
    if not config_file.exists():
        return None

    try:
        payload = json.loads(config_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None

    agent_cfg = payload.get("agent")
    if not isinstance(agent_cfg, dict):
        return None
    raw_root = agent_cfg.get("openClawProfileRootDir") or agent_cfg.get("openclaw_profile_root_dir")
    if not isinstance(raw_root, str):
        return None
    return normalize_openclaw_profile_root_dir(raw_root)


def main() -> None:
    parser = argparse.ArgumentParser(description="W-bot entrypoint")
    subparsers = parser.add_subparsers(dest="mode")

    onboard_parser = subparsers.add_parser("onboard", help="Initialize workspace scaffold")
    onboard_parser.add_argument(
        "--config",
        default="configs/app.json",
        help="Path to app config JSON used to resolve workspace root",
    )
    onboard_parser.add_argument(
        "--root",
        default=None,
        help="Workspace root directory override",
    )
    agent_parser = subparsers.add_parser("agent", help="Run agent CLI mode")
    agent_parser.add_argument(
        "--config",
        default="configs/app.json",
        help="Path to app config JSON for agent mode",
    )
    cli_parser = subparsers.add_parser("cli", help="Run agent CLI mode (legacy alias)")
    cli_parser.add_argument(
        "--config",
        default="configs/app.json",
        help="Path to app config JSON for agent mode",
    )
    feishu_parser = subparsers.add_parser("feishu", help="Run Feishu gateway mode")
    feishu_parser.add_argument(
        "--config",
        default="configs/app.json",
        help="Path to config JSON for Feishu gateway",
    )
    web_parser = subparsers.add_parser("web", help="Run Web gateway mode")
    web_parser.add_argument(
        "--config",
        default="configs/app.json",
        help="Path to config JSON for Web gateway",
    )

    args = parser.parse_args()
    mode = args.mode or "agent"
    if mode == "onboard":
        run_onboard(config_path=args.config, root_dir=args.root)
    elif mode == "feishu":
        run_feishu_gateway(config_path=args.config)
    elif mode == "web":
        from w_bot.channels.web.gateway import run_web_gateway

        run_web_gateway(config_path=args.config)
    else:
        run_cli(config_path=getattr(args, "config", "configs/app.json"))


if __name__ == "__main__":
    main()
