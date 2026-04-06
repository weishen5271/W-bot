from __future__ import annotations

import argparse
import json
from pathlib import Path

from w_bot.agents.core.config import (
    DEFAULT_APP_CONFIG_PATH,
    DEFAULT_OPENCLAW_PROFILE_ROOT_DIR,
    load_settings,
    normalize_openclaw_profile_root_dir,
)
from w_bot.agents.core.openclaw_profile import OpenClawProfileLoader
from w_bot.agents.core.session_store import SessionStateStore


def run_onboard(*, config_path: str = DEFAULT_APP_CONFIG_PATH, root_dir: str | None = None) -> None:
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
    parser = argparse.ArgumentParser(
        description="W-bot CLI entrypoint",
        epilog="Interactive CLI supports /help, /new, /resume, /session, /history, /stats, /cost, /vim, /config, /skills, /clear, /exit.",
    )
    subparsers = parser.add_subparsers(dest="mode")

    onboard_parser = subparsers.add_parser("onboard", help="Initialize workspace scaffold")
    onboard_parser.add_argument(
        "--config",
        default=DEFAULT_APP_CONFIG_PATH,
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
        default=DEFAULT_APP_CONFIG_PATH,
        help="Path to app config JSON for agent mode",
    )
    agent_parser.add_argument(
        "--session-id",
        default=None,
        help="Resume or pin a specific session id",
    )
    agent_parser.add_argument(
        "--new-session",
        action="store_true",
        help="Start a brand new session instead of restoring the last one",
    )
    cli_parser = subparsers.add_parser("cli", help="Run agent CLI mode (legacy alias)")
    cli_parser.add_argument(
        "--config",
        default=DEFAULT_APP_CONFIG_PATH,
        help="Path to app config JSON for agent mode",
    )
    cli_parser.add_argument(
        "--session-id",
        default=None,
        help="Resume or pin a specific session id",
    )
    cli_parser.add_argument(
        "--new-session",
        action="store_true",
        help="Start a brand new session instead of restoring the last one",
    )
    new_parser = subparsers.add_parser("new", help="Start CLI with a new session")
    new_parser.add_argument(
        "--config",
        default=DEFAULT_APP_CONFIG_PATH,
        help="Path to app config JSON for agent mode",
    )
    new_parser.add_argument(
        "--session-id",
        default=None,
        help="Optional custom session id",
    )
    resume_parser = subparsers.add_parser("resume", help="Start CLI and resume a specific session")
    resume_parser.add_argument(
        "session_id",
        help="Session id to resume",
    )
    resume_parser.add_argument(
        "--config",
        default=DEFAULT_APP_CONFIG_PATH,
        help="Path to app config JSON for agent mode",
    )
    sessions_parser = subparsers.add_parser("sessions", help="List recent CLI sessions")
    sessions_parser.add_argument(
        "--config",
        default=DEFAULT_APP_CONFIG_PATH,
        help="Path to app config JSON for agent mode",
    )
    feishu_parser = subparsers.add_parser("feishu", help="Run Feishu gateway mode")
    feishu_parser.add_argument(
        "--config",
        default=DEFAULT_APP_CONFIG_PATH,
        help="Path to config JSON for Feishu gateway",
    )
    web_parser = subparsers.add_parser("web", help="Run Web gateway mode")
    web_parser.add_argument(
        "--config",
        default=DEFAULT_APP_CONFIG_PATH,
        help="Path to config JSON for Web gateway",
    )

    args = parser.parse_args()
    mode = args.mode or "agent"
    if mode == "onboard":
        run_onboard(config_path=args.config, root_dir=args.root)
    elif mode == "feishu":
        from w_bot.channels.feishu.gateway import run_feishu_gateway

        run_feishu_gateway(config_path=args.config)
    elif mode == "web":
        from w_bot.channels.web.gateway import run_web_gateway

        run_web_gateway(config_path=args.config)
    elif mode == "sessions":
        settings = load_settings(config_path=args.config)
        sessions = SessionStateStore(settings.session_state_file_path).list_recent()
        if not sessions:
            print("No saved CLI sessions.")
            return
        print("Recent CLI sessions:")
        for record in sessions:
            print(f"- {record.session_id} ({record.updated_at})")
    elif mode == "resume":
        from w_bot.agents.core.cli import run_cli

        run_cli(config_path=args.config, session_id=args.session_id)
    elif mode == "new":
        from w_bot.agents.core.cli import run_cli

        run_cli(
            config_path=args.config,
            session_id=args.session_id,
            force_new_session=True,
        )
    else:
        from w_bot.agents.core.cli import run_cli

        run_cli(
            config_path=getattr(args, "config", DEFAULT_APP_CONFIG_PATH),
            session_id=getattr(args, "session_id", None),
            force_new_session=bool(getattr(args, "new_session", False)),
        )


if __name__ == "__main__":
    main()
