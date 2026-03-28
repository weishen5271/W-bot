# W-bot CLI Agent

[中文](./README.md) | [English](./README_EN.md)

Java version project: https://github.com/weishen5271/W-bot-java

W-bot is a CLI agent built with LangGraph.

### Memory Architecture

- Short-term memory: stored in PostgreSQL via `PostgresSaver`, isolated by `session_id`.
- Long-term memory: persisted to local `MEMORY.MD`.

`MEMORY.MD` structure:

- `User Information`
- `Preferences`
- `Project Context`
- `Important Notes`

The file is automatically maintained by the `save_memory` tool.

### Quick Start

1. Start PostgreSQL:

```bash
docker compose up -d
```

2. Install dependencies:

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
```

3. Configure app JSON:

```bash
cp configs/app.json.example configs/app.json
# Then fill your real keys/secrets in configs/app.json
```

4. Run:

```bash
python main.py cli --config configs/app.json
```

Feishu gateway mode:

```bash
python main.py feishu --config configs/app.json
```

Web UI mode:

```bash
python main.py web --config configs/app.json
```

Then open: `http://127.0.0.1:8000`

If `configs/app.json` does not exist, the app will auto-generate a template file.
Fill `channels.feishu.appId` and `channels.feishu.appSecret` before restarting.

Type `quit` or `exit` to leave.

By default, the CLI resumes the previous short-term session automatically.
Type `/new` in the CLI to start a brand new session context.

All runtime settings are configured in `configs/app.json`.
Short-term memory optimization is controlled by `agent.shortTermMemoryOptimization` (enabled by default), including:

- Tiered memory: keep only the latest `keepRecentCheckpoints` checkpoints.
- Summary replacement: roll old checkpoints into `checkpoint_rolling_summaries` by `summaryBatchSize`.
- Dedup + compression: archive blobs into `checkpoint_blob_store` + `checkpoint_cold_archive_entries`, then delete old raw records.

Example MCP server config:

```json
[
  {
    "name": "planner",
    "base_url": "http://127.0.0.1:8081",
    "enabled": true,
    "discovery_path": "/tools",
    "invoke_path_template": "/tools/{tool}",
    "headers": {
      "Authorization": "Bearer <token>"
    }
  }
]
```

### Skill Mechanism

This project supports file-based skills with progressive loading:

- Builtin skills: `w_bot/agents/skills_catalog/<skill>/SKILL.md`
- Workspace skills: `skills/<skill>/SKILL.md` (overrides builtin on same name)
- `always: true` skills are injected into system prompt when requirements are met.
- All skills are listed in a runtime summary, and the model can load full content on demand using `read_file`.

Skill requirements are read from frontmatter metadata:

- `metadata.requires.bins`
- `metadata.requires.env`
- `metadata.always` (or top-level `always`)

Builtin `clawhub` skill is included for downloading skills via:

- `npx --yes clawhub@latest ...`

To use it, make sure:

- `agent.enableExecTool=true` in config
- host has `npx` available

### Key Files

- `w_bot/agents/cli.py`: CLI runtime and Postgres checkpoint wiring.
- `w_bot/agents/agent.py`: LangGraph nodes and routing.
- `w_bot/agents/context.py`: system prompt assembly (memory + skills).
- `w_bot/agents/memory.py`: local `MEMORY.MD` long-term memory store.
- `w_bot/agents/skills.py`: skill discovery, availability checks, and summary rendering.
- `w_bot/agents/tools/runtime.py`: built-in tools and MCP dynamic tools.
- `w_bot/channels/feishu/gateway.py`: Feishu channel gateway (WebSocket + agent interaction).
- `w_bot/channels/web/gateway.py`: Web channel gateway (HTTP API + built-in chat page).
- `configs/app.json`: unified app configuration (`agent + channels`).
