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

If `configs/app.json` does not exist, the app will auto-generate a template file.
Fill `channels.feishu.appId` and `channels.feishu.appSecret` before restarting.

Type `quit` or `exit` to leave.

By default, the CLI resumes the previous short-term session automatically.
Type `/new` in the CLI to start a brand new session context.

All runtime settings are configured in `configs/app.json`.
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

- Builtin skills: `src/agents/skills_catalog/<skill>/SKILL.md`
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

- `src/agents/cli.py`: CLI runtime and Postgres checkpoint wiring.
- `src/agents/agent.py`: LangGraph nodes and routing.
- `src/agents/context.py`: system prompt assembly (memory + skills).
- `src/agents/memory.py`: local `MEMORY.MD` long-term memory store.
- `src/agents/skills.py`: skill discovery, availability checks, and summary rendering.
- `src/agents/tools/runtime.py`: built-in tools and MCP dynamic tools.
- `src/channels/feishu/gateway.py`: Feishu channel gateway (WebSocket + agent interaction).
- `configs/app.json`: unified app configuration (`agent + channels`).
