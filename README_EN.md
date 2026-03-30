# W-bot CLI Agent

[中文](./README.md) | [English](./README_EN.md)

Java version project: https://github.com/weishen5271/W-bot-java

W-bot is a CLI agent built with LangGraph.

### Memory Architecture

- Short-term memory: persisted in the workspace at `memory/short_term_memory.pkl`, isolated by `session_id`.
- Long-term memory: persisted to local `MEMORY.MD`.

`MEMORY.MD` structure:

- `User Information`
- `Preferences`
- `Project Context`
- `Important Notes`

The file is automatically maintained by the `save_memory` tool.

### Quick Start

1. Install dependencies:

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
```

2. Configure app JSON:

```bash
cp configs/app.json.example configs/app.json
# Then fill your real keys/secrets in configs/app.json
```

3. Initialize user profile files:

```bash
wbot onboard
```

This creates `~/.wbot/` in the user's home directory and copies missing template files from `w_bot/template/` into it without overwriting existing files.

4. Run the agent CLI:

```bash
wbot agent --config configs/app.json
```

Feishu gateway mode:

```bash
wbot feishu --config configs/app.json
```

Web UI mode:

```bash
wbot web --config configs/app.json
```

Then open: `http://127.0.0.1:8000`

If `configs/app.json` does not exist, the app will auto-generate a template file.
Fill `channels.feishu.appId` and `channels.feishu.appSecret` before restarting.

Type `quit` or `exit` to leave.

By default, the CLI resumes the previous short-term session automatically.
Type `/new` in the CLI to start a brand new session context.

All runtime settings are configured in `configs/app.json`.
Short-term memory is stored in `memory/short_term_memory.pkl` by default and can be changed with `agent.shortTermMemoryPath`.
The old `agent.shortTermMemoryOptimization` settings only applied to the PostgreSQL implementation and are ignored in the current local-file mode.

### Multi-Agent Notes

The current project runs as a single-agent workflow by default. It does not automatically decide to fan out into a real multi-agent execution plan.

- The main graph is `retrieve_memories -> agent -> action -> agent`.
- The `agent` node only decides whether the current model response contains `tool_calls`.
- The `spawn` tool exists, but currently only writes a pending job into `.w_bot_spawn_jobs.jsonl`.
- There is no built-in sub-agent launcher, result collector, or reducer loop yet.

So the current implementation is more accurately described as "single agent + tool calls", not automatic multi-agent orchestration.

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

- host has `npx` available

### OpenClaw Profile Files

OpenClaw-style profile templates now live under `w_bot/template/`. During onboarding or auto-init, missing files are copied into `~/.wbot/` without overwriting existing ones.

Default templates include:

- `AGENTS.md`, `SOUL.md`, `IDENTITY.md`, `USER.md`, `TOOLS.md`
- `BOOTSTRAP.md`, `BOOT.md`, `HEARTBEAT.md`
- `memory/MEMORY.md`, `memory/HISTORY.md`
- `skills/`

### Key Files

- `w_bot/agents/cli.py`: CLI runtime and local short-term memory checkpoint wiring.
- `w_bot/agents/agent.py`: LangGraph nodes and routing.
- `w_bot/agents/context.py`: system prompt assembly (memory + skills).
- `w_bot/agents/file_checkpointer.py`: workspace-local short-term memory persistence.
- `w_bot/agents/memory.py`: local `MEMORY.MD` long-term memory store.
- `w_bot/agents/skills.py`: skill discovery, availability checks, and summary rendering.
- `w_bot/agents/tools/runtime.py`: built-in tools and MCP dynamic tools. The `spawn` tool currently only records jobs and does not run a full multi-agent workflow.
- `w_bot/channels/feishu/gateway.py`: Feishu channel gateway (WebSocket + agent interaction).
- `w_bot/channels/web/gateway.py`: Web channel gateway (HTTP API + built-in chat page).
- `configs/app.json`: unified app configuration (`agent + channels`).
