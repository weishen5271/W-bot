# CyberCore CLI Agent

CyberCore is a CLI agent built with LangGraph.

## Memory architecture

- Short-term memory: stored in PostgreSQL via `PostgresSaver`, isolated by `session_id`.
- Long-term memory: persisted to local `MEMORY.MD`.

`MEMORY.MD` structure:

- `User Information`
- `Preferences`
- `Project Context`
- `Important Notes`

The file is automatically maintained by the `save_memory` tool.

## Quick start

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

3. Configure env:

```bash
cp .env.example .env
```

4. Run:

```bash
python main.py
```

Type `quit` or `exit` to leave.

By default, the CLI resumes the previous short-term session automatically.
Type `/new` in the CLI to start a brand new session context.

Optional env:

- `CYBERCORE_SESSION_STATE_FILE`: file used to persist the latest `session_id` (default: `.cybercore_session.json`).

## Key files

- `src/agents/cli.py`: CLI runtime and Postgres checkpoint wiring.
- `src/agents/agent.py`: LangGraph nodes and routing.
- `src/agents/memory.py`: local `MEMORY.MD` long-term memory store.
- `src/agents/tools/runtime.py`: tools (`execute_python`, `save_memory`).
