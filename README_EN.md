# W-bot

[中文](./README.md) | [English](./README_EN.md)

Java version project: [W-bot-java](https://github.com/weishen5271/W-bot-java)

W-bot is a local agent runtime built on top of LangGraph. It currently supports:

- CLI interaction
- Web chat UI and streaming API
- Feishu message gateway
- Long-term and short-term memory
- OpenClaw-style profile loading
- Skill system
- Explicit subagent collaboration
- MCP tool integration
- Multimodal input normalization
- Token and context optimization

At this point, the project is no longer just a CLI agent. It is a multi-entry agent runtime driven by one shared configuration model.

## Feature Overview

### 1. Multiple Runtime Entrypoints

- `wbot agent` / `wbot cli`: local terminal interaction
- `wbot web`: starts the web service and built-in chat page
- `wbot feishu`: starts the Feishu long-connection gateway

### 2. Sessions and Memory

- Short-term memory is persisted to `memory/short_term_memory.sqlite`
- Long-term memory defaults to `memory/MEMORY.md`
- The CLI automatically resumes the most recent session
- You can list, resume, and create sessions explicitly

### 3. Skill System

- Built-in skills: `w_bot/agents/skills_catalog/<skill>/SKILL.md`
- Workspace skills: `skills/<skill>/SKILL.md`
- Workspace skills override built-in ones when names collide
- Skills marked with `always: true` are injected into the system context
- The runtime exposes a skill summary so the model can decide when to load full `SKILL.md` content

Workspace skill examples currently in this repo:

- `juejin-blog-publisher`
- `juejin-hot-news`
- `pdf-smart-tool-cn`
- `skill-creator-2`
- `weather`

Built-in skills:

- `clawhub`
- `project-analyzer`

### 4. Explicit Subagent Collaboration

The main flow is still a single controlling agent, but the runtime now supports explicitly spawning real background subagents via tools:

- `spawn`
- `list_subagents`
- `wait_subagent`
- `run_skill`

The most accurate description is:

"single-agent control with explicit subagent collaboration", not fully automatic multi-agent orchestration.

### 5. Multimodal Processing

When `agent.multimodal.enabled=true`, W-bot normalizes incoming media through a shared processing pipeline:

- Images can be passed as native image input
- Audio, video, and documents go through the normalization pipeline first
- Documents can be converted into text excerpts
- Large media items are compacted into placeholders in message history to reduce context pressure

Model routing can be configured independently for:

- `textModelName`
- `imageModelName`
- `audioModelName`

### 6. Token Optimization

W-bot includes configurable context and token management features:

- recent-turn trimming
- conversation summaries
- token budget control
- git status injection
- project instruction file scanning

By default, it looks for these project instruction files:

- `CLAUDE.md`
- `AGENTS.md`
- `WBOT.md`

### 7. Tooling

The runtime currently registers these core tool categories by default:

- file read, write, and edit
- directory listing
- shell execution
- web search
- web fetch
- message tools
- long-term memory save
- skill execution
- subagent management
- MCP dynamic tools
- optional cron tools

## Quick Start

### Requirements

- Python `>=3.10`

### 1. Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

You can also install the project in editable mode:

```bash
pip install -e .
```

If `uv` is not available on the server, you can use the bundled bootstrap script to install `uv`, create a Python 3.11 virtual environment, and install the project in editable mode:

```bash
bash bootstrap.sh
source .venv/bin/activate
```

Optional environment variables:

- `PYTHON_VERSION=3.10 bash bootstrap.sh`
- `VENV_DIR=.venv-prod bash bootstrap.sh`

### 2. Prepare the Config

```bash
cp configs/app.json.example configs/app.json
```

Then fill in at least one provider configuration, for example:

- `providers.dashscope.apiKey`
- `providers.openai.apiKey`
- `providers.openrouter.apiKey`
- `providers.ollama.apiBase`

Then set the default model selection:

- `agents.defaults.provider`
- `agents.defaults.model`

If `configs/app.json` is missing, the app can also generate a template automatically on first launch.

### 3. Initialize the OpenClaw Profile

```bash
wbot onboard --config configs/app.json
```

By default, this initializes `~/.wbot/` and only fills in missing files without overwriting existing ones.

### 4. Start the Runtime

CLI:

```bash
wbot agent --config configs/app.json
```

or:

```bash
wbot cli --config configs/app.json
```

Web:

```bash
wbot web --config configs/app.json
```

Then open:

`http://127.0.0.1:8000`

Feishu:

```bash
wbot feishu --config configs/app.json
```

## CLI Usage

### Common Commands

```bash
wbot agent --config configs/app.json
wbot agent --config configs/app.json --new-session
wbot agent --config configs/app.json --session-id my_session
wbot new --config configs/app.json
wbot resume <session_id> --config configs/app.json
wbot sessions --config configs/app.json
```

### In-CLI Slash Commands

- `/help`
- `/new [session_id]`
- `/resume <session_id>`
- `/session`
- `/history [count]`
- `/stats`
- `/cost`
- `/vim [on|off|toggle|status]`
- `/config`
- `/skills [skill_name]`
- `/clear`
- `/exit`

You can also type `quit` or `exit` to leave.

## Configuration

All runtime settings live in `configs/app.json`. The top-level structure looks like this:

```json
{
  "agent": {},
  "agents": {
    "defaults": {}
  },
  "providers": {},
  "channels": {
    "feishu": {},
    "web": {}
  },
  "threadPrefix": "feishu"
}
```

### `agent`

Common fields:

- `memoryFilePath`: long-term memory file, default `memory/MEMORY.md`
- `shortTermMemoryPath`: short-term memory file, default `memory/short_term_memory.sqlite`
- `sessionStateFilePath`: CLI session state file, default `configs/session_state.json`
- `escalationStateFilePath`: escalation request state file, default `configs/escalations.json`
- `retrieveTopK`: number of memory retrieval hits
- `enableCronService`: whether to register cron tools
- `mcpServers`: MCP server definitions
- `enableSkills`: whether to enable skills
- `skillsWorkspaceDir`: workspace skills directory
- `modelRouting`: text/image/audio model routing
- `multimodal`: multimodal settings
- `tokenOptimization`: token optimization settings
- `enableOpenClawProfile`: whether to load the OpenClaw profile
- `openClawProfileRootDir`: profile directory, default `~/.wbot`
- `openClawAutoInit`: whether to auto-create missing profile files
- `loopGuard`: recursion and tool-call limits
- `enableStreaming`: streaming-related switch
- `restrictToWorkspace`: whether to restrict file/command tools to the current workspace, default `false`
- `enableConsoleLogs`: whether to print runtime logs to the console

Note:

- `shortTermMemoryOptimization` is currently ignored when using the local file-based checkpoint mode

### `agents.defaults`

Used to define the default provider and model:

- `provider`
- `model`
- `temperature`

### `providers`

The default template already includes placeholders for many OpenAI-compatible providers, including:

- `custom`
- `azureOpenai`
- `anthropic`
- `openai`
- `openrouter`
- `deepseek`
- `groq`
- `zhipu`
- `dashscope`
- `vllm`
- `ollama`
- `ovms`
- `gemini`
- `moonshot`
- `minimax`
- `mistral`
- `stepfun`
- `aihubmix`
- `siliconflow`
- `volcengine`
- `volcengineCodingPlan`
- `byteplus`
- `byteplusCodingPlan`

### `channels.web`

```json
{
  "enabled": true,
  "host": "127.0.0.1",
  "port": 8000
}
```

The web gateway provides:

- `GET /`
- `GET /api/health`
- `POST /api/session/new`
- `GET /api/history`
- `POST /api/chat`
- `POST /api/chat/stream`

### `channels.feishu`

At minimum, fill in:

- `appId`
- `appSecret`

Optional controls include:

- `allowFrom`
- `groupPolicy`
- `replyToMessage`
- `reactEmoji`
- `encryptKey`
- `verificationToken`

## OpenClaw Profile

When enabled, W-bot loads persona and operating guidance from `~/.wbot/`.

The default profile template includes:

- `AGENTS.md`
- `SOUL.md`
- `IDENTITY.md`
- `USER.md`
- `TOOLS.md`
- `BOOTSTRAP.md`
- `BOOT.md`
- `HEARTBEAT.md`
- `memory/MEMORY.md`
- `memory/HISTORY.md`
- `skills/`

Startup behavior:

- `IDENTITY`, `SOUL`, `AGENTS`, `USER`, `TOOLS`, `BOOT`, and `HEARTBEAT` are injected into the system prompt
- `BOOTSTRAP.md` is read once at startup and then deleted
- if `memoryFilePath` is still set to the default path, W-bot prefers the profile's `memory/MEMORY.md`

## MCP Config Example

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

## Key Directories and Files

- `w_bot/__main__.py`: unified CLI entrypoint
- `w_bot/agents/agent.py`: main LangGraph flow
- `w_bot/agents/cli.py`: CLI runtime and interactive commands
- `w_bot/agents/config.py`: config loading and default template
- `w_bot/agents/context.py`: system prompt and dynamic runtime context builder
- `w_bot/agents/file_checkpointer.py`: short-term memory persistence
- `w_bot/agents/memory.py`: long-term memory storage
- `w_bot/agents/openclaw_profile.py`: OpenClaw profile loading
- `w_bot/agents/skills.py`: skill discovery, checks, and summary rendering
- `w_bot/agents/subagent.py`: subagent execution and coordination
- `w_bot/agents/tools/runtime.py`: runtime tool registration
- `w_bot/channels/web/gateway.py`: web gateway
- `w_bot/channels/feishu/gateway.py`: Feishu gateway
- `configs/app.json.example`: config template

## Notes

- This README has been rewritten based on the current codebase behavior
- If `configs/app.json` is missing, the app will generate a template and ask you to fill in required values
- CLI, Web, and Feishu all share the same core agent config and tool system
