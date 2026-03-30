# W-bot CLI Agent

[中文](./README.md) | [English](./README_EN.md)

Java version project: https://github.com/weishen5271/W-bot-java

W-bot 是一个基于 LangGraph 构建的 CLI Agent。

### 记忆架构

- 短期记忆：持久化到当前工作区 `memory/short_term_memory.pkl`，按 `session_id` 隔离。
- 长期记忆：持久化到本地 `MEMORY.MD`。

`MEMORY.MD` 结构：

- `User Information`
- `Preferences`
- `Project Context`
- `Important Notes`

该文件由 `save_memory` 工具自动维护。

### 快速开始

1. 安装依赖：

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
```

2. 配置 app JSON：

```bash
cp configs/app.json.example configs/app.json
# 然后在 configs/app.json 中填入真实 keys/secrets
```

3. 初始化用户档案：

```bash
wbot onboard
```

该命令会在用户目录创建 `~/.wbot/`，并从 `w_bot/template/` 复制缺失模板进去，只补齐缺失文件，不覆盖已有文件。

4. 运行 Agent CLI：

```bash
wbot agent --config configs/app.json
```

飞书网关模式：

```bash
wbot feishu --config configs/app.json
```

Web 页面模式：

```bash
wbot web --config configs/app.json
```

启动后打开浏览器访问：`http://127.0.0.1:8000`

如果 `configs/app.json` 不存在，程序会自动生成模板文件。
重启前请先填写 `channels.feishu.appId` 和 `channels.feishu.appSecret`。

输入 `quit` 或 `exit` 退出。

默认情况下，CLI 会自动恢复上一次短期会话。
在 CLI 中输入 `/new` 可开启全新的会话上下文。

所有运行时配置统一放在 `configs/app.json`。
短期记忆默认写入工作区 `memory/short_term_memory.pkl`，可通过 `agent.shortTermMemoryPath` 修改路径。
原有 `agent.shortTermMemoryOptimization` 配置仅适用于 PostgreSQL 方案，当前本地文件模式下会被忽略，建议关闭。

### 多 Agent 说明

当前项目默认是单 Agent 执行模型，不会自动判断并开启真正的多 Agent 协作。

- 主流程是 `retrieve_memories -> agent -> action -> agent`。
- `agent` 节点只判断当前回复是否包含 `tool_calls`，有则进入 `ToolNode`，否则结束。
- 工具层虽然提供了 `spawn` 接口，但目前只是把任务写入 `.w_bot_spawn_jobs.jsonl`，还没有子 Agent 拉起、结果回收、汇总归并这套闭环。

因此目前更准确的描述是“单 Agent + 工具调用”，而不是“自动多 Agent 编排”。

多模态能力可通过 `agent.multimodal.enabled` 开关控制，默认模板已包含：

- 飞书图片会下载到本地 `media/` 并以原生图像块送入模型。
- 音频/视频/文档会进入统一归一化流程，当前默认以文本占位/摘录回退。
- 历史消息中的大媒体会自动压缩为占位文本，避免会话上下文过大。
- 可通过 `agent.modelRouting` 配置不同模态使用的模型（text/image/audio）。

MCP server 配置示例：

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

### Skill 机制

项目支持基于文件的 Skill，并按需渐进加载：

- 内置 skills：`w_bot/agents/skills_catalog/<skill>/SKILL.md`
- 工作区 skills：`skills/<skill>/SKILL.md`（同名时覆盖内置）
- 满足条件时，`always: true` 的 skill 会注入系统提示词。
- 所有 skill 会在运行时摘要中展示，模型可按需通过 `read_file` 加载完整内容。

Skill 依赖从 frontmatter 元数据读取：

- `metadata.requires.bins`
- `metadata.requires.env`
- `metadata.always`（或顶层 `always`）

内置了 `clawhub` skill，可用于下载技能：

- `npx --yes clawhub@latest ...`

使用前请确保：

- 运行环境有 `npx`

### OpenClaw 档案结构（可选）

支持按 OpenClaw 风格管理 Agent 档案。模板文件存放在 `w_bot/template/`，初始化时会复制到用户目录 `~/.wbot/` 并只补齐缺失项。

默认模板包含：

- `AGENTS.md`、`SOUL.md`、`IDENTITY.md`、`USER.md`、`TOOLS.md`
- `BOOTSTRAP.md`、`BOOT.md`、`HEARTBEAT.md`
- `memory/MEMORY.md`、`memory/HISTORY.md`
- `skills/`

启动行为：

- `IDENTITY/SOUL/AGENTS/USER/TOOLS/BOOT/HEARTBEAT` 会注入系统提示词上下文。
- `BOOTSTRAP.md` 在启动时读取一次后自动删除（用完即删）。
- 当 `memoryFilePath` 仍为默认值时，会优先使用 `memory/MEMORY.md` 作为长期记忆文件。

相关配置项（`agent` 下）：

- `enableOpenClawProfile`：是否启用 OpenClaw 档案加载，默认 `true`
- `openClawProfileRootDir`：档案根目录，默认 `~/.wbot`
- `openClawAutoInit`：是否自动补齐缺失档案文件，默认 `true`

### 关键文件

- `w_bot/agents/cli.py`：CLI 运行时与本地短期记忆 checkpoint 连接。
- `w_bot/agents/agent.py`：LangGraph 节点与路由。
- `w_bot/agents/context.py`：系统提示词组装（memory + skills）。
- `w_bot/agents/file_checkpointer.py`：工作区本地短期记忆持久化。
- `w_bot/agents/memory.py`：本地 `MEMORY.MD` 长期记忆存储。
- `w_bot/agents/openclaw_profile.py`：OpenClaw 档案加载与启动期处理。
- `w_bot/agents/skills.py`：skill 发现、可用性检查与摘要渲染。
- `w_bot/agents/tools/runtime.py`：内置工具与 MCP 动态工具。
- `w_bot/agents/tools/runtime.py` 中的 `spawn`：当前只记录待处理任务，不会自动执行多 Agent 编排。
- `w_bot/channels/feishu/gateway.py`：飞书通道网关（WebSocket + agent 交互）。
- `w_bot/channels/web/gateway.py`：Web 通道网关（HTTP API + 内置聊天页面）。
- `configs/app.json`：统一应用配置（`agent + channels`）。
