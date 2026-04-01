# W-bot

[中文](./README.md) | [English](./README_EN.md)

Java 版本项目：[W-bot-java](https://github.com/weishen5271/W-bot-java)

W-bot 是一个基于 LangGraph 的本地 Agent 运行框架，当前已经支持：

- CLI 交互模式
- Web 聊天页面与流式接口
- 飞书消息网关
- 长短期记忆
- OpenClaw 风格档案加载
- Skill 机制
- 显式子 Agent 协作
- MCP 工具接入
- 多模态输入归一化
- Token 上下文优化

项目当前不再只是“一个 CLI Agent”，而是统一配置驱动下的多入口 Agent Runtime。

## 功能概览

### 1. 多入口运行

- `wbot agent` / `wbot cli`：本地终端交互
- `wbot web`：启动 Web 服务和内置聊天页面
- `wbot feishu`：启动飞书长连接网关

### 2. 会话与记忆

- 短期记忆持久化到工作区 `memory/short_term_memory.pkl`
- 长期记忆默认写入工作区 `memory/MEMORY.md`
- CLI 会自动恢复最近一次会话
- 支持列出、恢复、新建会话

### 3. Skill 机制

- 支持内置 Skills：`w_bot/agents/skills_catalog/<skill>/SKILL.md`
- 支持工作区 Skills：`skills/<skill>/SKILL.md`
- 工作区同名 skill 会覆盖内置 skill
- `always: true` 的 skill 会自动注入系统上下文
- 运行时会向模型暴露技能摘要，模型可按需读取完整 `SKILL.md`

当前仓库内可直接使用的工作区 skill 示例：

- `juejin-blog-publisher`
- `juejin-hot-news`
- `pdf-smart-tool-cn`
- `skill-creator-2`
- `weather`

内置 skill：

- `clawhub`
- `project-analyzer`

### 4. 显式子 Agent 协作

主流程仍然是单 Agent 主控，但已经支持通过工具显式拉起后台子 Agent：

- `spawn`
- `list_subagents`
- `wait_subagent`
- `run_skill`

更准确的描述是：

“单 Agent 主控 + 显式子 Agent 协作”，而不是完全自动编排的多 Agent 框架。

### 5. 多模态处理

当 `agent.multimodal.enabled=true` 时，W-bot 会对输入媒体做统一归一化：

- 图片可按原生图像输入处理
- 音频、视频、文档会先进入标准化管线
- 文档可提取文本摘录
- 大媒体会在历史上下文中压缩为占位信息，避免上下文膨胀

模型路由支持分别配置：

- `textModelName`
- `imageModelName`
- `audioModelName`

### 6. Token 优化

支持基于配置的上下文压缩与动态提示词增强：

- 最近轮次裁剪
- 对话摘要
- token 预算控制
- Git 状态注入
- 项目指令文件扫描

默认会扫描这些项目指令文件：

- `CLAUDE.md`
- `AGENTS.md`
- `WBOT.md`

### 7. 工具系统

当前运行时默认会注册这些核心工具：

- 文件读写与编辑
- 目录浏览
- Shell 执行
- Web 搜索
- Web 抓取
- 消息工具
- 长期记忆保存
- Skill 执行
- 子 Agent 管理
- MCP 动态工具
- 可选 Cron 工具

## 快速开始

### 环境要求

- Python `>=3.10`

### 1. 安装依赖

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

也可以使用可编辑安装：

```bash
pip install -e .
```

### 2. 准备配置

```bash
cp configs/app.json.example configs/app.json
```

然后填写至少一组模型提供商配置，例如：

- `providers.dashscope.apiKey`
- `providers.openai.apiKey`
- `providers.openrouter.apiKey`
- `providers.ollama.apiBase`

再指定默认模型：

- `agents.defaults.provider`
- `agents.defaults.model`

如果 `configs/app.json` 不存在，程序首次启动时也会自动生成模板文件。

### 3. 初始化 OpenClaw 档案

```bash
wbot onboard --config configs/app.json
```

默认会在 `~/.wbot/` 初始化档案，只补齐缺失文件，不覆盖已有文件。

### 4. 启动

CLI：

```bash
wbot agent --config configs/app.json
```

或：

```bash
wbot cli --config configs/app.json
```

Web：

```bash
wbot web --config configs/app.json
```

启动后访问：

`http://127.0.0.1:8000`

飞书：

```bash
wbot feishu --config configs/app.json
```

## CLI 用法

### 常用命令

```bash
wbot agent --config configs/app.json
wbot agent --config configs/app.json --new-session
wbot agent --config configs/app.json --session-id my_session
wbot new --config configs/app.json
wbot resume <session_id> --config configs/app.json
wbot sessions --config configs/app.json
```

### CLI 内部 Slash Commands

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

输入 `quit` 或 `exit` 也可以退出。

## 配置说明

所有运行时配置统一放在 `configs/app.json`，核心结构如下：

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

常用字段：

- `memoryFilePath`：长期记忆文件，默认 `memory/MEMORY.md`
- `shortTermMemoryPath`：短期记忆文件，默认 `memory/short_term_memory.pkl`
- `sessionStateFilePath`：CLI 会话状态文件，默认 `.w_bot_session.json`
- `retrieveTopK`：长期记忆检索条数
- `enableCronService`：是否注册 Cron 工具
- `mcpServers`：MCP 服务列表
- `enableSkills`：是否启用 Skills
- `skillsWorkspaceDir`：工作区 Skills 目录
- `modelRouting`：文本/图像/音频模型路由
- `multimodal`：多模态配置
- `tokenOptimization`：token 优化配置
- `enableOpenClawProfile`：是否启用 OpenClaw 档案
- `openClawProfileRootDir`：档案目录，默认 `~/.wbot`
- `openClawAutoInit`：是否自动补齐档案
- `loopGuard`：单轮工具调用与递归限制
- `enableStreaming`：流式相关开关
- `enableConsoleLogs`：是否输出控制台日志

注意：

- `shortTermMemoryOptimization` 这组配置目前在本地文件 checkpoint 模式下会被忽略

### `agents.defaults`

用于指定默认模型来源和模型名：

- `provider`
- `model`
- `temperature`

### `providers`

当前模板已预留多种 OpenAI 兼容提供商配置，例如：

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

Web 网关提供：

- `GET /`
- `GET /api/health`
- `POST /api/session/new`
- `GET /api/history`
- `POST /api/chat`
- `POST /api/chat/stream`

### `channels.feishu`

启动前至少需要填写：

- `appId`
- `appSecret`

可选控制项包括：

- `allowFrom`
- `groupPolicy`
- `replyToMessage`
- `reactEmoji`
- `encryptKey`
- `verificationToken`

## OpenClaw 档案

启用后，W-bot 会从 `~/.wbot/` 加载人格与运行约束上下文。

默认模板包含：

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

启动行为：

- `IDENTITY`、`SOUL`、`AGENTS`、`USER`、`TOOLS`、`BOOT`、`HEARTBEAT` 会注入系统提示词
- `BOOTSTRAP.md` 启动时读取一次后自动删除
- 当 `memoryFilePath` 为默认值时，会优先使用 OpenClaw 档案下的 `memory/MEMORY.md`

## MCP 配置示例

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

## 关键目录

- `w_bot/__main__.py`：统一 CLI 入口
- `w_bot/agents/agent.py`：LangGraph 主流程
- `w_bot/agents/cli.py`：CLI 运行时与交互命令
- `w_bot/agents/config.py`：配置加载与默认模板
- `w_bot/agents/context.py`：系统提示词与动态上下文构建
- `w_bot/agents/file_checkpointer.py`：短期记忆持久化
- `w_bot/agents/memory.py`：长期记忆存储
- `w_bot/agents/openclaw_profile.py`：OpenClaw 档案加载
- `w_bot/agents/skills.py`：Skill 发现、检查与摘要
- `w_bot/agents/subagent.py`：子 Agent 执行与调度
- `w_bot/agents/tools/runtime.py`：运行时工具注册
- `w_bot/channels/web/gateway.py`：Web 网关
- `w_bot/channels/feishu/gateway.py`：飞书网关
- `configs/app.json.example`：配置模板

## 说明

- 当前 README 基于仓库现有代码能力重写
- 若 `configs/app.json` 缺失，程序会自动生成模板并提示补充必填项
- Web 与 Feishu 入口和 CLI 共用同一套 Agent 配置与工具体系
