# W-bot Subagent 技术方案

> 基于 claude-code Subagent 实现机制

**日期**: 2026-04-01
**项目**: W-bot Subagent 设计与实现
**参考**: claude-code `runAgent.ts` / `forkSubagent.ts` / `forkedAgent.ts`

---

## 一、现状对比

### 1.1 当前 W-bot 架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        W-bot 单一 Agent                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────┐    ┌──────────────┐    ┌─────────────────┐    │
│   │  retrieve   │───▶│  prepare     │───▶│     agent       │    │
│   │  memories   │    │  prompt      │    │  (LLM invoke)  │    │
│   └─────────────┘    └──────────────┘    └────────┬────────┘    │
│                                                   │              │
│                                          ┌────────▼────────┐     │
│                                          │     action      │     │
│                                          │ (tool executor) │     │
│                                          └────────┬────────┘     │
│                                                   │              │
│                              ┌────────────────────┼───────┐      │
│                              ▼                    ▼       ▼      │
│                        ┌──────────┐        ┌─────────┐ ┌────┐   │
│                        │  agent    │        │ recover │ │end │   │
│                        └──────────┘        └─────────┘ └────┘   │
│                                                                  │
│   问题：                                                          │
│   1. 所有任务在同一 Agent 内串行执行                               │
│   2. Skill 直接执行，无法隔离                                      │
│   3. 无独立 Token Budget 控制                                       │
│   4. 无法并行处理多任务                                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 claude-code Subagent 架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Main Agent (Parent)                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   query() ──▶ runAgent() ──▶ ┌─────────────────────────────────────┐   │
│                               │           fork context               │   │
│                               │  • forkContextMessages (共享历史)    │   │
│                               │  • 子 agent 独立系统提示              │   │
│                               │  • 独立 Token Budget                  │   │
│                               └──────────────────┬──────────────────┘   │
│                                                      │                  │
│                                    ┌─────────────────┼──────────────┐   │
│                                    ▼                 ▼              ▼   │
│                           ┌──────────────┐  ┌────────────┐  ┌─────────┐│
│                           │ Explore Agent│  │ Plan Agent │  │  Skill  ││
│                           │  (只读搜索)  │  │  (规划)    │  │ (Forked)││
│                           └──────┬───────┘  └─────┬──────┘  └───┬─────┘│
│                                  │                 │              │      │
│                                  ▼                 ▼              ▼      │
│                           ┌──────────────┐  ┌────────────┐  ┌─────────┐│
│                           │ 返回结构化    │  │ 返回计划   │  │ 返回    ││
│                           │ 发现结果     │  │ 建议       │  │ 执行结果││
│                           └──────────────┘  └────────────┘  └─────────┘│
│                                                                          │
│   特点：                                                                │
│   1. Parent 与 Subagent 共享部分上下文（forkContextMessages）            │
│   2. Subagent 有独立工具列表、权限模式、Token Budget                     │
│   3. 支持 Built-in Agent（Explore/Plan/Verify）和自定义 Agent            │
│   4. Skill 在 Forked Subagent 中执行，隔离主对话                        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 二、核心代码对比

### 2.1 Agent 定义与加载

#### W-bot - 简单 Skill 加载器

```python
# w_bot/agents/skills.py
class SkillsLoader:
    def list_skills(self) -> list[SkillSpec]:
        for item in self._scan_dir(self.workspace_skills_dir):
            merged[item.name] = item

    def load_skill(self, name: str) -> str | None:
        for skill in self.list_skills():
            if skill.name == name:
                return skill.path.read_text()
        return None
```

#### claude-code - 完整 Agent 定义

```typescript
// src/tools/AgentTool/loadAgentsDir.ts
export type AgentDefinition = {
  agentType: string
  whenToUse: string
  tools?: string[]
  disallowedTools?: string[]
  skills?: string[]           // 预加载的 Skills
  mcpServers?: AgentMcpServerSpec[]  // 独立的 MCP 服务器
  hooks?: HooksSettings       // 生命周期钩子
  model?: string | 'inherit'  // 可继承父模型
  effort?: EffortValue        // 努力程度配置
  permissionMode?: PermissionMode  // 独立权限模式
  maxTurns?: number           // 最大轮次限制
  background?: boolean        // 后台运行
  memory?: AgentMemoryScope   // 持久记忆范围
  isolation?: 'worktree' | 'remote'  // 隔离级别
  omitClaudeMd?: boolean       // 是否省略 CLAUDE.md
  getSystemPrompt: () => string
}

// 内置 Agent 示例
export const builtInAgents = {
  Explore: {
    agentType: 'Explore',
    tools: ['Grep', 'Glob', 'Read'],  // 只读工具
    maxTurns: 50,
    omitClaudeMd: true,  // 省略 CLAUDE.md 节省 token
    getSystemPrompt: () => 'You are a read-only research agent...',
  },
  Plan: {
    agentType: 'Plan',
    tools: ['Read', 'Bash'],  // 规划工具
    getSystemPrompt: () => 'You are a planning agent...',
  },
}
```

**对比总结**:

| 特性 | W-bot | claude-code |
|------|-------|-------------|
| Agent 类型 | 无 | 内置 + 自定义 |
| 工具列表 | 无 | per-agent 白名单/黑名单 |
| MCP 服务器 | 全局共享 | per-agent 独立配置 |
| 权限模式 | 统一 | per-agent 独立设置 |
| 模型继承 | 无 | 支持 `inherit` 关键字 |
| 最大轮次 | 全局 `max_tool_steps` | per-agent `maxTurns` |
| 记忆范围 | 统一 | per-agent `memory` 字段 |

---

### 2.2 Subagent 执行机制

#### W-bot - 直接执行（无隔离）

```python
# w_bot/agents/tools/skill.py (假设存在)
async def execute_skill(self, skill_name: str, args: dict) -> str:
    # 直接在当前 Agent 执行
    skill_content = self.skills_loader.load_skill(skill_name)
    # 执行 skill 内容
    result = await self.llm.agenerate([SystemMessage(content=skill_content), ...])
    return result
```

#### claude-code - Forked Subagent 执行

```typescript
// src/tools/AgentTool/runAgent.ts
export async function* runAgent({
  agentDefinition,
  promptMessages,
  toolUseContext,      // 父级上下文
  forkContextMessages, // 共享的上下文消息
  querySource,
  model,
  maxTurns,
  allowedTools,        // 独立的工具权限
  ...
}): AsyncGenerator<Message> {

  // 1. 创建独立的 Agent ID
  const agentId = override?.agentId ?? createAgentId()

  // 2. 处理上下文继承
  // 过滤掉不完整的 tool calls，避免 API 错误
  const contextMessages = forkContextMessages
    ? filterIncompleteToolCalls(forkContextMessages)
    : []
  const initialMessages = [...contextMessages, ...promptMessages]

  // 3. 独立的文件状态缓存
  const agentReadFileState = forkContextMessages !== undefined
    ? cloneFileStateCache(toolUseContext.readFileState)
    : createFileStateCacheWithSizeLimit(READ_FILE_STATE_CACHE_SIZE)

  // 4. 可选的 CLAUDE.md 省略（节省 token）
  const shouldOmitClaudeMd = agentDefinition.omitClaudeMd && ...
  const resolvedUserContext = shouldOmitClaudeMd
    ? omitClaudeMd(baseUserContext)
    : baseUserContext

  // 5. 独立的工具解析
  const availableTools = useExactTools
    ? toolUseContext.options.tools  // 直接使用父工具
    : resolveAgentTools(agentDefinition, toolUseContext)

  // 6. 运行独立的查询循环
  for await (const msg of query({
    messages: initialMessages,
    systemPrompt: resolvedSystemPrompt,
    tools: availableTools,
    canUseTool: wrappedCanUseTool,  // 包装的权限检查
    maxTurns: agentDefinition.maxTurns,
    ...
  })) {
    yield msg
  }
}
```

---

### 2.3 Forked Skill 执行

#### W-bot - 无 Forked 机制

```python
# W-bot 现状：Skill 直接在主 agent 中执行
# 问题：
# 1. Skill 消耗主对话的 Token Budget
# 2. Skill 执行失败会影响主对话
# 3. 无法并行执行多个 Skill
```

#### claude-code - 隔离的 Forked Skill

```typescript
// src/tools/SkillTool/SkillTool.ts
async function executeForkedSkill(
  command: Command & { type: 'prompt' },
  commandName: string,
  args: string | undefined,
  context: ToolUseContext,
  canUseTool: CanUseToolFn,
  parentMessage: AssistantMessage,
): Promise<ToolResult> {

  const startTime = Date.now()
  const agentId = createAgentId()
  const isBuiltIn = builtInCommandNames().has(commandName)
  const forkedSanitizedName = isBuiltIn || isBundled || isOfficialSkill
    ? commandName
    : 'custom'

  // 1. 构建 Skill 执行指令
  const skillPrompt = await command.getPromptForCommand(args, context)

  // 2. 创建子 Agent 上下文
  const forkContextMessages = [
    ...context.messages,  // 继承父消息历史
    createUserMessage({
      content: skillPrompt
    })
  ]

  // 3. 独立工具权限
  const allowedTools = ['Read', 'Write', 'Bash', 'Grep', 'Glob']

  // 4. Forked 执行（异步后台）
  const result = await runAgent({
    agentDefinition: {
      agentType: 'forked_skill',
      tools: ['*'],
      maxTurns: 50,
      permissionMode: 'bubble',  // 权限冒泡到父
    },
    promptMessages: forkContextMessages,
    canUseTool,
    forkContextMessages: context.messages,
    querySource: 'skill',
    availableTools: context.options.tools.filter(t =>
      allowedTools.includes(t.name)
    ),
  })

  // 5. 遥测数据收集
  logEvent('tengu_skill_tool_invocation', {
    command_name: forkedSanitizedName,
    execution_context: 'fork',
    invocation_trigger: queryDepth > 0 ? 'nested-skill' : 'claude-proactive',
  })

  return result
}
```

---

### 2.4 上下文共享机制

#### W-bot - 无

```python
# W-bot 现状：
# - 所有消息在同一上下文中
# - 无法共享部分上下文给子任务
# - Skill 执行污染主对话历史
```

#### claude-code - 精确的上下文共享

```typescript
// src/utils/forkedAgent.ts
export type CacheSafeParams = {
  // 必须与父请求完全一致才能共享 Prompt Cache
  systemPrompt: SystemPrompt
  userContext: { [k: string]: string }
  systemContext: { [k: string]: string }
  toolUseContext: ToolUseContext
  forkContextMessages: Message[]  // 共享的上下文消息
}

// Forked Agent 必须保证：
// 1. systemPrompt 完全一致 → 共享 Prompt Cache
// 2. tools 定义完全一致 → 共享 Tool Cache
// 3. model 完全一致 → 缓存键匹配
// 4. forkContextMessages 作为 API 请求的前缀消息
```

```typescript
// fork 子 Agent 的消息构建
export function buildForkedMessages(
  directive: string,
  assistantMessage: AssistantMessage,
): Message[] {
  // 1. 保留父 Agent 的完整回复（包括所有 tool_use blocks）
  const fullAssistantMessage = {
    ...assistantMessage,
    uuid: randomUUID(),
    message: {
      ...assistantMessage.message,
      content: [...assistantMessage.message.content]
    }
  }

  // 2. 为每个 tool_use 构建 placeholder tool_result
  // （所有子 Agent 使用相同的 placeholder 保证缓存命中）
  const toolUseBlocks = assistantMessage.message.content
    .filter((block): block is BetaToolUseBlock =>
      block.type === 'tool_use'
    )

  const toolResultBlocks = toolUseBlocks.map(block => ({
    type: 'tool_result' as const,
    tool_use_id: block.id,
    content: [{ type: 'text', text: 'Fork started — processing' }]
  }))

  // 3. 添加子 Agent 专属指令
  return [
    fullAssistantMessage,
    createUserMessage({
      content: [
        ...toolResultBlocks,
        { type: 'text', text: buildChildMessage(directive) }
      ]
    })
  ]
}
```

---

### 2.5 权限隔离

#### W-bot - 无权限隔离

```python
# W-bot 现状：
# - 所有工具共享同一权限检查
# - Skill 执行拥有完整工具权限
# - 无法限制特定 Skill 的工具访问
```

#### claude-code - 分层权限系统

```typescript
// src/tools/AgentTool/runAgent.ts
const agentGetAppState = () => {
  let toolPermissionContext = state.toolPermissionContext

  // 1. Agent 可定义独立权限模式
  if (agentPermissionMode && state.toolPermissionContext.mode !== 'bypassPermissions') {
    toolPermissionContext = {
      ...toolPermissionContext,
      mode: agentPermissionMode
    }
  }

  // 2. 异步 Agent 自动禁用权限提示
  if (shouldAvoidPrompts) {
    toolPermissionContext = {
      ...toolPermissionContext,
      shouldAvoidPermissionPrompts: true
    }
  }

  // 2. 独立工具列表（替换父级规则）
  if (allowedTools !== undefined) {
    toolPermissionContext = {
      ...toolPermissionContext,
      alwaysAllowRules: {
        cliArg: state.toolPermissionContext.alwaysAllowRules.cliArg,
        session: [...allowedTools]  // 仅允许列表中的工具
      }
    }
  }

  return { ...state, toolPermissionContext }
}
```

---

## 三、W-bot Subagent 技术方案

### 3.1 整体架构设计

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              W-bot Architecture                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                        Main WBotGraph                                │   │
│   │                                                                      │   │
│   │   ┌─────────────┐    ┌──────────────┐    ┌─────────────────────┐   │   │
│   │   │  retrieve   │───▶│  prepare     │───▶│       agent         │   │   │
│   │   │  memories   │    │  prompt      │    │   (LLM invoke)      │   │   │
│   │   └─────────────┘    └──────────────┘    └──────────┬──────────┘   │   │
│   │                                                       │             │   │
│   │                                          ┌────────────┼────────┐    │   │
│   │                                          ▼                         ▼    │   │
│   │                                   ┌──────────┐              ┌─────┐  │   │
│   │                                   │  action  │              │ end │  │   │
│   │                                   └─────┬────┘              └─────┘  │   │
│   │                                         │                          │   │
│   └─────────────────────────────────────────┼──────────────────────────┘   │
│                                             │                              │
│                              ┌──────────────┼──────────────────────┐     │
│                              ▼              ▼                              ▼     │
│   ┌──────────────────────────────────┐  ┌──────────────────────────────┐      │
│   │        SubAgent Executor        │  │     Skill Executor          │      │
│   │  ┌────────────────────────────┐  │  │  ┌────────────────────────┐ │      │
│   │  │ 1. 创建独立 Agent 实例     │  │  │  │ 1. Forked Skill 执行   │ │      │
│   │  │ 2. 继承部分父上下文        │  │  │  │ 2. 独立 Token Budget  │ │      │
│   │  │ 3. 独立工具列表/权限      │  │  │  │ 3. 结果回传主对话     │ │      │
│   │  │ 4. 独立 Turn 限制        │  │  │  └────────────────────────┘ │      │
│   │  └────────────────────────────┘  │  └──────────────────────────────┘      │
│   └──────────────────────────────────┘                                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 核心接口设计

```python
# w_bot/agents/subagent.py (新增)

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, AsyncGenerator
from enum import Enum
import asyncio
import uuid
import time
import json
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from .tools.base import Tool

class SubagentType(Enum):
    BUILTIN = "builtin"           # 内置 Agent (Explore, Plan)
    SKILL = "skill"              # Skill Forked 执行
    CUSTOM = "custom"            # 自定义 Agent
    FORK = "fork"                # 隐式 Fork

class PermissionMode(Enum):
    INHERIT = "inherit"          # 继承父级权限
    RESTRICTED = "restricted"    # 限制工具列表
    FULL = "full"                # 完整权限

@dataclass
class SubagentConfig:
    """Subagent 配置"""
    agent_type: SubagentType
    name: str
    description: str

    # 工具配置
    allowed_tools: list[str] | None = None  # None = 继承父级
    disallowed_tools: list[str] = field(default_factory=list)

    # 执行配置
    max_turns: int = 50
    timeout_seconds: int | None = None

    # 权限配置
    permission_mode: PermissionMode = PermissionMode.INHERIT

    # 上下文配置
    inherit_system_prompt: bool = True
    inherit_tools: bool = True
    fork_context_messages: list[BaseMessage] | None = None

    # 模型配置
    model: str | None = None  # None = 继承父模型

    # 记忆配置
    memory_scope: str = "local"  # user / project / local

    # 回调
    on_progress: Callable[[str], None] | None = None
    on_complete: Callable[[dict], None] | None = None

@dataclass
class SubagentResult:
    """Subagent 执行结果"""
    success: bool
    messages: list[BaseMessage]
    final_response: str
    tool_calls: list[dict]
    usage: dict | None = None
    error: str | None = None
    duration_seconds: float = 0.0

class SubagentExecutor:
    """Subagent 执行器"""

    def __init__(
        self,
        parent_graph: WBotGraph,
        tools: list[Tool],
    ) -> None:
        self._parent = parent_graph
        self._tools = tools
        self._subagents: dict[str, asyncio.Task] = {}

    async def execute_subagent(
        self,
        config: SubagentConfig,
        prompt: str,
        context_messages: list[BaseMessage] | None = None,
    ) -> SubagentResult:
        """执行一个 Subagent"""
        agent_id = str(uuid.uuid4())
        start_time = time.time()

        try:
            # 1. 构建子 Agent 消息
            messages = self._build_subagent_messages(
                config=config,
                prompt=prompt,
                context_messages=context_messages,
            )

            # 2. 解析工具
            tools = self._resolve_tools(config)

            # 3. 创建独立 LLM 实例
            llm = self._create_subagent_llm(config)

            # 4. 执行查询循环
            result_messages = []
            async for response in self._subagent_query_loop(
                agent_id=agent_id,
                config=config,
                messages=messages,
                llm=llm,
                tools=tools,
            ):
                result_messages.append(response)

            # 5. 构建结果
            return SubagentResult(
                success=True,
                messages=result_messages,
                final_response=self._extract_final_response(result_messages),
                tool_calls=self._extract_tool_calls(result_messages),
                duration_seconds=time.time() - start_time,
            )

        except asyncio.TimeoutError:
            return SubagentResult(
                success=False,
                messages=[],
                final_response="",
                tool_calls=[],
                error=f"Subagent timeout after {config.timeout_seconds}s",
                duration_seconds=time.time() - start_time,
            )
        except Exception as e:
            return SubagentResult(
                success=False,
                messages=[],
                final_response="",
                tool_calls=[],
                error=str(e),
                duration_seconds=time.time() - start_time,
            )

    async def execute_skill(
        self,
        skill_name: str,
        skill_content: str,
        arguments: dict,
        parent_messages: list[BaseMessage],
    ) -> SubagentResult:
        """Forked Skill 执行"""
        config = SubagentConfig(
            agent_type=SubagentType.SKILL,
            name=skill_name,
            description=f"Forked skill: {skill_name}",
            allowed_tools=['read_file', 'write_file', 'bash', 'grep', 'glob'],
            max_turns=30,
            permission_mode=PermissionMode.RESTRICTED,
            fork_context_messages=parent_messages,
        )

        # 替换 $ARGUMENTS 占位符
        prompt = skill_content.replace("$ARGUMENTS", json.dumps(arguments))

        return await self.execute_subagent(
            config=config,
            prompt=prompt,
            context_messages=parent_messages,
        )

    async def _subagent_query_loop(
        self,
        agent_id: str,
        config: SubagentConfig,
        messages: list[BaseMessage],
        llm: Any,
        tools: list[Tool],
    ) -> AsyncGenerator[BaseMessage, None]:
        """Subagent 专用查询循环"""
        turn_count = 0

        while turn_count < config.max_turns:
            turn_count += 1

            # LLM 调用
            response = await llm.agenerate(messages + [HumanMessage(content="")])

            if not response.tool_calls:
                yield response
                break

            # 工具执行
            tool_results = []
            for tool_call in response.tool_calls:
                result = await self._execute_tool(
                    tool_call=tool_call,
                    tools=tools,
                    config=config,
                )
                tool_results.append(result)
                yield result

            # 添加工具结果到消息
            messages.append(response)
            messages.extend(tool_results)

    def _resolve_tools(self, config: SubagentConfig) -> list[Tool]:
        """解析子 Agent 可用的工具"""
        if config.allowed_tools is not None:
            return [
                tool for tool in self._tools
                if tool.name in config.allowed_tools
            ]

        if config.disallowed_tools:
            return [
                tool for tool in self._tools
                if tool.name not in config.disallowed_tools
            ]

        return self._tools

    def _create_subagent_llm(self, config: SubagentConfig) -> Any:
        """创建子 Agent 专用 LLM"""
        base_llm = self._parent._llm_text_base

        if config.model:
            # 使用指定模型
            return base_llm  # TODO: 根据 config.model 创建新实例

        return base_llm

    def _build_subagent_messages(
        self,
        config: SubagentConfig,
        prompt: str,
        context_messages: list[BaseMessage] | None,
    ) -> list[BaseMessage]:
        """构建子 Agent 消息"""
        messages = []

        # Fork 上下文消息
        if config.fork_context_messages:
            messages.extend(config.fork_context_messages)

        # 系统提示
        if config.inherit_system_prompt:
            system_prompt = self._parent._prepared_system_prompt_base
            messages.append(SystemMessage(content=system_prompt))

        # 用户提示
        messages.append(HumanMessage(content=prompt))

        return messages

    def _extract_final_response(self, messages: list[BaseMessage]) -> str:
        """提取最终响应文本"""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and not msg.tool_calls:
                return msg.content or ""
        return ""

    def _extract_tool_calls(self, messages: list[BaseMessage]) -> list[dict]:
        """提取所有工具调用"""
        calls = []
        for msg in messages:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    calls.append({
                        "name": tc.name,
                        "args": tc.args,
                    })
        return calls

    async def _execute_tool(
        self,
        tool_call: Any,
        tools: list[Tool],
        config: SubagentConfig,
    ) -> BaseMessage:
        """执行单个工具调用"""
        tool_name = tool_call.name
        tool_args = tool_call.args

        tool = next((t for t in tools if t.name == tool_name), None)
        if not tool:
            return HumanMessage(
                content=f"Tool not found: {tool_name}",
                name="system"
            )

        # 权限检查
        if config.permission_mode == PermissionMode.RESTRICTED:
            if config.allowed_tools and tool_name not in config.allowed_tools:
                return HumanMessage(
                    content=f"Tool {tool_name} is not allowed in this context",
                    name="system"
                )

        try:
            result = await tool.execute(**tool_args)
            return HumanMessage(content=str(result), name=tool_name)
        except Exception as e:
            return HumanMessage(
                content=f"Tool execution failed: {str(e)}",
                name=tool_name
            )
```

### 3.3 内置 Subagent 定义

```python
# w_bot/agents/subagent_builtins.py (新增)

BUILTIN_SUBAGENTS = {
    "explore": {
        "name": "Explore",
        "description": "只读研究 Agent，用于搜索和分析代码库",
        "allowed_tools": ["read_file", "grep", "glob", "bash"],
        "disallowed_tools": ["write_file", "exec"],
        "max_turns": 50,
        "permission_mode": "restricted",
        "model": None,  # 继承父模型
        "system_prompt_template": """You are a read-only research agent.

## Your task
Research the codebase to find information requested by the parent agent.

## Rules
1. You can ONLY use: Read, Grep, Glob, Bash (read-only commands)
2. You CANNOT modify files or run write operations
3. Be thorough and report all relevant findings
4. Use structured output for your findings

## Output format
- Scope: <what you researched>
- Findings: <structured list of discoveries>
- Key files: <list of relevant file paths>""",
    },

    "plan": {
        "name": "Plan",
        "description": "规划 Agent，用于制定任务计划",
        "allowed_tools": ["read_file", "bash"],
        "disallowed_tools": ["write_file", "exec"],
        "max_turns": 30,
        "permission_mode": "restricted",
        "system_prompt_template": """You are a planning agent.

## Your task
Create a detailed plan to accomplish the given task.

## Rules
1. Analyze the requirements carefully
2. Break down into concrete steps
3. Consider dependencies and potential issues
4. Output a clear, actionable plan

## Output format
1. <step number>: <action description>
   - Files affected: <list>
   - Expected outcome: <description>""",
    },

    "verify": {
        "name": "Verify",
        "description": "验证 Agent，用于检查代码正确性",
        "allowed_tools": ["read_file", "bash", "grep", "glob"],
        "disallowed_tools": ["write_file"],
        "max_turns": 40,
        "permission_mode": "restricted",
        "system_prompt_template": """You are a verification agent.

## Your task
Verify the correctness of code changes or implementation.

## Rules
1. Only read and analyze code
2. Run tests or verification commands
3. Report findings objectively
4. Do not modify any files

## Output format
- Verified: <what was checked>
- Result: PASS / FAIL
- Details: <specific findings>""",
    },
}
```

### 3.4 集成到主 Agent

```python
# w_bot/agents/agent.py (修改)

class WBotGraph:
    async def _execute_skill_subagent(
        self,
        skill_name: str,
        arguments: dict,
    ) -> ToolMessage:
        """通过 Subagent 执行 Skill"""
        skill_content = self._skills_loader.load_skill(skill_name)
        if not skill_content:
            return ToolMessage(
                content=f"Skill not found: {skill_name}",
                tool_call_id="skill_error",
                name="skill"
            )

        # 获取当前消息历史作为 Fork 上下文
        current_messages = self._get_current_messages()

        # 创建 Subagent 执行器
        executor = SubagentExecutor(
            parent_graph=self,
            tools=list(self._tools_by_name.values()),
        )

        # Forked Skill 执行
        result = await executor.execute_skill(
            skill_name=skill_name,
            skill_content=skill_content,
            arguments=arguments,
            parent_messages=current_messages,
        )

        if result.success:
            return ToolMessage(
                content=result.final_response,
                tool_call_id=f"skill_{skill_name}",
                name="skill"
            )
        else:
            return ToolMessage(
                content=f"Skill execution failed: {result.error}",
                tool_call_id=f"skill_{skill_name}",
                name="skill"
            )

    async def _spawn_builtin_subagent(
        self,
        agent_type: str,
        task: str,
    ) -> str:
        """生成内置 Subagent"""
        if agent_type not in BUILTIN_SUBAGENTS:
            raise ValueError(f"Unknown subagent type: {agent_type}")

        config_dict = BUILTIN_SUBAGENTS[agent_type]
        config = SubagentConfig(
            agent_type=SubagentType.BUILTIN,
            name=config_dict["name"],
            description=config_dict["description"],
            allowed_tools=config_dict["allowed_tools"],
            disallowed_tools=config_dict["disallowed_tools"],
            max_turns=config_dict["max_turns"],
            permission_mode=PermissionMode.RESTRICTED,
        )

        executor = SubagentExecutor(
            parent_graph=self,
            tools=list(self._tools_by_name.values()),
        )

        result = await executor.execute_subagent(
            config=config,
            prompt=task,
            context_messages=self._get_current_messages(),
        )

        return result.final_response
```

### 3.5 CLI 命令集成

```python
# w_bot/agents/tools/subagent_cli.py (新增)

from .base import Tool, ToolResult

class SpawnSubagentTool(Tool):
    name = "spawn_subagent"
    description = "Spawn a subagent to handle a specific task"

    parameters = {
        "type": "object",
        "properties": {
            "agent_type": {
                "type": "string",
                "enum": ["explore", "plan", "verify"],
                "description": "Type of subagent to spawn"
            },
            "task": {
                "type": "string",
                "description": "Task description for the subagent"
            }
        },
        "required": ["agent_type", "task"]
    }

    async def execute(
        self,
        agent_type: str,
        task: str,
        agent: WBotGraph,
    ) -> ToolResult:
        try:
            result = await agent._spawn_builtin_subagent(
                agent_type=agent_type,
                task=task,
            )
            return ToolResult(success=True, content=result)
        except Exception as e:
            return ToolResult(success=False, content=str(e))
```

---

## 四、关键特性对比表

| 特性 | W-bot (现状) | W-bot (方案) | claude-code |
|------|-------------|---------------|-------------|
| **Subagent 类型** | 无 | 内置/Skill/Fork/自定义 | 内置/Skill/Fork/自定义 |
| **执行隔离** | 无 | Forked 进程隔离 | Forked 进程隔离 |
| **上下文共享** | 无 | forkContextMessages | forkContextMessages |
| **工具权限** | 统一权限 | per-Subagent 权限 | per-Subagent 权限 |
| **Token Budget** | 共享主对话 | 独立 Budget | 独立 Budget |
| **Turn 限制** | max_tool_steps_per_turn | per-Subagent maxTurns | per-Subagent maxTurns |
| **模型选择** | 固定模型 | 继承/指定模型 | 继承/指定模型 |
| **MCP 服务器** | 全局共享 | per-Subagent 独立 | per-Subagent 独立 |
| **记忆范围** | 统一存储 | per-Subagent memory scope | per-Subagent memory scope |
| **异步执行** | 无 | 后台 Task | 后台 Task |
| **结果回传** | 直接返回 | 结构化 Result | StreamEvent |
| **内置 Agent** | 无 | Explore/Plan/Verify | Explore/Plan/Verify/More |

---

## 五、实施计划

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           实施阶段规划                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Phase 1: 基础架构 (第1-2周)                                                │
│  ├── SubagentExecutor 核心类实现                                            │
│  ├── SubagentConfig / SubagentResult 数据结构                               │
│  ├── 基本的 Subagent 查询循环                                                │
│  └── 单元测试编写                                                           │
│                                                                              │
│  Phase 2: Skill Forked 执行 (第3-4周)                                       │
│  ├── Skill 执行集成到 SubagentExecutor                                      │
│  ├── 独立的 Tool 权限控制                                                   │
│  ├── Forked 上下文消息构建                                                  │
│  └── 集成测试                                                               │
│                                                                              │
│  Phase 3: 内置 Agent (第5-6周)                                               │
│  ├── 内置 Agent 定义 (Explore, Plan, Verify)                                │
│  ├── 内置 Agent 系统提示模板                                                │
│  ├── CLI 接口集成                                                           │
│  └── 端到端测试                                                             │
│                                                                              │
│  Phase 4: 高级特性 (第7-8周)                                                │
│  ├── 异步后台 Subagent                                                      │
│  ├── MCP 服务器 per-Subagent                                                │
│  ├── 记忆范围隔离                                                           │
│  └── 性能优化                                                               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 六、风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| Token Budget 超限 | Subagent 消耗过多资源 | 硬性 max_turns 限制 |
| 上下文污染 | Subagent 修改主对话状态 | 消息隔离，只读 forkContextMessages |
| 循环调用 | Subagent 再次调用主 Agent | 深度检测，禁止递归 |
| 权限逃逸 | Subagent 突破工具限制 | 工具白名单 + 权限模式检查 |
| 资源泄漏 | Subagent 任务未清理 | 独立的 Task 管理 + 超时机制 |

---

## 七、文件结构

```
w_bot/
├── agents/
│   ├── agent.py                    # 主 Agent (修改)
│   ├── subagent.py                  # SubagentExecutor (新增)
│   ├── subagent_builtins.py         # 内置 Agent 定义 (新增)
│   ├── subagent_cli.py              # CLI 工具 (新增)
│   ├── skills.py                    # Skill 加载器 (修改)
│   └── tools/
│       ├── base.py                  # Tool 基类 (修改)
│       └── ...
```

---

## 八、总结

通过参考 claude-code 的 Subagent 实现，W-bot 可以实现以下核心能力：

1. **任务隔离执行** - Skill 在独立 Subagent 中执行，不污染主对话
2. **资源独立控制** - per-Subagent Token Budget 和 Turn 限制
3. **灵活的权限管理** - per-Subagent 工具白名单/黑名单
4. **上下文共享** - forkContextMessages 机制复用父对话上下文
5. **内置专业 Agent** - Explore（只读研究）、Plan（规划）、Verify（验证）
6. **异步后台执行** - 支持后台 Subagent 任务
