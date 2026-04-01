# W-bot 项目优化建议

> 基于 claude-code 项目实现机制对比分析

**日期**: 2026-04-01
**项目**: W-bot vs claude-code

---

## 一、项目概述

### 1.1 W-bot 现状

```
w_bot/
├── agents/                    # Agent 核心
│   ├── agent.py              # LangGraph 状态机
│   ├── context.py            # 上下文构建（静态）
│   ├── memory.py             # 长期记忆（简单段落存储）
│   ├── short_memory_optimizer.py  # 后台压缩任务
│   ├── skills.py             # Skill 加载器
│   ├── tools/                # 工具集
│   │   ├── base.py           # 统一 Tool 基类
│   │   ├── filesystem.py     # 文件工具
│   │   ├── shell.py          # Shell 工具
│   │   ├── mcp.py            # MCP 工具
│   │   └── ...
│   └── multimodal/           # 多模态支持
├── channels/                 # 消息通道
├── template/                 # 提示词模板
└── configs/                   # 配置文件
```

### 1.2 claude-code 架构

```
src/
├── query.ts                  # 核心查询函数
├── QueryEngine.ts           # 查询引擎
├── context.ts               # 动态上下文构建
├── tools/                   # 独立工具目录
│   ├── BashTool/            # 每个工具有 UI.tsx + prompt.ts
│   ├── FileEditTool/
│   ├── SkillTool/           # Skill 执行（Forked）
│   └── ...
├── services/
│   ├── compact/             # 上下文压缩
│   │   ├── autoCompact.ts   # 自动压缩
│   │   ├── compact.ts       # 压缩逻辑
│   │   └── reactiveCompact.ts
│   └── api/                 # API 层
├── memdir/                  # 分层记忆系统
│   ├── memoryTypes.ts       # 4种记忆类型
│   └── memdir.ts
└── bootstrap/               # 状态管理
```

---

## 二、核心模块对比

### 2.1 上下文压缩与记忆管理

| 方面 | W-bot | claude-code |
|------|-------|-------------|
| **自动压缩触发** | 被动（基于配置） | 主动（基于 token 使用率 80%/90%/95%） |
| **压缩策略** | 简单历史摘要 | `compact.ts` + `reactiveCompact.ts` + `snipCompact.ts` |
| **压缩阈值** | 固定 `summary_trigger_messages=12` | 动态 `AUTOCOMPACT_BUFFER_TOKENS = 13,000` |
| **Token 追踪** | 估算 | API 真实 usage（含 cache tokens） |
| **记忆系统** | 简单段落存储 | 分层（user/feedback/project/reference） |
| **记忆写入指导** | 无 | 详细 `when_to_save` / `how_to_use` |

#### 代码对比

**W-bot - 被动压缩**

```python
# w_bot/agents/agent.py
def _prepare_optimized_context(self, *, state, history, config=None):
    # 简单保留最近 N 个用户回合
    recent = []
    user_turn_count = 0
    for msg in reversed(history):
        if isinstance(msg, HumanMessage):
            user_turn_count += 1
            if user_turn_count > self._token_opt.max_recent_user_turns:
                break
            recent.insert(0, msg)

    # 超过阈值时触发摘要
    if user_turn_count > self._token_opt.summary_trigger_messages:
        summary = _summarize_history(history)
```

**claude-code - 主动压缩**

```typescript
// src/services/compact/autoCompact.ts
const AUTOCOMPACT_BUFFER_TOKENS = 13_000
const WARNING_THRESHOLD_BUFFER_TOKENS = 20_000
const ERROR_THRESHOLD_BUFFER_TOKENS = 20_000

export function calculateTokenWarningState(tokenUsage: number, model: string) {
  const autoCompactThreshold = getAutoCompactThreshold(model)
  const threshold = isAutoCompactEnabled()
    ? autoCompactThreshold
    : getEffectiveContextWindowSize(model)

  return {
    percentLeft: Math.max(0, Math.round(((threshold - tokenUsage) / threshold) * 100)),
    isAboveWarningThreshold: tokenUsage >= warningThreshold,
    isAboveErrorThreshold: tokenUsage >= errorThreshold,
    isAboveAutoCompactThreshold: tokenUsage >= autoCompactThreshold,
    isAtBlockingLimit: tokenUsage >= blockingLimit,
  }
}
```

---

### 2.2 Token 使用追踪

| 方面 | W-bot | claude-code |
|------|-------|-------------|
| **输入/输出分离** | 混合计算 | `input_tokens` / `output_tokens` / `cache_tokens` 分离 |
| **缓存 Token** | 不追踪 | `cache_creation_input_tokens` / `cache_read_input_tokens` |
| **使用数据源** | 字符数估算 | API 响应真实数据 |
| **Task Budget** | 无 | 支持 `task_budget.total` 控制单次任务消耗 |

#### 代码对比

**W-bot - 简单估算**

```python
# w_bot/agents/agent.py
# 仅用字符数估算
def _estimate_tokens(self, text: str) -> int:
    return len(text) // 4  # 简单估算
```

**claude-code - 精确追踪**

```typescript
// src/utils/tokens.ts
export function getTokenCountFromUsage(usage: Usage): number {
  return (
    usage.input_tokens +
    (usage.cache_creation_input_tokens ?? 0) +
    (usage.cache_read_input_tokens ?? 0) +
    usage.output_tokens
  )
}

export function tokenCountWithEstimation(messages: Message[]): number {
  let total = 0
  for (const msg of messages) {
    const usage = getTokenUsage(msg)
    if (usage) {
      total += getTokenCountFromUsage(usage)  // 真实数据
    } else {
      total += roughTokenCountEstimation(msg)  // 估算
    }
  }
  return total
}
```

---

### 2.3 系统提示构建

| 方面 | W-bot | claude-code |
|------|-------|-------------|
| **构建时机** | 初始化时一次性构建 | 每次查询动态构建 |
| **缓存机制** | 无 | `memoize` 缓存整个会话 |
| **Git 状态** | 无 | 动态获取 `git status` |
| **CLAUDE.md** | 无 | 支持项目层级 `CLAUDE.md` 扫描 |
| **日期注入** | 无 | `Today's date is ${getLocalISODate()}` |

#### 代码对比

**W-bot - 静态构建**

```python
# w_bot/agents/context.py
class ContextBuilder:
    def build_static_system_prompt(self, *, base_prompt: str) -> str:
        blocks = [base_prompt.strip()]

        if self._openclaw_profile_loader:
            profile_context = self._openclaw_profile_loader.render_compact_profile_context()
            blocks.append(f"# OpenClaw Profile\n{profile_context}")

        return "\n\n---\n\n".join(blocks)
```

**claude-code - 动态缓存**

```typescript
// src/context.ts
export const getSystemContext = memoize(async (): Promise<{[k: string]: string}> => {
  const gitStatus = await getGitStatus()  // 动态获取
  return {
    ...(gitStatus && { gitStatus }),
    ...(feature('BREAK_CACHE_COMMAND') && injection
      ? { cacheBreaker: `[CACHE_BREAKER: ${injection}]` }
      : {}),
  }
})

export const getUserContext = memoize(async (): Promise<{[k: string]: string}> => {
  const claudeMd = shouldDisableClaudeMd ? null : getClaudeMds(await getMemoryFiles())
  return {
    ...(claudeMd && { claudeMd }),
    currentDate: `Today's date is ${getLocalISODate()}.`,
  }
})
```

---

### 2.4 工具系统架构

| 方面 | W-bot | claude-code |
|------|-------|-------------|
| **目录结构** | 扁平化（一个文件一个工具） | 独立目录（UI.tsx + prompt.ts + 实现） |
| **工具发现** | 手动注册 | 目录扫描 + 特性开关 |
| **UI 渲染** | 无 | 支持 React 组件渲染 |
| **动态可用性** | 无 | `isEnabled()` 动态控制 |
| **特性开关** | 无 | `feature('FLAG')` 动态加载 |

#### 代码对比

**W-bot - 统一基类**

```python
# w_bot/agents/tools/base.py
class Tool(ABC):
    @property
    @abstractmethod
    def name(self) -> str: pass

    @property
    @abstractmethod
    def description(self) -> str: pass

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]: pass

    @abstractmethod
    async def execute(self, **kwargs: Any) -> Any: pass

# 手动注册
class ToolRegistry:
    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool
```

**claude-code - 独立目录 + 条件加载**

```typescript
// src/tools/ (独立目录结构)
├── BashTool/
│   ├── BashTool.ts
│   ├── UI.tsx        // 渲染组件
│   └── prompt.ts     // 工具描述
├── FileEditTool/
│   └── ...
└── tools.ts          // 集中注册

// src/tools.ts
export function getAllBaseTools(): Tools {
  return [
    AgentTool,
    TaskOutputTool,
    BashTool,
    FileEditTool,
    // ... 50+ 工具
  ]
}

// 条件加载
const cronTools = feature('AGENT_TRIGGERS')
  ? [CronCreateTool, CronDeleteTool, CronListTool]
  : []

const WebBrowserTool = feature('WEB_BROWSER_TOOL')
  ? require('./tools/WebBrowserTool/WebBrowserTool.js').WebBrowserTool
  : null
```

---

### 2.5 Skill 系统

| 方面 | W-bot | claude-code |
|------|-------|-------------|
| **执行模式** | 当前 agent 直接执行 | Forked sub-agent 隔离执行 |
| **Token Budget** | 共享主对话 | 独立 budget |
| **记忆类型** | 简单段落 | 4 种分层类型 |
| **上下文注入** | 有限 | 完整项目上下文 |
| **发现机制** | 目录扫描 | 搜索 + 遥测 |

#### 代码对比

**W-bot - 直接执行**

```python
# w_bot/agents/skills.py
class SkillsLoader:
    def load_skill(self, name: str) -> str | None:
        for skill in self.list_skills():
            if skill.name == name:
                return skill.path.read_text()
        return None
    # 直接执行，无隔离
```

**claude-code - Forked 执行**

```typescript
// src/tools/SkillTool/SkillTool.ts
async function executeForkedSkill(
  command: Command & { type: 'prompt' },
  context: ToolUseContext,
): Promise<ToolResult> {
  const agentId = createAgentId()

  // 1. 创建独立 agent
  const result = await runAgent({
    agentDefinition: {
      agentType: 'forked_skill',
      tools: ['*'],
      maxTurns: 50,
      permissionMode: 'bubble',
    },
    // 2. Fork 上下文
    forkContextMessages: context.messages,
    // 3. 独立工具
    availableTools: context.options.tools.filter(t =>
      allowedTools.includes(t.name)
    ),
    // 4. 独立 Token Budget
  })

  // 5. 遥测
  logEvent('tengu_skill_tool_invocation', {
    command_name: commandName,
    execution_context: 'fork',
  })
}
```

---

### 2.6 权限系统

| 方面 | W-bot | claude-code |
|------|-------|-------------|
| **权限模式** | 简单检查 | `acceptEverything` / `denyOnConflict` / `manual` |
| **规则引擎** | 无 | 规则匹配 (`getDenyRuleForTool`) |
| **工具粒度** | 整体控制 | per-tool 权限 |
| **会话规则** | 无 | `alwaysAllowRules.session` |

#### 代码对比

**W-bot - 简单检查**

```python
# w_bot/agents/tools/shell.py
class ExecTool(_FsTool):
    async def execute(self, command: str | None = None, **kwargs):
        if not self._allowed_dir:
            raise PermissionError("Exec tool requires allowed_dir")
        # 直接执行
```

**claude-code - 分级权限**

```typescript
// src/utils/permissions/permissions.ts
export type PermissionMode =
  | 'acceptEverything'
  | 'denyOnConflict'
  | 'manual'

export function getDenyRuleForTool(
  tool: Tool,
  context: ToolUseContext,
): PermissionDecision | undefined {
  // 检查 alwaysAllowRules
  if (context.alwaysAllowRules?.command?.includes(tool.name)) {
    return { behavior: 'allow' }
  }

  // 检查 git 规则
  if (isGitOperation(tool.name) && context.gitAllowRule === 'deny') {
    return { behavior: 'deny', reason: 'git operations denied' }
  }
}
```

---

## 三、优化建议优先级

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        W-bot 优化路线图                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  高优先级:                                                                 │
│  ├── 1. Token 使用量精确追踪 + 多级警告阈值                                 │
│  │      - 从 API 响应提取真实 usage                                        │
│  │      - 实现 AUTOCOMPACT_BUFFER_TOKENS = 13,000 阈值                     │
│  │      └── 区分 input/output/cache_tokens                                 │
│  │                                                                           │
│  ├── 2. 自动上下文压缩                                                     │
│  │      - 监控 token 使用达到阈值时自动触发                                  │
│  │      - 支持多种压缩策略（摘要 / 语义合并）                               │
│  │      └── 压缩后保留关键决策点标记                                        │
│  │                                                                           │
│  └── 3. 增强记忆系统（分层、优先级）                                        │
│         - 引入 user/feedback/project/reference 类型                         │
│         - 记忆优先级机制                                                    │
│         └── 重要信息自动升级                                                │
│                                                                             │
│  中优先级:                                                                 │
│  ├── 4. 系统提示缓存机制                                                    │
│  │      - 引入 memoize 缓存                                                │
│  │      ├── 动态注入 git 状态                                               │
│  │      └── 支持 CLAUDE.md 项目扫描                                         │
│  │                                                                           │
│  ├── 5. 工具系统增强                                                        │
│  │      - 独立目录结构（UI + prompt + 实现）                                │
│  │      ├── 动态 isEnabled() 控制                                           │
│  │      └── 特性开关机制                                                    │
│  │                                                                           │
│  └── 6. 分级权限系统                                                        │
│         - acceptEverything / denyOnConflict / manual                        │
│         └── 工具粒度权限控制                                                │
│                                                                             │
│  低优先级:                                                                 │
│  ├── 7. Skill Forked 执行                                                  │
│  │      - 独立 Token Budget                                                │
│  │      └── Forked 进程隔离                                                │
│  │                                                                           │
│  └── 8. MCP OAuth 认证增强                                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 四、具体实施建议

### 4.1 Token 追踪增强

```python
# w_bot/agents/token_tracker.py (新增)

from dataclasses import dataclass
from typing import Any

@dataclass
class TokenUsage:
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    @property
    def total(self) -> int:
        return (
            self.input_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
            + self.output_tokens
        )

class TokenBudgetManager:
    # 阈值配置
    AUTOCOMPACT_BUFFER_TOKENS = 13_000
    WARNING_THRESHOLD_BUFFER_TOKENS = 20_000
    ERROR_THRESHOLD_BUFFER_TOKENS = 20_000

    def __init__(self, context_window: int, model: str):
        self.context_window = context_window
        self.model = model
        self.total_usage = TokenUsage(0, 0)

    def update_from_response(self, usage: dict[str, Any]) -> None:
        """从 API 响应更新使用量"""
        self.total_usage = TokenUsage(
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            cache_creation_input_tokens=usage.get("cache_creation_input_tokens", 0),
            cache_read_input_tokens=usage.get("cache_read_input_tokens", 0),
        )

    def calculate_state(self) -> dict[str, Any]:
        """计算当前 Token 状态"""
        threshold = self.context_window - self.AUTOCOMPACT_BUFFER_TOKENS
        usage = self.total_usage.total

        return {
            "percent_left": max(0, round(((threshold - usage) / threshold) * 100)),
            "is_above_warning": usage >= (threshold - self.WARNING_THRESHOLD_BUFFER_TOKENS),
            "is_above_error": usage >= (threshold - self.ERROR_THRESHOLD_BUFFER_TOKENS),
            "is_at_blocking": usage >= (threshold - 3_000),
            "should_auto_compact": usage >= threshold,
        }
```

### 4.2 自动压缩触发

```python
# w_bot/agents/auto_compactor.py (新增)

from .token_tracker import TokenBudgetManager

class AutoCompactor:
    def __init__(self, config: Any):
        self.enabled = config.get("autoCompactEnabled", True)
        self.threshold = config.get("autoCompactThreshold", 0.8)

    async def should_compact(
        self,
        messages: list[AnyMessage],
        token_manager: TokenBudgetManager,
    ) -> bool:
        if not self.enabled:
            return False

        state = token_manager.calculate_state()
        return (
            state["should_auto_compact"]
            or state["is_above_auto_compact_threshold"]
        )

    async def compact(
        self,
        messages: list[AnyMessage],
    ) -> list[AnyMessage]:
        """执行对话压缩"""
        # 1. 分离需要保留的消息（最近 N 轮）
        # 2. 压缩早期消息为摘要
        # 3. 插入压缩边界标记
        # 4. 返回压缩后的消息列表
```

### 4.3 分层记忆系统

```python
# w_bot/agents/memory/hierarchical.py (新增)

from enum import Enum
from dataclasses import dataclass
from datetime import datetime

class MemoryType(Enum):
    USER = "user"           # 用户信息、偏好
    FEEDBACK = "feedback"   # 用户反馈、纠正
    PROJECT = "project"     # 项目状态、目标
    REFERENCE = "reference" # 外部系统引用

@dataclass
class MemoryEntry:
    type: MemoryType
    content: str
    created_at: datetime
    priority: int  # 1-5, 越高越重要
    expires_at: datetime | None = None

    def should_upgrade(self) -> bool:
        """检查是否应该升级"""
        return self.priority >= 4

class HierarchicalMemoryStore:
    # 记忆写入指导
    TYPE_GUIDANCE = {
        MemoryType.USER: {
            "when_to_save": "学习到用户角色、偏好、职责时",
            "how_to_use": "根据用户背景调整回答风格和深度",
        },
        MemoryType.FEEDBACK: {
            "when_to_save": "用户纠正或确认方法时",
            "how_to_use": "指导未来行为避免重复错误",
        },
        MemoryType.PROJECT: {
            "when_to_save": "学习项目目标、截止日期、依赖时",
            "how_to_use": "理解请求背景，做出更相关建议",
        },
        MemoryType.REFERENCE: {
            "when_to_save": "学习外部系统引用时",
            "how_to_use": "需要时知道去哪里查找最新信息",
        },
    }
```

---

## 五、总结

通过对比 claude-code 的实现，W-bot 主要有以下优化方向：

| 优先级 | 领域 | 核心改进 |
|--------|------|----------|
| **高** | Token 追踪 | 从估算改为 API 真实数据追踪 |
| **高** | 上下文压缩 | 从被动触发改为基于 Token 使用率的主动压缩 |
| **高** | 记忆系统 | 从简单段落改为分层分类记忆 |
| **中** | 系统提示 | 增加动态 Git 状态、CLAUDE.md 支持 |
| **中** | 工具系统 | 独立目录结构 + UI 组件 + 特性开关 |
| **中** | 权限系统 | 分级权限模式 + 规则引擎 |
| **低** | Skill 执行 | Forked 隔离执行 |
| **低** | MCP | OAuth 认证增强 |

建议按照优先级逐步实施，优先解决 Token 追踪和上下文压缩问题，这两个是影响对话质量和稳定性的核心因素。
