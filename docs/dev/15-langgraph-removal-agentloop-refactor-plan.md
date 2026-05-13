# W-bot 去除 LangGraph 并改造 AgentLoop 重构方案

本文记录 W-bot 从当前 LangGraph 编排迁移到显式 AgentLoop 的重构方案。方案参考 `D:\github\hermes-agent` 中显式 agent loop、业务级会话存储和工具执行调度的设计思路，但不直接照搬 Hermes 代码。

目标是降低 LangGraph checkpoint 与业务状态耦合带来的维护成本，让 W-bot 的 CLI、Web、飞书等入口统一使用一个可审计、可测试、可逐步演进的 AgentRuntime。

## 1. 背景与问题

当前 W-bot 的主流程由 `w_bot/agents/core/agent.py` 中的 `WBotGraph` 构建：

```text
START
  ↓
retrieve_memories
  ↓
prepare_prompt_context
  ↓
agent
  ├─ tool_calls → action → agent/recover
  └─ no tool_calls → END
```

这个结构在早期能快速搭建 agent 闭环，但现在已经出现几个问题：

- 图节点数量很少，真实复杂逻辑集中在 `_agent` 和 `_action_async`，LangGraph 更像外层壳子。
- 会话状态主要依赖 LangGraph checkpoint，业务层难以直接查询、审计、迁移和修复。
- `WorkspaceFileCheckpointer` 为了适配 LangGraph checkpoint 结构维护了额外 SQLite 存储，复杂度较高。
- 工具调用当前默认 `asyncio.gather` 并发执行，对写文件、记忆写入、shell、子 agent 等有顺序风险。
- `graph.invoke()` 后还要额外刷新会话搜索索引、后台摘要和状态更新，容易形成双写和一致性问题。
- Web、飞书、CLI 都围绕 `graph.invoke/get_state` 展开，业务接口不够直接。

本次重构目标不是简单替换一个库，而是把 W-bot 的 agent 主循环变成显式业务流程。

## 2. 重构目标

目标架构如下：

```text
CLI / Web / Feishu
  ↓
AgentRuntime.run_turn(session_id, inbound_messages, config)
  ↓
SessionStore
  ├─ 读取历史消息
  ├─ 写入用户消息
  ├─ 写入 assistant/tool 消息
  └─ 维护摘要、token、元数据
  ↓
ContextBuilder / MemoryContextRetriever / ContextOptimizer
  ↓
ModelRouter / ModelRunner
  ↓
ToolExecutor / ToolPolicy
  ↓
PostTurnProcessor
  ├─ 会话搜索索引
  └─ 延迟摘要刷新
```

核心目标：

- 去掉 LangGraph 作为主编排和主状态来源。
- 保留现有 LangChain message 对象，降低迁移风险。
- 新增业务级 `SessionStore`，用 `sessions/messages` 表保存真实会话历史。
- 用显式 `while` loop 实现模型调用、工具调用、恢复和终止。
- 工具执行按读写风险选择并发或串行。
- 网关和 CLI 只依赖 `AgentRuntime`，不再直接依赖 graph。
- 保留现有长期记忆、技能、MCP、审批、子 agent、流式回调等能力。

## 3. 非目标

第一阶段不做这些事：

- 不重写所有工具注册体系。
- 不把 LangChain 消息对象整体替换成 Hermes 风格 dict。
- 不做旧 LangGraph checkpoint 的完整迁移。
- 不调整模型 provider 的整体抽象。
- 不顺手重构 Web UI、飞书平台协议或 CLI 交互。
- 不引入新的数据库或外部依赖。

旧 checkpoint 可以先保留文件，不作为新会话的主事实来源。

## 4. 建议新增模块

建议新增或拆分以下模块：

```text
w_bot/agents/core/session_db.py
w_bot/agents/core/session_models.py
w_bot/agents/core/runtime.py
w_bot/agents/core/turn.py
w_bot/agents/core/memory_context.py
w_bot/agents/core/model_runner.py
w_bot/agents/core/model_routing.py
w_bot/agents/core/context_optimizer.py
w_bot/agents/core/tool_executor.py
w_bot/agents/core/tool_policy.py
w_bot/agents/core/runtime_compat.py
w_bot/agents/core/post_turn.py
```

其中：

- `session_db.py`：业务级 SQLite 会话存储。
- `runtime.py`：显式 AgentRuntime 主入口。
- `turn.py`：运行配置、运行结果等轻量数据结构。
- `memory_context.py`：从长期记忆检索本轮上下文。
- `model_runner.py`：统一模型调用、流式回调、fallback。
- `model_routing.py`：承接当前文本/图像/音频路由逻辑。
- `context_optimizer.py`：承接上下文压缩、摘要、token budget 逻辑。
- `tool_executor.py`：执行工具调用。
- `tool_policy.py`：判断工具是否可并发、是否高风险。
- `runtime_compat.py`：给旧工具提供类似 graph 的兼容对象。
- `post_turn.py`：会话搜索索引与延迟摘要刷新。

## 5. SessionStore 设计

新增业务级 SQLite 存储，替代 LangGraph checkpoint 作为主事实来源。

建议表结构：

```sql
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    user_id TEXT,
    model TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    ended_at TEXT,
    message_count INTEGER NOT NULL DEFAULT 0,
    tool_call_count INTEGER NOT NULL DEFAULT 0,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    summary TEXT,
    summarized_message_count INTEGER NOT NULL DEFAULT 0,
    metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT,
    tool_call_id TEXT,
    tool_calls_json TEXT,
    tool_name TEXT,
    additional_kwargs_json TEXT,
    response_metadata_json TEXT,
    token_usage_json TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE INDEX IF NOT EXISTS idx_messages_session_id_id
    ON messages(session_id, id);
```

最小接口：

```python
class SessionStore:
    def ensure_session(self, *, session_id: str, source: str, user_id: str, model: str) -> None: ...
    def append_message(self, session_id: str, message: AnyMessage) -> None: ...
    def append_messages(self, session_id: str, messages: list[AnyMessage]) -> None: ...
    def get_messages(self, session_id: str, *, limit: int | None = None) -> list[AnyMessage]: ...
    def get_summary(self, session_id: str) -> tuple[str, int]: ...
    def update_summary(self, session_id: str, *, summary: str, summarized_message_count: int) -> None: ...
    def get_token_usage(self, session_id: str) -> dict[str, int]: ...
```

序列化要求：

- `HumanMessage`、`AIMessage`、`ToolMessage` 必须能完整往返。
- `AIMessage.tool_calls` 必须保存到 `tool_calls_json`。
- `ToolMessage.tool_call_id` 和 `name` 必须保存。
- `additional_kwargs` 用于多模态媒体等信息，不能丢失。
- `response_metadata` 和 token usage 应保存，便于后续统计与调试。

## 6. AgentRuntime 入口设计

新增统一入口：

```python
@dataclass
class RuntimeConfig:
    recursion_limit: int = 20
    defer_summary_update: bool = True
    status_callback: Callable[[str], None] | None = None
    stream_token_callback: Callable[[str], None] | None = None
    debug_callback: Callable[[str], None] | None = None
    tool_progress_callback: Callable[..., None] | None = None


@dataclass
class AgentTurnResult:
    session_id: str
    final_response: str
    messages: list[AnyMessage]
    tool_calls: int
    completed: bool
    error: str = ""


class AgentRuntime:
    def run_turn(
        self,
        *,
        session_id: str,
        inbound_messages: list[AnyMessage],
        config: RuntimeConfig,
    ) -> AgentTurnResult:
        ...
```

`RuntimeConfig` 替代当前通过 `RunnableConfig["configurable"]` 传递的回调和运行参数。

兼容期内可以提供转换函数：

```python
def runtime_config_from_graph_config(config: dict[str, Any]) -> RuntimeConfig:
    ...
```

## 7. 显式 AgentLoop 流程

新的主循环应尽量直观：

```python
def run_turn(...):
    session_store.ensure_session(...)
    session_store.append_messages(session_id, inbound_messages)

    messages = session_store.get_messages(session_id)
    summary, summarized_count = session_store.get_summary(session_id)

    consecutive_tool_failures = 0

    for iteration in range(max_model_iterations):
        memory_context = memory_retriever.retrieve_for_turn(...)
        optimized = context_optimizer.prepare(...)
        llm, route, selected_tools = model_router.select(...)
        assistant = model_runner.invoke(...)

        assistant = loop_guard.apply_agent_guards(...)
        session_store.append_message(session_id, assistant)
        messages.append(assistant)

        if not assistant.tool_calls:
            post_turn_processor.after_turn(session_id, config)
            return AgentTurnResult(..., completed=True)

        tool_messages = tool_executor.execute(...)
        session_store.append_messages(session_id, tool_messages)
        messages.extend(tool_messages)

        consecutive_tool_failures = update_failure_counter(tool_messages)
        if consecutive_tool_failures >= max_consecutive_tool_failures:
            recover = build_recover_message(...)
            session_store.append_message(session_id, recover)
            post_turn_processor.after_turn(session_id, config)
            return AgentTurnResult(..., completed=True)

    fallback = build_max_iteration_message(...)
    session_store.append_message(session_id, fallback)
    post_turn_processor.after_turn(session_id, config)
    return AgentTurnResult(..., completed=False)
```

需要保留的现有能力：

- 长期记忆检索。
- turn-level system prompt 构建。
- token budget 与上下文压缩。
- 文本/图像/音频模型路由。
- 模型失败 fallback。
- message length fallback。
- 非工具回复疑似未完成时的 continuation。
- 最大工具调用次数限制。
- 同一工具重复调用限制。
- 连续工具失败 recover。
- 流式 token 回调和 debug 回调。

## 8. ToolExecutor 与 ToolPolicy

当前 `_action_async` 对同一轮所有工具调用直接并发执行。新实现应引入工具策略。

建议策略：

```python
class ToolPolicy:
    def can_parallelize(self, tool_calls: list[dict[str, Any]]) -> bool:
        ...

    def is_readonly(self, tool_name: str, args: dict[str, Any]) -> bool:
        ...

    def is_mutating(self, tool_name: str, args: dict[str, Any]) -> bool:
        ...

    def requires_serial(self, tool_name: str, args: dict[str, Any]) -> bool:
        ...
```

默认规则：

- 全部只读工具可以并发。
- 包含文件写入、shell、记忆写入、cron、子 agent 调度时默认串行。
- 未知工具默认串行。
- 同一路径的多个文件操作强制串行。
- 高风险工具继续走现有审批机制。

工具执行接口：

```python
class ToolExecutor:
    async def execute(
        self,
        *,
        tool_calls: list[dict[str, Any]],
        messages: list[AnyMessage],
        session_id: str,
        config: RuntimeConfig,
    ) -> list[ToolMessage]:
        ...
```

工具上下文需要兼容旧工具：

```python
tool_context = {
    "runtime": runtime,
    "graph": runtime_compat_adapter,
    "state_messages": list(messages),
    "config": config,
    "status_callback": config.status_callback,
    "tool_progress_callback": config.tool_progress_callback,
    "thread_id": session_id,
    "subagent_depth": 0,
}
```

## 9. RuntimeCompatAdapter

部分现有工具会从 `_wbot_tool_context["graph"]` 获取能力。迁移期间必须提供兼容层，避免一次性改动所有工具。

建议能力：

```python
class RuntimeCompatAdapter:
    def get_state(self, config: dict[str, Any]) -> Any: ...
    def list_subagents(self, *, status: str | None = None, limit: int = 20) -> list[dict[str, Any]]: ...
    def wait_for_subagent(self, job_id: str, *, timeout_seconds: int = 60) -> dict[str, Any]: ...
    def spawn_subagent(...): ...
    async def run_skill_subagent(...): ...

    @property
    def session_search_llm(self) -> Any: ...
```

`get_state()` 可以用 `SessionStore` 组装一个近似旧 snapshot 的对象，至少提供：

```python
snapshot.values = {
    "messages": messages,
    "conversation_summary": summary,
    "summarized_message_count": summarized_count,
}
```

短期不建议支持完整 `update_state()` 语义。摘要更新应直接走 `SessionStore.update_summary()`。

## 10. PostTurnProcessor

旧逻辑在 `ScheduledGraphApp.invoke()` 后执行：

- `flush_session_search_index(config)`
- `schedule_deferred_summary(config)`

新逻辑改为：

```python
class PostTurnProcessor:
    def after_turn(self, *, session_id: str, config: RuntimeConfig) -> None:
        self.flush_session_search_index(session_id)
        self.schedule_summary_refresh(session_id, config)
```

摘要刷新不再读取 graph state，也不再 `graph.update_state`，而是：

```python
messages = session_store.get_messages(session_id)
summary, summarized_count = context_optimizer.refresh_summary(...)
session_store.update_summary(
    session_id,
    summary=summary,
    summarized_message_count=summarized_count,
)
```

会话搜索索引应以 `SessionStore` 的 messages 为来源。

## 11. 入口替换范围

需要替换的主要入口：

- `w_bot/channels/web/gateway.py`
- `w_bot/channels/feishu/gateway.py`
- `w_bot/agents/core/cli.py`

旧调用：

```python
result = graph.invoke(inputs, config=config)
latest_ai_text = _latest_ai_reply_from_result(result)
```

新调用：

```python
result = runtime.run_turn(
    session_id=session_id,
    inbound_messages=[HumanMessage(content=message)],
    config=runtime_config,
)
latest_ai_text = result.final_response
```

Web history 旧逻辑从 `graph.get_state(config)` 获取消息，新逻辑应改为：

```python
messages = runtime.get_session_messages(session_id)
```

飞书和 Web 现有按 `session_id` 加锁的逻辑必须保留，避免同一会话同时执行多个 turn。

## 12. 迁移阶段

### 阶段 0：冻结现状与确认边界

只读梳理，不改主流程。

确认：

- 旧 checkpoint 是否需要迁移。
- CLI、Web、飞书是否同步切换。
- 是否继续保留 LangChain 消息对象。
- 新会话数据库路径与配置项命名。

建议结论：

- 第一阶段不迁移旧 checkpoint。
- 保留 LangChain message 对象。
- 新会话全部写入业务 SessionStore。
- LangGraph 代码在兼容期保留，等入口切换完成后再删除。

### 阶段 1：新增 SessionStore

新增 `session_db.py` 和 `session_models.py`。

验证：

- 消息序列化往返。
- tool_calls、tool_call_id、tool_name 不丢失。
- 多模态 additional_kwargs 不丢失。
- token usage 能保存和累计。

### 阶段 2：新增 AgentRuntime 骨架

新增 `runtime.py`、`turn.py`，先不接入网关。

目标：

- 能创建 runtime。
- 能接收 `session_id` 和 `HumanMessage`。
- 能写入 SessionStore。
- 暂时可以只返回固定兜底或调用模型但不执行工具。

### 阶段 3：迁移记忆检索和上下文准备

从 `WBotGraph` 抽离：

- `_retrieve_memories`
- `_prepare_prompt_context`
- 相关 token budget snapshot 构建

验证：

- 命中长期记忆路径。
- 未命中长期记忆路径。
- system prompt 基本结构与旧逻辑一致。

### 阶段 4：迁移模型调用

拆分 `_agent` 中模型相关逻辑：

- context optimization
- model route selection
- streaming callback
- model invoke fallback
- token usage extraction
- incomplete response continuation

验证：

- 普通聊天一轮结束。
- 模型异常返回兜底。
- message length fallback 生效。
- 流式回调仍能输出 token。

### 阶段 5：实现 ToolExecutor

迁移 `_action_async`，并加入 ToolPolicy。

验证：

- 单工具成功。
- 单工具失败生成 ToolMessage。
- 多只读工具并发。
- 写操作串行。
- 连续工具失败触发 recover。

### 阶段 6：实现完整 AgentLoop

把模型调用与工具执行串成显式 loop。

验证：

- 用户消息 → 模型工具调用 → 工具结果 → 模型总结。
- 最大工具步数限制。
- 重复工具调用限制。
- 最大模型迭代限制。
- final_response 提取正确。

### 阶段 7：切换 Web 与飞书入口

先切 Web，再切飞书。

验证：

- `/api/chat` 正常返回。
- `/api/history` 从 SessionStore 读取。
- 会话绑定权限不变。
- 飞书群聊 mention 规则不变。
- 飞书新会话命令不变。

### 阶段 8：切换 CLI 入口

CLI 涉及状态展示、runtime usage、历史预览、子 agent 状态等，建议最后切。

验证：

- 普通问答。
- 工具调用。
- 流式输出。
- 历史展示。
- 会话摘要和 token 状态展示。

### 阶段 9：切换摘要与搜索索引

让摘要和 session search 都以 SessionStore 为来源。

验证：

- 本轮结束后 session search 可查到新消息。
- 摘要刷新后写入 sessions 表。
- 并发请求不会破坏摘要状态。

### 阶段 10：删除 LangGraph

入口全部切换并稳定后删除：

- `langgraph` 依赖。
- `WorkspaceFileCheckpointer`。
- `AgentState`。
- `ScheduledGraphApp`。
- `StateGraph` 构建逻辑。
- `graph.invoke/get_state/update_state` 主路径调用。

## 13. 配置兼容

现有配置中与 loop 相关的字段应尽量保留语义：

- `loop_guard.recursion_limit`
- `loop_guard.max_tool_steps_per_turn`
- `loop_guard.max_same_tool_call_repeats`

新 runtime 内部可以映射为：

```text
recursion_limit              → max_model_iterations 或总 loop 上限
max_tool_steps_per_turn      → 本轮最大工具步数
max_same_tool_call_repeats   → 同一工具重复调用上限
```

新增 SessionStore 路径配置时，建议不要复用 `short_term_memory_path` 的旧语义。可以新增：

```json
{
  "sessionStorePath": "memory/session_store.sqlite"
}
```

兼容期如果未配置，则可从 `short_term_memory_path` 推导默认路径，但不要继续写 LangGraph checkpoint 格式。

## 14. 风险与控制

### 14.1 旧会话兼容风险

LangGraph checkpoint 不适合直接完整迁移成业务消息表。

控制策略：

- 新 runtime 只处理新会话。
- 旧 checkpoint 文件保留，不删除。
- 如必须读取旧历史，单独实现只读导出工具，不混入主流程。

### 14.2 工具上下文兼容风险

旧工具依赖 `_wbot_tool_context["graph"]`。

控制策略：

- 先提供 `RuntimeCompatAdapter`。
- 工具内部后续逐步从 `graph` 命名迁移到 `runtime`。
- 每迁移一类工具都做局部测试。

### 14.3 流式输出退化风险

旧逻辑通过 `RunnableConfig` 传递 callback。

控制策略：

- 显式建模 `RuntimeConfig`。
- Web、飞书、CLI 各自适配一次。
- 保留 token callback、debug callback、tool progress callback。

### 14.4 并发一致性风险

业务状态改为 SessionStore 后，同会话并发写入风险更明显。

控制策略：

- 保留 Web、飞书现有 session lock。
- CLI 单进程内同样按 session_id 串行。
- SessionStore 写入使用事务。
- SQLite 开启 WAL，并提供失败 fallback。

### 14.5 工具执行顺序风险

从无脑并发改成策略执行可能改变部分工具组合的速度和行为。

控制策略：

- 未知工具默认串行。
- 写操作默认串行。
- 只读工具逐步加入白名单。
- 工具执行结果保持原 tool_calls 顺序写回消息。

## 15. 验证策略

优先复用现有测试：

```powershell
uv run pytest tests/unit/test_agent.py -q
uv run pytest tests/unit/test_tools.py -q
uv run pytest tests/unit/test_web_gateway_auth.py -q
uv run pytest tests/integration/test_agent_flow.py -q
```

建议新增测试：

- `test_session_store.py`
  - message 序列化和反序列化。
  - tool_calls 保存和恢复。
  - summary 更新。

- `test_agent_runtime.py`
  - 普通聊天。
  - 工具调用闭环。
  - 工具失败 recover。
  - 最大迭代限制。

- `test_tool_executor.py`
  - 只读工具并发。
  - 写工具串行。
  - 未知工具串行。
  - 异常工具转 ToolMessage。

- `test_runtime_compat.py`
  - `get_state()` 返回旧工具可用的 snapshot。
  - 子 agent 相关方法可转发。

## 16. 建议提交拆分

建议按以下顺序拆提交：

```text
feat:增加业务会话存储
feat:新增显式Agent运行时骨架
refactor:抽离记忆检索与上下文准备
refactor:抽离模型调用与路由逻辑
refactor:重构工具执行器
feat:接入显式AgentLoop
refactor:切换Web运行入口
refactor:切换飞书运行入口
refactor:切换CLI运行入口
refactor:统一摘要与会话搜索后处理
chore:移除LangGraph编排依赖
```

每个提交都应保证至少一条主路径可验证，不建议在一个提交中同时改存储、模型调用、工具执行和所有入口。

## 17. 推荐第一刀

推荐第一步只做：

```text
SessionStore + AgentRuntime.run_turn 最小闭环
```

最小闭环范围：

- 新增 SessionStore。
- 新增 AgentRuntime。
- `run_turn()` 能写入用户消息。
- 调用模型生成无工具回复。
- 写入 assistant 消息。
- 返回 `AgentTurnResult.final_response`。
- 不切换 Web、飞书、CLI 主入口。

这样可以在不影响现有 LangGraph 路径的情况下验证新架构，后续再逐步迁移工具 loop 和入口。

