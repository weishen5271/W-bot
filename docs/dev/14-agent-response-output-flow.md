# Agent 问答与回答输出流程

本文基于当前源码梳理 Hermes Agent 在一次用户问答中的核心执行链路，以及 CLI、Gateway 两类入口如何展示或发送 Agent 的回答。

## 1. 总览

一次问答不是“调用一次模型后直接打印”的简单流程，而是一个带工具循环、流式输出、上下文持久化和多端展示适配的完整执行过程。

核心路径如下：

```text
用户输入
  ↓
CLI / Gateway 入口整理会话上下文
  ↓
AIAgent.run_conversation()
  ↓
组装 messages、system prompt、tools、provider 参数
  ↓
调用模型（默认优先 streaming）
  ↓
模型返回：
  ├─ 有 tool_calls：校验工具调用 → 执行工具 → 工具结果写回 messages → 继续下一轮模型调用
  └─ 无 tool_calls：提取 assistant content → 清理思考块 → 得到 final_response
  ↓
保存会话、轨迹、插件 hook、memory review 等后处理
  ↓
CLI 渲染 Rich Panel / Gateway 发送平台消息
```

关键源码入口：

| 阶段 | 文件 | 说明 |
| --- | --- | --- |
| Agent 主循环 | `run_agent.py` | `AIAgent.run_conversation()` 是一次问答的核心入口 |
| 流式回调分发 | `run_agent.py` | `_fire_stream_delta()` 将模型增量文本分发给展示层 |
| 流式 API 调用 | `run_agent.py` | `_interruptible_streaming_api_call()` 统一处理不同 provider 的流式响应 |
| 工具执行 | `run_agent.py` | `_execute_tool_calls()` 根据工具调用数量和类型选择顺序或并发执行 |
| CLI 展示 | `cli.py` | `_stream_delta()`、`_flush_stream()`、`Panel(...)` 负责终端展示 |
| Gateway 展示 | `gateway/run.py`、`gateway/stream_consumer.py` | `GatewayStreamConsumer.on_delta()` 负责消息平台流式编辑/分段 |

## 2. `run_conversation()`：一次问答的核心入口

`AIAgent.run_conversation()` 位于 `run_agent.py`，负责执行“一条用户消息 → 一个最终回答”的完整生命周期。

入口签名包含这些关键参数：

- `user_message`：当前用户输入。
- `system_message`：可选临时 system prompt。
- `conversation_history`：历史消息，CLI/Gateway 会传入当前会话上下文。
- `task_id`：任务隔离标识，用于工具运行、VM、浏览器等资源隔离。
- `stream_callback`：TTS 等内部流式消费者回调。
- `persist_user_message`：用于持久化的干净用户消息，避免把 API-only 前缀写入历史。

进入函数后，会先做这些准备工作：

1. 安装安全 stdio 包装，避免 broken pipe 导致进程崩溃。
2. 恢复上一轮 fallback 后的 primary runtime，让每轮优先尝试首选模型。
3. 清理输入中的非法 surrogate 字符。
4. 保存流式回调到实例字段，供 API 调用阶段使用。
5. 生成或复用 `task_id`。
6. 重置本轮重试计数、上下文预算、interrupt 状态。
7. 组装初始 `messages`，包含历史消息、system prompt、memory/skill/plugin 注入内容等。

源码锚点：`run_agent.py:11613`。

## 3. 主循环：模型调用与工具调用闭环

Agent 的主循环本质是：**不断调用模型，直到模型不再请求工具，产生最终文本回答。**

伪代码如下：

```python
while api_call_count < max_iterations and iteration_budget.remaining > 0:
    api_messages = build_api_messages(messages)
    response = call_model(api_messages, tools)
    assistant_message = normalize_response(response)

    if assistant_message.tool_calls:
        validate_tool_names_and_args(assistant_message.tool_calls)
        messages.append(assistant_tool_call_message)
        execute_tool_calls(...)
        messages.append(tool_results)
        continue

    final_response = assistant_message.content
    final_response = strip_think_blocks(final_response)
    messages.append(final_assistant_message)
    break
```

### 3.1 API 调用策略

当前实现倾向于使用 streaming 调用，即使没有 CLI/Gateway/TTS 消费者，也会优先走流式路径，因为流式路径有更好的连接健康检查能力，可以避免 provider 只保持 SSE ping 但不返回正文时长期卡住。

核心判断在 `run_agent.py`：

- 有 stream consumer：走 `_interruptible_streaming_api_call()`，实时分发文本增量。
- 无 stream consumer：默认仍优先 streaming；测试 Mock client 等特殊情况会回退非流式调用。

源码锚点：`run_agent.py:12480`、`run_agent.py:12531`、`run_agent.py:7641`。

### 3.2 响应归一化

不同 provider/API mode 返回结构不完全一致。主循环拿到响应后，会把它归一化为统一的 `assistant_message` 形态，后续统一读取：

- `assistant_message.content`
- `assistant_message.tool_calls`
- `assistant_message.reasoning` / `reasoning_content` / `reasoning_details`
- `finish_reason`

这样后面的工具执行、最终回答提取、reasoning 展示可以共用同一套逻辑。

## 4. 工具调用分支

当模型返回 `tool_calls` 时，本轮不会结束。Agent 会先处理工具调用，再把工具结果交回模型，让模型基于工具结果继续生成。

### 4.1 工具调用校验

工具调用进入执行前会经过两类校验：

1. **工具名校验**：检查模型请求的工具是否在 `valid_tool_names` 中；如果名称疑似可修复，会尝试自动修正。
2. **参数 JSON 校验**：确保 `function.arguments` 是合法 JSON；空字符串会按空对象 `{}` 处理，非法 JSON 会把错误作为 tool result 回传给模型自我修正。

如果模型连续生成不存在的工具或非法参数，达到重试上限后会返回 partial 结果，不继续无限循环。

源码锚点：`run_agent.py:14503`。

### 4.2 工具消息写回

通过校验后，Agent 会将包含 `tool_calls` 的 assistant 消息写入 `messages`，然后执行工具。

执行完成后，每个工具结果会作为类似下面的消息追加：

```python
{
    "role": "tool",
    "tool_call_id": "...",
    "content": "..."
}
```

下一轮模型调用时，模型就能读取这些工具结果，并继续决定：

- 是否继续调用更多工具；
- 是否基于工具结果给用户最终回答。

### 4.3 顺序与并发执行

工具执行入口是 `_execute_tool_calls()`。它会根据当前工具调用集合判断是否适合并发执行：

- 可以安全并发的工具会走并发执行路径。
- 需要顺序保证、可能有副作用或存在依赖关系的工具走顺序执行路径。

源码锚点：`run_agent.py:10410`。

### 4.4 工具前后的输出边界

如果模型在工具调用前已经流式输出了一段文本，Agent 会在执行工具前调用：

```python
self.stream_delta_callback(None)
```

`None` 是一个“段落边界 / 工具边界”信号：

- CLI 收到后会刷新并关闭当前响应框，让后续工具日志显示在独立位置。
- Gateway 收到后会 finalize 当前草稿消息，后续文本会作为新消息或新编辑段落展示。

源码锚点：`run_agent.py:14695`、`gateway/stream_consumer.py:229`。

## 5. 最终回答分支

如果 `assistant_message.tool_calls` 为空，说明模型没有再请求工具，本轮内容就是最终回答。

处理步骤：

1. 读取 `assistant_message.content`。
2. 判断是否只有 reasoning / think block，而没有可见内容。
3. 如果上一轮“文本 + housekeeping 工具”已经产生过可见回答，则复用上一轮内容作为最终回答。
4. 如果是 reasoning-only，可最多尝试 prefill continuation，让模型继续补出可见文本。
5. 从用户可见回答中去除 `<think>`、`<reasoning>`、`<REASONING_SCRATCHPAD>` 等思考块。
6. 将最终 assistant 消息写入 `messages`。
7. 设置 `final_response` 并退出主循环。

源码锚点：`run_agent.py:14817`。

最终返回结构核心字段：

```python
{
    "final_response": final_response,
    "last_reasoning": last_reasoning,
    "messages": messages,
    "api_calls": api_call_count,
    "completed": completed,
    ...
}
```

源码锚点：`run_agent.py:15390`。

## 6. 流式输出分发

Agent 自身不直接决定“怎么画 UI”或“怎么发平台消息”。它只在模型流式返回文本时调用 `_fire_stream_delta()`，把文本增量交给外部消费者。

`_fire_stream_delta()` 会分发给两类回调：

- `self.stream_delta_callback`：CLI 或 Gateway 的展示回调。
- `self._stream_callback`：调用 `run_conversation(..., stream_callback=...)` 时传入的内部消费者，例如 TTS。

此外，它还会处理一个细节：工具执行后下一段文本到来时，自动补一个空行，避免工具结果和后续回答粘在一起。

源码锚点：`run_agent.py:7557`。

## 7. CLI 输出机制

CLI 的输出逻辑分为“实时流式展示”和“最终兜底展示”。

### 7.1 Agent 创建时注入回调

CLI 创建 `AIAgent` 时，如果开启了 streaming，会传入：

```python
stream_delta_callback=self._stream_delta if self.streaming_enabled else None
```

这意味着模型每产生一段可见文本，都会回调到 CLI 的 `_stream_delta()`。

源码锚点：`cli.py:4182`。

### 7.2 `_stream_delta()` 实时渲染

`_stream_delta()` 的职责：

- 接收模型输出 token / 文本片段。
- 过滤或转发 `<think>`、`<reasoning>` 等 reasoning 标签。
- 如果开启 `show_reasoning`，将 reasoning 内容渲染到单独的 Reasoning box。
- 普通回答文本进入响应框，按行缓冲输出，避免 prompt_toolkit 下显示错乱。
- 收到 `None` 时，刷新当前流式输出并重置状态，为工具日志让出显示位置。

源码锚点：`cli.py:3490`。

### 7.3 最终展示

`HermesCLI.chat()` 调用 `self.agent.run_conversation(...)` 后，从结果中读取：

```python
response = result.get("final_response", "")
```

如果回答已经通过流式框展示过，就不重复打印；如果没有流式展示，则使用 Rich `Panel` 一次性展示最终回答。

源码锚点：`cli.py:10446`、`cli.py:10713`。

## 8. CLI 中工具与 Skill 的非回答输出

除模型最终回答外，CLI 还会显示 Agent 调用工具、加载 Skill、更新 Skill、执行文件修改等过程状态。这些输出不是 LLM 最终回答本身，而是由 Agent 工具生命周期回调和 CLI 展示层共同生成。

### 8.1 回调注入

CLI 创建 `AIAgent` 时会注入多类工具展示回调：

```python
tool_progress_callback=self._on_tool_progress
tool_start_callback=self._on_tool_start if self._inline_diffs_enabled else None
tool_complete_callback=self._on_tool_complete if self._inline_diffs_enabled else None
tool_gen_callback=self._on_tool_gen_start if self.streaming_enabled else None
```

它们分别负责：

- `tool_progress_callback`：工具开始、完成、reasoning 可用等生命周期事件。
- `tool_start_callback`：写文件类工具执行前捕获本地快照，用于后续 diff。
- `tool_complete_callback`：写文件类工具执行后渲染 inline diff。
- `tool_gen_callback`：模型开始生成工具调用参数时，给用户一个“正在准备工具”的即时提示。

源码锚点：`cli.py:4179`、`cli.py:4180`、`cli.py:4181`、`cli.py:4183`。

### 8.2 工具参数生成阶段

当 streaming 开启时，模型可能会先生成大量工具参数，例如大段 `write_file` 内容。此时还没有真正开始执行工具，但用户如果只看终端会感觉卡住。

为避免这种体验，Agent 在检测到工具调用参数生成时触发 `tool_gen_callback`，CLI 的 `_on_tool_gen_start()` 会：

1. 如果当前正在流式显示回答框，则先 flush 并关闭回答框。
2. 关闭 reasoning box，避免工具状态行被绘制在 reasoning 框里。
3. 根据工具名取 emoji。
4. 打印一行类似 `preparing <tool_name>` 的短状态。

源码锚点：`cli.py:9173`。

### 8.3 工具开始执行阶段

工具真正执行前，Agent 会调用：

```python
self.tool_progress_callback("tool.started", function_name, preview, function_args)
```

CLI 的 `_on_tool_progress()` 在收到 `tool.started` 后会：

- 使用 `get_tool_emoji(function_name)` 获取工具图标。
- 使用工具 preview 或工具名作为 spinner 文案。
- 记录 `time.monotonic()`，让 TUI spinner 显示动态耗时。
- 将本次工具参数暂存到 `_pending_tool_info`，供完成后生成 scrollback 摘要。
- 如果 voice mode 开启，会播放一个短提示音。

源码锚点：`run_agent.py:11047`、`cli.py:9193`。

并发工具路径也会在每个可执行工具启动前触发同样的 `tool.started` 回调。源码锚点：`run_agent.py:10661`。

### 8.4 工具完成阶段

工具执行完成后，Agent 会调用：

```python
self.tool_progress_callback(
    "tool.completed",
    function_name,
    None,
    None,
    duration=tool_duration,
    is_error=_is_error_result,
)
```

CLI 收到 `tool.completed` 后会：

1. 清空当前工具计时。
2. 根据 `display.tool_progress` 判断是否打印持久 scrollback 摘要。
3. 从 `_pending_tool_info` 取出开始阶段保存的参数。
4. 调用 `agent.display.get_cute_tool_message()` 生成一行短摘要。
5. 如果工具结果被识别为错误，在摘要后追加 `[error]`。
6. 刷新 TUI。

源码锚点：`run_agent.py:11301`、`cli.py:9208`。

并发工具路径在收集每个工具结果后也会触发 `tool.completed`。源码锚点：`run_agent.py:10875`。

### 8.5 `display.tool_progress` 显示模式

CLI 通过 `display.tool_progress` 控制工具进度展示强度。初始化时读取配置并归一化：

```python
_raw_tp = CLI_CONFIG["display"].get("tool_progress", "all")
self.tool_progress_mode = "off" if _raw_tp is False else str(_raw_tp)
self.verbose = verbose if verbose is not None else (self.tool_progress_mode == "verbose")
```

支持的典型模式：

- `off`：关闭常规工具进度展示。
- `new`：只在工具名变化时打印 scrollback 摘要，跳过连续重复工具。
- `all`：工具完成后都打印一行 scrollback 摘要。
- `verbose`：进入更详细输出模式，Agent 会打印工具编号、参数预览、结果预览等。

源码锚点：`cli.py:2320`、`cli.py:2345`。

在 verbose 或非 quiet 模式下，Agent 本身也会打印工具调用和结果预览，例如：

- 工具开始：`Tool i: function_name([...]) - args_preview`
- 工具完成：`Tool i completed in X.XXs - result_preview`

源码锚点：`run_agent.py:11024`、`run_agent.py:11273`。

### 8.6 工具完成摘要格式

工具完成后的一行摘要由 `agent.display.get_cute_tool_message()` 生成。它根据工具名选择不同展示模板，而不是简单打印原始 JSON。

常见示例：

| 工具 | 摘要含义 |
| --- | --- |
| `terminal` | 显示 `$` 和命令预览 |
| `read_file` | 显示 `read` 和文件路径 |
| `write_file` | 显示 `write` 和文件路径 |
| `patch` | 显示 `patch` 和目标路径 |
| `search_files` | 显示 `find` 或 `grep` 与搜索模式 |
| `web_search` | 显示搜索关键词 |
| `browser_*` | 显示 navigate、click、type、snapshot 等浏览器动作 |
| `todo` | 显示任务数量或读取任务 |
| `memory` | 显示 add、replace、remove 等记忆动作 |
| `skills_list` | 显示 skill catalog 列表动作 |
| `skill_view` | 显示加载的 skill 名称 |
| `delegate_task` | 显示委托任务或并行任务数量 |

如果工具失败，`get_cute_tool_message()` 会根据结果内容中的错误特征追加失败标记。

源码锚点：`agent/display.py:841`。

### 8.7 工具结果不等于用户可见完整输出

工具返回值会作为 `role="tool"` 消息写入 `messages`，供下一轮模型继续读取和推理。

```python
tool_msg = {
    "role": "tool",
    "name": name,
    "content": _tool_content,
    "tool_call_id": tc.id,
}
messages.append(tool_msg)
```

CLI 默认不会把完整工具结果全部打印给用户，只展示摘要、必要预览、错误标记或 diff。这避免了大文件读取、搜索结果、终端输出等内容把终端刷屏。

源码锚点：`run_agent.py:10939`。

### 8.8 文件修改类工具的 inline diff

如果 `display.inline_diffs` 开启，CLI 会对写入类工具展示变更预览。

配置读取：

```python
self._inline_diffs_enabled = CLI_CONFIG["display"].get("inline_diffs", True)
```

流程如下：

1. 工具开始前，`_on_tool_start()` 调用 `capture_local_edit_snapshot()` 捕获修改前快照。
2. 工具完成后，`_on_tool_complete()` 调用 `render_edit_diff_with_delta()`。
3. `render_edit_diff_with_delta()` 针对 `write_file`、`patch`、`skill_manage` 等工具生成并输出 diff。
4. diff 通过 `_cprint` 输出，保证在 prompt_toolkit 环境下不破坏 TUI。

源码锚点：`cli.py:2357`、`cli.py:9291`、`cli.py:9302`、`agent/display.py:428`。

### 8.9 Skill slash command 的显示

用户直接输入 Skill 命令，例如 `/some-skill ...` 时，CLI 走 slash command 分支，而不是模型工具调用分支。

流程如下：

1. `process_command()` 检查用户输入的 `base_cmd` 是否在 `_skill_commands` 中。
2. 调用 `build_skill_invocation_message()` 读取并组装 Skill 内容。
3. CLI 打印一行 `Loading skill: <skill_name>`。
4. 将组装后的 Skill 消息放入 `_pending_input`。
5. 后续主循环把这条消息当作用户输入交给 Agent。

也就是说，Skill 内容本身不会直接完整打印到终端，而是作为新的用户消息上下文交给模型使用。

源码锚点：`cli.py:7647`、`agent/skill_commands.py:406`。

### 8.10 Agent 主动调用 Skill 工具时的显示

当模型主动调用 Skill 相关工具时，它们就是普通工具调用，显示机制完全复用工具生命周期：

- `skills_list`：列出可用 Skill 元信息，CLI 摘要显示为 skill catalog 列表动作。
- `skill_view`：加载某个 Skill 的完整内容或支持文件，CLI 摘要显示为加载的 Skill 名称。
- `skill_manage`：创建、更新、删除或写入 Skill 文件；如果涉及文件修改，还会走 inline diff。

源码锚点：`tools/skills_tool.py:674`、`tools/skills_tool.py:849`、`tools/skill_manager_tool.py:713`、`agent/display.py:958`、`agent/display.py:960`。

### 8.11 `/skills reload` 的直接 CLI 输出

`/skills reload` 这类命令不是模型工具调用，而是 CLI 命令。它会直接重新扫描 Skill 目录并打印结果：

- 没有变化：打印 `No new skills detected.` 和可用 Skill 总数。
- 有新增：打印 `Added Skills` 列表。
- 有移除：打印 `Removed Skills` 列表。
- 同时将一条 one-shot note 缓存到 `_pending_skills_reload_note`，让下一轮用户消息把 Skill 变更告知模型，但不破坏 prompt cache。

源码锚点：`cli.py:9097`。

## 9. Gateway 输出机制

Gateway 和 CLI 复用同一个 `AIAgent` 核心，只是展示层换成消息平台适配器。

### 9.1 注入 Gateway stream consumer

如果 gateway streaming 配置开启，Gateway 会创建 `GatewayStreamConsumer`，并将其 `on_delta` 作为 Agent 的流式输出回调：

```python
agent.stream_delta_callback = _stream_delta_cb
```

源码锚点：`gateway/run.py:13934`、`gateway/run.py:14900`。

### 9.2 `GatewayStreamConsumer.on_delta()`

`on_delta()` 是线程安全的回调：

- 收到普通文本：放入内部队列，由后台 worker 编辑或发送平台消息。
- 收到 `None`：调用 `on_segment_break()`，finalize 当前消息段，确保工具进度消息和后续回答顺序正确。

源码锚点：`gateway/stream_consumer.py:229`。

### 9.3 最终消息返回

Gateway 调用：

```python
result = agent.run_conversation(...)
final_response = result.get("final_response")
```

之后会继续处理：

- 空回答或错误回答兜底。
- `MEDIA:` 标签补全，确保工具生成的文件/图片/音频能被平台适配器发送。
- session split 后的 session id 同步。
- 会话标题自动生成。
- 返回统一结果给平台发送层。

源码锚点：`gateway/run.py:15228`、`gateway/run.py:15239`。

## 10. `chat()` 简化接口

`AIAgent.chat()` 是对 `run_conversation()` 的简单包装：

```python
result = self.run_conversation(message, stream_callback=stream_callback)
return result["final_response"]
```

适合只关心最终文本的调用方；但 CLI/Gateway 通常直接使用 `run_conversation()`，因为它们还需要完整 `messages`、reasoning、工具调用信息、错误状态和持久化信息。

源码锚点：`run_agent.py:15469`。

## 11. 异常与边界处理

这套流程还包含多类稳定性处理：

- **最大迭代限制**：防止模型不断调用工具导致无限循环。
- **工具名/参数重试上限**：模型连续产生非法工具调用时返回 partial。
- **空回答处理**：支持 thinking-only continuation；最终仍无可见内容时返回 `(empty)`。
- **长度截断续写**：当 provider 因长度中断时，会尝试 continuation，并拼接前缀。
- **中断处理**：CLI/Gateway 可请求 interrupt，Agent 会尽快跳出工具循环并返回当前状态。
- **上下文压缩**：工具循环中会根据 token 压力触发压缩，避免上下文无限膨胀。
- **思考块过滤**：流式输出阶段和最终 `final_response` 阶段都会避免把 raw reasoning 标签暴露给用户。

## 12. 关键结论

- `final_response` 只在模型不再请求工具时形成；工具调用结果本身不是最终回答。
- Agent 核心只负责生成、工具闭环、状态持久化和回调分发；具体“怎么显示”由 CLI/Gateway 决定。
- streaming 不只是体验优化，也承担连接健康检查和中断响应能力。
- `None` 类型的 stream delta 是重要边界信号，用于把“模型文本段”和“工具执行阶段”隔离开。
- CLI 与 Gateway 共享同一套问答内核，差异主要在 stream consumer 和最终发送/渲染逻辑。
