# CyberCore CLI Agent

基于技术文档实现的 1.0 MVP：
- LangGraph Agent 循环（retrieve_memories -> agent -> action）
- 百炼 Qwen（OpenAI 兼容模式）
- PostgreSQL Checkpointer（短期记忆）
- Milvus + BGE embedding（长期记忆）
- E2B Python 沙箱工具
- `rich` 面板流式展示 Thought / Action / Observation

## 1. 启动基础设施

```bash
docker compose up -d
```

## 2. 安装依赖

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## 3. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入 DASHSCOPE_API_KEY 和 E2B_API_KEY
# 可选：设置 CYBERCORE_LOG_LEVEL=DEBUG 查看更详细日志
```

## 4. 运行

```bash
cybercore
# 或
python main.py
```

输入 `quit` / `exit` 退出。

## 目录说明

- `src/agents/cli.py`: CLI 入口与 rich 输出
- `src/agents/agent.py`: LangGraph 状态图与路由逻辑
- `src/agents/memory.py`: Milvus 长期记忆读写
- `src/agents/tools/runtime.py`: E2B 与记忆工具
- `docker-compose.yml`: Postgres + Milvus 开发环境

## 注意

- 初次运行会通过 `PostgresSaver.setup()` 初始化短期记忆相关表。
- 长期记忆检索默认按 `user_id` 过滤（`CYBERCORE_USER_ID`）。
- 如果 Qwen 工具调用失败，请优先检查工具 schema 与 docstring 描述是否清晰。
