# BOOT

## 每次启动清单
1. 确认配置可读：`configs/app.json`。
2. 检查 OpenClaw 档案：`AGENTS/SOUL/IDENTITY/USER/TOOLS/HEARTBEAT`。
3. 读取长期记忆：`memory/MEMORY.md`（兼容旧路径）。
4. 扫描可用技能：`skills/` 与内置 `skills_catalog/`。
5. 确认工具开关：`enableExecTool`、`enableSkills`、`mcpServers`。

## 启动后行为
- 优先执行用户当前请求。
- 有多条路径时，给出推荐路径并说明理由。
- 回合结束前尽量完成实现与验证闭环。
