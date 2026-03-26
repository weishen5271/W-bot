# 飞书 Channel 配置说明

本文档用于说明本项目中 `channels.feishu` 的配置项来源、获取方式与启动方法。

## 1. 配置文件位置

项目使用统一配置文件：

- `configs/app.json`

首次可从示例复制：

```bash
cp configs/app.json.example configs/app.json
```

## 2. 字段说明与获取方式

`configs/app.json` 中飞书配置示例：

```json
"channels": {
  "feishu": {
    "enabled": true,
    "appId": "cli_xxx",
    "appSecret": "xxx",
    "encryptKey": "",
    "verificationToken": "",
    "allowFrom": ["*"],
    "groupPolicy": "mention",
    "replyToMessage": true,
    "reactEmoji": "THUMBSUP"
  }
}
```

各字段来源如下：

1. `enabled`
开启或关闭飞书网关。`true` 为开启。

2. `appId`
在飞书开放平台应用后台获取：
`飞书开放平台 -> 开发者后台 -> 你的应用 -> 凭证与基础信息 -> App ID`

3. `appSecret`
在飞书开放平台应用后台获取：
`飞书开放平台 -> 开发者后台 -> 你的应用 -> 凭证与基础信息 -> App Secret`

4. `encryptKey`（可选）
在事件订阅安全设置中获取。只有启用对应加密校验时需要填写。

5. `verificationToken`（可选）
在事件订阅安全设置中获取。只有启用对应校验时需要填写。

6. `allowFrom`
允许访问机器人的用户 `open_id` 列表。

- `["*"]`：允许所有用户（调试方便，不建议生产长期使用）
- `["ou_xxx", "ou_yyy"]`：仅允许白名单用户

获取 `open_id` 常见方式：

- 先临时配置 `["*"]`，让目标用户发一条消息，再从服务日志中获取其 `open_id`
- 通过飞书开放平台接口查询用户标识（按实际接入方式选择）

7. `groupPolicy`
群聊触发策略。

- `"mention"`：仅当机器人被 @ 时处理消息（推荐）
- 其他值可用于更宽松策略，但可能导致群消息触发过多

8. `replyToMessage`
是否以“回复该消息”形式发送结果。`true` 推荐保留。

9. `reactEmoji`
收到消息后添加表情反应类型（例如 `THUMBSUP`）。不需要可设为空字符串。

## 3. 最小可用配置

要让飞书网关启动，最少需要：

1. `enabled = true`
2. 正确填写 `appId`
3. 正确填写 `appSecret`

其他字段可先用默认值跑通，再逐步收紧策略。

## 4. 启动方式

```bash
python main.py feishu --config configs/app.json
```

## 5. 飞书后台检查清单

1. 机器人应用已创建并可用
2. 已开通消息相关权限（如 `im:message` 相关能力）
3. 事件订阅配置与本项目接入方式一致（本项目代码为 WebSocket 长连接模式）
4. 若开启安全校验，`encryptKey` 与 `verificationToken` 与平台一致

## 6. 安全建议

1. 不要将真实 `appSecret` 提交到 Git 仓库
2. 生产环境避免长期使用 `allowFrom = ["*"]`
3. 如密钥已泄露，应立即在飞书后台轮换并更新配置
