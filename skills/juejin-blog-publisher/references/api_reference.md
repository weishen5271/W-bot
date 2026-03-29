# 稀土掘金 API 参考

## 认证方式

掘金 API 主要通过 Cookie 进行认证。需要从浏览器获取以下关键字段：

### 必需字段
- `uid`: 用户ID
- `token` 或 `x-juejin-token`: 认证令牌

### 获取方法

1. 登录掘金网站：https://juejin.cn
2. 打开浏览器开发者工具 (F12)
3. 切换到 Network/网络 标签
4. 刷新页面
5. 任意请求中查看 Request Headers 中的 Cookie
6. 提取 `uid` 和 `token` 字段

或使用浏览器控制台：
```javascript
// 复制完整 Cookie
document.cookie

// 或提取特定字段
document.cookie.match(/uid=([^;]+)/)?.[1]
document.cookie.match(/token=([^;]+)/)?.[1]
```

## API 端点

### 用户相关

#### 获取当前用户信息
```
GET /user_api/v1/user/get
```

响应：
```json
{
  "err_no": 0,
  "err_msg": "success",
  "data": {
    "user_id": "123456789",
    "user_name": "用户名",
    "avatar_large": "头像URL",
    ...
  }
}
```

### 文章相关

#### 发布文章
```
POST /content_api/v1/article/publish
```

请求体：
```json
{
  "category_id": "6809637769959178254",
  "tag_ids": [{"tag_id": "5597ace9e4b0df71c31b1455"}],
  "link_url": "",
  "cover_image": "https://...",
  "title": "文章标题",
  "brief_content": "文章摘要",
  "edit_type": 10,
  "html_content": "deprecated",
  "mark_content": "# Markdown正文"
}
```

响应：
```json
{
  "err_no": 0,
  "err_msg": "success",
  "data": {
    "article_id": "1234567890",
    "url": "https://juejin.cn/post/1234567890"
  }
}
```

#### 更新文章
```
POST /content_api/v1/article/update
```

请求体包含 `id` 字段标识文章ID，其余与发布相同。

#### 保存草稿
```
POST /content_api/v1/article/draft/save
```

请求体与发布类似，但 `status` 为 1 表示草稿。

### 分类和标签

#### 获取分类列表
```
GET /tag_api/v1/query_category_briefs
```

响应包含所有可用的文章分类。

#### 搜索标签
```
GET /tag_api/v1/search
```

参数：
- `key_word`: 搜索关键词
- `cursor`: 分页游标
- `limit`: 数量限制

## 错误码

| 错误码 | 含义 | 解决方案 |
|-------|------|---------|
| 0 | 成功 | - |
| 401 | 未认证 | 检查 Cookie 是否有效 |
| 403 | 权限不足 | 检查账号权限 |
| 404 | 资源不存在 | 检查文章ID等参数 |
| 500 | 服务器错误 | 稍后重试 |

## 分类ID参考

| 分类 | ID |
|-----|-----|
| 后端 | 6809637769959178254 |
| 前端 | 6809637769959178253 |
| Android | 6809635626879549454 |
| iOS | 6809635626879549453 |
| 人工智能 | 6809637773935374343 |
| 开发工具 | 6809637771511070734 |
| 代码人生 | 6809637776263217160 |
| 阅读 | 6809637772874219534 |

## 注意事项

1. **频率限制**: 避免短时间内大量请求
2. **内容审核**: 发布的内容需符合掘金社区规范
3. **Cookie 有效期**: 定期更新 Cookie 以保持登录状态
4. **接口变更**: 掘金 API 可能随时调整，请关注官方文档
