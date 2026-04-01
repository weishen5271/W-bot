---
name: juejin-hot-news
description: 查询掘金热门文章和新闻。支持获取掘金社区的热门文章、技术资讯、最新动态，可按热度、时间排序，支持筛选技术分类。触发词：掘金、热门文章、技术新闻、掘金热榜、前端新闻、后端动态。
---

# 掘金热门新闻查询

本skill用于查询掘金社区的热门文章和新闻，支持多种排序方式和筛选条件。

## 核心功能

1. **获取热门文章**：查询掘金社区当前最热门的文章
2. **多种排序方式**：支持按热度、时间、综合推荐排序
3. **分类筛选**：可按技术分类筛选（前端、后端、移动端、AI等）
4. **分页查询**：支持分页获取更多结果
5. **详细文章信息**：获取文章标题、作者、阅读量、点赞数、评论数等

## 使用方法

### 基本查询

使用以下curl命令获取掘金热门文章：

```bash
curl -s "https://api.juejin.cn/recommend_api/v1/article/recommend_all_feed?aid=2608&uuid=7243086311112009224&spider=0" \
  -H "Content-Type: application/json" \
  -d '{"id_type":2,"client_type":2608,"sort_type":200,"cursor":"0","limit":10}'
```

### 参数说明

- `id_type`: 固定为2（文章类型）
- `client_type`: 客户端类型，2608表示Web端
- `sort_type`: 排序方式
  - `200`: 综合推荐（默认）
  - `300`: 最新发布
  - `400`: 最热文章
- `cursor`: 游标，用于分页，"0"表示第一页
- `limit`: 每页数量，最大20

### Python脚本示例

在`scripts/`目录中提供了Python脚本，可以直接使用：

```python
# 获取热门文章
python scripts/get_hot_articles.py --limit 10 --sort hot

# 获取最新文章
python scripts/get_hot_articles.py --limit 10 --sort new

# 获取前端相关文章
python scripts/get_hot_articles.py --limit 10 --tag frontend
```

## 数据解析

API返回的JSON数据结构：

```json
{
  "err_no": 0,
  "err_msg": "success",
  "data": [
    {
      "item_type": 2,
      "item_info": {
        "article_id": "文章ID",
        "article_info": {
          "title": "文章标题",
          "brief_content": "文章摘要",
          "view_count": 阅读量,
          "digg_count": 点赞数,
          "comment_count": 评论数,
          "collect_count": 收藏数,
          "hot_index": 热度指数,
          "read_time": "阅读时间",
          "ctime": "创建时间戳",
          "cover_image": "封面图片URL"
        },
        "author_user_info": {
          "user_name": "作者名",
          "avatar_large": "作者头像",
          "job_title": "职位"
        },
        "tags": [
          {"tag_name": "标签名", "color": "标签颜色"}
        ]
      }
    }
  ],
  "cursor": "下一页游标",
  "has_more": true
}
```

## 常用查询模式

### 1. 获取前10条热门文章
```bash
curl -s "https://api.juejin.cn/recommend_api/v1/article/recommend_all_feed?aid=2608&uuid=7243086311112009224&spider=0" \
  -H "Content-Type: application/json" \
  -d '{"id_type":2,"client_type":2608,"sort_type":400,"cursor":"0","limit":10}'
```

### 2. 获取最新发布的文章
```bash
curl -s "https://api.juejin.cn/recommend_api/v1/article/recommend_all_feed?aid=2608&uuid=7243086311112009224&spider=0" \
  -H "Content-Type: application/json" \
  -d '{"id_type":2,"client_type":2608,"sort_type":300,"cursor":"0","limit":10}'
```

### 3. 获取更多文章（分页）
```bash
curl -s "https://api.juejin.cn/recommend_api/v1/article/recommend_all_feed?aid=2608&uuid=7243086311112009224&spider=0" \
  -H "Content-Type: application/json" \
  -d '{"id_type":2,"client_type":2608,"sort_type":200,"cursor":"1","limit":10}'
```

## 输出格式化

建议将查询结果格式化为易读的Markdown表格：

| 排名 | 标题 | 作者 | 阅读量 | 点赞数 | 标签 |
|------|------|------|--------|--------|------|
| 1 | [标题1](链接) | 作者1 | 10k | 500 | 前端,JavaScript |
| 2 | [标题2](链接) | 作者2 | 8k | 300 | 后端,Java |

## 注意事项

1. **API限制**：掘金API可能有频率限制，请合理使用
2. **数据时效性**：热门文章数据实时更新，建议缓存结果
3. **错误处理**：检查`err_no`字段，0表示成功
4. **链接格式**：文章链接格式为：`https://juejin.cn/post/文章ID`

## 扩展功能

如需更高级功能（如定时抓取、数据存储、邮件推送等），可参考`references/advanced_usage.md`。