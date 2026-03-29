---
name: juejin-blog-publisher
description: |
  自动撰写并发布稀土掘金（Juejin）技术博客的 Skill。
  触发词：掘金、juejin、博客、发布、写文章、发表。
  功能：
  1. 根据主题自动生成技术博客内容
  2. 支持Markdown格式内容创作
  3. 自动发布到稀土掘金平台
  4. 支持草稿保存和文章管理
  使用场景：技术内容创作、知识分享、自动化运营。
---

# 稀土掘金博客发布器 (Juejin Blog Publisher)

## 能力概述

本 Skill 帮助你自动撰写并发布稀土掘金（Juejin.cn）技术博客文章。

## 使用流程

### 1. 快速开始

用户提供文章主题或初稿，即可自动完成内容创作和发布：

```
用户：帮我写一篇关于"Python装饰器进阶技巧"的掘金博客
```

执行流程：
1. 分析主题，确定目标读者
2. 生成技术博客内容（Markdown格式）
3. 用户确认内容
4. 调用发布脚本发布到掘金

### 2. 内容创作规范

掘金平台推荐的技术博客格式：

**标题结构：**
- 主标题：简洁有力，包含关键词
- 副标题（可选）：补充说明

**正文结构：**
```markdown
# 主标题

## 引言/背景
简要说明问题和文章价值

## 核心内容
- 原理说明
- 代码示例（含注释）
- 最佳实践

## 进阶/扩展（可选）
深度内容或相关技术

## 总结
要点回顾 + 行动建议

## 参考链接（可选）
```

**掘金特色功能：**
- 支持代码高亮（多种编程语言）
- 支持数学公式（LaTeX）
- 支持 Mermaid 图表
- 支持自定义封面图

### 3. 发布流程

**获取用户凭证：**
用户需要提供掘金账号的认证信息（Cookie或Token）。

**调用发布脚本：**
```bash
python scripts/publish.py \
  --title "文章标题" \
  --content "文章正文（Markdown）" \
  --cover "封面图URL（可选）" \
  --category "分类" \
  --tags "标签1,标签2" \
  --brief "文章摘要"
```

**发布结果：**
- 成功：返回文章链接
- 失败：返回错误信息和重试建议

## 注意事项

1. **内容原创性**：掘金鼓励原创内容，建议对AI生成的内容进行个性化调整
2. **发布频率**：避免短时间内大量发布，建议间隔合理
3. **账号安全**：用户需自行管理掘金账号凭证
4. **分类选择**：选择正确的技术分类有助于文章曝光

## 进阶功能

### 脚本直接使用

```bash
# 基础发布
python scripts/publish.py \
  --title "文章标题" \
  --content "# Markdown正文" \
  --cookie "your_cookie_here"

# 完整参数发布
python scripts/publish.py \
  --title "Python装饰器详解" \
  --content "$(cat article.md)" \
  --category "后端" \
  --tags "Python,后端,装饰器" \
  --brief "本文深入讲解Python装饰器的原理和高级用法" \
  --cookie "your_cookie_here" \
  --verbose

# 更新已有文章
python scripts/publish.py \
  --article-id "1234567890" \
  --title "更新后的标题" \
  --content "新内容" \
  --cookie "xxx"

# 模拟运行（不实际发布）
python scripts/publish.py \
  --title "测试" --content "内容" \
  --cookie "xxx" --dry-run
```

### 使用文件读取内容

```bash
# 从文件读取内容发布
python scripts/publish.py \
  --title "$(head -1 article.md | sed 's/# //')" \
  --content "$(cat article.md)" \
  --category "前端" \
  --cookie "xxx"
```

### 与其他工具配合

```bash
# 使用 AI 生成内容后发布（假设有生成脚本）
python generate_article.py "Python异步编程" | \
  xargs -I {} python scripts/publish.py \
    --title "Python异步编程完全指南" \
    --content {} \
    --cookie "xxx"
```
