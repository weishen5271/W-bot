# 掘金热门新闻高级使用指南

## 目录
- [定时任务](#定时任务)
- [数据存储](#数据存储)
- [邮件推送](#邮件推送)
- [Webhook集成](#webhook集成)
- [数据分析](#数据分析)
- [故障排除](#故障排除)

## 定时任务

### 使用crontab定时抓取
```bash
# 每天上午9点获取热门文章并保存到文件
0 9 * * * cd /path/to/skills/juejin-hot-news && python scripts/get_hot_articles.py --limit 20 --sort hot --format json > /tmp/juejin_hot_$(date +\%Y\%m\%d).json

# 每小时获取最新文章
0 * * * * cd /path/to/skills/juejin-hot-news && python scripts/get_hot_articles.py --limit 10 --sort new --format markdown > /tmp/juejin_latest.md
```

### 使用Python schedule库
```python
import schedule
import time
from scripts.get_hot_articles import JuejinHotNews

def daily_hot_news():
    juejin = JuejinHotNews()
    result = juejin.get_hot_articles(limit=20, sort_type="hot")
    articles = juejin.parse_articles(result)
    # 处理文章数据...
    print(f"已获取 {len(articles)} 篇热门文章")

# 每天9点执行
schedule.every().day.at("09:00").do(daily_hot_news)

# 每小时执行
schedule.every().hour.do(lambda: JuejinHotNews().get_hot_articles(limit=10, sort_type="new"))

while True:
    schedule.run_pending()
    time.sleep(60)
```

## 数据存储

### 存储到SQLite数据库
```python
import sqlite3
from datetime import datetime
from scripts.get_hot_articles import JuejinHotNews

def save_to_sqlite(articles):
    conn = sqlite3.connect('juejin_articles.db')
    cursor = conn.cursor()
    
    # 创建表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS articles (
        id TEXT PRIMARY KEY,
        title TEXT,
        author TEXT,
        views INTEGER,
        likes INTEGER,
        comments INTEGER,
        collects INTEGER,
        hot_index INTEGER,
        publish_time TEXT,
        tags TEXT,
        url TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # 插入数据
    for article in articles:
        cursor.execute('''
        INSERT OR REPLACE INTO articles 
        (id, title, author, views, likes, comments, collects, hot_index, publish_time, tags, url)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            article['id'],
            article['title'],
            article['author'],
            article['views'],
            article['likes'],
            article['comments'],
            article['collects'],
            article['hot_index'],
            article['publish_time'],
            ','.join(article['tags']),
            article['url']
        ))
    
    conn.commit()
    conn.close()

# 使用示例
juejin = JuejinHotNews()
result = juejin.get_hot_articles(limit=20)
articles = juejin.parse_articles(result)
save_to_sqlite(articles)
```

### 存储到JSON文件
```python
import json
from datetime import datetime
from scripts.get_hot_articles import JuejinHotNews

def save_to_json(articles, filename=None):
    if filename is None:
        filename = f"juejin_hot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    data = {
        "fetch_time": datetime.now().isoformat(),
        "article_count": len(articles),
        "articles": articles
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    return filename

# 使用示例
juejin = JuejinHotNews()
result = juejin.get_hot_articles(limit=15)
articles = juejin.parse_articles(result)
save_to_json(articles)
```

## 邮件推送

### 发送每日热门文章邮件
```python
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from scripts.get_hot_articles import JuejinHotNews

def send_hot_news_email(to_email, smtp_config):
    # 获取热门文章
    juejin = JuejinHotNews()
    result = juejin.get_hot_articles(limit=10, sort_type="hot")
    articles = juejin.parse_articles(result)
    
    # 构建邮件内容
    subject = f"掘金每日热门文章 - {datetime.now().strftime('%Y-%m-%d')}"
    
    html_content = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #333; }}
            .article {{ border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px; }}
            .title {{ font-size: 18px; font-weight: bold; margin-bottom: 5px; }}
            .meta {{ color: #666; font-size: 14px; margin-bottom: 5px; }}
            .tags {{ color: #007bff; font-size: 12px; }}
            .stats {{ background-color: #f8f9fa; padding: 5px; border-radius: 3px; }}
        </style>
    </head>
    <body>
        <h1>📰 掘金今日热门文章</h1>
        <p>共 {len(articles)} 篇文章，获取时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    """
    
    for i, article in enumerate(articles, 1):
        html_content += f"""
        <div class="article">
            <div class="title">{i}. <a href="{article['url']}">{article['title']}</a></div>
            <div class="meta">👤 {article['author']} | 🕐 {article['publish_time']} | ⏱️ {article['read_time']}</div>
            <div class="stats">
                👁️ {article['views']:,} | 👍 {article['likes']:,} | 💬 {article['comments']:,} | ⭐ {article['collects']:,}
            </div>
            <div class="tags">🏷️ {', '.join(article['tags'][:3]) if article['tags'] else '无标签'}</div>
            <p>{article['brief'][:150]}...</p>
        </div>
        """
    
    html_content += """
        <hr>
        <p style="color: #999; font-size: 12px;">
            此邮件由掘金热门新闻Skill自动生成<br>
            取消订阅请联系管理员
        </p>
    </body>
    </html>
    """
    
    # 发送邮件
    msg = MIMEMultipart()
    msg['From'] = smtp_config['from_email']
    msg['To'] = to_email
    msg['Subject'] = subject
    
    msg.attach(MIMEText(html_content, 'html'))
    
    with smtplib.SMTP(smtp_config['smtp_server'], smtp_config['smtp_port']) as server:
        server.starttls()
        server.login(smtp_config['username'], smtp_config['password'])
        server.send_message(msg)
    
    return True

# 配置示例
smtp_config = {
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'username': 'your-email@gmail.com',
    'password': 'your-password',
    'from_email': 'your-email@gmail.com'
}

# 发送邮件
send_hot_news_email('recipient@example.com', smtp_config)
```

## Webhook集成

### 发送到Slack
```python
import requests
import json
from scripts.get_hot_articles import JuejinHotNews

def send_to_slack(webhook_url, articles):
    blocks = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": "📰 掘金热门文章推送",
                "emoji": True
            }
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"共 {len(articles)} 篇热门文章"
            }
        }
    ]
    
    for i, article in enumerate(articles[:5], 1):  # 只发送前5篇
        blocks.append({
            "type": "divider"
        })
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*{i}. <{article['url']}|{article['title']}>*\n👤 {article['author']} | 🕐 {article['publish_time']}\n👁️ {article['views']:,} | 👍 {article['likes']:,} | 💬 {article['comments']:,}"
            }
        })
    
    payload = {"blocks": blocks}
    
    response = requests.post(
        webhook_url,
        data=json.dumps(payload),
        headers={'Content-Type': 'application/json'}
    )
    
    return response.status_code == 200

# 使用示例
juejin = JuejinHotNews()
result = juejin.get_hot_articles(limit=5)
articles = juejin.parse_articles(result)
send_to_slack('https://hooks.slack.com/services/XXX/XXX/XXX', articles)
```

### 发送到钉钉
```python
def send_to_dingtalk(webhook_url, articles):
    text = "## 📰 掘金热门文章\n\n"
    
    for i, article in enumerate(articles[:5], 1):
        text += f"### {i}. [{article['title']}]({article['url']})\n"
        text += f"- **作者**: {article['author']}\n"
        text += f"- **时间**: {article['publish_time']}\n"
        text += f"- **数据**: 阅读{article['views']:,} 点赞{article['likes']:,} 评论{article['comments']:,}\n"
        if article['tags']:
            text += f"- **标签**: {', '.join(article['tags'][:3])}\n"
        text += "\n"
    
    payload = {
        "msgtype": "markdown",
        "markdown": {
            "title": "掘金热门文章",
            "text": text
        }
    }
    
    response = requests.post(
        webhook_url,
        json=payload,
        headers={'Content-Type': 'application/json'}
    )
    
    return response.status_code == 200
```

## 数据分析

### 热门标签分析
```python
from collections import Counter
from scripts.get_hot_articles import JuejinHotNews

def analyze_tags(articles):
    all_tags = []
    for article in articles:
        all_tags.extend(article['tags'])
    
    tag_counter = Counter(all_tags)
    
    print("📊 热门标签分析:")
    for tag, count in tag_counter.most_common(10):
        print(f"  {tag}: {count}次")
    
    return tag_counter

# 使用示例
juejin = JuejinHotNews()
result = juejin.get_hot_articles(limit=50)
articles = juejin.parse_articles(result)
analyze_tags(articles)
```

### 作者影响力分析
```python
def analyze_authors(articles):
    author_stats = {}
    
    for article in articles:
        author = article['author']
        if author not in author_stats:
            author_stats[author] = {
                'article_count': 0,
                'total_views': 0,
                'total_likes': 0,
                'articles': []
            }
        
        author_stats[author]['article_count'] += 1
        author_stats[author]['total_views'] += article['views']
        author_stats[author]['total_likes'] += article['likes']
        author_stats[author]['articles'].append(article['title'])
    
    # 按总阅读量排序
    sorted_authors = sorted(
        author_stats.items(),
        key=lambda x: x[1]['total_views'],
        reverse=True
    )
    
    print("👥 作者影响力排名:")
    for i, (author, stats) in enumerate(sorted_authors[:10], 1):
        avg_views = stats['total_views'] / stats['article_count']
        print(f"  {i}. {author}: {stats['article_count']}篇, 总阅读{stats['total_views']:,}, 平均{avg_views:,.0f}")
    
    return author_stats
```

## 故障排除

### 常见问题

1. **API请求失败**
   - 检查网络连接
   - 验证API端点是否可用
   - 检查请求参数是否正确

2. **返回数据为空**
   - 确认排序参数是否正确
   - 检查游标参数
   - 可能是API限制或更新

3. **解析错误**
   - 检查JSON格式
   - 验证数据结构是否变化
   - 更新解析逻辑

### 调试模式
```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 调试请求
import http.client
http.client.HTTPConnection.debuglevel = 1
```

### 备用方案
如果主API不可用，可以考虑：
1. 使用掘金RSS源
2. 爬取掘金网页版
3. 使用第三方聚合API