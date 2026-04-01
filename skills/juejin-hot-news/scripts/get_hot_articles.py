#!/usr/bin/env python3
"""
掘金热门文章查询脚本
支持获取掘金社区的热门文章，可按热度、时间排序
"""

import json
import requests
import argparse
from datetime import datetime
from typing import Dict, List, Any

class JuejinHotNews:
    """掘金热门文章查询类"""
    
    def __init__(self):
        self.base_url = "https://api.juejin.cn/recommend_api/v1/article/recommend_all_feed"
        self.params = {
            "aid": "2608",
            "uuid": "7243086311112009224",
            "spider": "0"
        }
        self.headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        }
    
    def get_hot_articles(self, limit: int = 10, sort_type: str = "hot", cursor: str = "0") -> Dict[str, Any]:
        """
        获取热门文章
        
        Args:
            limit: 获取文章数量，最大20
            sort_type: 排序方式，可选值：hot（最热）、new（最新）、recommend（综合推荐）
            cursor: 分页游标，"0"表示第一页
        
        Returns:
            文章数据字典
        """
        # 映射排序类型
        sort_mapping = {
            "hot": 400,      # 最热
            "new": 300,      # 最新
            "recommend": 200 # 综合推荐
        }
        
        sort_code = sort_mapping.get(sort_type, 200)
        
        # 构建请求数据
        data = {
            "id_type": 2,
            "client_type": 2608,
            "sort_type": sort_code,
            "cursor": cursor,
            "limit": min(limit, 20)  # 限制最大20条
        }
        
        try:
            response = requests.post(
                self.base_url,
                params=self.params,
                headers=self.headers,
                data=json.dumps(data),
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"err_no": -1, "err_msg": f"请求失败: {str(e)}", "data": []}
        except json.JSONDecodeError as e:
            return {"err_no": -1, "err_msg": f"JSON解析失败: {str(e)}", "data": []}
    
    def parse_articles(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        解析文章数据
        
        Args:
            data: API返回的原始数据
        
        Returns:
            解析后的文章列表
        """
        if data.get("err_no") != 0:
            return []
        
        articles = []
        for item in data.get("data", []):
            if item.get("item_type") != 2:  # 只处理文章类型
                continue
            
            item_info = item.get("item_info", {})
            article_info = item_info.get("article_info", {})
            author_info = item_info.get("author_user_info", {})
            tags = item_info.get("tags", [])
            
            # 解析时间戳
            ctime = article_info.get("ctime", "")
            if ctime:
                try:
                    publish_time = datetime.fromtimestamp(int(ctime)).strftime("%Y-%m-%d %H:%M")
                except:
                    publish_time = "未知时间"
            else:
                publish_time = "未知时间"
            
            article = {
                "id": article_info.get("article_id", ""),
                "title": article_info.get("title", "无标题"),
                "brief": article_info.get("brief_content", "无摘要"),
                "author": author_info.get("user_name", "匿名作者"),
                "author_job": author_info.get("job_title", ""),
                "views": article_info.get("view_count", 0),
                "likes": article_info.get("digg_count", 0),
                "comments": article_info.get("comment_count", 0),
                "collects": article_info.get("collect_count", 0),
                "hot_index": article_info.get("hot_index", 0),
                "read_time": article_info.get("read_time", "未知"),
                "publish_time": publish_time,
                "cover_image": article_info.get("cover_image", ""),
                "tags": [tag.get("tag_name", "") for tag in tags if tag.get("tag_name")],
                "url": f"https://juejin.cn/post/{article_info.get('article_id', '')}"
            }
            articles.append(article)
        
        return articles
    
    def format_output(self, articles: List[Dict[str, Any]], output_format: str = "table") -> str:
        """
        格式化输出
        
        Args:
            articles: 文章列表
            output_format: 输出格式，可选值：table（表格）、json（JSON）、markdown（Markdown）
        
        Returns:
            格式化后的字符串
        """
        if not articles:
            return "未找到文章数据"
        
        if output_format == "json":
            return json.dumps(articles, ensure_ascii=False, indent=2)
        
        elif output_format == "markdown":
            output = "# 掘金热门文章\n\n"
            for i, article in enumerate(articles, 1):
                output += f"## {i}. [{article['title']}]({article['url']})\n"
                output += f"- **作者**: {article['author']}"
                if article['author_job']:
                    output += f" ({article['author_job']})"
                output += f"\n- **发布时间**: {article['publish_time']}\n"
                output += f"- **阅读量**: {article['views']:,} | **点赞**: {article['likes']:,} | **评论**: {article['comments']:,} | **收藏**: {article['collects']:,}\n"
                output += f"- **阅读时间**: {article['read_time']} | **热度指数**: {article['hot_index']}\n"
                if article['tags']:
                    output += f"- **标签**: {', '.join(article['tags'])}\n"
                output += f"- **摘要**: {article['brief'][:100]}...\n\n"
            return output
        
        else:  # table格式
            # 创建表格
            table = "| 排名 | 标题 | 作者 | 阅读量 | 点赞 | 评论 | 标签 |\n"
            table += "|------|------|------|--------|------|------|------|\n"
            
            for i, article in enumerate(articles, 1):
                title = article['title'][:30] + "..." if len(article['title']) > 30 else article['title']
                tags = ", ".join(article['tags'][:3]) if article['tags'] else "无标签"
                if len(tags) > 20:
                    tags = tags[:20] + "..."
                
                table += f"| {i} | [{title}]({article['url']}) | {article['author']} | {article['views']:,} | {article['likes']:,} | {article['comments']:,} | {tags} |\n"
            
            return table

def main():
    parser = argparse.ArgumentParser(description="查询掘金热门文章")
    parser.add_argument("--limit", type=int, default=10, help="获取文章数量（最大20）")
    parser.add_argument("--sort", choices=["hot", "new", "recommend"], default="hot", 
                       help="排序方式：hot(最热)、new(最新)、recommend(综合推荐)")
    parser.add_argument("--page", type=str, default="0", help="分页游标，0表示第一页")
    parser.add_argument("--format", choices=["table", "json", "markdown"], default="table",
                       help="输出格式：table(表格)、json(JSON)、markdown(Markdown)")
    parser.add_argument("--tag", type=str, help="筛选标签（暂不支持，仅作占位）")
    
    args = parser.parse_args()
    
    # 创建查询实例
    juejin = JuejinHotNews()
    
    # 获取文章数据
    print(f"正在获取掘金{args.sort}文章...")
    result = juejin.get_hot_articles(limit=args.limit, sort_type=args.sort, cursor=args.page)
    
    if result.get("err_no") != 0:
        print(f"错误: {result.get('err_msg', '未知错误')}")
        return
    
    # 解析文章
    articles = juejin.parse_articles(result)
    
    if not articles:
        print("未找到文章数据")
        return
    
    # 格式化输出
    output = juejin.format_output(articles, args.format)
    print(output)
    
    # 显示统计信息
    print(f"\n📊 统计信息:")
    print(f"  共获取 {len(articles)} 篇文章")
    print(f"  总阅读量: {sum(a['views'] for a in articles):,}")
    print(f"  总点赞数: {sum(a['likes'] for a in articles):,}")
    print(f"  总评论数: {sum(a['comments'] for a in articles):,}")

if __name__ == "__main__":
    main()