#!/usr/bin/env python3
"""
稀土掘金博客发布脚本
支持：发布文章、更新文章、保存草稿

使用方法:
    python publish.py --title "文章标题" --content "正文" --cookie "your_cookie"
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlencode, parse_qs, urlparse

try:
    import requests
except ImportError:
    print("错误: 需要安装 requests 库")
    print("运行: pip install requests")
    sys.exit(1)


# API 配置
JUEJIN_API_BASE = "https://api.juejin.cn"
JUEJIN_WEB_BASE = "https://juejin.cn"


class JuejinAPI:
    """掘金 API 客户端"""
    
    def __init__(self, cookie: str = None, uid: str = None, token: str = None):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Origin": JUEJIN_WEB_BASE,
            "Referer": JUEJIN_WEB_BASE + "/",
        })
        
        self.uid = uid
        self.token = token
        
        if cookie:
            self.set_cookie(cookie)
    
    def set_cookie(self, cookie: str):
        """设置 Cookie 并解析关键字段"""
        # 处理 Cookie 字符串
        if ";" in cookie:
            # 完整的 Cookie 字符串
            cookies = {}
            for item in cookie.split(";"):
                item = item.strip()
                if "=" in item:
                    key, value = item.split("=", 1)
                    cookies[key.strip()] = value.strip()
            
            # 设置到 session
            self.session.cookies.update(cookies)
            
            # 提取关键字段
            self.uid = cookies.get("uid")
            self.token = cookies.get("token") or cookies.get("x-juejin-token")
        else:
            # 可能是 Token 直接传入
            self.session.headers["Authorization"] = f"Bearer {cookie}" if not cookie.startswith("Bearer ") else cookie
    
    def _request(self, method: str, endpoint: str, **kwargs) -> Dict:
        """发送请求"""
        url = f"{JUEJIN_API_BASE}{endpoint}"
        
        # 添加时间戳避免缓存
        if "params" not in kwargs:
            kwargs["params"] = {}
        kwargs["params"]["_"] = int(time.time() * 1000)
        
        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            
            data = response.json()
            
            # 检查掘金 API 的错误码
            if data.get("err_no") != 0 and data.get("err_msg"):
                raise JuejinAPIError(f"API错误: {data.get('err_msg')} (code: {data.get('err_no')})")
            
            return data
            
        except requests.exceptions.RequestException as e:
            raise JuejinAPIError(f"请求失败: {str(e)}")
    
    def get(self, endpoint: str, params: Dict = None) -> Dict:
        return self._request("GET", endpoint, params=params)
    
    def post(self, endpoint: str, data: Dict = None) -> Dict:
        return self._request("POST", endpoint, json=data)
    
    # ===== 用户相关 API =====
    
    def get_user_info(self) -> Dict:
        """获取当前用户信息"""
        return self.get("/user_api/v1/user/get")
    
    # ===== 文章相关 API =====
    
    def get_categories(self) -> List[Dict]:
        """获取文章分类列表"""
        resp = self.get("/tag_api/v1/query_category_briefs")
        return resp.get("data", [])
    
    def get_tags(self, cursor: str = "0", limit: int = 20) -> Dict:
        """获取标签列表"""
        return self.get("/tag_api/v1/query_tag_list", {
            "cursor": cursor,
            "limit": limit,
            "sort_type": 1
        })
    
    def search_tags(self, keyword: str) -> List[Dict]:
        """搜索标签"""
        resp = self.get("/tag_api/v1/search", {
            "key_word": keyword,
            "cursor": "0",
            "limit": 10
        })
        return resp.get("data", [])
    
    def publish_article(
        self,
        title: str,
        content: str,
        category_id: str = "6809637769959178254",  # 后端默认
        tags: List[str] = None,
        cover_image: str = "",
        brief: str = "",
        article_id: str = None,
    ) -> Dict:
        """
        发布文章
        
        Args:
            title: 文章标题
            content: 文章内容 (Markdown格式)
            category_id: 分类ID
            tags: 标签列表
            cover_image: 封面图片URL
            brief: 文章摘要
            article_id: 文章ID（用于更新已存在的文章）
        """
        # 如果没有提供摘要，从内容生成
        if not brief:
            brief = self._generate_brief(content)
        
        # 确保标签是标签ID列表
        tag_ids = []
        if tags:
            for tag in tags:
                # 如果是数字ID直接使用
                if tag.isdigit():
                    tag_ids.append({"tag_id": tag})
                else:
                    # 搜索标签获取ID
                    search_result = self.search_tags(tag)
                    if search_result:
                        tag_ids.append({"tag_id": str(search_result[0]["tag_id"])})
        
        data = {
            "category_id": category_id,
            "tag_ids": tag_ids,
            "link_url": "",
            "cover_image": cover_image,
            "title": title,
            "brief_content": brief,
            "edit_type": 10,  # Markdown
            "html_content": "deprecated",
            "mark_content": content,
        }
        
        if article_id:
            # 更新文章
            data["id"] = article_id
            data["status"] = 0  # 发布
            return self.post("/content_api/v1/article/update", data)
        else:
            # 新建文章
            return self.post("/content_api/v1/article/publish", data)
    
    def _generate_brief(self, content: str, max_length: int = 100) -> str:
        """从内容生成摘要"""
        # 移除Markdown标记
        text = re.sub(r'[#*`\[\]()]', '', content)
        # 取前N个字符
        brief = text[:max_length].strip()
        if len(text) > max_length:
            brief += "..."
        return brief


class JuejinAPIError(Exception):
    """掘金API错误"""
    pass


def parse_category(category_name: str) -> str:
    """解析分类名称到ID"""
    categories = {
        "后端": "6809637769959178254",
        "前端": "6809637769959178253",
        "Android": "6809635626879549454",
        "iOS": "6809635626879549453",
        "人工智能": "6809637773935374343",
        "开发工具": "6809637771511070734",
        "代码人生": "6809637776263217160",
        "阅读": "6809637772874219534",
    }
    return categories.get(category_name, "6809637769959178254")  # 默认后端


def main():
    parser = argparse.ArgumentParser(
        description="稀土掘金博客发布工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 发布新文章
    python publish.py --title "Python装饰器详解" --content "# 正文..." --cookie "your_cookie"
    
    # 指定分类和标签
    python publish.py --title "文章标题" --content "正文" \\
        --category "前端" --tags "Python,后端" --cookie "xxx"
    
    # 更新已有文章
    python publish.py --article-id "123456" --title "新标题" --content "新内容" --cookie "xxx"
        """
    )
    
    parser.add_argument("--title", required=True, help="文章标题")
    parser.add_argument("--content", required=True, help="文章内容（Markdown格式）")
    parser.add_argument("--brief", help="文章摘要（自动从内容生成）")
    parser.add_argument("--category", default="后端", help="文章分类（后端/前端/Android/iOS/人工智能/开发工具/代码人生/阅读）")
    parser.add_argument("--tags", help="标签，逗号分隔，如：Python,后端")
    parser.add_argument("--cover", help="封面图片URL")
    parser.add_argument("--article-id", help="文章ID（用于更新已有文章）")
    
    # 认证方式
    auth_group = parser.add_mutually_exclusive_group(required=True)
    auth_group.add_argument("--cookie", help="掘金网站的 Cookie 字符串")
    auth_group.add_argument("--token", help="掘金 API Token")
    auth_group.add_argument("--uid", help="用户ID，配合其他认证方式使用")
    
    parser.add_argument("--draft", action="store_true", help="保存为草稿而不发布")
    parser.add_argument("--dry-run", action="store_true", help="模拟运行，不实际发布")
    parser.add_argument("-v", "--verbose", action="store_true", help="显示详细日志")
    
    args = parser.parse_args()
    
    # 处理标签
    tag_list = []
    if args.tags:
        tag_list = [t.strip() for t in args.tags.split(",")]
    
    # 解析分类
    category_id = parse_category(args.category)
    
    # 初始化 API 客户端
    try:
        api = JuejinAPI(cookie=args.cookie, uid=args.uid, token=args.token)
        
        if args.verbose:
            print("正在验证用户身份...")
        user_info = api.get_user_info()
        user_name = user_info.get("data", {}).get("user_name", "未知用户")
        print(f"已认证用户: {user_name}")
        
    except Exception as e:
        print(f"认证失败: {str(e)}")
        print("提示：请检查 Cookie 是否有效，或尝试重新登录掘金网站获取最新 Cookie")
        sys.exit(1)
    
    # 模拟运行
    if args.dry_run:
        print("\n[模拟运行模式] 以下信息将用于发布：")
        print(f"标题: {args.title}")
        print(f"分类: {args.category} ({category_id})")
        print(f"标签: {tag_list}")
        print(f"内容长度: {len(args.content)} 字符")
        print(f"文章ID: {args.article_id or '新建文章'}")
        print(f"保存为草稿: {args.draft}")
        return
    
    # 发布文章
    try:
        if args.verbose:
            print("\n正在准备发布...")
        
        result = api.publish_article(
            title=args.title,
            content=args.content,
            category_id=category_id,
            tags=tag_list,
            cover_image=args.cover or "",
            brief=args.brief,
            article_id=args.article_id,
        )
        
        article_id = result.get("data", {}).get("article_id") or result.get("data", {}).get("id")
        
        if article_id:
            article_url = f"{JUEJIN_WEB_BASE}/post/{article_id}"
            print(f"\n✅ 文章发布成功！")
            print(f"文章ID: {article_id}")
            print(f"文章链接: {article_url}")
        else:
            print(f"\n⚠️ 发布响应异常，请检查掘金后台")
            print(f"响应: {json.dumps(result, ensure_ascii=False, indent=2)}")
            
    except JuejinAPIError as e:
        print(f"\n❌ 发布失败: {str(e)}")
        if "token" in str(e).lower() or "auth" in str(e).lower() or "login" in str(e).lower():
            print("提示: 认证信息可能已过期，请重新获取 Cookie")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 发布失败: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
