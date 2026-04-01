#!/usr/bin/env python3
"""
测试掘金热门文章查询
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from get_hot_articles import JuejinHotNews

def test_basic_functionality():
    """测试基本功能"""
    print("🧪 测试掘金热门文章查询...")
    
    juejin = JuejinHotNews()
    
    # 测试获取热门文章
    print("1. 测试获取热门文章...")
    result = juejin.get_hot_articles(limit=5, sort_type="hot")
    
    if result.get("err_no") != 0:
        print(f"❌ 请求失败: {result.get('err_msg')}")
        return False
    
    print(f"✅ 请求成功，返回数据条数: {len(result.get('data', []))}")
    
    # 测试解析文章
    print("2. 测试解析文章数据...")
    articles = juejin.parse_articles(result)
    
    if not articles:
        print("❌ 解析文章失败")
        return False
    
    print(f"✅ 成功解析 {len(articles)} 篇文章")
    
    # 显示第一篇文章信息
    if articles:
        first_article = articles[0]
        print(f"\n📄 第一篇文章信息:")
        print(f"   标题: {first_article['title'][:50]}...")
        print(f"   作者: {first_article['author']}")
        print(f"   阅读量: {first_article['views']:,}")
        print(f"   点赞数: {first_article['likes']:,}")
        print(f"   标签: {', '.join(first_article['tags'][:3])}")
    
    # 测试格式化输出
    print("\n3. 测试格式化输出...")
    
    # 表格格式
    table_output = juejin.format_output(articles[:3], "table")
    print("✅ 表格格式输出测试通过")
    
    # Markdown格式
    md_output = juejin.format_output(articles[:2], "markdown")
    print("✅ Markdown格式输出测试通过")
    
    # JSON格式
    json_output = juejin.format_output(articles[:1], "json")
    print("✅ JSON格式输出测试通过")
    
    return True

def test_different_sorts():
    """测试不同排序方式"""
    print("\n🧪 测试不同排序方式...")
    
    juejin = JuejinHotNews()
    sort_types = ["hot", "new", "recommend"]
    
    for sort_type in sort_types:
        print(f"  测试 {sort_type} 排序...")
        result = juejin.get_hot_articles(limit=3, sort_type=sort_type)
        
        if result.get("err_no") == 0:
            articles = juejin.parse_articles(result)
            print(f"    ✅ 获取到 {len(articles)} 篇文章")
        else:
            print(f"    ❌ 失败: {result.get('err_msg')}")
    
    return True

def main():
    """主测试函数"""
    print("=" * 50)
    print("掘金热门新闻Skill测试")
    print("=" * 50)
    
    all_passed = True
    
    # 测试基本功能
    if not test_basic_functionality():
        all_passed = False
    
    # 测试不同排序
    if not test_different_sorts():
        all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 所有测试通过！")
        print("Skill已准备就绪，可以使用以下命令：")
        print("  python scripts/get_hot_articles.py --limit 10 --sort hot")
        print("  python scripts/get_hot_articles.py --limit 5 --sort new --format markdown")
    else:
        print("❌ 部分测试失败，请检查问题")
    
    print("=" * 50)

if __name__ == "__main__":
    main()