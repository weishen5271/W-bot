#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF Reader Skill 简单使用示例
"""

import os
import sys

# 添加父目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from scripts.pdf_extractor import PDFExtractor, PDFExtractionResult
except ImportError:
    print("无法导入 PDFExtractor，请确保已安装依赖")
    sys.exit(1)


def main():
    """主函数"""
    print("PDF Reader Skill 示例")
    print("=" * 60)
    
    # 创建提取器
    print("创建 PDF 提取器...")
    extractor = PDFExtractor({
        'use_pdfplumber': True,
        'use_pypdf2': True,
        'enable_ocr': False,  # 默认不启用 OCR
        'output_encoding': 'utf-8'
    })
    
    # 测试文件路径（使用相对路径）
    test_files = [
        'sample_text.pdf',      # 文本型 PDF
        'sample_scanned.pdf',   # 扫描型 PDF
        'sample_mixed.pdf'      # 混合型 PDF
    ]
    
    for filename in test_files:
        if os.path.exists(filename):
            print(f"\n处理文件: {filename}")
            print("-" * 40)
            
            result = extractor.extract_text(filename)
            
            if result.success:
                print(f"✓ 提取成功")
                print(f"  页数: {result.page_count}")
                print(f"  提取方法: {result.extraction_method}")
                print(f"  编码: {result.encoding_used}")
                print(f"  文本长度: {len(result.text)} 字符")
                
                # 保存提取的文本
                output_file = f"{os.path.splitext(filename)[0]}_extracted.txt"
                if result.save_text(output_file):
                    print(f"✓ 文本已保存到: {output_file}")
                
                # 显示元数据
                if result.metadata:
                    print(f"\n  元数据:")
                    for key, value in result.metadata.items():
                        print(f"    {key}: {value}")
                
                # 显示文本预览
                preview = result.text[:200]
                if len(result.text) > 200:
                    preview += "..."
                print(f"\n  文本预览: {preview}")
                
            else:
                print(f"✗ 提取失败: {result.error}")
                
                # 如果是扫描型 PDF，建议启用 OCR
                if "扫描" in str(result.error).lower() or "scanned" in str(result.error).lower():
                    print("  建议: 启用 OCR 功能处理扫描版 PDF")
                    print("        config = {'enable_ocr': True, 'ocr_lang': 'chi_sim+eng'}")
        else:
            print(f"\n文件不存在: {filename}")
            print("请准备测试文件或使用其他 PDF 文件")
    
    print("\n" + "=" * 60)
    print("示例完成")
    
    # 提供更多使用示例
    print("\n更多使用方式:")
    print("1. 启用 OCR 处理扫描版 PDF:")
    print("   config = {'enable_ocr': True, 'ocr_lang': 'chi_sim+eng'}")
    print("   extractor = PDFExtractor(config)")
    print("   result = extractor.extract_text('scanned.pdf')")
    
    print("\n2. 批量处理目录:")
    print("   from scripts.batch_processor import PDFBatchProcessor")
    print("   processor = PDFBatchProcessor()")
    print("   results = processor.process_directory('./pdfs')")
    
    print("\n3. 自定义输出:")
    print("   result = extractor.extract_text('document.pdf')")
    print("   # 保存为 JSON")
    print("   import json")
    print("   with open('output.json', 'w', encoding='utf-8') as f:")
    print("       json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)")


if __name__ == "__main__":
    main()