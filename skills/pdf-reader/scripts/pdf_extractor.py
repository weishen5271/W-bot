#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF 提取工具类
支持文本提取和 OCR 识别，处理跨平台编码问题
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict
import warnings

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class PDFExtractionResult:
    """PDF 提取结果"""
    success: bool
    text: str
    page_count: int
    metadata: Dict
    error: Optional[str] = None
    extraction_method: Optional[str] = None
    encoding_used: Optional[str] = None
    file_path: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return asdict(self)
    
    def to_json(self) -> str:
        """转换为 JSON 字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    def save_text(self, output_path: str) -> bool:
        """保存文本到文件"""
        try:
            # 确保目录存在
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            # 根据文件扩展名决定编码
            _, ext = os.path.splitext(output_path)
            encoding = 'utf-8'
            
            with open(output_path, 'w', encoding=encoding) as f:
                f.write(self.text)
            
            logger.info(f"文本已保存到: {output_path}")
            return True
        except Exception as e:
            logger.error(f"保存文本时出错: {e}")
            return False


class PDFExtractor:
    """PDF 提取器，支持文本提取和 OCR 识别"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化 PDF 提取器
        
        Args:
            config: 配置字典，包含以下选项：
                - use_pdfplumber: 是否使用 pdfplumber（默认为 True）
                - use_pypdf2: 是否使用 PyPDF2（默认为 True）
                - enable_ocr: 是否启用 OCR（默认为 False）
                - ocr_lang: OCR 语言（默认 'chi_sim+eng'）
                - output_encoding: 输出编码（默认 'utf-8'）
                - fallback_encodings: 备用编码列表
        """
        self.config = config or {}
        self._setup_config()
        self._check_dependencies()
        
    def _setup_config(self):
        """设置默认配置"""
        defaults = {
            'use_pdfplumber': True,
            'use_pypdf2': True,
            'enable_ocr': False,
            'ocr_lang': 'chi_sim+eng',
            'output_encoding': 'utf-8',
            'fallback_encodings': ['utf-8', 'gbk', 'gb2312', 'latin-1'],
            'max_pages': None,  # None 表示所有页面
            'extract_images': False,
            'preserve_layout': False,
        }
        
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value
    
    def _check_dependencies(self):
        """检查依赖"""
        self.dependencies = {
            'PyPDF2': False,
            'pdfplumber': False,
            'pytesseract': False,
            'PIL': False,
        }
        
        # 检查 PyPDF2
        try:
            import PyPDF2
            self.dependencies['PyPDF2'] = True
            logger.debug("PyPDF2 可用")
        except ImportError:
            logger.warning("PyPDF2 未安装，部分功能可能不可用")
        
        # 检查 pdfplumber
        try:
            import pdfplumber
            self.dependencies['pdfplumber'] = True
            logger.debug("pdfplumber 可用")
        except ImportError:
            logger.warning("pdfplumber 未安装，部分功能可能不可用")
        
        # 检查 OCR 相关依赖
        if self.config['enable_ocr']:
            try:
                import pytesseract
                from PIL import Image
                self.dependencies['pytesseract'] = True
                self.dependencies['PIL'] = True
                logger.debug("OCR 依赖可用")
            except ImportError as e:
                logger.warning(f"OCR 依赖未完全安装: {e}")
                self.config['enable_ocr'] = False
    
    def _normalize_path(self, file_path: str) -> str:
        """规范化文件路径，处理跨平台问题"""
        # 将路径转换为绝对路径
        abs_path = os.path.abspath(file_path)
        
        # 处理 Windows 路径分隔符
        if os.name == 'nt':  # Windows
            # 确保使用正确的分隔符
            abs_path = abs_path.replace('/', '\\')
        else:  # Unix/Linux/macOS
            abs_path = abs_path.replace('\\', '/')
        
        return abs_path
    
    def _detect_pdf_type(self, file_path: str) -> str:
        """检测 PDF 类型（文本型或扫描型）"""
        try:
            if self.dependencies['PyPDF2']:
                import PyPDF2
                with open(file_path, 'rb') as f:
                    pdf = PyPDF2.PdfReader(f)
                    
                    # 检查前几页是否有文本
                    text_found = False
                    pages_to_check = min(3, len(pdf.pages))
                    
                    for i in range(pages_to_check):
                        page_text = pdf.pages[i].extract_text()
                        if page_text and page_text.strip():
                            text_found = True
                            break
                    
                    if text_found:
                        return "text"
                    else:
                        return "scanned"
        except Exception as e:
            logger.warning(f"检测 PDF 类型时出错: {e}")
        
        return "unknown"
    
    def extract_with_pypdf2(self, file_path: str) -> Tuple[str, Dict]:
        """使用 PyPDF2 提取文本"""
        if not self.dependencies['PyPDF2']:
            return "", {}
        
        try:
            import PyPDF2
            
            text_parts = []
            metadata = {}
            
            with open(file_path, 'rb') as f:
                pdf = PyPDF2.PdfReader(f)
                
                # 获取元数据
                if pdf.metadata:
                    metadata = dict(pdf.metadata)
                
                # 确定要提取的页面数
                max_pages = self.config['max_pages']
                if max_pages is None:
                    pages_to_extract = len(pdf.pages)
                else:
                    pages_to_extract = min(max_pages, len(pdf.pages))
                
                # 提取每页文本
                for i in range(pages_to_extract):
                    try:
                        page_text = pdf.pages[i].extract_text()
                        if page_text and page_text.strip():
                            text_parts.append(f"=== 第 {i+1} 页 ===\n{page_text.strip()}")
                    except Exception as e:
                        logger.warning(f"提取第 {i+1} 页时出错: {e}")
                        text_parts.append(f"=== 第 {i+1} 页（提取失败: {e}）===")
            
            full_text = '\n\n'.join(text_parts)
            return full_text, metadata
            
        except Exception as e:
            logger.error(f"PyPDF2 提取失败: {e}")
            return "", {}
    
    def extract_with_pdfplumber(self, file_path: str) -> Tuple[str, Dict]:
        """使用 pdfplumber 提取文本"""
        if not self.dependencies['pdfplumber']:
            return "", {}
        
        try:
            import pdfplumber
            
            text_parts = []
            metadata = {}
            
            with pdfplumber.open(file_path) as pdf:
                # 获取元数据
                if hasattr(pdf, 'metadata') and pdf.metadata:
                    metadata = dict(pdf.metadata)
                
                # 确定要提取的页面数
                max_pages = self.config['max_pages']
                if max_pages is None:
                    pages_to_extract = len(pdf.pages)
                else:
                    pages_to_extract = min(max_pages, len(pdf.pages))
                
                # 提取每页文本
                for i in range(pages_to_extract):
                    try:
                        page = pdf.pages[i]
                        page_text = page.extract_text()
                        
                        if page_text and page_text.strip():
                            text_parts.append(f"=== 第 {i+1} 页 ===\n{page_text.strip()}")
                    except Exception as e:
                        logger.warning(f"提取第 {i+1} 页时出错: {e}")
                        text_parts.append(f"=== 第 {i+1} 页（提取失败: {e}）===")
            
            full_text = '\n\n'.join(text_parts)
            return full_text, metadata
            
        except Exception as e:
            logger.error(f"pdfplumber 提取失败: {e}")
            return "", {}
    
    def extract_with_ocr(self, file_path: str) -> Tuple[str, Dict]:
        """使用 OCR 提取文本（适用于扫描版 PDF）"""
        if not self.config['enable_ocr'] or not self.dependencies['pytesseract']:
            return "", {}
        
        try:
            import pytesseract
            from PIL import Image
            import PyPDF2
            import io
            
            text_parts = []
            metadata = {}
            
            # 首先尝试获取 PDF 元数据
            try:
                with open(file_path, 'rb') as f:
                    pdf = PyPDF2.PdfReader(f)
                    if pdf.metadata:
                        metadata = dict(pdf.metadata)
            except:
                pass
            
            # 将 PDF 转换为图像并进行 OCR
            # 注意：这里需要 pdf2image 库，简化版本只处理已转换为图像的情况
            # 在实际使用中，应该使用 pdf2image 或类似的库
            
            logger.warning("OCR 提取需要 pdf2image 库，当前为简化版本")
            logger.info("请安装 pdf2image: pip install pdf2image")
            
            # 返回空结果，提示用户安装依赖
            return "", metadata
            
        except Exception as e:
            logger.error(f"OCR 提取失败: {e}")
            return "", {}
    
    def extract_text(self, file_path: str) -> PDFExtractionResult:
        """
        提取 PDF 文本
        
        Args:
            file_path: PDF 文件路径
            
        Returns:
            PDFExtractionResult 对象
        """
        # 规范化路径
        normalized_path = self._normalize_path(file_path)
        
        # 检查文件是否存在
        if not os.path.exists(normalized_path):
            return PDFExtractionResult(
                success=False,
                text="",
                page_count=0,
                metadata={},
                error=f"文件不存在: {normalized_path}",
                file_path=normalized_path
            )
        
        # 检测 PDF 类型
        pdf_type = self._detect_pdf_type(normalized_path)
        logger.info(f"PDF 类型: {pdf_type}")
        
        # 根据类型选择提取方法
        extraction_method = "unknown"
        extracted_text = ""
        metadata = {}
        
        try:
            # 首先尝试 pdfplumber（如果可用且配置为使用）
            if self.config['use_pdfplumber'] and self.dependencies['pdfplumber']:
                extracted_text, metadata = self.extract_with_pdfplumber(normalized_path)
                if extracted_text and extracted_text.strip():
                    extraction_method = "pdfplumber"
                    logger.info("使用 pdfplumber 成功提取文本")
            
            # 如果 pdfplumber 失败或未启用，尝试 PyPDF2
            if (not extracted_text or not extracted_text.strip()) and \
               self.config['use_pypdf2'] and self.dependencies['PyPDF2']:
                extracted_text, metadata = self.extract_with_pypdf2(normalized_path)
                if extracted_text and extracted_text.strip():
                    extraction_method = "PyPDF2"
                    logger.info("使用 PyPDF2 成功提取文本")
            
            # 如果文本提取失败且启用了 OCR，尝试 OCR
            if (not extracted_text or not extracted_text.strip()) and \
               self.config['enable_ocr'] and pdf_type == "scanned":
                extracted_text, metadata = self.extract_with_ocr(normalized_path)
                if extracted_text and extracted_text.strip():
                    extraction_method = "OCR (pytesseract)"
                    logger.info("使用 OCR 成功提取文本")
            
            # 获取页数
            page_count = 0
            try:
                if self.dependencies['PyPDF2']:
                    import PyPDF2
                    with open(normalized_path, 'rb') as f:
                        pdf = PyPDF2.PdfReader(f)
                        page_count = len(pdf.pages)
            except:
                pass
            
            # 处理编码问题
            encoding_used = self.config['output_encoding']
            if extracted_text:
                # 尝试使用配置的编码
                try:
                    # 确保文本是字符串
                    if isinstance(extracted_text, bytes):
                        extracted_text = extracted_text.decode(encoding_used)
                except UnicodeDecodeError:
                    # 尝试备用编码
                    for enc in self.config['fallback_encodings']:
                        try:
                            if isinstance(extracted_text, bytes):
                                extracted_text = extracted_text.decode(enc)
                            encoding_used = enc
                            logger.info(f"使用备用编码: {enc}")
                            break
                        except:
                            continue
            
            success = bool(extracted_text and extracted_text.strip())
            
            return PDFExtractionResult(
                success=success,
                text=extracted_text if extracted_text else "",
                page_count=page_count,
                metadata=metadata,
                error=None if success else "未能提取文本内容",
                extraction_method=extraction_method,
                encoding_used=encoding_used,
                file_path=normalized_path
            )
            
        except Exception as e:
            logger.error(f"提取 PDF 时发生错误: {e}")
            return PDFExtractionResult(
                success=False,
                text="",
                page_count=0,
                metadata={},
                error=f"提取失败: {str(e)}",
                file_path=normalized_path
            )
    
    def batch_extract(self, directory_path: str, output_dir: Optional[str] = None) -> List[PDFExtractionResult]:
        """
        批量提取目录中的所有 PDF 文件
        
        Args:
            directory_path: 目录路径
            output_dir: 输出目录（可选）
            
        Returns:
            提取结果列表
        """
        results = []
        
        # 规范化目录路径
        normalized_dir = self._normalize_path(directory_path)
        
        if not os.path.isdir(normalized_dir):
            logger.error(f"目录不存在: {normalized_dir}")
            return results
        
        # 查找所有 PDF 文件
        pdf_files = []
        for root, dirs, files in os.walk(normalized_dir):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(root, file))
        
        logger.info(f"找到 {len(pdf_files)} 个 PDF 文件")
        
        # 设置输出目录
        if output_dir is None:
            output_dir = os.path.join(normalized_dir, "extracted_texts")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 批量处理
        for i, pdf_file in enumerate(pdf_files):
            logger.info(f"处理文件 {i+1}/{len(pdf_files)}: {os.path.basename(pdf_file)}")
            
            result = self.extract_text(pdf_file)
            results.append(result)
            
            # 如果提取成功，保存文本
            if result.success:
                output_filename = f"{os.path.splitext(os.path.basename(pdf_file))[0]}.txt"
                output_path = os.path.join(output_dir, output_filename)
                result.save_text(output_path)
        
        return results


def main():
    """命令行入口点"""
    import argparse
    
    parser = argparse.ArgumentParser(description='PDF 文本提取工具')
    parser.add_argument('input', help='PDF 文件路径或目录路径')
    parser.add_argument('--output', '-o', help='输出文件路径或目录')
    parser.add_argument('--method', '-m', choices=['auto', 'pypdf2', 'pdfplumber', 'ocr'], 
                       default='auto', help='提取方法')
    parser.add_argument('--ocr-lang', default='chi_sim+eng', help='OCR 语言')
    parser.add_argument('--encoding', default='utf-8', help='输出编码')
    parser.add_argument('--max-pages', type=int, help='最大提取页数')
    parser.add_argument('--batch', '-b', action='store_true', help='批量处理目录')
    parser.add_argument('--config', '-c', help='配置文件路径')
    
    args = parser.parse_args()
    
    # 加载配置
    config = {}
    if args.config and os.path.exists(args.config):
        try:
            with open(args.config, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
    
    # 更新配置
    if args.method != 'auto':
        config['use_pdfplumber'] = (args.method == 'pdfplumber')
        config['use_pypdf2'] = (args.method == 'pypdf2')
        config['enable_ocr'] = (args.method == 'ocr')
    
    if args.ocr_lang:
        config['ocr_lang'] = args.ocr_lang
    
    if args.encoding:
        config['output_encoding'] = args.encoding
    
    if args.max_pages:
        config['max_pages'] = args.max_pages
    
    # 创建提取器
    extractor = PDFExtractor(config)
    
    # 处理输入
    if args.batch:
        # 批量处理目录
        if not os.path.isdir(args.input):
            print(f"错误: {args.input} 不是目录")
            sys.exit(1)
        
        results = extractor.batch_extract(args.input, args.output)
        
        # 输出结果
        success_count = sum(1 for r in results if r.success)
        total_count = len(results)
        
        print(f"\n批量处理完成")
        print(f"总文件数: {total_count}")
        print(f"成功处理: {success_count}")
        print(f"失败: {total_count - success_count}")
        
        if args.output:
            print(f"输出目录: {args.output}")
        
    else:
        # 处理单个文件
        if not os.path.isfile(args.input):
            print(f"错误: {args.input} 不是文件")
            sys.exit(1)
        
        result = extractor.extract_text(args.input)
        
        # 输出结果
        if result.success:
            print(f"\n✓ 提取成功")
            print(f"  文件: {result.file_path}")
            print(f"  页数: {result.page_count}")
            print(f"  提取方法: {result.extraction_method}")
            print(f"  编码: {result.encoding_used}")
            print(f"  文本长度: {len(result.text)} 字符")
            
            # 保存输出
            if args.output:
                if result.save_text(args.output):
                    print(f"✓ 文本已保存到: {args.output}")
                else:
                    print(f"✗ 保存失败")
            
            # 显示文本预览
            preview = result.text[:500]
            if len(result.text) > 500:
                preview += "..."
            print(f"\n文本预览:")
            print("-" * 60)
            print(preview)
            print("-" * 60)
            
        else:
            print(f"\n✗ 提取失败")
            print(f"  错误: {result.error}")
            sys.exit(1)


if __name__ == "__main__":
    main()