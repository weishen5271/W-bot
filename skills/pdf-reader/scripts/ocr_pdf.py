#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF OCR 识别脚本
使用 pytesseract 对扫描版 PDF 进行 OCR 识别
"""

import os
import sys
import json
import logging
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import warnings

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PDFOCRProcessor:
    """PDF OCR 处理器"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化 OCR 处理器
        
        Args:
            config: 配置字典，包含以下选项：
                - ocr_lang: OCR 语言（默认 'chi_sim+eng'）
                - tesseract_cmd: tesseract 命令路径（可选）
                - dpi: 图像 DPI（默认 300）
                - output_format: 输出格式（'text', 'hocr', 'pdf'）
                - preserve_interword_spaces: 是否保留单词间空格
                - timeout: 超时时间（秒）
        """
        self.config = config or {}
        self._setup_config()
        self._check_dependencies()
    
    def _setup_config(self):
        """设置默认配置"""
        defaults = {
            'ocr_lang': 'chi_sim+eng',
            'dpi': 300,
            'output_format': 'text',
            'preserve_interword_spaces': True,
            'timeout': 300,  # 5分钟
            'page_seg_mode': 3,  # 自动页面分割
            'ocr_engine_mode': 3,  # 默认引擎模式
        }
        
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value
        
        # 设置 tesseract 命令路径（跨平台处理）
        if 'tesseract_cmd' not in self.config:
            # 尝试自动检测
            self.config['tesseract_cmd'] = self._detect_tesseract_path()
    
    def _detect_tesseract_path(self) -> Optional[str]:
        """检测 tesseract 命令路径"""
        import subprocess
        
        # 常见路径
        common_paths = [
            'tesseract',
            '/usr/bin/tesseract',
            '/usr/local/bin/tesseract',
            'C:\\Program Files\\Tesseract-OCR\\tesseract.exe',
            'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe',
        ]
        
        for path in common_paths:
            try:
                # 检查命令是否存在
                if os.name == 'nt':  # Windows
                    # 在 Windows 上，需要检查 .exe 扩展名
                    if not path.endswith('.exe') and os.path.exists(path + '.exe'):
                        path = path + '.exe'
                    
                    # 使用 where 命令查找
                    result = subprocess.run(
                        ['where', path],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                else:  # Unix/Linux/macOS
                    # 使用 which 命令查找
                    result = subprocess.run(
                        ['which', path],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                
                if result.returncode == 0:
                    detected_path = result.stdout.strip().split('\n')[0]
                    logger.info(f"检测到 tesseract: {detected_path}")
                    return detected_path
                    
            except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
                continue
        
        logger.warning("未检测到 tesseract 命令")
        return None
    
    def _check_dependencies(self):
        """检查依赖"""
        self.dependencies = {
            'pytesseract': False,
            'PIL': False,
            'pdf2image': False,
        }
        
        # 检查 pytesseract
        try:
            import pytesseract
            self.dependencies['pytesseract'] = True
            
            # 设置 tesseract 命令路径
            if self.config.get('tesseract_cmd'):
                pytesseract.pytesseract.tesseract_cmd = self.config['tesseract_cmd']
            
            logger.debug("pytesseract 可用")
        except ImportError:
            logger.warning("pytesseract 未安装")
        
        # 检查 PIL/Pillow
        try:
            from PIL import Image
            self.dependencies['PIL'] = True
            logger.debug("PIL/Pillow 可用")
        except ImportError:
            logger.warning("PIL/Pillow 未安装")
        
        # 检查 pdf2image
        try:
            import pdf2image
            self.dependencies['pdf2image'] = True
            logger.debug("pdf2image 可用")
        except ImportError:
            logger.warning("pdf2image 未安装，无法直接处理 PDF")
    
    def _convert_pdf_to_images(self, pdf_path: str, dpi: int = 300) -> List:
        """
        将 PDF 转换为图像列表
        
        Args:
            pdf_path: PDF 文件路径
            dpi: 图像 DPI
            
        Returns:
            图像对象列表
        """
        if not self.dependencies['pdf2image']:
            logger.error("需要 pdf2image 库来转换 PDF")
            return []
        
        try:
            import pdf2image
            
            logger.info(f"将 PDF 转换为图像 (DPI: {dpi})")
            
            # 使用临时目录保存图像
            with tempfile.TemporaryDirectory() as temp_dir:
                # 转换 PDF 为图像
                images = pdf2image.convert_from_path(
                    pdf_path,
                    dpi=dpi,
                    output_folder=temp_dir,
                    fmt='png',
                    thread_count=2,
                    grayscale=True,  # 灰度图像通常 OCR 效果更好
                    size=None
                )
                
                logger.info(f"成功转换 {len(images)} 页")
                return images
                
        except Exception as e:
            logger.error(f"转换 PDF 为图像时出错: {e}")
            return []
    
    def _ocr_image(self, image, page_num: int = 1) -> Tuple[str, Dict]:
        """
        对单个图像进行 OCR 识别
        
        Args:
            image: PIL 图像对象或图像路径
            page_num: 页码
            
        Returns:
            (识别文本, 元数据)
        """
        if not self.dependencies['pytesseract'] or not self.dependencies['PIL']:
            return "", {}
        
        try:
            import pytesseract
            from PIL import Image
            
            logger.info(f"对第 {page_num} 页进行 OCR 识别")
            
            # 如果 image 是路径字符串，加载图像
            if isinstance(image, str):
                image_obj = Image.open(image)
            else:
                image_obj = image
            
            # 配置 OCR 参数
            config_params = f'--psm {self.config["page_seg_mode"]}'
            if self.config['preserve_interword_spaces']:
                config_params += ' --oem 1'
            
            # 执行 OCR
            text = pytesseract.image_to_string(
                image_obj,
                lang=self.config['ocr_lang'],
                config=config_params
            )
            
            # 获取 OCR 元数据
            metadata = {
                'page': page_num,
                'ocr_lang': self.config['ocr_lang'],
                'dpi': self.config.get('dpi', 300),
            }
            
            # 如果启用了详细输出，获取更多信息
            if self.config['output_format'] == 'hocr':
                hocr_data = pytesseract.image_to_pdf_or_hocr(
                    image_obj,
                    lang=self.config['ocr_lang'],
                    extension='hocr'
                )
                metadata['hocr'] = hocr_data.decode('utf-8') if isinstance(hocr_data, bytes) else hocr_data
            
            return text.strip(), metadata
            
        except Exception as e:
            logger.error(f"第 {page_num} 页 OCR 失败: {e}")
            return f"[OCR 失败: {str(e)}]", {'page': page_num, 'error': str(e)}
    
    def ocr_pdf(self, pdf_path: str) -> Dict:
        """
        对 PDF 文件进行 OCR 识别
        
        Args:
            pdf_path: PDF 文件路径
            
        Returns:
            包含识别结果的字典
        """
        # 检查文件是否存在
        if not os.path.exists(pdf_path):
            return {
                'success': False,
                'error': f"文件不存在: {pdf_path}",
                'text': '',
                'pages': 0,
                'metadata': {}
            }
        
        # 检查依赖
        if not all([self.dependencies['pytesseract'], 
                   self.dependencies['PIL'], 
                   self.dependencies['pdf2image']]):
            return {
                'success': False,
                'error': '缺少必要的 OCR 依赖',
                'text': '',
                'pages': 0,
                'metadata': {}
            }
        
        try:
            # 将 PDF 转换为图像
            images = self._convert_pdf_to_images(pdf_path, self.config['dpi'])
            
            if not images:
                return {
                    'success': False,
                    'error': '无法将 PDF 转换为图像',
                    'text': '',
                    'pages': 0,
                    'metadata': {}
                }
            
            # 对每页图像进行 OCR
            all_text_parts = []
            all_metadata = []
            
            for i, image in enumerate(images):
                page_num = i + 1
                text, metadata = self._ocr_image(image, page_num)
                
                if text:
                    all_text_parts.append(f"=== 第 {page_num} 页 ===\n{text}")
                else:
                    all_text_parts.append(f"=== 第 {page_num} 页（无文本）===")
                
                all_metadata.append(metadata)
                
                # 进度提示
                if page_num % 10 == 0 or page_num == len(images):
                    logger.info(f"已完成 {page_num}/{len(images)} 页")
            
            # 合并所有文本
            full_text = '\n\n'.join(all_text_parts)
            
            return {
                'success': True,
                'text': full_text,
                'pages': len(images),
                'metadata': {
                    'ocr_lang': self.config['ocr_lang'],
                    'dpi': self.config['dpi'],
                    'page_metadata': all_metadata
                },
                'file_path': pdf_path
            }
            
        except Exception as e:
            logger.error(f"OCR 处理失败: {e}")
            return {
                'success': False,
                'error': f"OCR 失败: {str(e)}",
                'text': '',
                'pages': 0,
                'metadata': {},
                'file_path': pdf_path
            }
    
    def ocr_image(self, image_path: str) -> Dict:
        """
        对单个图像文件进行 OCR 识别
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            包含识别结果的字典
        """
        if not os.path.exists(image_path):
            return {
                'success': False,
                'error': f"文件不存在: {image_path}",
                'text': '',
                'metadata': {}
            }
        
        # 检查依赖
        if not self.dependencies['pytesseract'] or not self.dependencies['PIL']:
            return {
                'success': False,
                'error': '缺少必要的 OCR 依赖',
                'text': '',
                'metadata': {}
            }
        
        try:
            text, metadata = self._ocr_image(image_path, 1)
            
            return {
                'success': bool(text and text.strip()),
                'text': text,
                'metadata': metadata,
                'file_path': image_path
            }
            
        except Exception as e:
            logger.error(f"图像 OCR 失败: {e}")
            return {
                'success': False,
                'error': f"OCR 失败: {str(e)}",
                'text': '',
                'metadata': {},
                'file_path': image_path
            }


def main():
    """命令行入口点"""
    import argparse
    
    parser = argparse.ArgumentParser(description='PDF OCR 识别工具')
    parser.add_argument('input', help='PDF 文件或图像文件路径')
    parser.add_argument('--output', '-o', help='输出文件路径')
    parser.add_argument('--lang', '-l', default='chi_sim+eng', help='OCR 语言')
    parser.add_argument('--dpi', type=int, default=300, help='图像 DPI')
    parser.add_argument('--format', '-f', choices=['text', 'hocr', 'pdf'], 
                       default='text', help='输出格式')
    
    args = parser.parse_args()
    
    # 创建处理器
    config = {
        'ocr_lang': args.lang,
        'dpi': args.dpi,
        'output_format': args.format
    }
    
    processor = PDFOCRProcessor(config)
    
    # 检查文件类型
    input_path = args.input
    _, ext = os.path.splitext(input_path.lower())
    
    if ext == '.pdf':
        # 处理 PDF 文件
        result = processor.ocr_pdf(input_path)
    else:
        # 处理图像文件
        result = processor.ocr_image(input_path)
    
    # 输出结果
    if result['success']:
        print(f"✓ OCR 识别成功")
        print(f"  文件: {result.get('file_path', 'N/A')}")
        print(f"  页数: {result.get('pages', 1)}")
        print(f"  语言: {args.lang}")
        print(f"  文本长度: {len(result['text'])} 字符")
        
        # 保存到文件
        if args.output:
            try:
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(result['text'])
                print(f"✓ 结果已保存到: {args.output}")
            except Exception as e:
                print(f"✗ 保存文件失败: {e}")
        
        # 预览前500字符
        print("\n=== 文本预览 ===")
        preview = result['text'][:500]
        if len(result['text']) > 500:
            preview += "..."
        print(preview)
        
    else:
        print(f"✗ OCR 识别失败")
        print(f"  错误: {result.get('error', '未知错误')}")
        
        # 提供安装建议
        print("\n=== 安装建议 ===")
        print("请确保已安装以下依赖:")
        print("1. Tesseract OCR:")
        print("   - Windows: 下载安装包从 https://github.com/UB-Mannheim/tesseract/wiki")
        print("   - macOS: brew install tesseract")
        print("   - Linux: sudo apt-get install tesseract-ocr")
        print("2. Python 库:")
        print("   pip install pytesseract pillow pdf2image")
        
        sys.exit(1)


if __name__ == "__main__":
    main()