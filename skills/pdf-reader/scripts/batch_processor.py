#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF 批量处理脚本
批量处理目录中的 PDF 文件，支持文本提取和 OCR 识别
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import concurrent.futures
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 导入 PDF 提取器
try:
    from pdf_extractor import PDFExtractor, PDFExtractionResult
except ImportError:
    # 如果无法直接导入，尝试从当前目录导入
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pdf_extractor import PDFExtractor, PDFExtractionResult


class PDFBatchProcessor:
    """PDF 批量处理器"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化批量处理器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self._setup_config()
        self.extractor = PDFExtractor(self.config)
    
    def _setup_config(self):
        """设置默认配置"""
        defaults = {
            'output_dir': 'extracted_texts',
            'output_format': 'txt',
            'encoding': 'utf-8',
            'max_workers': 4,
            'skip_existing': True,
            'generate_summary': True,
            'generate_report': True,
            'recursive': True,
        }
        
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value
    
    def _find_pdf_files(self, directory: str) -> List[str]:
        """
        查找目录中的所有 PDF 文件
        
        Args:
            directory: 目录路径
            
        Returns:
            PDF 文件路径列表
        """
        pdf_files = []
        
        if self.config['recursive']:
            # 递归查找
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.lower().endswith('.pdf'):
                        pdf_files.append(os.path.join(root, file))
        else:
            # 仅查找当前目录
            for file in os.listdir(directory):
                if file.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(directory, file))
        
        # 排序
        pdf_files.sort()
        
        return pdf_files
    
    def _process_single_pdf(self, pdf_path: str, output_dir: str) -> Dict:
        """
        处理单个 PDF 文件
        
        Args:
            pdf_path: PDF 文件路径
            output_dir: 输出目录
            
        Returns:
            处理结果字典
        """
        try:
            # 生成输出文件名
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]
            
            # 检查是否跳过已存在的文件
            output_path = os.path.join(output_dir, f"{base_name}.txt")
            if self.config['skip_existing'] and os.path.exists(output_path):
                logger.info(f"跳过已存在的文件: {base_name}")
                return {
                    'file': pdf_path,
                    'status': 'skipped',
                    'output': output_path,
                    'error': None
                }
            
            # 提取文本
            logger.info(f"处理文件: {os.path.basename(pdf_path)}")
            result = self.extractor.extract_text(pdf_path)
            
            if result.success:
                # 保存文本
                success = result.save_text(output_path)
                
                # 保存 JSON 元数据
                json_path = os.path.join(output_dir, f"{base_name}.json")
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
                
                return {
                    'file': pdf_path,
                    'status': 'success' if success else 'failed',
                    'output': output_path,
                    'error': None if success else '保存失败',
                    'metadata': result.metadata,
                    'page_count': result.page_count,
                    'extraction_method': result.extraction_method,
                    'text_length': len(result.text)
                }
            else:
                logger.warning(f"提取失败: {os.path.basename(pdf_path)} - {result.error}")
                return {
                    'file': pdf_path,
                    'status': 'failed',
                    'output': None,
                    'error': result.error,
                    'metadata': {},
                    'page_count': 0,
                    'extraction_method': None,
                    'text_length': 0
                }
                
        except Exception as e:
            logger.error(f"处理文件时出错 {pdf_path}: {e}")
            return {
                'file': pdf_path,
                'status': 'error',
                'output': None,
                'error': str(e),
                'metadata': {},
                'page_count': 0,
                'extraction_method': None,
                'text_length': 0
            }
    
    def process_directory(self, input_dir: str, output_dir: Optional[str] = None) -> Dict:
        """
        处理目录中的所有 PDF 文件
        
        Args:
            input_dir: 输入目录
            output_dir: 输出目录（可选）
            
        Returns:
            处理结果汇总
        """
        # 检查输入目录
        if not os.path.isdir(input_dir):
            return {
                'success': False,
                'error': f"输入目录不存在: {input_dir}",
                'results': [],
                'summary': {}
            }
        
        # 设置输出目录
        if output_dir is None:
            output_dir = os.path.join(input_dir, self.config['output_dir'])
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 查找 PDF 文件
        pdf_files = self._find_pdf_files(input_dir)
        
        if not pdf_files:
            return {
                'success': False,
                'error': f"在目录中未找到 PDF 文件: {input_dir}",
                'results': [],
                'summary': {}
            }
        
        logger.info(f"找到 {len(pdf_files)} 个 PDF 文件")
        
        # 处理文件
        results = []
        
        if self.config['max_workers'] > 1:
            # 使用线程池并行处理
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.config['max_workers']) as executor:
                # 提交任务
                future_to_file = {
                    executor.submit(self._process_single_pdf, pdf_file, output_dir): pdf_file 
                    for pdf_file in pdf_files
                }
                
                # 收集结果
                for future in concurrent.futures.as_completed(future_to_file):
                    pdf_file = future_to_file[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logger.error(f"处理文件时出错 {pdf_file}: {e}")
                        results.append({
                            'file': pdf_file,
                            'status': 'error',
                            'output': None,
                            'error': str(e)
                        })
        else:
            # 串行处理
            for pdf_file in pdf_files:
                result = self._process_single_pdf(pdf_file, output_dir)
                results.append(result)
        
        # 生成汇总
        summary = self._generate_summary(results)
        
        # 生成报告
        if self.config['generate_report']:
            self._generate_report(results, summary, output_dir)
        
        return {
            'success': True,
            'results': results,
            'summary': summary,
            'output_dir': output_dir
        }
    
    def _generate_summary(self, results: List[Dict]) -> Dict:
        """
        生成处理结果汇总
        
        Args:
            results: 处理结果列表
            
        Returns:
            汇总字典
        """
        total_files = len(results)
        successful = sum(1 for r in results if r['status'] == 'success')
        failed = sum(1 for r in results if r['status'] == 'failed')
        errors = sum(1 for r in results if r['status'] == 'error')
        skipped = sum(1 for r in results if r['status'] == 'skipped')
        
        # 统计文本长度
        total_text_length = sum(r.get('text_length', 0) for r in results if r['status'] == 'success')
        avg_text_length = total_text_length / successful if successful > 0 else 0
        
        # 统计页数
        total_pages = sum(r.get('page_count', 0) for r in results if r['status'] == 'success')
        avg_pages = total_pages / successful if successful > 0 else 0
        
        # 统计提取方法
        extraction_methods = {}
        for r in results:
            if r['status'] == 'success':
                method = r.get('extraction_method', 'unknown')
                extraction_methods[method] = extraction_methods.get(method, 0) + 1
        
        return {
            'total_files': total_files,
            'successful': successful,
            'failed': failed,
            'errors': errors,
            'skipped': skipped,
            'success_rate': successful / total_files if total_files > 0 else 0,
            'total_text_length': total_text_length,
            'average_text_length': avg_text_length,
            'total_pages': total_pages,
            'average_pages': avg_pages,
            'extraction_methods': extraction_methods,
            'timestamp': datetime.now().isoformat()
        }
    
    def _generate_report(self, results: List[Dict], summary: Dict, output_dir: str):
        """
        生成处理报告
        
        Args:
            results: 处理结果列表
            summary: 汇总信息
            output_dir: 输出目录
        """
        try:
            # 生成文本报告
            report_path = os.path.join(output_dir, "batch_processing_report.txt")
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write("PDF 批量处理报告\n")
                f.write("=" * 60 + "\n\n")
                
                f.write(f"生成时间: {summary['timestamp']}\n")
                f.write(f"输出目录: {output_dir}\n\n")
                
                f.write("=== 处理汇总 ===\n")
                f.write(f"总文件数: {summary['total_files']}\n")
                f.write(f"成功处理: {summary['successful']}\n")
                f.write(f"处理失败: {summary['failed']}\n")
                f.write(f"处理错误: {summary['errors']}\n")
                f.write(f"跳过文件: {summary['skipped']}\n")
                f.write(f"成功率: {summary['success_rate']:.2%}\n\n")
                
                f.write(f"总文本长度: {summary['total_text_length']} 字符\n")
                f.write(f"平均文本长度: {summary['average_text_length']:.0f} 字符\n")
                f.write(f"总页数: {summary['total_pages']}\n")
                f.write(f"平均页数: {summary['average_pages']:.1f}\n\n")
                
                f.write("=== 提取方法统计 ===\n")
                for method, count in summary['extraction_methods'].items():
                    f.write(f"  {method}: {count} 个文件\n")
                f.write("\n")
                
                f.write("=== 文件详情 ===\n")
                for i, result in enumerate(results, 1):
                    f.write(f"\n{i}. {os.path.basename(result['file'])}\n")
                    f.write(f"   状态: {result['status']}\n")
                    
                    if result['status'] == 'success':
                        f.write(f"   输出文件: {os.path.basename(result['output'])}\n")
                        f.write(f"   页数: {result.get('page_count', 'N/A')}\n")
                        f.write(f"   提取方法: {result.get('extraction_method', 'N/A')}\n")
                        f.write(f"   文本长度: {result.get('text_length', 0)} 字符\n")
                    elif result['error']:
                        f.write(f"   错误: {result['error']}\n")
            
            logger.info(f"报告已生成: {report_path}")
            
            # 生成 JSON 报告
            json_report_path = os.path.join(output_dir, "batch_processing_report.json")
            with open(json_report_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'summary': summary,
                    'results': results,
                    'output_dir': output_dir
                }, f, ensure_ascii=False, indent=2)
            
            logger.info(f"JSON 报告已生成: {json_report_path}")
            
        except Exception as e:
            logger.error(f"生成报告时出错: {e}")


def main():
    """命令行入口点"""
    parser = argparse.ArgumentParser(description='PDF 批量处理工具')
    parser.add_argument('input_dir', help='输入目录路径')
    parser.add_argument('--output-dir', '-o', help='输出目录路径')
    parser.add_argument('--config', '-c', help='配置文件路径')
    parser.add_argument('--max-workers', '-w', type=int, default=4, help='最大工作线程数')
    parser.add_argument('--recursive', '-r', action='store_true', help='递归查找子目录')
    parser.add_argument('--skip-existing', '-s', action='store_true', help='跳过已存在的输出文件')
    parser.add_argument('--no-report', action='store_true', help='不生成报告')
    
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
    config['max_workers'] = args.max_workers
    config['recursive'] = args.recursive
    config['skip_existing'] = args.skip_existing
    config['generate_report'] = not args.no_report
    
    # 创建处理器
    processor = PDFBatchProcessor(config)
    
    # 处理目录
    print("=" * 60)
    print("PDF 批量处理工具")
    print("=" * 60)
    print(f"输入目录: {args.input_dir}")
    print(f"最大线程数: {args.max_workers}")
    print(f"递归查找: {args.recursive}")
    print(f"跳过已存在: {args.skip_existing}")
    print("=" * 60)
    
    result = processor.process_directory(args.input_dir, args.output_dir)
    
    if result['success']:
        summary = result['summary']
        
        print("\n✓ 批量处理完成")
        print(f"\n=== 处理结果汇总 ===")
        print(f"总文件数: {summary['total_files']}")
        print(f"成功处理: {summary['successful']}")
        print(f"处理失败: {summary['failed']}")
        print(f"处理错误: {summary['errors']}")
        print(f"跳过文件: {summary['skipped']}")
        print(f"成功率: {summary['success_rate']:.2%}")
        
        print(f"\n=== 输出信息 ===")
        print(f"输出目录: {result['output_dir']}")
        
        if config['generate_report']:
            print(f"报告文件: {os.path.join(result['output_dir'], 'batch_processing_report.txt')}")
        
        # 显示提取方法统计
        if summary['extraction_methods']:
            print(f"\n=== 提取方法统计 ===")
            for method, count in summary['extraction_methods'].items():
                print(f"  {method}: {count} 个文件")
        
        # 显示失败的文件（如果有）
        failed_files = [r for r in result['results'] if r['status'] in ['failed', 'error']]
        if failed_files:
            print(f"\n=== 失败的文件 ===")
            for i, r in enumerate(failed_files[:10], 1):  # 最多显示10个
                print(f"{i}. {os.path.basename(r['file'])}: {r.get('error', '未知错误')}")
            
            if len(failed_files) > 10:
                print(f"... 还有 {len(failed_files) - 10} 个失败文件")
        
    else:
        print(f"\n✗ 批量处理失败")
        print(f"错误: {result.get('error', '未知错误')}")
        sys.exit(1)


if __name__ == "__main__":
    main()