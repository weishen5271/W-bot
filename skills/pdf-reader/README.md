# PDF Reader Skill for W-bot

一个功能强大的 PDF 读取技能，支持文本提取和 OCR 识别，专门处理跨平台编码问题。

## 功能特点

- **多格式支持**: 处理文本型、扫描型和混合型 PDF
- **OCR 识别**: 使用 Tesseract OCR 引擎识别扫描文档
- **跨平台编码**: 自动处理 Windows、Linux、macOS 的编码差异
- **多语言支持**: 中文、英文、日文、韩文等
- **批量处理**: 支持目录级别的批量处理
- **多种输出格式**: 文本、JSON、HTML 等

## 快速开始

### 安装依赖

```bash
# 基础功能
pip install PyPDF2 pdfplumber

# 完整功能（包含 OCR）
pip install pytesseract Pillow pdf2image

# 系统依赖
# Windows: 安装 Tesseract OCR 和 poppler
# macOS: brew install tesseract poppler
# Linux: sudo apt-get install tesseract-ocr poppler-utils
```

### 基本使用

```python
from scripts.pdf_extractor import PDFExtractor

# 创建提取器
extractor = PDFExtractor()

# 提取文本
result = extractor.extract_text('document.pdf')

if result.success:
    print(f"提取成功: {len(result.text)} 字符")
    result.save_text('output.txt')
else:
    print(f"提取失败: {result.error}")
```

### 启用 OCR

```python
config = {
    'enable_ocr': True,
    'ocr_lang': 'chi_sim+eng',  # 中文+英文
    'dpi': 300
}

extractor = PDFExtractor(config)
result = extractor.extract_text('scanned_document.pdf')
```

## 目录结构

```
pdf-reader/
├── SKILL.md                    # 技能主文档
├── README.md                   # 本文件
├── requirements.txt            # Python 依赖
├── __init__.py
├── scripts/                    # 核心脚本
│   ├── __init__.py
│   ├── pdf_extractor.py       # PDF 提取器主类
│   ├── ocr_pdf.py             # OCR 处理器
│   └── batch_processor.py     # 批量处理器
├── references/                 # 参考文档
│   ├── installation.md        # 安装指南
│   ├── encoding_guide.md      # 编码处理指南
│   └── troubleshooting.md     # 故障排除
├── assets/                     # 资源文件
│   └── config_template.json   # 配置模板
└── examples/                   # 示例代码
    └── simple_usage.py        # 简单使用示例
```

## 配置选项

详细配置选项请参考 `assets/config_template.json` 或查看 `SKILL.md`。

### 常用配置

```python
config = {
    # 提取方法
    'use_pdfplumber': True,      # 使用 pdfplumber（推荐）
    'use_pypdf2': True,          # 使用 PyPDF2 作为后备
    'enable_ocr': False,         # 是否启用 OCR
    
    # OCR 设置
    'ocr_lang': 'chi_sim+eng',   # OCR 语言
    'dpi': 300,                  # 图像 DPI
    
    # 编码设置
    'output_encoding': 'utf-8',  # 输出编码
    'fallback_encodings': ['utf-8', 'gbk', 'gb2312', 'latin-1'],
    
    # 性能
    'max_pages': None,           # 最大处理页数
    'max_workers': 4             # 并行处理线程数
}
```

## 操作系统注意事项

### Windows
- 需要安装 Tesseract OCR: https://github.com/UB-Mannheim/tesseract/wiki
- 需要安装 poppler: https://github.com/oschwartz10612/poppler-windows/releases
- 技能会自动修复控制台编码问题

### macOS
```bash
brew install tesseract poppler
brew install tesseract-lang  # 语言包
```

### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr tesseract-ocr-chi-sim poppler-utils
```

## 故障排除

常见问题及解决方案：

1. **"TesseractNotFoundError"**: 安装 Tesseract OCR 并确保在 PATH 中
2. **中文乱码**: 确保使用 UTF-8 编码
3. **扫描 PDF 无文本**: 启用 OCR 功能
4. **内存不足**: 使用 `max_pages` 参数分块处理

详细故障排除请参考 `references/troubleshooting.md`。

## 示例

### 批量处理目录

```python
from scripts.batch_processor import PDFBatchProcessor

processor = PDFBatchProcessor({
    'max_workers': 4,
    'skip_existing': True,
    'generate_report': True
})

results = processor.process_directory('./pdf_documents')
```

### 生成结构化输出

```python
import json
from scripts.pdf_extractor import PDFExtractor

extractor = PDFExtractor()
result = extractor.extract_text('report.pdf')

# 保存为结构化 JSON
with open('report_data.json', 'w', encoding='utf-8') as f:
    json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
```

## 许可证

MIT License

## 支持

如有问题，请：
1. 查看 `references/` 目录中的文档
2. 检查依赖是否已正确安装
3. 提供详细的错误信息和系统环境