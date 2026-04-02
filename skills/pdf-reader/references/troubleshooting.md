# PDF Reader Skill 故障排除指南

## 概述

本文档提供 PDF Reader Skill 常见问题的解决方案和调试技巧。

## 常见问题分类

### 1. 安装和依赖问题
### 2. 文件读取问题
### 3. 文本提取问题
### 4. OCR 识别问题
### 5. 性能问题
### 6. 编码问题

## 1. 安装和依赖问题

### 问题 1.1: "ModuleNotFoundError: No module named 'PyPDF2'"

**症状**:
```
ModuleNotFoundError: No module named 'PyPDF2'
```

**原因**: PyPDF2 库未安装。

**解决方案**:
```bash
# 安装 PyPDF2
pip install PyPDF2

# 或者安装最新版本
pip install PyPDF2 --upgrade
```

### 问题 1.2: "TesseractNotFoundError: tesseract is not installed or it's not in your PATH"

**症状**:
```
pytesseract.pytesseract.TesseractNotFoundError: tesseract is not installed or it's not in your PATH
```

**原因**: Tesseract OCR 未安装或不在系统 PATH 中。

**解决方案**:

#### Windows:
1. 下载并安装 Tesseract OCR: https://github.com/UB-Mannheim/tesseract/wiki
2. 安装时勾选 "中文简体" 语言包
3. 将安装目录（如 `C:\Program Files\Tesseract-OCR`）添加到系统 PATH
4. 重启命令行或 IDE

#### macOS:
```bash
brew install tesseract
brew install tesseract-lang  # 安装语言包
```

#### Linux (Ubuntu/Debian):
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
sudo apt-get install tesseract-ocr-chi-sim  # 中文语言包
```

#### 验证安装:
```bash
tesseract --version
tesseract --list-langs
```

### 问题 1.3: "pdf2image 无法找到 poppler"

**症状**:
```
pdf2image.exceptions.PDFInfoNotInstalledError: Unable to get page count. Is poppler installed and in PATH?
```

**解决方案**:

#### Windows:
1. 下载 poppler: https://github.com/oschwartz10612/poppler-windows/releases
2. 解压到 `C:\poppler` 或自定义目录
3. 将 `bin` 目录添加到系统 PATH

#### macOS:
```bash
brew install poppler
```

#### Linux:
```bash
sudo apt-get install poppler-utils
```

### 问题 1.4: 内存不足错误

**症状**:
```
MemoryError: Unable to allocate array with shape...
```

**解决方案**:
1. 减少同时处理的文件数量
2. 降低图像 DPI:
   ```python
   config = {'dpi': 200}  # 默认 300
   ```
3. 使用分页处理
4. 增加系统虚拟内存

## 2. 文件读取问题

### 问题 2.1: "FileNotFoundError: [Errno 2] No such file or directory"

**症状**:
```
FileNotFoundError: [Errno 2] No such file or directory: 'document.pdf'
```

**解决方案**:
1. 检查文件路径是否正确
2. 使用绝对路径而不是相对路径
3. 检查文件权限
4. 使用路径规范化:

```python
import os

# 转换为绝对路径
abs_path = os.path.abspath('document.pdf')
print(f"绝对路径: {abs_path}")

# 检查文件是否存在
if os.path.exists(abs_path):
    print("文件存在")
else:
    print("文件不存在")
```

### 问题 2.2: 权限被拒绝

**症状**:
```
PermissionError: [Errno 13] Permission denied: 'document.pdf'
```

**解决方案**:
1. 检查文件权限
2. 以管理员身份运行程序（Windows）
3. 使用 `sudo`（Linux/macOS）
4. 修改文件权限:

```bash
# Linux/macOS
chmod 644 document.pdf

# 或更改所有权
sudo chown $USER document.pdf
```

### 问题 2.3: 文件损坏或格式错误

**症状**:
```
PyPDF2.errors.PdfReadError: PDF starts with '...', but '%PDF-' expected
```

**解决方案**:
1. 验证 PDF 文件完整性
2. 尝试修复 PDF:
   ```python
   from pdf_extractor import PDFExtractor
   
   extractor = PDFExtractor()
   try:
       result = extractor.extract_text('corrupted.pdf')
       if not result.success:
           print(f"提取失败: {result.error}")
   except Exception as e:
       print(f"文件可能损坏: {e}")
   ```
3. 使用其他工具验证:
   ```bash
   # Linux/macOS
   file document.pdf
   
   # 检查 PDF 结构
   pdftk document.pdf dump_data
   ```

## 3. 文本提取问题

### 问题 3.1: 提取不到文本（空白结果）

**症状**: 提取结果为空或只有很少的文本。

**原因**:
1. PDF 是扫描版（图像）
2. 文本被编码为图像
3. 使用了特殊的字体编码

**解决方案**:
1. 启用 OCR 功能:
   ```python
   config = {
       'enable_ocr': True,
       'ocr_lang': 'chi_sim+eng'
   }
   extractor = PDFExtractor(config)
   ```
2. 尝试不同的提取库:
   ```python
   config = {
       'use_pdfplumber': True,  # 默认
       'use_pypdf2': True       # 同时尝试
   }
   ```
3. 检查 PDF 类型:
   ```python
   from pdf_extractor import PDFExtractor
   
   extractor = PDFExtractor()
   # 内部会检测 PDF 类型
   result = extractor.extract_text('document.pdf')
   print(f"提取方法: {result.extraction_method}")
   ```

### 问题 3.2: 文本顺序混乱

**症状**: 提取的文本顺序不正确，特别是多栏布局。

**解决方案**:
1. 使用 pdfplumber 的布局分析:
   ```python
   import pdfplumber
   
   with pdfplumber.open('document.pdf') as pdf:
       page = pdf.pages[0]
       # 尝试不同的提取策略
       text = page.extract_text(layout=True)  # 布局模式
       # 或
       text = page.extract_text(x_tolerance=3, y_tolerance=3)
   ```
2. 手动调整参数:
   ```python
   config = {
       'preserve_layout': True,
       'x_tolerance': 3,
       'y_tolerance': 3
   }
   ```

### 问题 3.3: 特殊字符丢失或乱码

**症状**: 数学符号、特殊字符显示不正确。

**解决方案**:
1. 检查字体嵌入:
   ```python
   import PyPDF2
   
   with open('document.pdf', 'rb') as f:
       pdf = PyPDF2.PdfReader(f)
       for page in pdf.pages:
           if '/Font' in page['/Resources']:
               print("页面包含字体")
   ```
2. 使用 OCR 重新识别特殊字符区域
3. 尝试不同的编码:

```python
# 在配置中指定备用编码
config = {
    'fallback_encodings': ['utf-8', 'gbk', 'gb2312', 'latin-1', 'cp1252']
}
```

## 4. OCR 识别问题

### 问题 4.1: OCR 识别率低

**症状**: OCR 识别结果包含大量错误。

**解决方案**:
1. 提高图像质量:
   ```python
   config = {
       'dpi': 400,  # 增加 DPI
       'grayscale': True  # 使用灰度图像
   }
   ```
2. 调整 OCR 参数:
   ```python
   config = {
       'ocr_lang': 'chi_sim+eng',
       'page_seg_mode': 6,  # 假设为统一文本块
       'preserve_interword_spaces': True
   }
   ```
3. 预处理图像:
   ```python
   from PIL import Image, ImageFilter, ImageEnhance
   
   def preprocess_image(image_path):
       img = Image.open(image_path)
       
       # 转换为灰度
       img = img.convert('L')
       
       # 增强对比度
       enhancer = ImageEnhance.Contrast(img)
       img = enhancer.enhance(2.0)
       
       # 锐化
       img = img.filter(ImageFilter.SHARPEN)
       
       return img
   ```

### 问题 4.2: OCR 速度慢

**症状**: OCR 处理时间过长。

**解决方案**:
1. 降低 DPI:
   ```python
   config = {'dpi': 200}  # 默认 300
   ```
2. 限制处理页面:
   ```python
   config = {'max_pages': 50}  # 只处理前50页
   ```
3. 使用并行处理:
   ```python
   config = {'max_workers': 4}  # 使用4个线程
   ```
4. 缓存中间结果

### 问题 4.3: 语言识别错误

**症状**: OCR 错误识别语言。

**解决方案**:
1. 明确指定语言:
   ```python
   config = {'ocr_lang': 'chi_sim'}  # 仅中文
   ```
2. 多语言组合:
   ```python
   config = {'ocr_lang': 'chi_sim+eng'}  # 中文+英文
   ```
3. 自动检测语言（高级）:
   ```python
   import langdetect
   
   def detect_language(text_sample):
       try:
           return langdetect.detect(text_sample)
       except:
           return 'en'
   ```

## 5. 性能问题

### 问题 5.1: 处理大文件时内存不足

**解决方案**:
1. 使用流式处理:
   ```python
   import PyPDF2
   
   def extract_text_streaming(pdf_path, batch_size=10):
       with open(pdf_path, 'rb') as f:
           pdf = PyPDF2.PdfReader(f)
           
           for start in range(0, len(pdf.pages), batch_size):
               end = min(start + batch_size, len(pdf.pages))
               batch_text = []
               
               for i in range(start, end):
                   text = pdf.pages[i].extract_text()
                   batch_text.append(text)
                   
               # 处理批次文本
               yield '\n'.join(batch_text)
   ```
2. 使用磁盘缓存:
   ```python
   import tempfile
   
   with tempfile.TemporaryDirectory() as temp_dir:
       # 在临时目录中处理文件
       pass
   ```

### 问题 5.2: 批量处理速度慢

**解决方案**:
1. 增加并行度:
   ```python
   config = {'max_workers': os.cpu_count()}
   ```
2. 使用进度条:
   ```python
   from tqdm import tqdm
   
   files = [...]  # 文件列表
   for file in tqdm(files, desc="处理PDF"):
       process_file(file)
   ```
3. 跳过已处理文件:
   ```python
   config = {'skip_existing': True}
   ```

## 6. 编码问题

### 问题 6.1: 中文显示乱码

**症状**: 中文文本显示为乱码字符。

**解决方案**:
1. 确保使用 UTF-8 编码:
   ```python
   with open('output.txt', 'w', encoding='utf-8') as f:
       f.write(text)
   ```
2. 设置系统编码:
   ```python
   import sys
   import io
   
   # Windows 控制台
   if sys.platform == 'win32':
       sys.stdout = io.TextIOWrapper(
           sys.stdout.buffer, 
           encoding='utf-8'
       )
   ```
3. 使用正确的字体:
   ```python
   # 在输出中指定字体（如生成HTML）
   html = f'''
   <html>
   <head>
       <meta charset="UTF-8">
       <style>
           body {{ font-family: "Microsoft YaHei", sans-serif; }}
       </style>
   </head>
   <body>{text}</body>
   </html>
   '''
   ```

### 问题 6.2: 文件路径包含非 ASCII 字符

**症状**: 无法打开包含中文等字符的文件。

**解决方案**:
1. 使用 `os.path` 处理路径:
   ```python
   import os
   
   # 安全打开文件
   def safe_open(path, mode='r', encoding='utf-8'):
       try:
           return open(path, mode, encoding=encoding)
       except:
           # 尝试其他编码
           return open(path.encode('utf-8'), mode)
   ```
2. 使用 `pathlib`:
   ```python
   from pathlib import Path
   
   path = Path('中文文件.pdf')
   if path.exists():
       content = path.read_text(encoding='utf-8')
   ```

## 调试技巧

### 1. 启用详细日志

```python
import logging

# 设置详细日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 在代码中添加调试信息
logger = logging.getLogger(__name__)
logger.debug(f"处理文件: {file_path}")
```

### 2. 创建测试用例

```python
def test_pdf_extraction():
    """测试 PDF 提取功能"""
    test_files = [
        ('text_only.pdf', '文本型PDF'),
        ('scanned.pdf', '扫描型PDF'),
        ('mixed.pdf', '混合型PDF')
    ]
    
    extractor = PDFExtractor({'enable_ocr': True})
    
    for file_path, description in test_files:
        print(f"\n测试: {description} ({file_path})")
        
        if os.path.exists(file_path):
            result = extractor.extract_text(file_path)
            print(f"  成功: {result.success}")
            print(f"  方法: {result.extraction_method}")
            print(f"  页数: {result.page_count}")
            print(f"  文本长度: {len(result.text)}")
        else:
            print(f"  文件不存在")
```

### 3. 使用交互式调试

```python
# 在代码中插入调试点
import pdb

def debug_extraction(file_path):
    result = extractor.extract_text(file_path)
    
    if not result.success:
        pdb.set_trace()  # 进入调试模式
        # 检查 result.error
        # 检查中间状态
    
    return result
```

### 4. 生成诊断报告

```python
def generate_diagnostic_report(pdf_path):
    """生成 PDF 诊断报告"""
    report = []
    
    # 文件信息
    report.append(f"文件: {pdf_path}")
    report.append(f"大小: {os.path.getsize(pdf_path)} 字节")
    report.append(f"修改时间: {os.path.getmtime(pdf_path)}")
    
    # PDF 信息
    try:
        import PyPDF2
        with open(pdf_path, 'rb') as f:
            pdf = PyPDF2.PdfReader(f)
            report.append(f"页数: {len(pdf.pages)}")
            report.append(f"加密: {pdf.is_encrypted}")
            report.append(f"元数据: {pdf.metadata}")
    except Exception as e:
        report.append(f"PDF 读取错误: {e}")
    
    # 提取测试
    extractor = PDFExtractor()
    result = extractor.extract_text(pdf_path)
    report.append(f"提取成功: {result.success}")
    report.append(f"提取方法: {result.extraction_method}")
    report.append(f"文本预览: {result.text[:200]}")
    
    return '\n'.join(report)
```

## 获取帮助

如果以上解决方案都无法解决问题：

1. **检查文档**: 查看 `installation.md` 和 `encoding_guide.md`
2. **搜索错误信息**: 在互联网上搜索具体的错误信息
3. **提供详细信息**:
   - 操作系统和版本
   - Python 版本
   - 安装的库版本
   - 完整的错误信息
   - 重现步骤
4. **简化测试**: 创建一个最小化的测试用例
5. **检查更新**: 确保所有库都是最新版本

```bash
# 检查版本
python --version
pip list | grep -E "(PyPDF2|pdfplumber|pytesseract|pillow|pdf2image)"

# 更新所有库
pip install --upgrade PyPDF2 pdfplumber pytesseract pillow pdf2image
```

通过系统性的故障排除，可以解决大多数 PDF 读取问题。如果问题仍然存在，请提供详细的错误信息和环境信息以便进一步诊断。