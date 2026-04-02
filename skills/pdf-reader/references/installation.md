# PDF Reader Skill 安装指南

## 概述

PDF Reader Skill 需要以下依赖：
1. **核心 Python 库**: PyPDF2, pdfplumber
2. **OCR 相关库**: pytesseract, Pillow, pdf2image
3. **系统依赖**: Tesseract OCR 引擎

## 快速安装

### 1. 基础安装（仅文本提取）

```bash
pip install PyPDF2 pdfplumber
```

### 2. 完整安装（包含 OCR 功能）

```bash
# 安装 Python 库
pip install PyPDF2 pdfplumber pytesseract pillow pdf2image

# 安装系统依赖
```

## 系统依赖安装

### Windows

1. **Tesseract OCR**:
   - 下载地址: https://github.com/UB-Mannheim/tesseract/wiki
   - 推荐版本: tesseract-ocr-w64-setup-5.5.0.20241119.exe
   - 安装时勾选 "中文简体" 和 "英文" 语言包

2. **poppler** (用于 pdf2image):
   - 下载地址: https://github.com/oschwartz10612/poppler-windows/releases
   - 解压到 `C:\poppler` 或添加 `bin` 目录到 PATH

### macOS

```bash
# 使用 Homebrew 安装
brew install tesseract poppler

# 安装中文语言包
brew install tesseract-lang
```

### Linux (Ubuntu/Debian)

```bash
# 安装系统依赖
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-chi-sim poppler-utils

# 安装 Python 库
pip install PyPDF2 pdfplumber pytesseract pillow pdf2image
```

### Linux (其他发行版)

```bash
# 安装 Tesseract OCR
# Arch Linux
sudo pacman -S tesseract tesseract-data-chi-sim poppler

# Fedora
sudo dnf install tesseract tesseract-langpack-chi_sim poppler-utils
```

## 配置 Tesseract OCR 路径

在某些系统上，可能需要手动配置 Tesseract 路径：

### Windows 配置

```python
import pytesseract

# 设置 Tesseract 路径
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 或者在配置文件中设置
config = {
    'tesseract_cmd': r'C:\Program Files\Tesseract-OCR\tesseract.exe',
    'ocr_lang': 'chi_sim+eng'
}
```

### Linux/macOS 配置

通常 Tesseract 会自动找到，如果需要手动配置：

```python
import pytesseract

# macOS
pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'

# Linux
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
```

## 验证安装

### 1. 验证 Python 库

```python
import PyPDF2
import pdfplumber
import pytesseract
from PIL import Image
import pdf2image

print("✓ 所有 Python 库已正确安装")
```

### 2. 验证 Tesseract OCR

```bash
# 命令行验证
tesseract --version

# 列出可用语言
tesseract --list-langs
```

### 3. 验证 PDF 处理

创建测试脚本 `test_installation.py`:

```python
#!/usr/bin/env python3
import sys
import os

def test_dependencies():
    print("测试 PDF Reader Skill 依赖...")
    
    # 测试 PyPDF2
    try:
        import PyPDF2
        print(f"✓ PyPDF2: {PyPDF2.__version__}")
    except ImportError:
        print("✗ PyPDF2 未安装")
    
    # 测试 pdfplumber
    try:
        import pdfplumber
        print(f"✓ pdfplumber: {pdfplumber.__version__}")
    except ImportError:
        print("✗ pdfplumber 未安装")
    
    # 测试 OCR 依赖
    try:
        import pytesseract
        from PIL import Image
        print(f"✓ pytesseract: {pytesseract.get_tesseract_version()}")
        print(f"✓ PIL/Pillow: {Image.__version__}")
    except ImportError:
        print("✗ OCR 依赖未完全安装")
    
    # 测试 pdf2image
    try:
        import pdf2image
        print("✓ pdf2image: 可用")
    except ImportError:
        print("✗ pdf2image 未安装")
    
    print("\n安装状态:")
    print("-" * 40)
    print("文本提取功能: 需要 PyPDF2 或 pdfplumber")
    print("OCR 功能: 需要 pytesseract, Pillow, pdf2image")
    print("-" * 40)

if __name__ == "__main__":
    test_dependencies()
```

运行测试：
```bash
python test_installation.py
```

## 常见问题

### 1. "TesseractNotFoundError: tesseract is not installed or it's not in your PATH"

**解决方案**:
- Windows: 确保 Tesseract 安装目录在 PATH 中
- Linux/macOS: 使用包管理器安装 tesseract

### 2. "pdf2image 无法找到 poppler"

**解决方案**:
- Windows: 下载 poppler 并设置环境变量
- Linux: 安装 poppler-utils
- macOS: `brew install poppler`

### 3. 中文识别效果差

**解决方案**:
1. 确保安装了中文语言包
2. 调整 OCR 参数:
   ```python
   config = {
       'ocr_lang': 'chi_sim+eng',
       'dpi': 300,
       'preserve_interword_spaces': True
   }
   ```

### 4. 内存不足处理大 PDF

**解决方案**:
1. 使用分页处理
2. 降低图像 DPI
3. 增加系统内存

## 性能优化

### 1. 并行处理

```python
config = {
    'max_workers': 4,  # 根据 CPU 核心数调整
    'timeout': 300     # 超时时间（秒）
}
```

### 2. 图像处理优化

```python
config = {
    'dpi': 200,        # 降低 DPI 提高速度
    'grayscale': True, # 使用灰度图像
    'size': (800, None) # 限制图像大小
}
```

### 3. 缓存管理

```python
import tempfile

# 使用临时目录
with tempfile.TemporaryDirectory() as temp_dir:
    # 处理 PDF
    pass
```

## 升级

### 升级 Python 库

```bash
pip install --upgrade PyPDF2 pdfplumber pytesseract pillow pdf2image
```

### 升级 Tesseract OCR

- Windows: 重新下载安装包
- macOS: `brew upgrade tesseract`
- Linux: `sudo apt-get upgrade tesseract-ocr`

## 卸载

### 卸载 Python 库

```bash
pip uninstall PyPDF2 pdfplumber pytesseract pillow pdf2image
```

### 卸载系统依赖

- Windows: 通过控制面板卸载
- macOS: `brew uninstall tesseract poppler`
- Linux: `sudo apt-get remove tesseract-ocr poppler-utils`

## 支持的语言

PDF Reader Skill 支持以下 OCR 语言：

| 语言代码 | 语言名称 | 备注 |
|---------|---------|------|
| `eng` | 英语 | 默认包含 |
| `chi_sim` | 简体中文 | 需要单独安装 |
| `chi_tra` | 繁体中文 | 需要单独安装 |
| `jpn` | 日语 | 需要单独安装 |
| `kor` | 韩语 | 需要单独安装 |
| `fra` | 法语 | 需要单独安装 |
| `deu` | 德语 | 需要单独安装 |
| `spa` | 西班牙语 | 需要单独安装 |

使用多个语言：`chi_sim+eng`（中文+英文）