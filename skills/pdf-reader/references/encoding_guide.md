# 跨平台编码问题处理指南

## 概述

PDF Reader Skill 需要处理不同操作系统和 PDF 文件的编码问题。本指南提供解决跨平台编码问题的策略和最佳实践。

## 常见编码问题

### 1. 文件路径编码问题

**问题**: Windows、Linux 和 macOS 使用不同的文件系统编码。

**解决方案**:

```python
import os
import sys

def normalize_path(file_path: str) -> str:
    """规范化文件路径，处理跨平台问题"""
    # 转换为绝对路径
    abs_path = os.path.abspath(file_path)
    
    # 处理 Windows 路径分隔符
    if os.name == 'nt':  # Windows
        # 确保使用正确的分隔符
        abs_path = abs_path.replace('/', '\\')
    else:  # Unix/Linux/macOS
        abs_path = abs_path.replace('\\', '/')
    
    return abs_path

# 使用示例
path = "C:\\Users\\test\\文档.pdf"  # Windows 路径
normalized = normalize_path(path)
print(f"规范化路径: {normalized}")
```

### 2. 文本编码问题

**问题**: PDF 中的文本可能使用不同的编码（UTF-8, GBK, GB2312, Latin-1 等）。

**解决方案**:

```python
import codecs

def decode_text(text_bytes: bytes, fallback_encodings=None) -> str:
    """尝试多种编码解码文本"""
    if fallback_encodings is None:
        fallback_encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1', 'cp1252']
    
    for encoding in fallback_encodings:
        try:
            return text_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    
    # 如果所有编码都失败，使用 errors='replace'
    try:
        return text_bytes.decode('utf-8', errors='replace')
    except:
        return text_bytes.decode('latin-1', errors='replace')

# 使用示例
text_bytes = b'\xc4\xe3\xba\xc3'  # GBK 编码的 "你好"
decoded = decode_text(text_bytes)
print(f"解码文本: {decoded}")
```

## 操作系统特定问题

### Windows 编码问题

#### 1. 控制台编码

**问题**: Windows 控制台默认使用 GBK 编码，可能导致中文显示乱码。

**解决方案**:

```python
import sys
import io

def setup_windows_console():
    """设置 Windows 控制台编码"""
    if sys.platform == 'win32':
        # 设置标准输出编码
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, 
            encoding='utf-8',
            errors='replace'
        )
        sys.stderr = io.TextIOWrapper(
            sys.stderr.buffer,
            encoding='utf-8',
            errors='replace'
        )
        
        # 设置控制台代码页
        import ctypes
        ctypes.windll.kernel32.SetConsoleOutputCP(65001)  # UTF-8
        ctypes.windll.kernel32.SetConsoleCP(65001)  # UTF-8

# 在程序开始时调用
setup_windows_console()
```

#### 2. 文件名编码

**问题**: Windows 文件名可能包含非 ASCII 字符。

**解决方案**:

```python
import os

def safe_open_file(file_path: str, mode='r', encoding='utf-8'):
    """安全打开文件，处理编码问题"""
    try:
        if 'b' in mode:
            # 二进制模式
            return open(file_path, mode)
        else:
            # 文本模式，指定编码
            return open(file_path, mode, encoding=encoding, errors='replace')
    except UnicodeDecodeError:
        # 尝试其他编码
        for enc in ['gbk', 'gb2312', 'latin-1']:
            try:
                return open(file_path, mode, encoding=enc, errors='replace')
            except:
                continue
        raise

# 使用示例
try:
    with safe_open_file('中文文件.txt', 'r') as f:
        content = f.read()
except Exception as e:
    print(f"打开文件失败: {e}")
```

### Linux/macOS 编码问题

#### 1. 区域设置

**问题**: 系统区域设置可能影响文件编码。

**解决方案**:

```python
import locale

def get_system_locale():
    """获取系统区域设置"""
    try:
        return locale.getlocale()[0]
    except:
        return None

def setup_system_encoding():
    """设置系统编码"""
    # 设置默认编码为 UTF-8
    import sys
    if sys.version_info >= (3, 7):
        sys.stdout.reconfigure(encoding='utf-8')
    
    # 设置环境变量
    import os
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['LC_ALL'] = 'C.UTF-8'
    os.environ['LANG'] = 'C.UTF-8'

# 使用示例
setup_system_encoding()
locale_info = get_system_locale()
print(f"系统区域设置: {locale_info}")
```

## PDF 特定编码问题

### 1. PDF 元数据编码

**问题**: PDF 元数据可能使用不同的编码。

**解决方案**:

```python
def decode_pdf_metadata(metadata_dict):
    """解码 PDF 元数据"""
    decoded = {}
    
    for key, value in metadata_dict.items():
        if isinstance(value, bytes):
            # 尝试解码字节数据
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    decoded[key] = value.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                # 所有编码都失败
                decoded[key] = str(value)
        else:
            decoded[key] = value
    
    return decoded

# 使用示例
metadata = {
    '/Title': b'\xe6\x96\x87\xe6\xa1\xa3\xe6\xa0\x87\xe9\xa2\x98',  # UTF-8 编码
    '/Author': b'Author Name'
}

decoded = decode_pdf_metadata(metadata)
print(f"解码后的元数据: {decoded}")
```

### 2. PDF 文本内容编码

**问题**: PDF 中的文本可能使用非标准编码。

**解决方案**:

```python
import re

def clean_pdf_text(text: str) -> str:
    """清理 PDF 文本，处理编码问题"""
    if not text:
        return ""
    
    # 移除不可打印字符
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    
    # 替换常见的编码问题字符
    replacements = {
        'â€™': "'",      # Windows-1252 单引号
        'â€œ': '"',      # Windows-1252 左双引号
        'â€': '"',       # Windows-1252 右双引号
        'â€"': '-',      # Windows-1252 破折号
        'Ã©': 'é',       # Latin-1 重音字符
        'Ã¨': 'è',
        'Ãª': 'ê',
        'Ã': 'à',
    }
    
    for wrong, correct in replacements.items():
        text = text.replace(wrong, correct)
    
    # 标准化空白字符
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

# 使用示例
dirty_text = "This is â€™a testâ€" with encoding issuesÃ©"
clean_text = clean_pdf_text(dirty_text)
print(f"清理后的文本: {clean_text}")
```

## 跨平台最佳实践

### 1. 统一的编码策略

```python
class EncodingManager:
    """编码管理器"""
    
    def __init__(self):
        self.system_encoding = self._detect_system_encoding()
        self.fallback_encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1', 'cp1252']
    
    def _detect_system_encoding(self):
        """检测系统编码"""
        import sys
        import locale
        
        if sys.platform == 'win32':
            return 'gbk'  # Windows 默认
        else:
            try:
                return locale.getpreferredencoding()
            except:
                return 'utf-8'
    
    def decode(self, data, hint=None):
        """解码数据"""
        if isinstance(data, str):
            return data
        
        if not isinstance(data, bytes):
            data = str(data).encode('utf-8')
        
        # 如果有提示编码，先尝试
        if hint and hint in self.fallback_encodings:
            try:
                return data.decode(hint)
            except:
                pass
        
        # 尝试所有备选编码
        for encoding in self.fallback_encodings:
            try:
                return data.decode(encoding)
            except UnicodeDecodeError:
                continue
        
        # 最后手段
        try:
            return data.decode('utf-8', errors='replace')
        except:
            return data.decode('latin-1', errors='replace')
    
    def encode(self, text, target_encoding='utf-8'):
        """编码文本"""
        if isinstance(text, bytes):
            return text
        
        try:
            return text.encode(target_encoding)
        except UnicodeEncodeError:
            # 使用 UTF-8 作为后备
            return text.encode('utf-8', errors='replace')

# 使用示例
manager = EncodingManager()
text = "中文文本"
encoded = manager.encode(text)
decoded = manager.decode(encoded)
print(f"编码/解码测试: {decoded}")
```

### 2. 文件操作包装器

```python
import os
import sys

class SafeFileHandler:
    """安全的文件处理器"""
    
    def __init__(self, default_encoding='utf-8'):
        self.default_encoding = default_encoding
        self.fallback_encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
    
    def read_file(self, file_path, encoding=None):
        """读取文件，自动处理编码"""
        if encoding is None:
            encoding = self.default_encoding
        
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            # 尝试其他编码
            for enc in self.fallback_encodings:
                if enc == encoding:
                    continue
                try:
                    with open(file_path, 'r', encoding=enc) as f:
                        return f.read()
                except:
                    continue
        
        # 所有编码都失败，使用二进制模式
        with open(file_path, 'rb') as f:
            content = f.read()
            for enc in self.fallback_encodings:
                try:
                    return content.decode(enc)
                except:
                    continue
        
        # 最后手段
        return content.decode('latin-1', errors='replace')
    
    def write_file(self, file_path, content, encoding=None):
        """写入文件，确保目录存在"""
        if encoding is None:
            encoding = self.default_encoding
        
        # 确保目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # 写入文件
        with open(file_path, 'w', encoding=encoding, errors='replace') as f:
            f.write(content)
    
    def list_files(self, directory, pattern='*.pdf'):
        """列出文件，处理路径编码"""
        import glob
        
        files = []
        try:
            files = glob.glob(os.path.join(directory, pattern))
        except UnicodeEncodeError:
            # 处理非 ASCII 路径
            directory_bytes = directory.encode('utf-8')
            pattern_bytes = pattern.encode('utf-8')
            # 使用字节模式
            pass
        
        return files

# 使用示例
handler = SafeFileHandler()
content = handler.read_file('中文文件.txt')
handler.write_file('output.txt', content)
```

### 3. 日志记录编码

```python
import logging
import sys

def setup_logging():
    """设置日志记录，处理编码问题"""
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # 文件处理器（使用 UTF-8 编码）
    file_handler = logging.FileHandler(
        'pdf_reader.log',
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    
    # 配置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # 设置特定模块的日志级别
    logging.getLogger('pdfplumber').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)

# 使用示例
setup_logging()
logger = logging.getLogger(__name__)
logger.info("日志记录已设置，使用 UTF-8 编码")
```

## 测试编码处理

```python
def test_encoding_handling():
    """测试编码处理功能"""
    test_cases = [
        (b'Hello World', 'utf-8', 'Hello World'),
        (b'\xe4\xb8\xad\xe6\x96\x87', 'utf-8', '中文'),
        (b'\xd6\xd0\xce\xc4', 'gbk', '中文'),
        (b'\xa1\xa2\xa1\xa3', 'gb2312', '··'),
    ]
    
    manager = EncodingManager()
    
    for data, expected_encoding, expected_text in test_cases:
        decoded = manager.decode(data)
        print(f"数据: {data}")
        print(f"预期编码: {expected_encoding}")
        print(f"预期文本: {expected_text}")
        print(f"实际解码: {decoded}")
        print(f"匹配: {decoded == expected_text}")
        print("-" * 40)

if __name__ == "__main__":
    test_encoding_handling()
```

## 总结

处理跨平台编码问题的关键策略：

1. **始终使用 UTF-8 作为内部表示**
2. **提供编码回退机制**
3. **操作系统特定的适配**
4. **PDF 格式的特殊处理**
5. **统一的错误处理**

通过遵循这些最佳实践，PDF Reader Skill 可以在不同操作系统上稳定运行，正确处理各种编码的 PDF 文件。