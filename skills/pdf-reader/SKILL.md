---
name: pdf-reader
description: Comprehensive PDF text extraction and OCR tool with cross-platform encoding support. Use when Claude needs to read, extract, or analyze PDF documents. Supports both text-based PDFs and scanned/image-based PDFs with OCR capabilities. Handles Windows, Linux, and macOS encoding issues, Chinese and multilingual text, batch processing, and various output formats.
license: MIT
---

# PDF Reader Skill

A comprehensive tool for extracting text from PDF documents with support for both direct text extraction and OCR (Optical Character Recognition) for scanned documents. Designed to handle cross-platform encoding issues and multilingual content.

## When to Use This Skill

Use this skill when you need to:
- Extract text from PDF documents (both digital and scanned)
- Process multiple PDF files in batch mode
- Handle PDFs with Chinese or other non-Latin characters
- Work across different operating systems (Windows, Linux, macOS)
- Convert PDF content to text, JSON, or other formats
- Analyze PDF structure and metadata

## Quick Start

### Basic Text Extraction

```python
from pdf_extractor import PDFExtractor

# Create extractor with default settings
extractor = PDFExtractor()

# Extract text from a PDF file
result = extractor.extract_text('document.pdf')

if result.success:
    print(f"Extracted {len(result.text)} characters")
    print(f"Page count: {result.page_count}")
    print(f"Method used: {result.extraction_method}")
    
    # Save to file
    result.save_text('output.txt')
else:
    print(f"Extraction failed: {result.error}")
```

### With OCR Support

```python
config = {
    'enable_ocr': True,
    'ocr_lang': 'chi_sim+eng',  # Chinese + English
    'dpi': 300
}

extractor = PDFExtractor(config)
result = extractor.extract_text('scanned_document.pdf')
```

### Batch Processing

```python
from batch_processor import PDFBatchProcessor

processor = PDFBatchProcessor()
results = processor.process_directory('/path/to/pdf/folder')
```

## Core Features

### 1. Text Extraction Methods

The skill supports multiple extraction methods, automatically selecting the best one:

1. **pdfplumber** (Recommended): Better layout preservation and accuracy
2. **PyPDF2**: Reliable fallback for basic PDFs
3. **OCR with pytesseract**: For scanned/image-based PDFs

### 2. Cross-Platform Encoding Support

Handles encoding issues across different operating systems:

- **Windows**: GBK, GB2312, UTF-8 with console encoding fixes
- **Linux/macOS**: UTF-8, locale-aware encoding detection
- **Automatic fallback**: Tries multiple encodings for robustness

### 3. Multilingual Support

- **Chinese**: Simplified (chi_sim) and Traditional (chi_tra)
- **English**: Default OCR language
- **Other languages**: Japanese, Korean, French, German, Spanish
- **Mixed languages**: Support for multi-language documents

### 4. Output Formats

- **Plain text**: Clean, readable text output
- **JSON**: Structured data with metadata
- **HTML**: Formatted output with styling options
- **Custom formats**: Extensible output system

## Detailed Usage

### Configuration Options

Create a custom configuration dictionary:

```python
config = {
    # Extraction methods
    'use_pdfplumber': True,
    'use_pypdf2': True,
    'enable_ocr': False,  # Enable for scanned PDFs
    
    # OCR settings
    'ocr_lang': 'chi_sim+eng',
    'tesseract_cmd': None,  # Auto-detect
    'dpi': 300,
    
    # Encoding settings
    'output_encoding': 'utf-8',
    'fallback_encodings': ['utf-8', 'gbk', 'gb2312', 'latin-1'],
    
    # Performance
    'max_pages': None,  # Process all pages
    'max_workers': 4,   # For batch processing
    
    # Output
    'preserve_layout': False,
    'extract_images': False
}
```

### Handling Different PDF Types

#### Text-based PDFs

```python
# For digital/text PDFs (recommended settings)
config = {
    'use_pdfplumber': True,
    'use_pypdf2': False,  # pdfplumber is usually better
    'enable_ocr': False
}
```

#### Scanned PDFs

```python
# For scanned/image PDFs
config = {
    'use_pdfplumber': False,  # Won't find text
    'use_pypdf2': False,
    'enable_ocr': True,
    'ocr_lang': 'chi_sim+eng',
    'dpi': 400  # Higher DPI for better accuracy
}
```

#### Mixed PDFs

```python
# For PDFs with both text and images
config = {
    'use_pdfplumber': True,
    'use_pypdf2': True,
    'enable_ocr': True  # Fallback to OCR if needed
}
```

### Advanced Features

#### Custom OCR Processing

```python
from ocr_pdf import PDFOCRProcessor

ocr_config = {
    'ocr_lang': 'chi_sim+eng',
    'dpi': 300,
    'output_format': 'hocr',  # Get structured OCR output
    'preprocess_images': True
}

processor = PDFOCRProcessor(ocr_config)
result = processor.ocr_pdf('document.pdf')
```

#### Metadata Extraction

```python
result = extractor.extract_text('document.pdf')

# Access metadata
metadata = result.metadata
print(f"Title: {metadata.get('/Title', 'N/A')}")
print(f"Author: {metadata.get('/Author', 'N/A')}")
print(f"Creation date: {metadata.get('/CreationDate', 'N/A')}")
```

#### Batch Processing with Progress

```python
from batch_processor import PDFBatchProcessor
from tqdm import tqdm

config = {
    'max_workers': 4,
    'skip_existing': True,
    'generate_report': True
}

processor = PDFBatchProcessor(config)

# Process with progress bar
import os
pdf_files = [f for f in os.listdir('.') if f.endswith('.pdf')]

for file in tqdm(pdf_files, desc="Processing PDFs"):
    result = extractor.extract_text(file)
    # Handle result...
```

## Operating System Specific Notes

### Windows

1. **Tesseract Installation**: Download from https://github.com/UB-Mannheim/tesseract/wiki
2. **Poppler for pdf2image**: Download from https://github.com/oschwartz10612/poppler-windows/releases
3. **Console Encoding**: The skill automatically fixes Windows console encoding issues

### macOS

```bash
# Install dependencies
brew install tesseract poppler
brew install tesseract-lang  # For language packs
```

### Linux (Ubuntu/Debian)

```bash
# Install dependencies
sudo apt-get update
sudo apt-get install tesseract-ocr tesseract-ocr-chi-sim poppler-utils
```

## Troubleshooting Common Issues

### 1. "TesseractNotFoundError"

**Solution**: Install Tesseract OCR and ensure it's in PATH.

### 2. Chinese Text Shows as Garbled Characters

**Solution**: Ensure UTF-8 encoding is used for both input and output.

### 3. No Text Extracted from Scanned PDF

**Solution**: Enable OCR with `enable_ocr: True` in configuration.

### 4. Memory Issues with Large PDFs

**Solution**: Process in chunks using `max_pages` parameter.

## Reference Files

For detailed information, refer to:

- [Installation Guide](references/installation.md): Complete installation instructions
- [Encoding Guide](references/encoding_guide.md): Cross-platform encoding solutions
- [Troubleshooting Guide](references/troubleshooting.md): Common issues and solutions

## Examples

### Example 1: Extract and Summarize

```python
from pdf_extractor import PDFExtractor

extractor = PDFExtractor()
result = extractor.extract_text('report.pdf')

if result.success:
    text = result.text
    # Simple summary - first 500 characters
    summary = text[:500] + '...' if len(text) > 500 else text
    print(f"Summary: {summary}")
```

### Example 2: Process All PDFs in Folder

```python
import os
from pdf_extractor import PDFExtractor

extractor = PDFExtractor({'enable_ocr': True})
pdf_folder = './documents'

for filename in os.listdir(pdf_folder):
    if filename.endswith('.pdf'):
        filepath = os.path.join(pdf_folder, filename)
        result = extractor.extract_text(filepath)
        
        print(f"Processed: {filename}")
        print(f"  Pages: {result.page_count}")
        print(f"  Success: {result.success}")
        print(f"  Method: {result.extraction_method}")
        print()
```

### Example 3: Generate Structured Output

```python
from pdf_extractor import PDFExtractor
import json

extractor = PDFExtractor()
result = extractor.extract_text('document.pdf')

# Save as JSON with metadata
output_data = {
    'filename': 'document.pdf',
    'extraction_result': result.to_dict(),
    'summary': {
        'page_count': result.page_count,
        'text_length': len(result.text),
        'extraction_method': result.extraction_method
    }
}

with open('output.json', 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)
```

## Performance Tips

1. **For text PDFs**: Use `use_pdfplumber: True`, `use_pypdf2: False`
2. **For scanned PDFs**: Increase `dpi` for accuracy, decrease for speed
3. **Batch processing**: Adjust `max_workers` based on CPU cores
4. **Memory management**: Set `max_pages` for large documents

## License

This skill is licensed under the MIT License. See the LICENSE file for details.

## Support

For issues and questions:
1. Check the troubleshooting guide
2. Verify installation dependencies
3. Provide error messages and system information