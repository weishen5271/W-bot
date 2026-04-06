---
name: html-to-pdf
description: Convert HTML files to PDF documents using WeasyPrint
homepage: "https://doc.courtbouillon.org/weasyprint/latest/"
metadata:
  {
    "openclaw":
      {
        "emoji": "📄",
        "requires": { "pip": ["weasyprint"] },
        "install":
          [
            {
              "id": "weasyprint",
              "kind": "pip",
              "package": "weasyprint",
              "label": "Install WeasyPrint",
            },
          ],
        "version": "1.0.0",
      },
  }
---

# HTML to PDF Skill

Convert HTML files to PDF documents using WeasyPrint.

## Tool API

### convert
Converts an HTML file to PDF format.

**Parameters:**
- `file_path` (string, required): Path to the HTML file to convert.
- `--output` (string, optional): Output path for the PDF file. Defaults to replacing the .html extension with .pdf.

**Usage:**
```bash
python3 skills/html-to-pdf/converter.py convert docs/resume.html
python3 skills/html-to-pdf/converter.py convert docs/resume.html --output docs/resume.pdf
```

**Output:** PDF file saved to the specified output path.

## Example

```bash
# Convert resume.html to resume.pdf
python3 skills/html-to-pdf/converter.py convert docs/resume.html

# Specify custom output path
python3 skills/html-to-pdf/converter.py convert docs/resume.html --output /path/to/output.pdf
```
