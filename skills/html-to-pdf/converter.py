#!/usr/bin/env python3
"""HTML to PDF converter using Playwright."""

import argparse
import asyncio
import sys
from pathlib import Path


async def convert_async(input_path: str, output_path: str | None = None) -> int:
    """Convert HTML file to PDF using Playwright."""
    from playwright.async_api import async_playwright

    html_file = Path(input_path)

    if not html_file.exists():
        print(f"Error: File not found: {input_path}", file=sys.stderr)
        return 1

    if not html_file.suffix.lower() == '.html':
        print(f"Error: Input file must be .html: {input_path}", file=sys.stderr)
        return 1

    if output_path:
        pdf_file = Path(output_path)
    else:
        pdf_file = html_file.with_suffix('.pdf')

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            await page.goto(f"file://{html_file.absolute()}")
            await page.pdf(
                path=str(pdf_file),
                format='A4',
                margin={'top': '8mm', 'bottom': '8mm', 'left': '10mm', 'right': '10mm'},
                print_background=True,
                scale=0.98
            )
            await browser.close()
        print(f"PDF saved to: {pdf_file}")
        return 0
    except Exception as e:
        print(f"Error converting HTML to PDF: {e}", file=sys.stderr)
        return 1


def convert(input_path: str, output_path: str | None = None) -> int:
    """Convert HTML file to PDF."""
    return asyncio.run(convert_async(input_path, output_path))


def main() -> int:
    parser = argparse.ArgumentParser(description='Convert HTML to PDF using Playwright')
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    convert_parser = subparsers.add_parser('convert', help='Convert HTML to PDF')
    convert_parser.add_argument('file_path', help='Path to the HTML file')
    convert_parser.add_argument('--output', '-o', help='Output path for PDF file')

    args = parser.parse_args()

    if args.command == 'convert':
        return convert(args.file_path, args.output)

    parser.print_help()
    return 1


if __name__ == '__main__':
    sys.exit(main())
