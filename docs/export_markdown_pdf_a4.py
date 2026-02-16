"""Export a professional A4 PDF from a Markdown file.

Designed for sharing with teams:
- Clean typography + tables
- Code blocks styled for print
- Mermaid diagrams rendered (if present)
- Page numbers + document title in header/footer

Usage (PowerShell):
  C:/Users/varun/Desktop/InerviewAst/.venv/Scripts/python.exe export_markdown_pdf_a4.py --input TECHNICAL_DOCUMENTATION.md

Notes:
- Requires: playwright
- After installing playwright, run once:
    C:/Users/varun/Desktop/InerviewAst/.venv/Scripts/python.exe -m playwright install chromium
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as _dt
import html
import re
from pathlib import Path

import markdown


def _preprocess_markdown(md_text: str) -> str:
    # Convert ```mermaid blocks into <div class="mermaid"> blocks.
    # Escape so tags like <br/> remain text for Mermaid.
    def _mermaid_block_repl(m: re.Match) -> str:
        raw = m.group(1).strip("\n")
        escaped = html.escape(raw, quote=False)
        return f"\n<div class=\"mermaid\">\n{escaped}\n</div>\n"

    md_text = re.sub(
        r"```mermaid\s*\n(.*?)\n```",
        _mermaid_block_repl,
        md_text,
        flags=re.DOTALL,
    )

    # Normalize manual page breaks.
    md_text = md_text.replace(
        '<div style="page-break-after: always;"></div>',
        '<div class="page-break"></div>',
    )
    return md_text


def render_html(*, md_path: Path, title: str | None = None) -> str:
    md_text = md_path.read_text(encoding="utf-8")
    md_text = _preprocess_markdown(md_text)

    # Render markdown -> HTML
    body_html = markdown.markdown(
        md_text,
        extensions=[
            "tables",
            "fenced_code",
            "toc",
            "sane_lists",
        ],
    )

    # Keep headings and their immediately-following Mermaid diagram together.
    # This prevents a common print-PDF artifact where a heading is placed on a
    # page and the diagram is pushed to the next page, leaving a mostly-blank page.
    body_html = re.sub(
      r'(?s)(<h([23])\b[^>]*>.*?</h\2>\s*)(<div class="mermaid">.*?</div>)',
      r'<div class="keep-with-diagram">\1\3</div>',
      body_html,
    )

    # Special-case: Security Model diagram tends to be tall and can overflow an A4 page,
    # causing Chromium to split the wrapper after the heading (heading-only page).
    body_html = re.sub(
      r'(<div class="keep-with-diagram">)(\s*<h2\b[^>]*id="32-security-model-high-level"[^>]*>)',
      r'<div class="keep-with-diagram diagram-security">\2',
      body_html,
    )

    # If the markdown ends with manual page breaks, they can create a blank trailing page.
    body_html = re.sub(r'(?s)(<div class="page-break"></div>\s*)+\Z', "", body_html)

    resolved_title = (title or md_path.stem).strip()

    css = """
    <style>
      @page { size: A4; margin: 18mm 16mm; }

      body {
        font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
        font-size: 10.5pt;
        line-height: 1.45;
        color: #111827;
        margin: 0;
        padding: 0;
      }

      h1 {
        font-size: 22pt;
        font-weight: 700;
        color: #0f172a;
        margin: 0 0 10pt 0;
        border-bottom: 3px solid #2563eb;
        padding-bottom: 6pt;
        page-break-after: avoid;
        break-after: avoid-page;
      }

      h2 {
        font-size: 15pt;
        font-weight: 600;
        color: #2563eb;
        margin: 12pt 0 6pt 0;
        page-break-after: avoid;
        break-after: avoid-page;
      }

      h3 {
        font-size: 12.5pt;
        font-weight: 600;
        color: #1e40af;
        margin: 10pt 0 5pt 0;
        page-break-after: avoid;
        break-after: avoid-page;
      }

      p { margin: 0 0 6pt 0; }

      code {
        background: #f3f4f6;
        padding: 2pt 6pt;
        border-radius: 3pt;
        font-family: Consolas, 'Courier New', monospace;
        font-size: 10pt;
        color: #1e40af;
      }

      pre {
        background: #0f172a;
        color: #e2e8f0;
        padding: 10pt;
        border-radius: 6pt;
        overflow-x: auto;
        margin: 10pt 0;
        font-family: Consolas, 'Courier New', 'Lucida Console', monospace;
        font-size: 9.5pt;
        line-height: 1.4;
        page-break-inside: auto;
        break-inside: auto;
        white-space: pre;
      }

      pre code {
        background: transparent;
        color: #e2e8f0;
        padding: 0;
        font-size: 9.5pt;
        display: block;
        white-space: pre;
      }

      table {
        width: 100%;
        border-collapse: collapse;
        margin: 10pt 0;
        font-size: 9.5pt;
        page-break-inside: auto;
        break-inside: auto;
      }

      thead { display: table-header-group; }
      tfoot { display: table-footer-group; }

      tr {
        page-break-inside: avoid;
        break-inside: avoid;
      }

      th {
        background: #2563eb;
        color: white;
        padding: 8pt 10pt;
        text-align: left;
        font-weight: 600;
        border: 1px solid #1e40af;
      }

      td {
        padding: 8pt 10pt;
        border: 1px solid #d1d5db;
      }

      tr:nth-child(even) { background: #f9fafb; }

      ul, ol { margin: 6pt 0; padding-left: 25pt; }
      li { margin: 3pt 0; }

      blockquote {
        border-left: 4pt solid #2563eb;
        margin: 14pt 0;
        color: #4b5563;
        background: #f8fafc;
        padding: 12pt 16pt;
      }

      .page-break { break-after: page; page-break-after: always; height: 0; }

      .keep-with-diagram {
        page-break-inside: avoid;
        break-inside: avoid;
      }

      /* Force-fit tall diagrams (especially the Security Model) onto a single page. */
      .diagram-security .mermaid svg {
        max-height: 180mm !important;
      }

      /* Mermaid blocks get SVG inserted by Mermaid */
      .mermaid {
        margin: 10pt 0;
        display: flex;
        justify-content: center;
      }
      .mermaid svg {
        width: auto !important;
        max-width: 100% !important;
        height: auto !important;
        display: block;
      }

      /* Cover page */
      .cover {
        page-break-after: always;
        padding-top: 14mm;
      }
      .cover .title {
        font-size: 28pt;
        font-weight: 800;
        color: #0f172a;
        margin: 0 0 10pt 0;
      }
      .cover .subtitle {
        font-size: 12pt;
        color: #374151;
        margin: 0 0 18pt 0;
      }
      .cover .meta {
        font-size: 10.5pt;
        color: #6b7280;
      }
    </style>
    """

    # Mermaid (render diagrams if any)
    mermaid = """
    <script type="module">
      import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
      mermaid.initialize({
        startOnLoad: true,
        theme: 'base',
        securityLevel: 'loose',
        themeVariables: {
          fontFamily: "Segoe UI, Helvetica Neue, Arial, sans-serif",
          // Keep diagrams compact for print/PDF.
          fontSize: "12px",
          primaryColor: "#EEF2FF",
          primaryTextColor: "#0F172A",
          primaryBorderColor: "#6366F1",
          lineColor: "#475569",
          secondaryColor: "#E0F2FE",
          tertiaryColor: "#F8FAFC"
        },
        flowchart: {
          // Prevent Mermaid from stretching diagrams to container width.
          useMaxWidth: false,
          nodeSpacing: 18,
          rankSpacing: 18,
          padding: 8,
          curve: 'linear'
        },
        sequence: {
          useMaxWidth: false,
          diagramMarginX: 10,
          diagramMarginY: 10,
          actorMargin: 20,
          boxMargin: 6,
          boxTextMargin: 6,
          noteMargin: 6,
          messageMargin: 18
        }
      });
    </script>
    """

    today = _dt.date.today().isoformat()

    # Basic cover page
    cover_html = f"""
    <div class="cover">
      <div class="title">{html.escape(resolved_title)}</div>
      <div class="subtitle">Exported from {html.escape(md_path.name)}</div>
      <div class="meta">Date: {today}</div>
    </div>
    """

    full = f"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1"/>
    <title>{html.escape(resolved_title)}</title>
    {css}
  </head>
  <body>
    {cover_html}
    {body_html}
    {mermaid}
  </body>
</html>
"""

    return full


async def export_pdf(*, html_path: Path, pdf_path: Path, doc_title: str) -> None:
    from playwright.async_api import async_playwright

    file_url = html_path.absolute().as_uri()

    header_template = f"""
    <div style="font-size:8px;width:100%;padding:0 12mm;color:#6b7280;display:flex;justify-content:space-between;">
      <div>{html.escape(doc_title)}</div>
      <div>{_dt.date.today().isoformat()}</div>
    </div>
    """

    footer_template = """
    <div style="font-size:8px;width:100%;padding:0 12mm;color:#6b7280;display:flex;justify-content:flex-end;">
      <div>Page <span class=\"pageNumber\"></span> / <span class=\"totalPages\"></span></div>
    </div>
    """

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1240, "height": 1754})

        await page.goto(file_url, wait_until="networkidle")

        # Wait for Mermaid to render if present.
        await page.wait_for_function(
            """
            () => {
              const blocks = Array.from(document.querySelectorAll('.mermaid'));
              if (blocks.length === 0) return true;
              return blocks.every(b => b.querySelector('svg'));
            }
            """,
            timeout=60_000,
        )

        await page.pdf(
            path=str(pdf_path),
            format="A4",
            print_background=True,
            display_header_footer=True,
            header_template=header_template,
            footer_template=footer_template,
            margin={"top": "18mm", "right": "14mm", "bottom": "18mm", "left": "14mm"},
        )

        await browser.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a styled A4 PDF from Markdown.")
    parser.add_argument("--input", default="TECHNICAL_DOCUMENTATION.md", help="Input Markdown file")
    parser.add_argument("--output", default="", help="Output PDF path (default: <stem>_A4.pdf)")
    parser.add_argument("--title", default="", help="Title to show in header/cover")
    args = parser.parse_args()

    md_path = Path(args.input)
    if not md_path.exists():
        raise SystemExit(f"Input markdown not found: {md_path}")

    doc_title = (args.title or md_path.stem).strip()

    out_pdf = Path(args.output) if args.output else Path(f"{md_path.stem}_A4.pdf")
    out_html = Path(f"{md_path.stem}_A4.html")

    out_html.write_text(render_html(md_path=md_path, title=doc_title), encoding="utf-8")

    asyncio.run(export_pdf(html_path=out_html, pdf_path=out_pdf, doc_title=doc_title))

    size_mb = out_pdf.stat().st_size / 1024 / 1024
    print(f"✅ PDF generated: {out_pdf.absolute()}")
    print(f"📄 Size: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
