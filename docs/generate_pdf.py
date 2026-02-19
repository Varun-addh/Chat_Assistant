"""Generate professional HTML (A4-ready) from Stratax AI Technical Documentation.

This file produces `Stratax_AI_Technical_Documentation.html` with:
- consistent typography + table styling
- Mermaid diagrams rendered (via Mermaid JS)

Note: PDF export is handled by a separate script that prints this HTML via headless Chromium.
"""

from __future__ import annotations

import html
import re
import webbrowser
from pathlib import Path

import markdown


def _remove_h2_section(md_text: str, title: str) -> str:
    """Remove a section that starts with an H2 heading (## {title}) up to the next H1/H2."""
    pattern = rf"(?ms)^##\s+{re.escape(title)}\s*\n.*?(?=^##\s+|^#\s+|\Z)"
    return re.sub(pattern, "", md_text)


def _filter_markdown_for_audience(md_text: str, audience: str) -> str:
    audience = (audience or "engineering").strip().lower()
    if audience not in {"client", "engineering"}:
        raise ValueError("audience must be 'client' or 'engineering'")

    # Remove the decorative end-of-document block (causes awkward last page / client doesn't want it)
    md_text = re.sub(
        r"(?ms)^\*End of Technical Documentation\*\s*\n.*\Z",
        "",
        md_text,
    )

    if audience == "engineering":
        return md_text

    # Client version: keep architecture-centric story; drop dev-heavy sections and all appendices.
    md_text = _remove_h2_section(md_text, "Key Files by Complexity")
    md_text = _remove_h2_section(md_text, "Common Developer Tasks")
    md_text = _remove_h2_section(md_text, "Environment Configuration Priority")
    md_text = _remove_h2_section(md_text, "Debugging Tips")

    # Drop everything from "# Appendices" onwards.
    md_text = re.sub(r"(?ms)^#\s+Appendices\s*\n.*\Z", "", md_text)
    return md_text

def generate_html(audience: str = "engineering"):
    """Convert markdown to styled HTML"""
    
    # Read markdown
    md_path = Path("Stratax_AI_Technical_Documentation.md")
    with open(md_path, 'r', encoding='utf-8') as f:
        md_content = f.read()

    md_content = _filter_markdown_for_audience(md_content, audience)
    
    # --- Mermaid handling ---
    # Python-Markdown doesn't natively render Mermaid; we convert ```mermaid blocks into
    # <div class="mermaid"> blocks and HTML-escape them so tags like <br/> remain TEXT.
    def _mermaid_block_repl(m: re.Match) -> str:
        raw = m.group(1).strip("\n")
        escaped = html.escape(raw, quote=False)
        return f"\n<div class=\"mermaid\">\n{escaped}\n</div>\n"

    md_preprocessed = re.sub(
        r"```mermaid\s*\n(.*?)\n```",
        _mermaid_block_repl,
        md_content,
        flags=re.DOTALL,
    )

    # Normalize manual page breaks so we can style them consistently.
    md_preprocessed = md_preprocessed.replace(
        '<div style="page-break-after: always;"></div>',
        '<div class="page-break"></div>',
    )

    # Convert to HTML (avoid codehilite because it wraps code with spans and breaks Mermaid extraction)
    html_content = markdown.markdown(
        md_preprocessed,
        extensions=[
            "tables",
            "fenced_code",
            "toc",
            "sane_lists",
        ],
    )

    # If the markdown ends with a manual page break, it creates a blank trailing page in the PDF.
    html_content = re.sub(r"(?s)(<div class=\"page-break\"></div>\s*)+\Z", "", html_content)
    
    # Professional CSS styling
    css = '''
    <style>
        @page {
            size: A4;
            margin: 18mm 16mm;
        }

        /* Print-first layout */
        @media print {
            body {
                padding: 0;
            }
        }
        
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
            margin-top: 0;
            margin-bottom: 10pt;
            border-bottom: 3px solid #2563eb;
            padding-bottom: 6pt;
            page-break-after: avoid;
            break-after: avoid-page;
        }
        
        h2 {
            font-size: 15pt;
            font-weight: 600;
            color: #2563eb;
            margin-top: 12pt;
            margin-bottom: 6pt;
            page-break-after: avoid;
            break-after: avoid-page;
        }
        
        h3 {
            font-size: 12.5pt;
            font-weight: 600;
            color: #1e40af;
            margin-top: 10pt;
            margin-bottom: 5pt;
            page-break-after: avoid;
            break-after: avoid-page;
        }
        
        h4 {
            font-size: 11pt;
            font-weight: 600;
            color: #374151;
            margin-top: 8pt;
            margin-bottom: 4pt;
        }
        
        p {
            margin: 0 0 6pt 0;
            text-align: left;
        }
        
        code {
            background: #f3f4f6;
            padding: 2pt 6pt;
            border-radius: 3pt;
            font-family: 'Consolas', 'Courier New', monospace;
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
            font-family: 'Consolas', 'Courier New', 'Lucida Console', monospace;
            font-size: 9.5pt;
            line-height: 1.4;
            page-break-inside: avoid;
            white-space: pre;
            word-wrap: normal;
            letter-spacing: 0;
        }
        
        pre code {
            background: transparent;
            color: #e2e8f0;
            padding: 0;
            font-family: 'Consolas', 'Courier New', 'Lucida Console', monospace;
            font-size: 9.5pt;
            white-space: pre;
            display: block;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 10pt 0;
            font-size: 9.5pt;
            page-break-inside: avoid;
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
        
        tr:nth-child(even) {
            background: #f9fafb;
        }
        
        ul, ol {
            margin: 6pt 0;
            padding-left: 25pt;
        }
        
        li {
            margin: 3pt 0;
        }
        
        blockquote {
            border-left: 4pt solid #2563eb;
            padding-left: 16pt;
            margin: 14pt 0;
            color: #4b5563;
            font-style: italic;
            background: #f8fafc;
            padding: 12pt 16pt;
        }
        
        hr {
            border: none;
            border-top: 2px solid #e5e7eb;
            margin: 14pt 0;
        }

        .page-break {
            break-after: page;
            page-break-after: always;
            height: 0;
            margin: 0;
            padding: 0;
        }
        
        a {
            color: #2563eb;
            text-decoration: none;
        }
        
        strong {
            font-weight: 700;
            color: #1f2937;
        }
        
        em {
            font-style: italic;
            color: #4b5563;
        }
        
        .header {
            text-align: right;
            font-size: 9pt;
            color: #666;
            margin-bottom: 10pt;
            border-bottom: 1px solid #e5e7eb;
            padding-bottom: 5pt;
        }
        
        /* Footer removed intentionally (user requested no bottom page description) */
    </style>
    '''
    
    # Wrap HTML with proper document structure and Mermaid support
    full_html = f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Stratax AI - Technical Documentation</title>
        <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
        <script>
            mermaid.initialize({{ 
                startOnLoad: true,
                theme: 'default',
                flowchart: {{
                    useMaxWidth: true,
                    htmlLabels: true,
                    curve: 'basis'
                }}
            }});
        </script>
        {css}
        <style>
            .mermaid {{
                text-align: center;
                margin: 8pt 0;
                /* Avoid isolating a whole page due to an over-strict keep-together rule */
                page-break-inside: auto;
                break-inside: auto;
            }}
            .mermaid svg {{
                max-width: 100%;
                height: auto;
                /* Keep large diagrams from blowing up spacing */
                max-height: 240mm;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            Stratax AI Technical Documentation | Version 1.0 | January 2026
        </div>
        {html_content}
    </body>
    </html>
    '''
    
    # Save HTML
    suffix = "Engineering" if audience.strip().lower() == "engineering" else "Client"
    output_path = Path(f"Stratax_AI_Technical_Documentation_{suffix}.html")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_html)
    
    print(f"✅ HTML generated successfully: {output_path.absolute()}")
    print(f"📄 File size: {output_path.stat().st_size / 1024:.2f} KB")
    print(f"\n🖨️  To convert to PDF:")
    print(f"   1. Open the HTML file in Microsoft Edge or Chrome")
    print(f"   2. Press Ctrl+P (Print)")
    print(f"   3. Select 'Save as PDF' as the printer")
    print(f"   4. Click 'Save' and choose location")
    print(f"\n   Recommended settings:")
    print(f"   - Layout: Portrait")
    print(f"   - Paper size: A4")
    print(f"   - Margins: Default")
    print(f"   - Background graphics: ON")
    
    # Try to open in default browser
    try:
        webbrowser.open(str(output_path.absolute()))
        print(f"\n✨ Opening in your default browser...")
    except Exception:
        pass
    
    return output_path

if __name__ == "__main__":
    generate_html()

