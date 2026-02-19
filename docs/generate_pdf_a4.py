"""Generate a polished A4 PDF (with tables + Mermaid diagrams) from markdown.

How it works:
1) Builds HTML via `generate_pdf.generate_html()` (Markdown -> HTML with Mermaid blocks)
2) Uses Playwright (headless Chromium) to render the HTML and export an A4 PDF

This avoids WeasyPrint/GTK dependencies on Windows and preserves Mermaid diagrams.
"""

from __future__ import annotations

import asyncio
import argparse
from pathlib import Path


async def _export_pdf(html_path: Path, pdf_path: Path) -> None:
    from playwright.async_api import async_playwright

    file_url = html_path.absolute().as_uri()

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1240, "height": 1754})

        await page.goto(file_url, wait_until="networkidle")

        # Wait for Mermaid to render.
        # We wait until every .mermaid block contains an <svg>.
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
            margin={"top": "16mm", "right": "14mm", "bottom": "16mm", "left": "14mm"},
        )

        await browser.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate A4 PDFs (client and/or engineering) with Mermaid diagrams.")
    parser.add_argument(
        "--audience",
        choices=["client", "engineering", "both"],
        default="both",
        help="Which document variant to generate",
    )
    args = parser.parse_args()

    import generate_pdf

    targets: list[str]
    if args.audience == "both":
        targets = ["client", "engineering"]
    else:
        targets = [args.audience]

    for audience in targets:
        html_path = Path(generate_pdf.generate_html(audience=audience))
        suffix = "Client" if audience == "client" else "Engineering"
        pdf_path = Path(f"Stratax_AI_Technical_Documentation_{suffix}_A4.pdf")

        asyncio.run(_export_pdf(html_path, pdf_path))

        size_mb = pdf_path.stat().st_size / 1024 / 1024
        print(f"✅ PDF generated: {pdf_path.absolute()}")
        print(f"📄 Size: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
