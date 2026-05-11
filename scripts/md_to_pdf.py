"""Convert REPORT.md to REPORT.pdf using markdown + WeasyPrint + MathML.

Mermaid flowcharts (```mermaid ... ``` fenced blocks) are rendered to PNG via
the public ``mermaid.ink`` service and cached locally in ``assets/mermaid/``
so the PDF embeds real diagrams rather than raw source text. PNG is used
instead of SVG because Mermaid emits node labels inside ``<foreignObject>``
which WeasyPrint cannot render, producing empty boxes in the PDF.
"""

from __future__ import annotations

import base64
import hashlib
import re
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

import markdown
import latex2mathml.converter
from weasyprint import CSS, HTML


# ---------------------------------------------------------------------------
# Mermaid rendering
# ---------------------------------------------------------------------------

MERMAID_FENCE_RE = re.compile(r"```mermaid\n(.*?)\n```", re.DOTALL)
# Use the PNG endpoint with white background and 2x scale for crispness.
# SVG output from mermaid.ink embeds labels inside <foreignObject>, which
# WeasyPrint cannot render (boxes appear empty in the PDF).
MERMAID_INK_URL = "https://mermaid.ink/img/{encoded}?type=png&bgColor=white&width=1600"


def _render_mermaid(source: str, cache_dir: Path) -> Path | None:
    """Render a Mermaid diagram to PNG using mermaid.ink and cache it.

    Returns the path to the cached PNG file, or None if rendering failed.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(source.strip().encode("utf-8")).hexdigest()[:16]
    png_path = cache_dir / f"mermaid_{digest}.png"

    if png_path.exists() and png_path.stat().st_size > 0:
        return png_path

    encoded = base64.urlsafe_b64encode(source.strip().encode("utf-8")).decode("ascii")
    url = MERMAID_INK_URL.format(encoded=encoded)
    # mermaid.ink rejects non-browser User-Agents on the /img/ endpoint (403).
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
        ),
        "Accept": "image/png,image/*;q=0.8,*/*;q=0.5",
    }
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        png_path.write_bytes(data)
        return png_path
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"  [warn] Could not render Mermaid diagram via mermaid.ink: {exc}")
        return None


def preprocess_mermaid(text: str, report_root: Path) -> str:
    """Replace ```mermaid``` fenced blocks with <img> tags pointing to SVGs."""
    cache_dir = report_root / "assets" / "mermaid"

    def repl(match: re.Match) -> str:
        source = match.group(1)
        img_path = _render_mermaid(source, cache_dir)
        if img_path is None:
            escaped = (
                source.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            )
            return (
                '<pre class="mermaid-fallback"><code>'
                + escaped
                + "</code></pre>"
            )
        rel = img_path.relative_to(report_root).as_posix()
        return (
            f'<div class="mermaid-diagram"><img src="{rel}" '
            f'alt="Mermaid diagram"/></div>'
        )

    return MERMAID_FENCE_RE.sub(repl, text)


# ---------------------------------------------------------------------------
# Math extraction helpers
# ---------------------------------------------------------------------------

MATH_FENCE_RE = re.compile(r"```math\n(.*?)\n```", re.DOTALL)
# Display math: $$ ... $$  (possibly spanning multiple lines)
DISPLAY_MATH_RE = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
# Inline math: \( ... \)
INLINE_MATH_RE = re.compile(r"\\\((.+?)\\\)", re.DOTALL)
# Also handle \[ ... \] display math
DISPLAY_BRACKET_RE = re.compile(r"\\\[(.+?)\\\]", re.DOTALL)


def _latex_to_mathml(latex: str, display: str) -> str:
    """Convert a LaTeX string to a MathML element string."""
    try:
        return latex2mathml.converter.convert(latex.strip(), display=display)
    except Exception:
        # Fallback: wrap in a styled span so it's at least visible
        tag = "div" if display == "block" else "span"
        safe = latex.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return f'<{tag} class="math-fallback">{safe}</{tag}>'


def preprocess_md(text: str) -> tuple[str, dict[str, str]]:
    """
    1. Convert ```math ... ``` fenced blocks to $$ ... $$.
    2. Extract all math expressions (display and inline) and replace with
       unique placeholder tokens so the markdown parser never sees them.

    Returns (processed_text, placeholder_map) where placeholder_map maps
    each token back to the final MathML HTML string.
    """
    # Step 1: convert ```math ... ``` → $$ ... $$
    text = MATH_FENCE_RE.sub(lambda m: f"\n$$\n{m.group(1)}\n$$\n", text)

    placeholder_map: dict[str, str] = {}

    def make_token() -> str:
        return f"MATHTOKEN{uuid.uuid4().hex}END"

    def replace_display(m: re.Match) -> str:
        latex = m.group(1)
        token = make_token()
        mathml = _latex_to_mathml(latex, display="block")
        placeholder_map[token] = f'<div class="math-display">{mathml}</div>'
        return f"\n\n{token}\n\n"

    def replace_display_bracket(m: re.Match) -> str:
        latex = m.group(1)
        token = make_token()
        mathml = _latex_to_mathml(latex, display="block")
        placeholder_map[token] = f'<div class="math-display">{mathml}</div>'
        return f"\n\n{token}\n\n"

    def replace_inline(m: re.Match) -> str:
        latex = m.group(1)
        token = make_token()
        mathml = _latex_to_mathml(latex, display="inline")
        placeholder_map[token] = f'<span class="math-inline">{mathml}</span>'
        return token

    # Order matters: display first (so $$ isn't confused with inline $)
    text = DISPLAY_MATH_RE.sub(replace_display, text)
    text = DISPLAY_BRACKET_RE.sub(replace_display_bracket, text)
    text = INLINE_MATH_RE.sub(replace_inline, text)

    return text, placeholder_map


def restore_math(html: str, placeholder_map: dict[str, str]) -> str:
    """Replace placeholder tokens in the rendered HTML with their MathML."""
    # The markdown renderer may have wrapped tokens in <p> tags.
    # We need to handle the case where a display token is wrapped in <p>...</p>
    for token, mathml_html in placeholder_map.items():
        # Replace standalone <p>TOKEN</p>  → mathml_html (unwrapped)
        html = re.sub(
            rf"<p>\s*{re.escape(token)}\s*</p>",
            mathml_html,
            html,
        )
        # Replace any remaining bare token
        html = html.replace(token, mathml_html)
    return html


# ---------------------------------------------------------------------------
# HTML builder
# ---------------------------------------------------------------------------

def build_html(md_path: Path, report_root: Path) -> tuple[str, str]:
    raw = md_path.read_text(encoding="utf-8")

    print("Rendering Mermaid diagrams (via mermaid.ink, cached)...")
    raw = preprocess_mermaid(raw, report_root)

    processed, placeholder_map = preprocess_md(raw)

    md_ext = [
        "tables",
        "fenced_code",
        "toc",
        "attr_list",
        "sane_lists",
        "md_in_html",
    ]
    body_html = markdown.markdown(processed, extensions=md_ext)

    # Restore MathML blocks
    body_html = restore_math(body_html, placeholder_map)

    # Fix relative image paths so WeasyPrint can find the files
    def fix_img(m: re.Match) -> str:
        src = m.group(1)
        if src.startswith("http"):
            return m.group(0)
        abs_src = (report_root / src).resolve()
        return f'<img src="file://{abs_src}"'

    body_html = re.sub(r'<img src="([^"]+)"', fix_img, body_html)

    css = """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Fira+Code&display=swap');

    @page {
        size: A4;
        margin: 2.5cm 2.2cm 2.5cm 2.2cm;
        @bottom-right {
            content: counter(page);
            font-size: 9pt;
            color: #666;
        }
    }

    body {
        font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif;
        font-size: 10.5pt;
        line-height: 1.65;
        color: #1a1a1a;
        max-width: 100%;
    }

    h1 {
        font-size: 18pt;
        font-weight: 700;
        color: #0d1b2a;
        border-bottom: 2.5px solid #1565c0;
        padding-bottom: 6px;
        margin-top: 0;
        page-break-after: avoid;
    }

    h2 {
        font-size: 13.5pt;
        font-weight: 700;
        color: #1565c0;
        border-bottom: 1px solid #bbdefb;
        padding-bottom: 4px;
        margin-top: 28px;
        page-break-after: avoid;
    }

    h3 {
        font-size: 11.5pt;
        font-weight: 600;
        color: #1976d2;
        margin-top: 20px;
        page-break-after: avoid;
    }

    h4 {
        font-size: 10.5pt;
        font-weight: 600;
        color: #37474f;
        margin-top: 14px;
        page-break-after: avoid;
    }

    p { margin: 0.5em 0 0.8em 0; }

    /* Inline code */
    code {
        font-family: 'Fira Code', 'Courier New', monospace;
        font-size: 9pt;
        background: #f3f4f6;
        border: 1px solid #e5e7eb;
        border-radius: 3px;
        padding: 1px 4px;
    }

    /* Code blocks */
    pre {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 3px solid #1565c0;
        border-radius: 4px;
        padding: 10px 14px;
        overflow-x: auto;
        font-size: 8.5pt;
        line-height: 1.5;
        page-break-inside: avoid;
    }

    pre code {
        background: none;
        border: none;
        padding: 0;
        font-size: inherit;
    }

    /* Tables */
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 14px 0;
        font-size: 9.5pt;
        page-break-inside: avoid;
    }

    th {
        background: #1565c0;
        color: white;
        font-weight: 600;
        padding: 7px 10px;
        text-align: left;
        border: 1px solid #1565c0;
    }

    td {
        padding: 6px 10px;
        border: 1px solid #d1d5db;
    }

    tr:nth-child(even) td { background: #f0f4ff; }
    tr:nth-child(odd) td  { background: #ffffff; }

    /* Images */
    img {
        max-width: 100%;
        height: auto;
        display: block;
        margin: 16px auto;
        border: 1px solid #e2e8f0;
        border-radius: 4px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
        page-break-inside: avoid;
    }

    /* Display math blocks */
    .math-display {
        display: block;
        text-align: center;
        margin: 14px auto;
        page-break-inside: avoid;
        overflow-x: auto;
    }

    .math-display math {
        display: block;
        margin: 0 auto;
    }

    /* Inline math */
    .math-inline {
        display: inline;
        vertical-align: middle;
    }

    /* Fallback for failed math conversion */
    .math-fallback {
        font-family: 'Fira Code', 'Courier New', monospace;
        font-size: 9pt;
        color: #b91c1c;
        background: #fef2f2;
        border: 1px solid #fca5a5;
        border-radius: 3px;
        padding: 1px 4px;
    }

    /* MathML sizing */
    math {
        font-size: 10.5pt;
    }

    blockquote {
        border-left: 4px solid #1565c0;
        background: #e8f0fe;
        padding: 8px 14px;
        margin: 12px 0;
        border-radius: 0 4px 4px 0;
        font-style: italic;
    }

    hr {
        border: none;
        border-top: 1.5px solid #d1d5db;
        margin: 20px 0;
    }

    ul, ol { padding-left: 1.6em; margin: 0.5em 0 0.8em 0; }
    li { margin: 3px 0; }

    a { color: #1565c0; text-decoration: none; }

    /* Mermaid diagrams: rendered to SVG via mermaid.ink */
    .mermaid-diagram {
        text-align: center;
        margin: 18px auto;
        page-break-inside: avoid;
    }

    .mermaid-diagram img {
        max-width: 100%;
        height: auto;
        display: inline-block;
        margin: 0 auto;
        border: 1px solid #e2e8f0;
        border-radius: 4px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
        background: #ffffff;
        padding: 8px;
    }

    /* Fallback rendering when mermaid.ink is unreachable */
    .mermaid-fallback {
        background: #f0f4ff;
        border-left: 3px solid #7c3aed;
        color: #4b5563;
        font-style: italic;
    }
    """

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>RIS Channel Estimation Report</title>
</head>
<body>
{body_html}
</body>
</html>"""

    return full_html, css


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    md_path = project_root / "REPORT.md"
    pdf_path = project_root / "REPORT.pdf"

    print(f"Reading: {md_path}")
    html_content, css_content = build_html(md_path, project_root)

    html_out = project_root / "REPORT.html"
    html_out.write_text(html_content, encoding="utf-8")
    print(f"Intermediate HTML written: {html_out}")

    print("Converting to PDF (this may take 30–60 seconds)...")
    HTML(string=html_content, base_url=str(project_root)).write_pdf(
        str(pdf_path),
        stylesheets=[CSS(string=css_content)],
    )
    print(f"PDF saved: {pdf_path}")
    size_mb = pdf_path.stat().st_size / 1024 / 1024
    print(f"File size: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
