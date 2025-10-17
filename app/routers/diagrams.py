from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
import httpx

from app.utils.security import verify_api_key


router = APIRouter()


def _sanitize_code(raw: str) -> str:
    """Remove surrounding markdown fences if present."""
    text = raw.strip()
    if text.startswith("```"):
        # Remove first line of fence
        lines = text.split("\n")
        if lines:
            # drop first line and any closing fence line
            body = "\n".join(lines[1:])
            if body.rstrip().endswith("```"):
                body = body[: body.rfind("```")].rstrip()
            return body
    return text


def _convert_layer_nodes_to_subgraphs(code: str) -> str:
    """Best-effort transform: turn standalone nodes whose labels end with
    the word "Layer" into Mermaid subgraphs.

    Rationale: Some generators model architectural layers as simple nodes
    (e.g., `CL[Client Layer]`). This function rewrites such headers into
    `subgraph` blocks so that contained content is visually grouped.

    Rules (conservative):
    - If the code already contains any `subgraph` token, leave unchanged.
    - Detect header lines shaped like: <ID>[<... Layer>] or <ID>(<... Layer>) or <ID>{<... Layer>}.
    - Start a subgraph at each detected header line and automatically close it
      right before the next detected header (or end of document).
    - Header node definitions are removed (replaced by the subgraph title).
    """
    src = code
    if "subgraph" in src:
        return src

    import re as _re

    lines = src.split("\n")
    header_regex = _re.compile(r"^\s*([A-Za-z0-9_]+)\s*([\[\(\{])\s*(.+?)\s*([\]\)\}])\s*$")

    # Gather edge references to avoid converting nodes that are used in edges
    edge_ref_regex = _re.compile(r"(^|\W)([A-Za-z0-9_]+)\s*[-=~]+[ox]?\>|\<[-=~]+[ox]?\s*([A-Za-z0-9_]+)(\W|$)")
    edge_refs: set[str] = set()
    for line in lines:
        for m in edge_ref_regex.finditer(line):
            # Matches either source in group 2 or target in group 3
            if m.group(2):
                edge_refs.add(m.group(2))
            if m.group(3):
                edge_refs.add(m.group(3))

    header_indices: list[tuple[int, str, str]] = []  # (line_index, id, label)
    for idx, line in enumerate(lines):
        m = header_regex.match(line)
        if not m:
            continue
        node_id, _open, label, _close = m.groups()
        label_clean = label.strip()
        lower = label_clean.lower()
        looks_like_layer = (
            lower.endswith("layer") or
            " layer" in lower or
            "plane" in lower or
            lower in {"file/external", "external", "file layer"}
        )
        # Only convert if it looks like a grouping header and is not referenced in edges
        if looks_like_layer and node_id not in edge_refs:
            header_indices.append((idx, node_id, label.strip()))

    if not header_indices:
        return src

    # Build new code with subgraphs spanning header-to-next-header-1
    result: list[str] = []
    i = 0
    header_ptr = 0
    current_block_end = -1
    while i < len(lines):
        if header_ptr < len(header_indices) and i == header_indices[header_ptr][0]:
            # Open new subgraph
            _idx, node_id, label = header_indices[header_ptr]
            # Determine end
            if header_ptr + 1 < len(header_indices):
                current_block_end = header_indices[header_ptr + 1][0]
            else:
                current_block_end = len(lines)

            # Emit subgraph header (escaped quotes inside label)
            safe_label = label.replace('"', '\\"')
            result.append(f"subgraph {node_id}[\"{safe_label}\"]")
            # Skip the header node line itself
            i += 1
            # Emit content until next header (exclusive)
            while i < current_block_end:
                result.append(lines[i])
                i += 1
            # Close block
            result.append("end")
            header_ptr += 1
            continue

        # Lines before the first header or between already processed blocks
        result.append(lines[i])
        i += 1

    # Cleanup: remove accidental double blank lines
    out = "\n".join(result)
    out = _re.sub(r"\n{3,}", "\n\n", out).strip()
    return out


def _prettify_edge_labels(code: str) -> str:
    """Convert numeric step labels like `-- 1. Foo -->` into circled numerals
    to improve aesthetics. Conservative: only changes numbers 1-20.
    """
    import re as _re
    circled = {
        1: "①", 2: "②", 3: "③", 4: "④", 5: "⑤",
        6: "⑥", 7: "⑦", 8: "⑧", 9: "⑨", 10: "⑩",
        11: "⑪", 12: "⑫", 13: "⑬", 14: "⑭", 15: "⑮",
        16: "⑯", 17: "⑰", 18: "⑱", 19: "⑲", 20: "⑳",
    }

    def repl(m: _re.Match[str]) -> str:
        n = int(m.group(1))
        symbol = circled.get(n, m.group(1))
        return f" -- {symbol} "

    # Edge label patterns:  A -- 1. Text --> B  or A ---|1. Text| B
    code = _re.sub(r"\s--\s*(\d+)\.(\s|\|)", lambda m: repl(m), code)
    code = _re.sub(r"\|\s*(\d+)\.(\s|\|)", lambda m: f"| {circled.get(int(m.group(1)), m.group(1))} ", code)
    return code


def _add_sequential_step_numbers(code: str) -> str:
    """Add sequential step numbers directly to node labels to show workflow sequence.
    Modifies existing node definitions to include step numbers inline.
    """
    import re as _re
    
    lines = code.split('\n')
    node_pattern = _re.compile(r'^\s*([A-Za-z0-9_]+)\s*[\[\(\{]([^\]\)\}]+)[\]\)\}]\s*')
    
    # Find all node definitions and track their order
    node_definitions = []
    for i, line in enumerate(lines):
        match = node_pattern.match(line)
        if match:
            node_id, label = match.groups()
            node_definitions.append((i, node_id, label.strip()))
    
    if not node_definitions:
        return code
    
    # Modify lines to add step numbers to node labels
    result_lines = lines.copy()
    step_num = 1
    
    for line_idx, node_id, label in node_definitions:
        # Add step number to the beginning of the label
        new_label = f"{step_num}. {label}"
        # Replace the line with updated label
        result_lines[line_idx] = _re.sub(
            r'^\s*([A-Za-z0-9_]+)\s*[\[\(\{]([^\]\)\}]+)[\]\)\}]\s*',
            f'{node_id}[{new_label}]',
            result_lines[line_idx]
        )
        step_num += 1
    
    return '\n'.join(result_lines)


@router.post("/render_mermaid")
async def render_mermaid(payload: dict):
    """Render Mermaid code to SVG via Kroki backend.

    Expected payload: { "code": "flowchart LR...", "theme": "default|dark|forest|neutral" }
    Returns raw SVG content.
    """
    code = _sanitize_code(payload.get("code") or "")
    if not code:
        raise HTTPException(status_code=400, detail="Missing 'code' in payload")

    # Basic guardrail: hard-limit size to avoid abuse
    if len(code) > 40_000:
        raise HTTPException(status_code=413, detail="Diagram too large")

    # Attempt to group layer headers into subgraphs before rendering
    try:
        code = _convert_layer_nodes_to_subgraphs(code)
    except Exception:
        # Do not fail rendering if transformation has issues
        pass

    theme = (payload.get("theme") or "").strip() or "default"

    # Optional: style preset for modern elegant look without changing semantics
    style = (payload.get("style") or "").strip().lower()
    if style == "modern" and not code.lstrip().startswith("%%{init"):
        init = (
            "%%{init: {\n"
            "  'theme': 'neutral',\n"
            "  'themeVariables': {\n"
            "    'fontSize':'13px', 'fontFamily':'Inter, sans-serif',\n"
            "    'lineColor':'#666', 'primaryColor':'#f8f9fa',\n"
            "    'edgeLabelBackground':'#ffffff', 'padding':16, 'curve':'basis',\n"
            "    'textWrapWidth': 340\n"
            "  },\n"
            "  'flowchart': { 'htmlLabels': true, 'useMaxWidth': false,\n"
            "                 'nodeSpacing': 50, 'rankSpacing': 60,\n"
            "                 'diagramPadding': 16, 'wrap': true }\n"
            "}}%%\n"
        )
        # Add compact spacing helpers
        code = (
            init
            + code
            + "\nlinkStyle default stroke:#666,stroke-width:1.3px;\n"
            + "classDef client fill:#e3f2fd,stroke:#1976d2,color:#000;\n"
            + "classDef network fill:#fff3e0,stroke:#e65100,color:#000;\n"
            + "classDef service fill:#fff8e1,stroke:#f57f17,color:#000;\n"
            + "classDef storage fill:#f1f8e9,stroke:#2e7d32,color:#000;\n"
            + "classDef queue fill:#e0f7fa,stroke:#006064,color:#000;\n"
            + "classDef cache fill:#f3e5f5,stroke:#6a1b9a,color:#000;\n"
        )

    # Optional: prettify numeric edge labels and add step numbers
    if style == "modern":
        try:
            code = _prettify_edge_labels(code)
            code = _add_sequential_step_numbers(code)
        except Exception:
            pass

    # Try multiple Mermaid rendering services for better reliability
    services = [
        "https://mermaid.ink/svg",
        "https://kroki.io/mermaid/svg"
    ]
    
    headers = {
        "Content-Type": "text/plain; charset=utf-8",
    }

    # Some themes are supported by Mermaid directly; inject theme directive if provided
    if theme and theme != "default" and not code.lstrip().startswith("%%{init"):
        # Prepend Mermaid init directive using valid JSON (double quotes)
        code = f"%%{{init: {{ \"theme\": \"{theme}\" }} }}%%\n" + code

    import requests
    import base64
    
    # Try mermaid.ink first (more reliable)
    try:
        print(f"DEBUG: Trying mermaid.ink")
        print(f"DEBUG: Code: {code[:100]}...")
        
        # mermaid.ink uses base64 encoded diagram in URL
        encoded_code = base64.b64encode(code.encode('utf-8')).decode('ascii')
        url = f"https://mermaid.ink/svg/{encoded_code}"
        
        resp = requests.get(url, timeout=10)
        print(f"DEBUG: mermaid.ink response: {resp.status_code}")
        
        if resp.status_code == 200 and resp.text.strip().startswith("<svg"):
            svg = resp.text
        else:
            raise Exception(f"mermaid.ink failed: {resp.status_code}")
            
    except Exception as exc:
        print(f"DEBUG: mermaid.ink failed: {exc}")
        # Fallback to Kroki with shorter timeout
        try:
            print(f"DEBUG: Trying Kroki as fallback")
            url = "https://kroki.io/mermaid/svg"
            resp = requests.post(url, data=code.encode("utf-8"), headers=headers, timeout=5)
            print(f"DEBUG: Kroki response: {resp.status_code}")
            
            if resp.status_code != 200:
                error_text = resp.text[:200] if resp.text else "No error details"
                raise Exception(f"Kroki failed: {resp.status_code} - {error_text}")
                
            svg = resp.text
            if not svg.strip().startswith("<svg"):
                raise Exception("Invalid SVG from Kroki")
                
        except Exception as kroki_exc:
            print(f"DEBUG: Both services failed. Kroki error: {kroki_exc}")
            raise HTTPException(status_code=502, detail=f"All rendering services failed. Last error: {str(kroki_exc)}")

    # Final sanity check
    if not svg.strip().startswith("<svg"):
        raise HTTPException(status_code=502, detail="Invalid SVG returned from renderer")

    return Response(content=svg, media_type="image/svg+xml")


@router.get("/render_mermaid")
async def render_mermaid_get(
    code: str = Query(default=""),
    theme: str = Query(default="default"),
):
    """GET variant for <img src> compatibility.

    Accepts `code` and optional `theme` as query params and returns SVG.
    """
    payload = {"code": code, "theme": theme}
    return await render_mermaid(payload)


