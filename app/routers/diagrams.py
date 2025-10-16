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


def _normalize_text(s: str) -> str:
    """Normalize diagram text to avoid Mermaid parse issues from pasted content.
    - Normalize newlines to \n
    - Replace smart quotes/dashes with ASCII
    - Remove zero‑width and non-breaking spaces
    - Strip UTF-8 BOM if present
    """
    import re as _re
    if s.startswith("\ufeff"):
        s = s.lstrip("\ufeff")
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    trans = {
        "\u2018": "'", "\u2019": "'", "\u201C": '"', "\u201D": '"',
        "\u2013": "-", "\u2014": "-", "\u00A0": " ",
    }
    for k, v in trans.items():
        s = s.replace(k, v)
    # Remove zero-width characters
    s = _re.sub("[\u200B\u200C\u200D\uFEFF]", "", s)
    # Collapse trailing spaces that can break labels
    s = _re.sub(r"[ \t]+\n", "\n", s)
    return s


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
    """Convert numeric step labels like `-- 1. Foo -->` into clean numeric format
    removing the dot and extra spaces for professional look.
    """
    import re as _re

    def repl(m: _re.Match[str]) -> str:
        n = m.group(1)
        return f" -- {n} "

    # Edge label patterns:  A -- 1. Text --> B  or A ---|1. Text| B
    # Convert "1. Send Msg" to just "1"
    code = _re.sub(r"\s--\s*(\d+)\.(\s|\|)", lambda m: repl(m), code)
    code = _re.sub(r"\|\s*(\d+)\.(\s|\|)", lambda m: f"| {m.group(1)} ", code)
    return code


def _fix_parenthetical_edge_labels(code: str) -> str:
    """Convert invalid parenthetical edge labels like `A --> B (W)` to
    Mermaid-compliant `A -- W --> B` or `A -- (W) --> B`.
    Also supports spaced labels: `( Read )`. Conservative regex.
    """
    import re as _re
    # Pattern: NodeId --> NodeId (Label)
    pattern = _re.compile(r"(^|\n)\s*([A-Za-z0-9_]+)\s*--?>\s*([A-Za-z0-9_]+)\s*\(([^)]+)\)")
    def repl(m: _re.Match[str]) -> str:
        prefix = m.group(1)
        a = m.group(2)
        b = m.group(3)
        label = m.group(4).strip()
        return f"{prefix}{a} -- {label} --> {b}"
    return pattern.sub(repl, code)


def _auto_insert_line_breaks(code: str, max_len: int = 28) -> str:
    """Hard-wrap node labels by inserting <br/> at word boundaries when labels
    exceed max_len. Guarantees visibility without truncation.
    Examples converted:
      NodeId[This is a very long title] -> NodeId[This is a very<br/>long title]
    """
    import re as _re

    def wrap_label(text: str) -> str:
        words = text.split()
        lines: list[str] = []
        current = ""
        for w in words:
            if not current:
                current = w
                continue
            if len(current) + 1 + len(w) <= max_len:
                current += " " + w
            else:
                lines.append(current)
                current = w
        if current:
            lines.append(current)
        return "<br/>".join(lines)

    pattern = _re.compile(r"([A-Za-z0-9_]+)\s*\[(.*?)\]")
    def repl(m: _re.Match[str]) -> str:
        node = m.group(1)
        label = m.group(2)
        if '<br/>' in label or len(label) <= max_len:
            return m.group(0)
        return f"{node}[{wrap_label(label)}]"

    return pattern.sub(repl, code)


def _redirect_edges_to_subgraphs(code: str) -> str:
    """Mermaid doesn't allow edges to subgraph IDs. If found, create an anchor
    node inside each subgraph and redirect edges to that anchor.
    Example: `Web --> E` becomes `Web --> E_anchor`, and we inject
    `E_anchor(( )):::anchor` as the first line inside subgraph E.
    """
    import re as _re

    # Find subgraph ids and their line indices
    lines = code.split("\n")
    subgraph_re = _re.compile(r"^\s*subgraph\s+([A-Za-z0-9_]+)\s*\[")
    id_to_line: dict[str, int] = {}
    for idx, line in enumerate(lines):
        m = subgraph_re.match(line)
        if m:
            id_to_line[m.group(1)] = idx

    if not id_to_line:
        return code

    # Detect edges referencing subgraph ids
    edge_re = _re.compile(r"(^|\n)(\s*)([A-Za-z0-9_]+)\s*--[\-|>\s\w\.\(\)]*?>\s*([A-Za-z0-9_]+)")
    used: set[str] = set()

    def edge_replacer(m: _re.Match[str]) -> str:
        prefix, indent, src, dst = m.group(1), m.group(2), m.group(3), m.group(4)
        new_src = src
        new_dst = dst
        if src in id_to_line:
            new_src = f"{src}_anchor"
            used.add(src)
        if dst in id_to_line:
            new_dst = f"{dst}_anchor"
            used.add(dst)
        # Reconstruct the original edge text by keeping everything between src and dst
        # Simpler: replace only the ids at ends to avoid changing labels
        s = m.group(0)
        s = _re.sub(rf"(^|\n){indent}{src}(?=\s*--)", f"{prefix}{indent}{new_src}", s)
        s = _re.sub(rf"(?<=>)\s*{dst}(?=\b)", new_dst, s)
        return s

    new_code = edge_re.sub(edge_replacer, code)
    if not used:
        return new_code

    # Inject anchors into corresponding subgraphs (right after 'subgraph ...' line)
    insertion_offset = 0
    for sid, line_idx in sorted(id_to_line.items(), key=lambda kv: kv[1]):
        if sid not in used:
            continue
        insert_at = line_idx + 1 + insertion_offset
        lines.insert(insert_at, f"  {sid}_anchor(( )):::anchor")
        insertion_offset += 1

    return "\n".join(lines)


@router.post("/render_mermaid")
async def render_mermaid(payload: dict):
    """Render Mermaid code to SVG via Kroki backend.

    Expected payload: { "code": "flowchart LR...", "theme": "default|dark|forest|neutral" }
    Returns raw SVG content.
    """
    code = _sanitize_code(payload.get("code") or "")
    code = _normalize_text(code)
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

    # Optional: style preset for modern/enterprise look (visual only)
    style = (payload.get("style") or "").strip().lower()
    if style in ("modern", "enterprise") and not code.lstrip().startswith("%%{init"):
        is_enterprise = style == "enterprise"
        init = (
            "%%{init: {\n"
            "  'theme': 'neutral',\n"
            "  'themeVariables': {\n"
            f"    'fontSize':'12px', 'fontFamily':'Inter, sans-serif',\n"
            f"    'lineColor':'#666', 'primaryColor':'#f8f9fa',\n"
            f"    'edgeLabelBackground':'#ffffff', 'padding':12, 'curve':'{'step' if is_enterprise else 'basis'}',\n"
            f"    'textWrapWidth': {240 if is_enterprise else 220}\n"
            "  },\n"
            f"  'flowchart': {{ 'htmlLabels': true, 'useMaxWidth': false,\n"
            f"                 'nodeSpacing': {60 if is_enterprise else 40}, 'rankSpacing': {80 if is_enterprise else 50},\n"
            f"                 'diagramPadding': {16 if is_enterprise else 8}, 'wrap': true }}\n"
            "}}%%\n"
        )
        # Add compact spacing helpers
        # Force LR for cleaner edges when requested
        direction_prefix = "flowchart LR\n" if code.lstrip().startswith("flowchart ") else ""
        code = (
            init
            + (direction_prefix if direction_prefix else "")
            + code
            + ("\nlinkStyle default stroke:#444,stroke-width:1.6px;\n" if is_enterprise else "\nlinkStyle default stroke:#666,stroke-width:1.3px;\n")
            + "classDef client fill:#e3f2fd,stroke:#1976d2,color:#000;\n"
            + "classDef network fill:#fff3e0,stroke:#e65100,color:#000;\n"
            + "classDef service fill:#fff8e1,stroke:#f57f17,color:#000;\n"
            + "classDef storage fill:#f1f8e9,stroke:#2e7d32,color:#000;\n"
            + "classDef queue fill:#e0f7fa,stroke:#006064,color:#000;\n"
            + "classDef cache fill:#f3e5f5,stroke:#6a1b9a,color:#000;\n"
        )

    # Optional: prettify numeric edge labels
    # Always apply syntax tolerant transforms so diagrams render reliably
    try:
        code = _fix_parenthetical_edge_labels(code)
        code = _prettify_edge_labels(code)
        code = _auto_insert_line_breaks(code, max_len=26)
        code = _redirect_edges_to_subgraphs(code)
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


