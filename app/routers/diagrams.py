from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
import httpx
import logging
import os
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

from app.utils.security import verify_api_key
from app.schemas import (
    GenerateArchitectureRequest,
    ArchitecturePackageOut,
    ArchitectureViewOut,
    ArchitectureViewType,
    DiagramStyle,
    RenderViewRequest
)
from app.services.architecture_generator import get_architecture_generator
from app.services.llm_service import get_llm_service


router = APIRouter()


def _fix_duplicate_diagram_declarations(text: str) -> str:
    """Fix invalid Mermaid where LLM generated multiple diagram type declarations.
    
    Example bad syntax:
        flowchart LR
          graph TD
          A --> B
    
    This function keeps only the FIRST diagram type declaration and removes duplicates.
    """
    import re as _re
    
    lines = text.split('\n')
    if len(lines) < 2:
        return text
    
    diagram_types = ['flowchart', 'graph', 'sequenceDiagram', 'classDiagram', 'stateDiagram', 
                     'erDiagram', 'journey', 'gantt', 'pie', 'gitGraph', 'mindmap', 'timeline']
    
    first_declaration_found = False
    cleaned_lines = []
    
    for line in lines:
        stripped = line.strip().lower()
        
        # Check if this line is a diagram type declaration
        is_diagram_declaration = any(stripped.startswith(dtype.lower()) for dtype in diagram_types)
        
        if is_diagram_declaration:
            if not first_declaration_found:
                # Keep the first declaration
                cleaned_lines.append(line)
                first_declaration_found = True
            # Skip any subsequent declarations (they're duplicates)
            continue
        else:
            # Keep all non-declaration lines
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)


def _sanitize_code(raw: str) -> str:
    """Remove surrounding markdown fences if present and fix escaped newlines."""
    text = raw.strip()
    
    # Fix escaped newlines (\\n -> \n) that LLMs sometimes generate
    if '\\n' in text:
        text = text.replace('\\n', '\n')
    
    if text.startswith("```"):
        # Remove first line of fence
        lines = text.split("\n")
        if lines:
            # drop first line and any closing fence line
            body = "\n".join(lines[1:])
            if body.rstrip().endswith("```"):
                body = body[: body.rfind("```")].rstrip()
            text = body
    
    # Fix duplicate diagram declarations (LLM sometimes generates "flowchart LR\n  graph TD")
    text = _fix_duplicate_diagram_declarations(text)
    
    return text


def _remove_non_ascii(text: str) -> str:
    """Remove non-ASCII characters that can break some Mermaid renderers (and URLs).
    Keeps basic punctuation and replaces unicode arrows/emojis with ASCII equivalents.
    """
    replacements = {
        '→': '->', '←': '<-', '⇒': '=>', '⇐': '<=', '↔': '<->',
        '“': '"', '”': '"', '’': "'", '–': '-', '—': '-', '…': '...',
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    # Strip any remaining non-ASCII
    return ''.join(ch if ord(ch) < 128 else '' for ch in text)


def _strip_mermaid_features(text: str) -> str:
    """Reduce diagram to a minimal, broadly compatible subset:
    - Remove Mermaid init blocks (%%{init ... }%%)
    - Remove classDef and linkStyle lines
    - Remove class assignments (:::class)
    - Remove HTML labels directive if present
    - Remove any CSS/style artifacts (corrupted retries)
    """
    import re as _re
    lines = []
    for line in text.split('\n'):
        s = line.strip()
        if s.startswith('%%{init'):  # drop init block start
            continue
        if s.endswith('}%%') and '%%{init' in text:  # drop init block end line if isolated
            continue
        if s.lower().startswith('classdef'):
            continue
        if s.lower().startswith('linkstyle'):
            continue
        # Skip CSS artifacts that got mixed into Mermaid code
        if any(css_marker in s for css_marker in ['@keyframes', '@import', '#container', '.edge-', '#mermaid-svg']):
            continue
        if ':::' in s:
            # remove class assignment but keep node/edge
            line = _re.sub(r':::\w+', '', line)
        lines.append(line)
    out = '\n'.join(lines)
    # Remove htmlLabels directive if present inside init-like string remnants
    out = _re.sub(r"'htmlLabels'\s*:\s*(true|false)", '', out, flags=_re.IGNORECASE)
    return out.strip()


def _ultra_simplify_mermaid(text: str) -> str:
    """Last-resort simplification when renderers reject Mermaid.

    Goal: produce *some* SVG instead of a placeholder by stripping the diagram down
    to a basic flowchart with unlabeled edges.
    """
    import re as _re

    if not text:
        return ""

    # Normalize newlines
    code = text.replace("\r\n", "\n").replace("\r", "\n")
    code = _strip_mermaid_features(code)
    code = _remove_non_ascii(code)

    # Force a flowchart header if missing
    lines = [ln for ln in code.split("\n") if ln.strip()]
    if not lines:
        return "flowchart TD\n  A[Empty]"
    first = lines[0].strip().lower()
    if not (first.startswith("flowchart") or first.startswith("graph")):
        lines.insert(0, "flowchart LR")

    code = "\n".join(lines)

    # Drop edge labels: -->|label| becomes -->
    code = _re.sub(r"-->\|[^\|]{1,80}\|", "-->", code)
    code = _re.sub(r"-\.->\|[^\|]{1,80}\|", "-.->", code)
    code = _re.sub(r"==>\|[^\|]{1,80}\|", "==>", code)

    # Drop any remaining pipe labels (best-effort)
    code = _re.sub(r"\|\s*\([^\)]{1,6}\)\s*\|", "| |", code)
    code = _re.sub(r"\|[^\|]{1,80}\|", "| |", code)

    return code.strip()


def _svg_placeholder(message: str) -> str:
        """Return a tiny SVG placeholder indicating render failure.
        Keeps UI consistent by always returning an SVG instead of a 5xx.
        """
        safe_msg = (message or "Mermaid render failed").replace("<", "&lt;").replace(">", "&gt;")
        return (
                """
<svg xmlns="http://www.w3.org/2000/svg" width="640" height="120" viewBox="0 0 640 120" role="img" aria-label="Mermaid render failed">
    <rect x="1" y="1" width="638" height="118" rx="8" ry="8" fill="#fff8e1" stroke="#f57f17"/>
    <text x="20" y="50" font-family="Inter, Arial, sans-serif" font-size="16" fill="#333">Mermaid render unavailable</text>
    <text x="20" y="80" font-family="Inter, Arial, sans-serif" font-size="13" fill="#555">""" + safe_msg + """</text>
</svg>
                """
        ).strip()


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
    """Prettify numeric step labels like `-- 1. Foo -->`.

    IMPORTANT: Must stay ASCII-safe because we do a final non-ASCII stripping
    pass before rendering (and non-ASCII labels can break Mermaid syntax when
    partially removed).
    """
    import re as _re

    def repl(m: _re.Match[str]) -> str:
        n = int(m.group(1))
        # Keep it ASCII: (1), (2), ...
        return f" -- ({n}) "

    # Edge label patterns:  A -- 1. Text --> B  or A ---|1. Text| B
    code = _re.sub(r"\s--\s*(\d+)\.(\s|\|)", lambda m: repl(m), code)
    code = _re.sub(r"\|\s*(\d+)\.(\s|\|)", lambda m: f"| ({int(m.group(1))}) ", code)
    return code


def _sanitize_edge_labels(code: str) -> str:
    """Sanitize edge labels to prevent Kroki parse errors.
    
    Common issues:
    - Brackets inside pipe delimiters: -->| (8) [Mobile Push]| breaks parsing
    - Solution: Remove brackets from pipe labels or convert to parentheses
    """
    import re as _re
    
    # Pattern: -->| ... | or -.-| ... | or ==>| ... |
    # Replace any [...] inside pipe delimiters with (...)
    def sanitize_pipe_label(match):
        arrow = match.group(1)
        label = match.group(2)
        
        # Replace square brackets with parentheses inside labels
        sanitized_label = label.replace('[', '(').replace(']', ')')
        
        return f"{arrow}|{sanitized_label}|"
    
    # Match arrow with pipe labels: -->|...|  or  -.-|...|  or  ==>|...|
    code = _re.sub(r'([-=\.]+>)\|([^\|]+)\|', sanitize_pipe_label, code)
    
    return code


def _fix_mermaid_syntax_errors(code: str) -> str:
    """Fix common Mermaid syntax errors to prevent rendering failures.
    Handles bidirectional arrows, edge labels, and other common issues.
    """
    import re as _re
    
    # Fix bidirectional arrows: A <--> B becomes A --> B and B --> A
    def fix_bidirectional_arrows(text):
        # Find all bidirectional arrows
        bidirectional_pattern = _re.compile(r'^\s*([A-Za-z0-9_]+)\s*<-->\s*([A-Za-z0-9_]+)\s*$', _re.MULTILINE)
        
        def replace_bidirectional(match):
            from_node = match.group(1)
            to_node = match.group(2)
            return f"  {from_node} --> {to_node}\n  {to_node} --> {from_node}"
        
        return bidirectional_pattern.sub(replace_bidirectional, text)
    
    # Fix edge labels: A -- Label --> B becomes A -->|Label| B
    def fix_edge_labels(text):
        # Pattern for A -- Label --> B
        edge_label_pattern = _re.compile(r'^\s*([A-Za-z0-9_]+)\s*--\s*([^-\n]+?)\s*-->\s*([A-Za-z0-9_]+)\s*$', _re.MULTILINE)
        
        def replace_edge_label(match):
            from_node = match.group(1).strip()
            label = match.group(2).strip()
            to_node = match.group(3).strip()
            return f"  {from_node} -->|{label}| {to_node}"
        
        return edge_label_pattern.sub(replace_edge_label, text)
    
    # Fix malformed edges like A --> B -- Label --> C
    def fix_malformed_edges(text):
        # Pattern for A --> B -- Label --> C (incorrect)
        malformed_pattern = _re.compile(r'^\s*([A-Za-z0-9_]+)\s*-->\s*([A-Za-z0-9_]+)\s*--\s*([^-\n]+?)\s*-->\s*([A-Za-z0-9_]+)\s*$', _re.MULTILINE)
        
        def replace_malformed(match):
            from_node = match.group(1).strip()
            middle_node = match.group(2).strip()
            label = match.group(3).strip()
            to_node = match.group(4).strip()
            return f"  {from_node} --> {middle_node}\n  {middle_node} -->|{label}| {to_node}"
        
        return malformed_pattern.sub(replace_malformed, text)

    # Fix nested parentheses in labels: ID((Label (Text))) becomes ID(("Label (Text)"))
    def fix_nested_labels(text):
        # Match ID((label)), ID([label]), ID[label], ID(label), ID{label}
        patterns = [
            (r'([A-Za-z0-9_]+)\(\(([^"]+?)\)\)', r'\1(("\2"))'),  # circle
            (r'([A-Za-z0-9_]+)\[\(([^"]+?)\)\]', r'\1([ "\2" ])'),  # stadium
            (r'([A-Za-z0-9_]+)\[([^"\]]+?)\]', r'\1["\2"]'),      # square
            (r'([A-Za-z0-9_]+)\(([^"\)]+?)\)', r'\1("\2")'),      # round
            (r'([A-Za-z0-9_]+)\{([^"\}]+?)\}', r'\1{"\2"}'),      # diamond
        ]
        
        for pattern, replacement in patterns:
            def quote_if_needed(match):
                node_id = match.group(1)
                label = match.group(2).strip()
                if any(c in label for c in "()/\\:-"):
                    if "((" in pattern: return f'{node_id}(("{label}"))'
                    if "[(" in pattern: return f'{node_id}(["{label}"])'
                    if "[" in pattern: return f'{node_id}["{label}"]'
                    if "(" in pattern: return f'{node_id}("{label}")'
                    if "{" in pattern: return f'{node_id}{{"{label}"}}'
                return match.group(0)
            text = _re.sub(pattern, quote_if_needed, text)
        return text

    # New Fixes imported from llm_service (Step Id: 82)
    def fix_special_chars(text):
        lines = text.split('\n')
        out_lines = []
        for line in lines:
            # 1. Fix double colons (::) BUT preserve class assignments (:::)
            # Replace :: only when NOT part of ::: (class assignments)
            # Use negative lookahead/lookbehind to avoid breaking :::
            import re as _re
            # Replace :: that is NOT preceded or followed by another :
            line = _re.sub(r'(?<!:)::(?!:)', '-', line)
            
            # 2. Fix slashes in node labels: [text/with/slashes] -> ["text-with-slashes"]
            # Iteratively fix to handle multiple occurrences
            while '/' in line and '[' in line:
                old_line = line
                line = _re.sub(r'\[([^\]]*)/([^\]]*)\]', r'["\1-\2"]', line)
                if old_line == line: break
            
            # 3. Fix colons in node labels: [text:with:colons] -> ["text-with-colons"]
            # Exclude lines that already have quoted labels to avoid damaging them
            # Also exclude lines with ::: (class assignments)
            if '["' not in line and '"]' not in line and ':::' not in line:
                 while ':' in line and '[' in line and '::' not in line:
                    old_line = line
                    line = _re.sub(r'\[([^\]]*):([^\]]*)\]', r'["\1-\2"]', line)
                    if old_line == line: break
            
            out_lines.append(line)
        return '\n'.join(out_lines)

    # Fix wrong diagram type: erDiagram used with flowchart syntax
    def fix_wrong_diagram_type(text):
        """Fix when LLM generates erDiagram but uses flowchart syntax (subgraph, -->)."""
        import re as _re
        lines = text.split('\n')
        if not lines:
            return text
        
        # Find the diagram type line (skip init blocks)
        diagram_type_idx = -1
        for i, line in enumerate(lines):
            stripped = line.strip().lower()
            # Skip init blocks and empty lines
            if stripped.startswith('%%{') or stripped.startswith("'") or stripped.startswith('}') or stripped == '':
                continue
            # Skip lines that are part of init JSON
            if stripped in ['{', '},', '}}%%', '}%%']:
                continue
            # Found potential diagram type declaration
            if stripped in ['erdiagram', 'er-diagram', 'sequencediagram', 'classiagram']:
                diagram_type_idx = i
                break
            # If it's already a flowchart/graph, no fix needed
            if stripped.startswith('flowchart') or stripped.startswith('graph'):
                return text
        
        # Check if erDiagram is used but has flowchart syntax
        if diagram_type_idx >= 0:
            has_flowchart_syntax = any(
                kw in text.lower() for kw in ['subgraph', '-->', '---', '-.->', ':::']
            )
            if has_flowchart_syntax:
                # Replace erDiagram with flowchart TD
                lines[diagram_type_idx] = 'flowchart TD'
                return '\n'.join(lines)
        return text
    
    # Fix invalid brace syntax: NodeName { Label }:::class -> NodeName["Label"]:::class
    def fix_invalid_brace_nodes(text):
        """Fix nodes using { } braces incorrectly (multi-line brace blocks are invalid)."""
        import re as _re
        
        # Pattern: NodeName {\n  Label\n}:::class (multi-line brace node)
        # This is INVALID for flowcharts
        pattern = _re.compile(
            r'([A-Za-z0-9_]+)\s*\{\s*\n\s*([^\n}]+?)\s*\n\s*\}(:::?\w+)?',
            _re.MULTILINE
        )
        
        def fix_brace(match):
            node_id = match.group(1)
            label = match.group(2).strip()
            class_def = match.group(3) or ''
            # Convert to proper square bracket node with quoted label
            return f'{node_id}["{label}"]{class_def}'
        
        return pattern.sub(fix_brace, text)

    # Fix orphaned/floating subgraphs by identifying which have no connections
    def find_and_warn_orphaned_subgraphs(text):
        """Log warning about subgraphs that have no external connections."""
        import re as _re
        
        # Find all subgraph IDs
        subgraph_pattern = _re.compile(r'subgraph\s+(\w+)', _re.IGNORECASE)
        subgraphs = set(subgraph_pattern.findall(text))
        
        # Find all nodes referenced in edges (before and after arrows)
        edge_pattern = _re.compile(r'(\w+)\s*[-=~]+[>\|]|[>\|][-=~]*\s*(\w+)')
        nodes_in_edges = set()
        for match in edge_pattern.finditer(text):
            if match.group(1):
                nodes_in_edges.add(match.group(1))
            if match.group(2):
                nodes_in_edges.add(match.group(2))
        
        # Find nodes inside each subgraph
        subgraph_nodes = {}
        current_subgraph = None
        node_def_pattern = _re.compile(r'^\s*(\w+)[\[\(\{]')
        
        for line in text.split('\n'):
            stripped = line.strip().lower()
            subgraph_match = _re.match(r'subgraph\s+(\w+)', line, _re.IGNORECASE)
            if subgraph_match:
                current_subgraph = subgraph_match.group(1)
                subgraph_nodes[current_subgraph] = []
            elif stripped == 'end':
                current_subgraph = None
            elif current_subgraph:
                node_match = node_def_pattern.match(line.strip())
                if node_match:
                    subgraph_nodes[current_subgraph].append(node_match.group(1))
        
        # Check which subgraphs have no nodes in edges
        orphaned = []
        for sg, nodes in subgraph_nodes.items():
            has_connection = any(node in nodes_in_edges for node in nodes)
            if not has_connection and nodes:
                orphaned.append(sg)
        
        if orphaned:
            logger.warning(f"[MERMAID] Orphaned subgraphs detected (no connections): {orphaned}")
        
        return text  # Return unchanged, just log warning for now

    # Apply all fixes
    fixed_code = code
    fixed_code = fix_wrong_diagram_type(fixed_code)  # Fix diagram type FIRST
    fixed_code = fix_invalid_brace_nodes(fixed_code)  # Fix brace syntax
    fixed_code = fix_special_chars(fixed_code) # Apply this to clean up labels
    fixed_code = fix_bidirectional_arrows(fixed_code)
    fixed_code = fix_edge_labels(fixed_code)
    fixed_code = fix_malformed_edges(fixed_code)
    fixed_code = fix_nested_labels(fixed_code)
    fixed_code = find_and_warn_orphaned_subgraphs(fixed_code)  # Log orphaned subgraphs
    
    return fixed_code


def _add_sequential_step_numbers(code: str, force: bool = False) -> str:
    """Add sequential step numbers to edges/arrows to show workflow sequence.
    Numbers appear on the connections between nodes: 1st arrow gets "1", 2nd gets "2", etc.
    
    Args:
        code: Mermaid diagram code
        force: If True, add numbers even if edges already have labels. If False, skip if edges already numbered.
    """
    import re as _re
    
    # Check if edges already have step numbers (e.g., |1. or |2. or |1|)
    if not force and _re.search(r'\|\d+[\.\)\|]', code):
        logger.debug("[MERMAID] Edges already have step numbers, skipping auto-numbering")
        return code
    
    lines = code.split('\n')
    
    # Pattern to find edges and capture the FULL line structure
    # Matches: A[Label] --> B[Label], A -->|text| B, etc.
    edge_line_pattern = _re.compile(
        r'^(\s*)'  # indent
        r'(\w+(?:\[.*?\]|\(\[.*?\]\)|\[\(.*?\)\]|\(\(.*?\)\)|\{.*?\})?)'  # source with optional shape
        r'\s*(--+>|--+|->|=+>|~+>|-\.+->?)'  # arrow
        r'(\|[^|]*\|)?'  # optional existing label
        r'\s*(\w+(?:\[.*?\]|\(\[.*?\]\)|\[\(.*?\)\]|\(\(.*?\)\)|\{.*?\})?)'  # target with optional shape
        r'\s*$'
    )
    
    # Find all edges
    edges = []
    for i, line in enumerate(lines):
        stripped = line.strip().lower()
        # Skip non-edge lines
        if any(stripped.startswith(kw) for kw in ['classdef', 'linkstyle', 'subgraph', 'end', '%%', 'flowchart', 'graph', '%']):
            continue
        if not stripped:
            continue
        
        m = edge_line_pattern.match(line)
        if m:
            indent = m.group(1) or ''
            source = m.group(2)  # Full source including [Label]
            arrow = m.group(3)
            existing_label = m.group(4)  # May be None or |text|
            target = m.group(5)  # Full target including [Label]
            edges.append((i, indent, source, arrow, existing_label, target))
    
    if not edges:
        logger.debug("[MERMAID] No edges found for numbering")
        return code
    
    logger.debug(f"[MERMAID] Found {len(edges)} edges to number")
    
    # Add step numbers to each edge
    result_lines = lines.copy()
    step_num = 1
    
    for line_idx, indent, source, arrow, existing_label, target in edges:
        if existing_label:
            # Has existing label like |text|, prepend step number
            inner_text = existing_label[1:-1]  # Remove | delimiters
            new_label = f"|{step_num}. {inner_text}|"
        else:
            # No label, add step number only
            new_label = f"|{step_num}|"
        
        # Reconstruct the line preserving node labels
        result_lines[line_idx] = f"{indent}{source} {arrow}{new_label} {target}"
        step_num += 1
    
    return '\n'.join(result_lines)


@router.post("/render_mermaid")
async def render_mermaid(payload: dict):
    """Render Mermaid code to SVG via Kroki backend.

    Expected payload: { "code": "flowchart LR...", "theme": "default|dark|forest|neutral", "addStepNumbers": true/false }
    Returns raw SVG content.
    """
    code = _sanitize_code(payload.get("code") or "")
    if not code:
        raise HTTPException(status_code=400, detail="Missing 'code' in payload")

    # If the caller accidentally sends already-rendered SVG, return it as-is.
    # This avoids turning an upstream UI wiring mistake into a broken placeholder.
    if code.lstrip().startswith("<svg"):
        return Response(
            content=code,
            media_type="image/svg+xml",
            headers={"Cache-Control": "public, max-age=3600"},
        )

    # Guard against payloads that look like SVG/CSS instead of Mermaid.
    # We see this occasionally when UI passes the wrong field or retries corrupt the code.
    suspicious_css = ("@keyframes" in code) or ("#container[" in code) or ("@import url" in code) or (".edge-animation" in code)
    lc = code.lower()
    if suspicious_css and not any(tok in lc for tok in ("flowchart", "graph", "sequencediagram", "erdiagram")):
        logger.warning("[MERMAID] Payload does not look like Mermaid. Returning placeholder.")
        return Response(
            content=_svg_placeholder("Input did not look like Mermaid code (received CSS/SVG-like content)."),
            media_type="image/svg+xml",
            headers={"Cache-Control": "no-store"},
        )
    
    # Even if it contains a diagram type, reject if it's mostly CSS (corruption signal)
    if suspicious_css:
        css_line_count = sum(1 for line in code.split('\n') if any(x in line for x in ['@keyframes', '@import', '#container', '.edge-']))
        total_lines = len([l for l in code.split('\n') if l.strip()])
        if total_lines > 0 and (css_line_count / total_lines) > 0.3:
            logger.warning(f"[MERMAID] Rejecting corrupted input: {css_line_count}/{total_lines} lines are CSS")
            return Response(
                content=_svg_placeholder("Input appears corrupted (too much CSS markup). Use plain Mermaid syntax."),
                media_type="image/svg+xml",
                headers={"Cache-Control": "no-store"},
            )

    # Basic guardrail: hard-limit size to avoid abuse
    if len(code) > 40_000:
        raise HTTPException(status_code=413, detail="Diagram too large")

    # Log first 200 chars for debugging
    logger.debug(f"[MERMAID] Input code (first 200 chars): {code[:200]}")

    # Fix common Mermaid syntax errors first
    try:
        code = _fix_mermaid_syntax_errors(code)
        logger.debug(f"[MERMAID] After syntax fix (first 200 chars): {code[:200]}")
    except Exception as e:
        logger.warning(f"[MERMAID] Syntax fix failed: {e}")
    
    # Sanitize edge labels to prevent parse errors from brackets in pipe delimiters
    try:
        code = _sanitize_edge_labels(code)
        logger.debug("[MERMAID] Edge labels sanitized")
    except Exception as e:
        logger.warning(f"[MERMAID] Edge label sanitization failed: {e}")

    # Optional: Add step numbers to edges.
    # IMPORTANT: Default is FALSE for reliability. (Auto-numbering increases diagram
    # size and can introduce edge-label syntax failures with some renderers.)
    add_step_numbers = str(payload.get("addStepNumbers", "false")).strip().lower()
    if add_step_numbers == "true":
        try:
            code = _add_sequential_step_numbers(code, force=True)
            logger.debug("[MERMAID] Added step numbers to edges (forced)")
        except Exception as e:
            logger.warning(f"[MERMAID] Failed to add step numbers: {e}")
    elif add_step_numbers == "auto":
        # Auto mode: add numbers only if edges don't already have them
        try:
            code = _add_sequential_step_numbers(code, force=False)
        except Exception as e:
            logger.warning(f"[MERMAID] Failed to add step numbers: {e}")

    # Attempt to group layer headers into subgraphs before rendering
    try:
        code = _convert_layer_nodes_to_subgraphs(code)
    except Exception:
        # Do not fail rendering if transformation has issues
        pass

    theme = (payload.get("theme") or "").strip() or "default"

    # Optional: style preset for modern elegant look without changing semantics
    # DISABLED for reliability - style injection causes parse errors in Kroki
    # Frontend should use Mermaid.js client-side rendering for styled diagrams
    style = (payload.get("style") or "").strip().lower()
    # Avoid adding heavy style blocks for very large diagrams
    if False and style == "modern" and not code.lstrip().startswith("%%{init") and len(code) < 3000:
        size = (payload.get("size") or "large").strip().lower()  # Changed default from "medium" to "large"
        responsive = str(payload.get("responsive", "true")).strip().lower() == "true"
        
        if size == "compact":
            font_size = "11px"; padding_val = 10; wrap_w = 280; node_sp = 40; rank_sp = 50; diag_pad = 10
        elif size == "large":
            font_size = "16px"; padding_val = 20; wrap_w = 450; node_sp = 80; rank_sp = 120; diag_pad = 24  # Increased all values
        else:  # medium
            font_size = "14px"; padding_val = 16; wrap_w = 350; node_sp = 60; rank_sp = 80; diag_pad = 16  # Increased all values

        # Build init block with ACTUAL newlines (not escaped)
        init = "%%{init: {\n"
        init += "  'theme': 'neutral',\n"
        init += "  'themeVariables': {\n"
        init += f"    'fontSize':'{font_size}', 'fontFamily':'Inter, sans-serif',\n"
        init += "    'lineColor':'#666', 'primaryColor':'#f8f9fa',\n"
        init += f"    'edgeLabelBackground':'#ffffff', 'padding':{padding_val}, 'curve':'basis'\n"
        init += "  },\n"
        init += "  'flowchart': {\n"
        init += "    'htmlLabels': true,\n"
        init += f"    'useMaxWidth': {str(responsive).lower()},\n"
        init += f"    'nodeSpacing': {node_sp},\n"
        init += f"    'rankSpacing': {rank_sp},\n"
        init += f"    'diagramPadding': {diag_pad}\n"
        init += "  }\n"
        init += "}}%%\n"
        
        # Only add classDefs if they don't already exist (avoid duplicates)
        style_additions = ""
        if "linkStyle" not in code:
            style_additions += "\nlinkStyle default stroke:#666,stroke-width:1.3px;\n"
        
        # Only add default classDefs if user hasn't defined their own
        if "classDef" not in code:
            style_additions += (
                "classDef client fill:#e3f2fd,stroke:#1976d2,color:#000\n"
                "classDef network fill:#fff3e0,stroke:#e65100,color:#000\n"
                "classDef service fill:#fff8e1,stroke:#f57f17,color:#000\n"
                "classDef storage fill:#f1f8e9,stroke:#2e7d32,color:#000\n"
                "classDef queue fill:#e0f7fa,stroke:#006064,color:#000\n"
                "classDef cache fill:#f3e5f5,stroke:#6a1b9a,color:#000\n"
            )
        
        code = init + code + style_additions

    # NOTE: Do not apply additional edge-label transforms here.
    # (Historically this double-applied numbering and produced invalid labels like
    # `-->| (8) [Mobile Push]|`, which Kroki rejects.)

    # Simple cache key based on code + theme
    import hashlib
    cache_key = hashlib.md5(f"{code}|{theme}".encode()).hexdigest()
    
    # Check in-memory cache (simple dict for now)
    if not hasattr(render_mermaid, '_cache'):
        render_mermaid._cache = {}
    
    if cache_key in render_mermaid._cache:
        logger.debug(f"✅ Cache hit for diagram {cache_key[:8]}")
        return Response(
            content=render_mermaid._cache[cache_key],
            media_type="image/svg+xml",
            headers={"Cache-Control": "public, max-age=3600"}
        )

    # Some themes are supported by Mermaid directly; inject theme directive if provided
    if theme and theme != "default" and not code.lstrip().startswith("%%{init") and len(code) < 3000:
        # Prepend Mermaid init directive using valid JSON (double quotes)
        code = f"%%{{init: {{ \"theme\": \"{theme}\" }} }}%%\n" + code

    import base64
    
    # Use async httpx instead of blocking requests
    import httpx
    
    # Final sanitization pass to reduce renderer failures
    code = _remove_non_ascii(code)

    # Decide renderer strategy based on size
    svg = None
    timeout = httpx.Timeout(20.0, connect=5.0)  # Increased timeout
    prefer_kroki = len(code) > 1200

    if not prefer_kroki:
        # Try mermaid.ink first for small diagrams
        try:
            logger.debug(f"Trying mermaid.ink")
            encoded_code = base64.b64encode(code.encode('utf-8')).decode('ascii')
            url = f"https://mermaid.ink/svg/{encoded_code}"
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.get(url)
                # transient outage retry (mermaid.ink sometimes returns 503)
                if resp.status_code in {429, 502, 503, 504}:
                    import asyncio as _asyncio
                    await _asyncio.sleep(0.25)
                    resp = await client.get(url)
            logger.debug(f"mermaid.ink response: {resp.status_code}")
            if resp.status_code == 200 and resp.text.strip().startswith("<svg"):
                svg = resp.text
        except Exception as exc:
            logger.error(f"mermaid.ink failed: {exc}")

    if not svg:
        # Use Kroki (POST) with sanitization retries
        kroki_primary = (os.getenv("KROKI_URL") or "").strip()
        kroki_urls = [u for u in [kroki_primary, "https://kroki.io/mermaid/svg"] if u]
        kroki_urls = list(dict.fromkeys(kroki_urls))  # de-dupe, preserve order

        for url in kroki_urls:
            for attempt in range(4):
                try:
                    logger.debug(f"Trying Kroki {url} (attempt {attempt + 1})")
                    code_to_send = code
                    if attempt == 1:
                        code_to_send = _strip_mermaid_features(code_to_send)
                    elif attempt == 2:
                        code_to_send = _remove_non_ascii(_strip_mermaid_features(code_to_send))
                    elif attempt == 3:
                        code_to_send = _ultra_simplify_mermaid(code_to_send)

                    async with httpx.AsyncClient(timeout=timeout) as client:
                        resp = await client.post(
                            url,
                            content=code_to_send,
                            headers={"Content-Type": "text/plain; charset=utf-8"}
                        )
                    if resp.status_code == 200 and resp.text.strip().startswith("<svg"):
                        svg = resp.text
                        break

                    # Log a small slice of the error body for diagnosis
                    body_preview = (resp.text or "").strip().replace("\n", " ")[:280]
                    logger.error(f"Kroki returned {resp.status_code}: {body_preview}")
                except Exception as kroki_exc:
                    logger.error(f"Kroki attempt {attempt + 1} failed: {kroki_exc}")
            if svg:
                break

        if not svg:
            # Final fallback: try mermaid.ink even for large diagrams (may work if not too large)
            try:
                slim_code = _remove_non_ascii(_strip_mermaid_features(code))
                encoded_code = base64.b64encode(slim_code.encode('utf-8')).decode('ascii')
                url = f"https://mermaid.ink/svg/{encoded_code}"
                async with httpx.AsyncClient(timeout=timeout) as client:
                    resp = await client.get(url)
                logger.debug(f"mermaid.ink final fallback response: {resp.status_code}")
                if resp.status_code == 200 and resp.text.strip().startswith("<svg"):
                    svg = resp.text
            except Exception as exc:
                logger.error(f"mermaid.ink final fallback failed: {exc}")

        if not svg:
            # Do not break UI; return a placeholder SVG explaining the failure
            msg = "Kroki returned errors and mermaid.ink fallback was unavailable. Showing placeholder."
            svg = _svg_placeholder(msg)

    if not svg:
        logger.error(f"❌ Mermaid Rendering Failed. Input Code:\n{code}")
        # Return a generic placeholder to keep UI stable
        svg = _svg_placeholder("Unexpected renderer state; using placeholder")
    
    # Cache the successful result
    render_mermaid._cache[cache_key] = svg
    
    # Limit cache size to prevent memory issues
    if len(render_mermaid._cache) > 100:
        # Remove oldest 20 entries
        keys_to_remove = list(render_mermaid._cache.keys())[:20]
        for k in keys_to_remove:
            del render_mermaid._cache[k]
    
    logger.debug(f"✅ Successfully rendered diagram {cache_key[:8]}")
    
    return Response(
        content=svg,
        media_type="image/svg+xml",
        headers={"Cache-Control": "public, max-age=3600"}
    )


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


# ===== 🚀 WORLD-CLASS MULTI-VIEW ARCHITECTURE GENERATION =====

@router.post("/generate_architecture", response_model=ArchitecturePackageOut)
async def generate_architecture(
    request: GenerateArchitectureRequest,
    _api_key: str = Depends(verify_api_key)
):
    """
    🏗️ Generate a complete, multi-view architecture package.
    
    This endpoint transforms a system description into multiple coordinated
    architectural views, each answering a specific question about the system.
    
    **Why Multi-View?**
    - One diagram = one idea (not everything at once)
    - Interview-ready and explainable
    - Production-grade and maintainable
    - Properly abstracted for different audiences
    
    **Example Request:**
    ```json
    {
        "system_description": "Event management platform with real-time notifications and ticket sales",
        "user_level": "mid",
        "style": "modern"
    }
    ```
    
    **Returns:**
    - System Overview (high-level building blocks)
    - Request Flow (critical business path)
    - Async Processing (background work)
    - Data Model (storage strategy)
    - Deployment (infrastructure) - for senior+
    - Observability (monitoring) - for senior+
    - Security (auth/authz) - if relevant
    """
    try:
        print(f"\n🚀 [API] Request received: Generate Architecture")
        print(f"   System: {request.system_description[:50]}...")
        print(f"   Level: {request.user_level}")
        
        logger.info(f"🏗️ Generating multi-view architecture for: {request.system_description[:100]}...")
        
        arch_generator = get_architecture_generator()
        llm_service = get_llm_service()
        
        # Determine which views to generate
        if request.specific_views:
            views_to_generate = request.specific_views
        else:
            views_to_generate = arch_generator.get_recommended_views(
                request.system_description,
                request.user_level
            )
        
        print(f"📊 Views selected: {[v.value for v in views_to_generate]}")
        logger.info(f"📊 Generating {len(views_to_generate)} views: {[v.value for v in views_to_generate]}")
        
        # Extract system name from description (first few words)
        system_name = " ".join(request.system_description.split()[:5])
        if len(system_name) > 50:
            system_name = system_name[:47] + "..."
        
        # Generate each view
        generated_views: List[ArchitectureViewOut] = []
        
        for view_type in views_to_generate:
            try:
                logger.info(f"🎨 Generating {view_type.value}...")
                
                # Get prompts for this view
                prompts = arch_generator.get_view_prompt(view_type, request.system_description)
                metadata = arch_generator.get_view_metadata(view_type)
                
                # Generate diagram using LLM
                response = await llm_service.generate_response(
                    session_id=request.session_id or "architecture_gen",
                    question=prompts["user_prompt"],
                    system_prompt=prompts["system_prompt"],
                    save_to_history=False  # Don't pollute history with internal prompts
                )
                
                # Extract mermaid code from response
                mermaid_code = _sanitize_code(response.get("answer", ""))
                
                # Validate complexity
                validation = arch_generator.validate_diagram_complexity(
                    mermaid_code,
                    view_type,
                    max_nodes=prompts.get("max_nodes"),
                    max_edges=prompts.get("max_edges"),
                )
                if not validation["valid"]:
                    logger.warning(f"⚠️ Diagram complexity issues for {view_type.value}: {validation['issues']}")
                    # Continue anyway, but log the issues
                
                # Apply syntax fixes
                mermaid_code = _fix_mermaid_syntax_errors(mermaid_code)
                
                # Generate key insights if requested
                key_insights = []
                if request.include_explanations:
                    # Generate 3-5 key insights about this view
                    insights_prompt = f"""Based on this {view_type.value} diagram for {request.system_description}, 
list 3-5 key insights or takeaways that someone should understand from this view.

Format as a simple bullet list, one insight per line, no markdown formatting."""
                    
                    insights_response = await llm_service.generate_response(
                        session_id=request.session_id or "architecture_gen",
                        question=insights_prompt,
                        save_to_history=False
                    )
                    
                    # Parse insights
                    insights_text = insights_response.get("answer", "")
                    key_insights = [
                        line.strip().lstrip("-•*").strip()
                        for line in insights_text.split("\n")
                        if line.strip() and not line.strip().startswith("#")
                    ][:5]  # Max 5 insights
                
                # Create view object
                view = ArchitectureViewOut(
                    view_type=view_type,
                    title=metadata.get("title", view_type.value),
                    description=metadata.get("description", ""),
                    mermaid_code=mermaid_code,
                    key_insights=key_insights,
                    complexity_level=metadata.get("complexity_level", "mid"),
                    estimated_explanation_time=metadata.get("estimated_explanation_time", "2-3 min"),
                    audience=metadata.get("audience", "Engineers"),
                    key_question=metadata.get("key_question", "")
                )
                
                generated_views.append(view)
                logger.info(f"✅ Generated {view_type.value} successfully")
                
            except Exception as e:
                logger.error(f"❌ Failed to generate {view_type.value}: {e}")
                # Continue with other views
                continue
        
        if not generated_views:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate any architecture views. Please try again."
            )
        
        # Create architecture package
        package = ArchitecturePackageOut(
            system_name=system_name,
            description=request.system_description,
            views=generated_views,
            view_order=views_to_generate,
            total_views=len(generated_views),
            metadata={
                "user_level": request.user_level,
                "style": request.style.value,
                "generation_method": "ai_multi_view"
            }
        )
        
        logger.info(f"🎉 Successfully generated {len(generated_views)} architecture views!")
        
        return package
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Architecture generation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate architecture: {str(e)}"
        )


@router.get("/architecture/available_views")
async def get_available_views():
    """
    📋 Get list of all available architecture view types with descriptions.
    
    Returns metadata about each view type to help users understand
    what each view shows and who it's for.
    """
    arch_generator = get_architecture_generator()
    
    views_info = []
    for view_type in ArchitectureViewType:
        metadata = arch_generator.get_view_metadata(view_type)
        views_info.append({
            "view_type": view_type.value,
            "title": metadata.get("title", ""),
            "description": metadata.get("description", ""),
            "complexity_level": metadata.get("complexity_level", ""),
            "audience": metadata.get("audience", ""),
            "key_question": metadata.get("key_question", ""),
            "estimated_time": metadata.get("estimated_explanation_time", "")
        })
    
    return {
        "total_views": len(views_info),
        "views": views_info,
        "recommendation": "Start with system_overview, then request_flow. Add others based on your needs."
    }


@router.post("/architecture/recommend_views")
async def recommend_views(
    system_description: str,
    user_level: str = "mid"
):
    """
    🎯 Get recommended views for a system description.
    
    AI analyzes the system description and recommends which views
    would be most valuable based on the system's characteristics.
    
    **Parameters:**
    - system_description: Description of the system
    - user_level: junior|mid|senior|architect
    
    **Returns:**
    - List of recommended view types
    - Reasoning for each recommendation
    """
    arch_generator = get_architecture_generator()
    
    recommended_views = arch_generator.get_recommended_views(
        system_description,
        user_level
    )
    
    # Get metadata for each recommended view
    views_with_metadata = []
    for view_type in recommended_views:
        metadata = arch_generator.get_view_metadata(view_type)
        views_with_metadata.append({
            "view_type": view_type.value,
            "title": metadata.get("title", ""),
            "description": metadata.get("description", ""),
            "why_recommended": _get_recommendation_reason(view_type, system_description, user_level)
        })
    
    return {
        "system_description": system_description,
        "user_level": user_level,
        "recommended_views": views_with_metadata,
        "total_recommended": len(recommended_views),
        "estimated_total_time": f"{len(recommended_views) * 2.5:.0f}-{len(recommended_views) * 3.5:.0f} min"
    }


def _get_recommendation_reason(view_type: ArchitectureViewType, system_desc: str, user_level: str) -> str:
    """Get human-readable reason for recommending a view."""
    
    reasons = {
        ArchitectureViewType.SYSTEM_OVERVIEW: "Essential for all system designs - provides high-level context",
        ArchitectureViewType.REQUEST_FLOW: "Critical for understanding user journeys and business logic",
        ArchitectureViewType.ASYNC_PROCESSING: "System has async/event-driven components",
        ArchitectureViewType.DATA_MODEL: "Important for understanding data persistence strategy",
        ArchitectureViewType.DEPLOYMENT: f"Recommended for {user_level} level - shows infrastructure",
        ArchitectureViewType.OBSERVABILITY: f"Recommended for {user_level} level - shows operational health",
        ArchitectureViewType.SECURITY: "System has authentication/authorization requirements"
    }
    
    return reasons.get(view_type, "Recommended for complete system understanding")


@router.post("/architecture/export_markdown")
async def export_architecture_markdown(package: ArchitecturePackageOut):
    """
    📄 Export architecture package as formatted Markdown.
    
    Creates a beautiful, interview-ready markdown document with:
    - System overview
    - All diagrams with explanations
    - Key insights for each view
    - Usage tips
    """
    from datetime import datetime
    
    md_lines = []
    
    # Header
    md_lines.append(f"# {package.system_name}")
    md_lines.append(f"\n**Generated:** {package.generated_at.strftime('%Y-%m-%d %H:%M UTC')}")
    md_lines.append(f"\n**Total Views:** {package.total_views}")
    md_lines.append(f"\n## System Description\n")
    md_lines.append(package.description)
    md_lines.append(f"\n## How to Use These Views\n")
    md_lines.append(package.how_to_use)
    md_lines.append(f"\n### Interview Tips\n")
    for tip in package.interview_tips:
        md_lines.append(f"- {tip}")
    
    # Each view
    md_lines.append(f"\n---\n")
    md_lines.append(f"\n## Architecture Views\n")
    
    for i, view in enumerate(package.views, 1):
        md_lines.append(f"\n### {i}. {view.title}\n")
        md_lines.append(f"**Answers:** {view.key_question}\n")
        md_lines.append(f"**Audience:** {view.audience}\n")
        md_lines.append(f"**Explanation Time:** {view.estimated_explanation_time}\n")
        md_lines.append(f"\n{view.description}\n")
        
        if view.key_insights:
            md_lines.append(f"\n**Key Insights:**\n")
            for insight in view.key_insights:
                md_lines.append(f"- {insight}")
        
        md_lines.append(f"\n**Diagram:**\n")
        md_lines.append(f"```mermaid\n{view.mermaid_code}\n```\n")
        md_lines.append(f"\n---\n")
    
    # Footer
    md_lines.append(f"\n## Next Steps\n")
    md_lines.append("1. Review each view in order")
    md_lines.append("2. Practice explaining each diagram in 2-3 minutes")
    md_lines.append("3. Be ready to dive deeper into any specific view")
    md_lines.append("4. Use these views to structure your system design discussion")
    
    markdown_content = "\n".join(md_lines)
    
    return Response(
        content=markdown_content,
        media_type="text/markdown",
        headers={
            "Content-Disposition": f"attachment; filename=\"{package.system_name.replace(' ', '_')}_architecture.md\""
        }
    )
