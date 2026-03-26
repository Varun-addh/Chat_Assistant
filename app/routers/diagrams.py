from __future__ import annotations

import asyncio
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import Response
import httpx
import logging
import os
import re
import time
from collections import OrderedDict
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
from app.services.architecture.architecture_generator import get_architecture_generator
from app.services.chat.llm_service import get_llm_service
from app.utils.mermaid_sanitizer import MermaidSanitizer
from app.config import settings
from app.services.core.redis_client import get_redis, redis_enabled


router = APIRouter()

# Module-level Mermaid render cache (LRU)
_mermaid_cache: OrderedDict = OrderedDict()
_MERMAID_CACHE_MAX = 100

# Per-IP rate limiter for the render endpoint
_render_ip_hits: dict = {}
_render_lock = asyncio.Lock()
_RENDER_RATE_LIMIT = 60
_RENDER_RATE_WINDOW = 60


async def _rate_limit_render(request: Request):
    """Lightweight per-IP burst limiter for the render endpoint."""
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    async with _render_lock:
        hits = _render_ip_hits.get(client_ip, [])
        hits = [t for t in hits if now - t < _RENDER_RATE_WINDOW]
        if len(hits) >= _RENDER_RATE_LIMIT:
            raise HTTPException(status_code=429, detail="Render rate limit exceeded")
        hits.append(now)
        _render_ip_hits[client_ip] = hits
        # Evict all stale IPs when dict grows too large to prevent unbounded memory.
        # Preserve only the current client's entry to avoid losing their rate state.
        if len(_render_ip_hits) > 5000:
            current_hits = _render_ip_hits[client_ip]
            _render_ip_hits.clear()
            _render_ip_hits[client_ip] = current_hits


def _fix_duplicate_diagram_declarations(text: str) -> str:
    """Fix invalid Mermaid where LLM generated multiple diagram type declarations."""
    return MermaidSanitizer.fix_duplicate_diagram_declarations(text)


def _sanitize_code(raw: str) -> str:
    """Remove surrounding markdown fences if present and fix escaped newlines."""
    return MermaidSanitizer.sanitize_code_block(raw)


def _remove_non_ascii(text: str) -> str:
    """Remove non-ASCII characters that can break some Mermaid renderers (and URLs)."""
    return MermaidSanitizer.to_ascii(text)


def _strip_mermaid_features(text: str) -> str:
    """Reduce diagram to a minimal, broadly compatible subset."""
    return MermaidSanitizer.strip_features(text)


def _ultra_simplify_mermaid(text: str) -> str:
    """Last-resort simplification when renderers reject Mermaid."""
    return MermaidSanitizer.ultra_simplify(text)


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
    if len(src) > 20_000:
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
    return MermaidSanitizer.sanitize_edge_labels(code)


def _fix_mermaid_syntax_errors(code: str) -> str:
    """Fix common Mermaid syntax errors to prevent rendering failures."""
    return MermaidSanitizer.fix_mermaid_syntax_errors(code)


def _add_sequential_step_numbers(code: str, force: bool = False) -> str:
    """Add sequential step numbers to edges/arrows to show workflow sequence.
    Numbers appear on the connections between nodes: 1st arrow gets "1", 2nd gets "2", etc.
    
    Args:
        code: Mermaid diagram code
        force: If True, add numbers even if edges already have labels. If False, skip if edges already numbered.
    """
    import re as _re
    
    if len(code) > 20_000:
        return code
    
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
async def render_mermaid(payload: dict, _rl=Depends(_rate_limit_render)):
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

    # Single-source Mermaid sanitization pipeline (debuggable and deterministic)
    sanitize_result = MermaidSanitizer.sanitize(code, mode="render")
    if sanitize_result.stages:
        logger.debug(f"[MERMAID] Sanitizer stages: {sanitize_result.stages}")
    code = sanitize_result.code

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

    import base64
    
    # Use async httpx instead of blocking requests
    import httpx
    
    # Build explicit attempt variants.
    # - "theme": try honoring theme if requested (may fail on some renderers)
    # - "base": renderer-safe subset (no init/style)
    # - "ultra": last-resort simplification
    code_base = code
    attempts: list[tuple[str, str]] = []

    if theme and theme != "default" and not code_base.lstrip().startswith("%%{init") and len(code_base) < 3000:
        # Prepend Mermaid init directive using valid JSON (double quotes)
        code_theme = f"%%{{init: {{ \"theme\": \"{theme}\" }} }}%%\n" + code_base
        attempts.append(("theme", code_theme))

    attempts.append(("base", code_base))

    ultra_result = MermaidSanitizer.sanitize(code_base, mode="ultra")
    if ultra_result.code and ultra_result.code != code_base:
        logger.debug(f"[MERMAID] Ultra sanitizer stages: {ultra_result.stages}")
        attempts.append(("ultra", ultra_result.code))

    # Cache is keyed by the actual attempt variant + code so we don't permanently
    # pin a themed request to a previously-failed themed render.
    import hashlib

    def _attempt_cache_key(label: str, attempt_code: str) -> str:
        return hashlib.md5(f"{label}|{attempt_code}".encode()).hexdigest()

    redis = None
    if redis_enabled():
        try:
            redis = await get_redis()
        except Exception:
            redis = None

    def _redis_cache_key(ck: str) -> str:
        prefix = (getattr(settings, "redis_key_prefix", "stratax") or "stratax").strip() or "stratax"
        return f"{prefix}:cache:mermaid:{ck}"

    for label, attempt_code in attempts:
        ck = _attempt_cache_key(label, attempt_code)

        # First-level cache: in-process LRU
        if ck in _mermaid_cache:
            # LRU: mark as recently used
            try:
                _mermaid_cache.move_to_end(ck)
            except Exception:
                pass
            logger.debug(f"✅ Cache hit (local) for diagram {ck[:8]} ({label})")
            return Response(
                content=_mermaid_cache[ck],
                media_type="image/svg+xml",
                headers={"Cache-Control": "public, max-age=3600"}
            )

        # Second-level cache: Redis (shared across workers/instances)
        if redis is not None:
            try:
                cached_svg = await redis.get(_redis_cache_key(ck))
            except Exception:
                cached_svg = None

            if cached_svg:
                try:
                    _mermaid_cache[ck] = cached_svg
                    _mermaid_cache.move_to_end(ck)
                except Exception:
                    pass
                logger.debug(f"✅ Cache hit (redis) for diagram {ck[:8]} ({label})")
                return Response(
                    content=cached_svg,
                    media_type="image/svg+xml",
                    headers={"Cache-Control": "public, max-age=3600"}
                )

    # Decide renderer strategy based on size
    svg = None
    timeout = httpx.Timeout(20.0, connect=5.0)  # Increased timeout
    prefer_kroki = len(code_base) > 1200

    if not prefer_kroki:
        # Try mermaid.ink first for small diagrams
        for label, attempt_code in attempts[:2]:
            try:
                logger.debug(f"Trying mermaid.ink ({label})")
                encoded_code = base64.b64encode(attempt_code.encode('utf-8')).decode('ascii')
                url = f"https://mermaid.ink/svg/{encoded_code}"
                async with httpx.AsyncClient(timeout=timeout) as client:
                    resp = await client.get(url)
                    # transient outage retry (mermaid.ink sometimes returns 503)
                    if resp.status_code in {429, 502, 503, 504}:
                        import asyncio as _asyncio
                        await _asyncio.sleep(0.25)
                        resp = await client.get(url)
                logger.debug(f"mermaid.ink ({label}) response: {resp.status_code}")
                if resp.status_code == 200 and resp.text.strip().startswith("<svg"):
                    svg = resp.text
                    used_label = label
                    used_code = attempt_code
                    break
            except Exception as exc:
                logger.error(f"mermaid.ink ({label}) failed: {exc}")

    if not svg:
        # Use Kroki (POST) with sanitization retries
        kroki_primary = (os.getenv("KROKI_URL") or "").strip()
        kroki_urls = [u for u in [kroki_primary, "https://kroki.io/mermaid/svg"] if u]
        kroki_urls = list(dict.fromkeys(kroki_urls))  # de-dupe, preserve order

        for url in kroki_urls:
            for label, attempt_code in attempts:
                try:
                    logger.debug(f"Trying Kroki {url} ({label})")
                    code_to_send = attempt_code

                    async with httpx.AsyncClient(timeout=timeout) as client:
                        resp = await client.post(
                            url,
                            content=code_to_send,
                            headers={"Content-Type": "text/plain; charset=utf-8"}
                        )
                    if resp.status_code == 200 and resp.text.strip().startswith("<svg"):
                        svg = resp.text
                        used_label = label
                        used_code = attempt_code
                        break

                    # Log a small slice of the error body for diagnosis
                    body_preview = (resp.text or "").strip().replace("\n", " ")[:280]
                    logger.error(f"Kroki ({label}) returned {resp.status_code}: {body_preview}")
                except Exception as kroki_exc:
                    logger.error(f"Kroki ({label}) failed: {kroki_exc}")
            if svg:
                break

        if not svg:
            # Final fallback: try mermaid.ink even for large diagrams (may work if not too large)
            try:
                fallback_label, fallback_code = attempts[-1]
                logger.debug(f"Trying mermaid.ink final fallback ({fallback_label})")
                encoded_code = base64.b64encode(fallback_code.encode('utf-8')).decode('ascii')
                url = f"https://mermaid.ink/svg/{encoded_code}"
                async with httpx.AsyncClient(timeout=timeout) as client:
                    resp = await client.get(url)
                logger.debug(f"mermaid.ink final fallback ({fallback_label}) response: {resp.status_code}")
                if resp.status_code == 200 and resp.text.strip().startswith("<svg"):
                    svg = resp.text
                    used_label = fallback_label
                    used_code = fallback_code
            except Exception as exc:
                logger.error(f"mermaid.ink final fallback failed: {exc}")

        if not svg:
            # Do not break UI; return a placeholder SVG explaining the failure
            msg = "Kroki returned errors and mermaid.ink fallback was unavailable. Showing placeholder."
            svg = _svg_placeholder(msg)
            used_label = "placeholder"
            used_code = code_base

    if not svg:
        logger.error(f"❌ Mermaid Rendering Failed. Input Code:\n{code}")
        # Return a generic placeholder to keep UI stable
        svg = _svg_placeholder("Unexpected renderer state; using placeholder")
    
    # Cache the successful result (avoid caching placeholders)
    if svg and "Mermaid render unavailable" not in svg:
        label_for_cache = (locals().get("used_label") or "base")
        code_for_cache = (locals().get("used_code") or code_base)
        cache_key = _attempt_cache_key(label_for_cache, code_for_cache)
        _mermaid_cache[cache_key] = svg
        # LRU: mark as recently used
        try:
            _mermaid_cache.move_to_end(cache_key)
        except Exception:
            pass

        if redis is not None:
            try:
                ttl = int(getattr(settings, "redis_cache_ttl_seconds", 3600) or 3600)
                ttl = max(60, ttl)
                await redis.set(_redis_cache_key(cache_key), svg, ex=ttl)
            except Exception:
                # Fail-open: never break diagram rendering if Redis is down
                pass
    else:
        cache_key = "nocache"
    
    # Limit cache size to prevent memory issues (LRU eviction)
    try:
        while len(_mermaid_cache) > _MERMAID_CACHE_MAX:
            _mermaid_cache.popitem(last=False)
    except Exception:
        if len(_mermaid_cache) > _MERMAID_CACHE_MAX:
            keys_to_remove = list(_mermaid_cache.keys())[:20]
            for k in keys_to_remove:
                del _mermaid_cache[k]
    
    if cache_key != "nocache":
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
    _rl=Depends(_rate_limit_render),
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
        logger.debug("[API] Request received: Generate Architecture")
        logger.debug("System: %s...", (request.system_description or "")[:50])
        logger.debug("Level: %s", request.user_level)
        
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
        
        logger.debug("Views selected: %s", [v.value for v in views_to_generate])
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
            detail="Architecture generation failed. Please try again."
        )


@router.get("/architecture/available_views")
async def get_available_views(_api_key: str = Depends(verify_api_key)):
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
    system_description: str = Query(..., min_length=10, max_length=2000),
    user_level: str = Query(default="mid", pattern=r"^(junior|mid|senior|architect)$"),
    _api_key: str = Depends(verify_api_key),
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
async def export_architecture_markdown(
    package: ArchitecturePackageOut,
    _api_key: str = Depends(verify_api_key),
):
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
    
    safe_name = re.sub(r'[^\w\s-]', '', package.system_name).strip().replace(' ', '_')[:100]
    
    return Response(
        content=markdown_content,
        media_type="text/markdown",
        headers={
            "Content-Disposition": f"attachment; filename=\"{safe_name}_architecture.md\""
        }
    )
