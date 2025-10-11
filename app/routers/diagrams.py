from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
import httpx
import re

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

    theme = (payload.get("theme") or "").strip() or "default"

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
        # Prepend Mermaid init directive for theme; keep code intact otherwise
        code = f"%%{{init: {{ 'theme': '{theme}' }} }}%%\n" + code
    
    # Basic syntax validation
    if not code.strip():
        raise HTTPException(status_code=400, detail="Empty Mermaid code")
    
    # Check for basic Mermaid structure
    if not re.search(r'^(flowchart|sequenceDiagram|classDiagram|erDiagram|stateDiagram|gantt|journey|pie|mindmap|timeline)\s+', code, re.MULTILINE):
        raise HTTPException(status_code=400, detail="Invalid Mermaid syntax: missing diagram type declaration")

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


