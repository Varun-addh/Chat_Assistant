from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
import httpx

from app.utils.security import verify_api_key


router = APIRouter()


@router.post("/render_mermaid")
async def render_mermaid(payload: dict, _: None = Depends(verify_api_key)):
    """Render Mermaid code to SVG via Kroki backend.

    Expected payload: { "code": "flowchart LR...", "theme": "default|dark|forest|neutral" }
    Returns raw SVG content.
    """
    code = (payload.get("code") or "").strip()
    if not code:
        raise HTTPException(status_code=400, detail="Missing 'code' in payload")

    # Basic guardrail: hard-limit size to avoid abuse
    if len(code) > 40_000:
        raise HTTPException(status_code=413, detail="Diagram too large")

    theme = (payload.get("theme") or "").strip() or "default"

    # Kroki mermaid rendering endpoint
    url = "https://kroki.io/mermaid/svg"

    # Kroki accepts either plain text body or JSON; use text to reduce overhead
    headers = {
        "Content-Type": "text/plain; charset=utf-8",
    }

    # Some themes are supported by Mermaid directly; inject theme directive if provided
    if theme and theme != "default" and not code.lstrip().startswith("%%{init"):
        # Prepend Mermaid init directive for theme; keep code intact otherwise
        code = f"%%{{init: {{ 'theme': '{theme}' }} }}%%\n" + code

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(url, content=code.encode("utf-8"), headers=headers)
            if resp.status_code != 200:
                raise HTTPException(status_code=502, detail=f"Kroki render failed: {resp.status_code}")
    except HTTPException:
        raise
    except Exception as exc:  # Network/timeout/etc
        raise HTTPException(status_code=502, detail=f"Render error: {exc}")

    svg = resp.text
    # Minimal sanity check
    if not svg.strip().startswith("<svg"):
        raise HTTPException(status_code=502, detail="Invalid SVG returned from renderer")

    return Response(content=svg, media_type="image/svg+xml")


