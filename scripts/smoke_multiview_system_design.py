"""Smoke test: multi-view system design output contract.

Validates that:
- /api/session works
- /api/question (architecture_mode=multi-view) returns 5 views
- Each view contains a Mermaid block
- Layered views contain Layer 1..5 headings + Final Summary
- /api/render_mermaid works (and is not rate-limited)

Usage:
- Start backend
- Run this script (it will default to http://localhost:7860)

Optional env vars:
- BACKEND_BASE_URL (e.g. http://localhost:7860)
- X_API_KEY (Groq key) or X_GEMINI_KEY (Gemini key)
- AUTH_BEARER (Authorization token value, without 'Bearer ')

Note: This script is intentionally lightweight and prints a crisp verdict.
"""

from __future__ import annotations

import os
import re
import sys
from typing import Dict, List, Tuple

import requests


def _headers() -> Dict[str, str]:
    h: Dict[str, str] = {"Content-Type": "application/json"}

    x_api_key = (os.getenv("X_API_KEY") or "").strip()
    x_gemini_key = (os.getenv("X_GEMINI_KEY") or "").strip()
    bearer = (os.getenv("AUTH_BEARER") or "").strip()

    if x_api_key:
        h["X-API-Key"] = x_api_key
    if x_gemini_key:
        h["X-Gemini-Key"] = x_gemini_key
    if bearer:
        h["Authorization"] = f"Bearer {bearer}"

    return h


def _post(base_url: str, path: str, payload: dict) -> requests.Response:
    url = base_url.rstrip("/") + path
    resp = requests.post(url, json=payload, headers=_headers(), timeout=120)
    return resp


def _get(base_url: str, path: str) -> requests.Response:
    url = base_url.rstrip("/") + path
    resp = requests.get(url, headers=_headers(), timeout=60)
    return resp


def _extract_views(answer: str) -> List[Tuple[str, str]]:
    """Return list of (view_title_line, view_body) split by '## ' headings."""
    if not answer:
        return []

    # We expect: "## <title>" per view
    parts = re.split(r"(?m)^##\s+", answer)
    if len(parts) <= 1:
        return []

    views: List[Tuple[str, str]] = []
    for chunk in parts[1:]:
        # chunk begins with title line
        lines = chunk.splitlines()
        if not lines:
            continue
        title = lines[0].strip()
        body = "\n".join(lines[1:]).strip()
        views.append((title, body))
    return views


def _has_mermaid(body: str) -> bool:
    return bool(re.search(r"```mermaid\s*[\s\S]*?```", body, flags=re.IGNORECASE))


def _has_layers_1_to_5(body: str) -> bool:
    for i in range(1, 6):
        if not re.search(rf"(?m)^###\s*Layer\s*{i}\b", body, flags=re.IGNORECASE):
            return False
    return True


def _has_final_summary(body: str) -> bool:
    return bool(re.search(r"(?m)^###\s*Final\b.*Summary\b", body, flags=re.IGNORECASE))


def _is_system_overview(title: str) -> bool:
    t = title.lower()
    return "system overview" in t


def main() -> int:
    base_url = (os.getenv("BACKEND_BASE_URL") or "http://localhost:7860").strip()

    print(f"Base URL: {base_url}")

    # 1) Create session
    r = _post(base_url, "/api/session", {})
    if r.status_code != 200:
        print("❌ POST /api/session failed")
        print(r.status_code, r.text)
        return 1

    session_id = r.json().get("session_id")
    if not session_id:
        print("❌ Missing session_id")
        print(r.text)
        return 1

    print(f"✅ Session: {session_id}")

    # 2) Ask multi-view system design
    # Use an explicit trigger phrase to avoid any ambiguity in architecture detection.
    question = (
        os.getenv("SMOKE_QUESTION")
        or "System design: Build Facebook Marketplace with listings, search, chat, payments, and moderation at scale"
    )
    architecture_mode = (os.getenv("ARCH_MODE") or "multi-view").strip()
    payload = {
        "session_id": session_id,
        "question": question,
        "architecture_mode": architecture_mode,
        "stream": False,
        "save_to_history": False,
    }

    r = _post(base_url, "/api/question", payload)
    if r.status_code != 200:
        print("❌ POST /api/question failed")
        print(r.status_code, r.text)
        return 1

    resp_json = r.json() or {}
    answer = resp_json.get("answer") or ""

    # Quick early diagnostic
    if isinstance(answer, str) and "Choose your preferred architecture format" in answer:
        print("❌ Backend asked for architecture mode selection (arch_mode was not applied).")
        print("Answer:")
        print(answer)
        return 1

    views = _extract_views(answer)
    if len(views) != 5:
        print(f"❌ Expected 5 views, got {len(views)}")
        print("Response keys:", sorted(list(resp_json.keys())))
        # Print a tiny sample for debugging
        print("--- Answer preview ---")
        print((answer or "")[:1200])
        return 1

    print("✅ Got 5 views")

    failures: List[str] = []
    for idx, (title, body) in enumerate(views, 1):
        if not _has_mermaid(body):
            failures.append(f"View {idx} '{title}': missing mermaid block")

        if _is_system_overview(title):
            # System overview uses bullets + Goal, not layers
            if not re.search(r"(?m)^- ", body):
                failures.append(f"View {idx} '{title}': expected bullets")
            if not re.search(r"(?m)^Goal:\s+", body):
                failures.append(f"View {idx} '{title}': missing Goal:")
        else:
            if not _has_layers_1_to_5(body):
                failures.append(f"View {idx} '{title}': missing Layer 1..5 headings")
            if not _has_final_summary(body):
                failures.append(f"View {idx} '{title}': missing Final Summary heading")

    if failures:
        print("❌ Contract validation failed:")
        for f in failures:
            print(" -", f)
        return 1

    print("✅ Contract validation passed (mermaid + structure + 5-layer explanations)")

    # 3) Quick render smoke (should never be rate-limited)
    mermaid_code = "flowchart TD\n  A[Client] --> B[API]\n  B --> C[(DB)]"
    r = _post(base_url, "/api/render_mermaid", {"code": mermaid_code, "theme": "default"})
    if r.status_code != 200:
        print("❌ POST /api/render_mermaid failed")
        print(r.status_code, r.text[:500])
        return 1

    print("✅ Mermaid render endpoint OK")

    return 0


if __name__ == "__main__":
    sys.exit(main())
