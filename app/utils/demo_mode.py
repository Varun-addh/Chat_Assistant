"""Demo Mode utilities.

This module centralizes:
- user_type inference: demo vs registered
- demo session id generation (demo_<hash>)
- smart API key routing for demo/registered users

Rules (product contract):
- If no auth + no API key -> user_type=demo, use STRATAX_DEMO_API_KEY server-side
- Else -> user_type=registered, require a user-provided key (header or stored)

Security note:
- STRATAX_DEMO_API_KEY must never be returned to the client.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Literal, Optional

from fastapi import HTTPException, Request

from app.config import settings
from app.models import User
from app.services.demo_key_pool import demo_key_pool


UserType = Literal["demo", "registered"]


DEMO_ERROR_PAYLOAD = {
    "error": "DEMO_LIMIT_REACHED",
    "message": "Create a free account to continue",
}


def _client_ip(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for")
    if xff:
        ip = xff.split(",")[0].strip()
        if ip:
            return ip
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def _fingerprint_digest(request: Request, *, length: int = 16) -> str:
    ip = _client_ip(request)
    ua = request.headers.get("user-agent", "")
    raw = f"{ip}|{ua}".encode("utf-8", errors="ignore")
    return hashlib.sha256(raw).hexdigest()[: max(8, int(length))]


def demo_session_id(request: Request) -> str:
    """Return stable demo session id: demo_<hash>."""
    sid = getattr(request.state, "user_id", None)
    if sid:
        # Hash to avoid leaking internal guest ids and keep the demo id compact.
        return f"demo_{hashlib.sha256(str(sid).encode('utf-8', errors='ignore')).hexdigest()[:16]}"
    return f"demo_{_fingerprint_digest(request)}"


def _looks_like_llm_key(token: str) -> bool:
    t = (token or "").strip()
    if not t:
        return False
    # Groq keys commonly start with gsk_
    if t.startswith("gsk_"):
        return True
    # Gemini/Google keys often start with AIza
    if t.startswith("AIza") or t.startswith("ATza"):
        return True
    return False


def extract_user_provided_api_key(
    request: Request,
    *,
    x_api_key: Optional[str] = None,
    x_gemini_key: Optional[str] = None,
    authorization: Optional[str] = None,
) -> Optional[str]:
    """Extract a user-provided LLM API key from headers.

    We prefer explicit bridge headers (X-Gemini-Key, X-API-Key).

    Authorization is tricky because it may be a JWT (auth) OR an LLM key.
    We only accept Authorization as an LLM key when:
    - request.state.user is None (unauthenticated), and
    - the Bearer token looks like an LLM key prefix.
    """

    def _clean_header_val(v: Optional[str]) -> Optional[str]:
        t = (v or "").strip()
        if not t:
            return None
        if t.lower() in ("null", "undefined", "none"):
            return None
        return t

    groq_key = _clean_header_val(x_api_key)
    gemini_key = _clean_header_val(x_gemini_key)

    # Only accept bridge headers if they look like a supported LLM key.
    # This prevents accidental values (e.g. 'undefined') from disabling demo mode.
    key = gemini_key or groq_key
    if key and _looks_like_llm_key(key):
        return key

    auth_val = (authorization or "").strip()
    if not auth_val:
        return None

    if auth_val.lower() in ("null", "undefined", "none"):
        return None

    # If authenticated user exists, Authorization is assumed to be JWT.
    if getattr(request.state, "user", None) is not None:
        return None

    if auth_val.lower().startswith("bearer "):
        token = auth_val.split(" ", 1)[1].strip() if " " in auth_val else ""
        return token if _looks_like_llm_key(token) else None

    # Non-bearer auth header: accept only if it looks like an LLM key.
    return auth_val if _looks_like_llm_key(auth_val) else None


def infer_user_type(
    request: Request,
    *,
    user: Optional[User] = None,
    user_provided_key: Optional[str] = None,
) -> UserType:
    """Infer demo vs registered.

    demo = no auth AND no API key
    registered = everything else
    """

    if user is None:
        user = getattr(request.state, "user", None)

    if user_provided_key is None:
        user_provided_key = extract_user_provided_api_key(
            request,
            x_api_key=request.headers.get("X-API-Key"),
            x_gemini_key=request.headers.get("X-Gemini-Key"),
            authorization=request.headers.get("Authorization"),
        )

    if user is None and not user_provided_key:
        return "demo"
    return "registered"


def resolve_effective_api_key(
    request: Request,
    *,
    x_api_key: Optional[str] = None,
    x_gemini_key: Optional[str] = None,
    authorization: Optional[str] = None,
) -> tuple[UserType, Optional[str], bool]:
    """Resolve (user_type, api_key, needs_api_key).

    - demo: uses STRATAX_DEMO_API_KEY; needs_api_key=False
    - registered: requires user key (header or stored); needs_api_key=True if missing

    For authenticated users, we also consider stored keys on the user model.
    """

    user: Optional[User] = getattr(request.state, "user", None)
    provided = extract_user_provided_api_key(
        request,
        x_api_key=x_api_key,
        x_gemini_key=x_gemini_key,
        authorization=authorization,
    )

    utype = infer_user_type(request, user=user, user_provided_key=provided)

    if utype == "demo":
        # IMPORTANT: In local development, you usually want the UI to use your
        # developer/server key (GROQ_API_KEY / GEMINI_API_KEY), not consume demo keys.
        # Only use demo keys when explicitly enabled.
        if not settings.is_demo_key_pool_enabled():
            return "demo", None, False

        demo_key = demo_key_pool.get_key()
        if not demo_key:
            # Either misconfiguration OR all keys currently exhausted.
            raise HTTPException(
                status_code=503,
                detail="Demo temporarily unavailable (no demo keys available)",
            )
        return "demo", demo_key, False

    # registered
    if provided:
        return "registered", provided, False

    # If authenticated, fall back to stored user key
    if user is not None:
        provider = (settings.llm_provider or "groq").lower()
        if provider == "gemini":
            stored = (user.user_gemini_api_key or "").strip() or None
        else:
            stored = (user.user_groq_api_key or "").strip() or None
        if stored:
            return "registered", stored, False

    # Missing key
    return "registered", None, True


@dataclass(frozen=True)
class DemoUsage:
    questions_used: int
    designs_used: int
    practice_rounds_used: int


