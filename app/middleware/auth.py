"""
User authentication middleware for Stratax AI

Extracts user from JWT token and attaches to request state
for rate limiting, quotas, and personalized features.
"""
from __future__ import annotations

from fastapi import Request
from typing import Optional
import logging
import hashlib
import re

logger = logging.getLogger(__name__)


def _looks_like_jwt(token: str) -> bool:
    """Heuristic: JWTs are dot-separated (header.payload.signature).

    Important: Authorization is also reused for LLM keys in some flows.
    We must not attempt JWT decode for non-JWT bearer tokens.
    """
    t = (token or "").strip()
    return t.count(".") == 2


_GUEST_ID_COOKIE_NAME = "stratax_guest_id"
_GUEST_ID_HEADER_CANDIDATES = (
    "x-stratax-guest-id",
    "x-client-id",
)


def _is_safe_guest_id(value: str) -> bool:
    """Validate an identifier that will be used as a user_id and filesystem segment."""
    v = (value or "").strip()
    if not v:
        return False
    if len(v) > 64:
        return False
    # Allow simple, path-safe chars only.
    return re.fullmatch(r"[A-Za-z0-9_\-]+", v) is not None


def _extract_user_from_token(token: str):
    """
    Extract user from JWT token
    
    Returns User object or None
    """
    try:
        # Use the shared auth decoder so claims/expiry validation is consistent
        # across middleware and route dependencies.
        from app.auth import decode_access_token
        from app.database import get_db_context
        from app.models import User

        token_data = decode_access_token(token)
        user_id = getattr(token_data, "user_id", None)
        
        if not user_id:
            return None
        
        # Get user from database
        with get_db_context() as db:
            user = db.query(User).filter(User.id == user_id).first()
            if user is None:
                return None
            if not getattr(user, "is_active", True):
                return None
            return user
    
    except Exception as e:
        logger.debug(f"Token extraction failed: {e}")
        return None


async def user_auth_middleware(request: Request, call_next):
    """
    Middleware to extract and attach user to request state
    
    Checks for Authorization header with Bearer token (JWT)
    Attaches user object to request.state.user for downstream use
    
    If no valid token, request proceeds as guest (rate limited)
    """
    user = None

    # If we generate/choose a guest id that should be persisted client-side,
    # set it on the response after call_next.
    guest_id_to_persist: Optional[str] = None

    def _get_client_ip() -> str:
        # Prefer proxy-aware header if present (first hop)
        xff = request.headers.get("x-forwarded-for")
        if xff:
            # XFF can contain a chain: client, proxy1, proxy2...
            ip = xff.split(",")[0].strip()
            if ip:
                return ip
        if request.client and request.client.host:
            return request.client.host
        return "unknown"

    def _guest_user_id() -> str:
        nonlocal guest_id_to_persist

        # 1) Prefer explicit client-provided stable id (for SPAs / cross-origin clients).
        for header_name in _GUEST_ID_HEADER_CANDIDATES:
            h = request.headers.get(header_name)
            if h and _is_safe_guest_id(h):
                return h

        # 2) Prefer cookie-stored stable id (best for same-origin UI).
        cookie_val = request.cookies.get(_GUEST_ID_COOKIE_NAME)
        if cookie_val and _is_safe_guest_id(cookie_val):
            return cookie_val

        # 3) Backward-compatible fallback: hash IP + UA so existing histories remain reachable.
        # IMPORTANT:
        # - Do NOT use a single global "guest" bucket (it makes all guests share sessions/quota)
        # - Avoid storing raw IPs in session paths; hash to a short stable ID
        # - Windows filesystem does not allow ':' in directory names
        ip = _get_client_ip()
        ua = request.headers.get("user-agent", "")
        raw = f"{ip}|{ua}".encode("utf-8", errors="ignore")
        digest = hashlib.sha256(raw).hexdigest()[:16]
        legacy = f"guest_{digest}"

        # Persist this identifier so future requests don't depend on IP/UA.
        guest_id_to_persist = legacy
        return legacy
    
    # Check Authorization header (JWT token).
    # NOTE: Authorization may also contain an LLM API key for unauthenticated flows.
    # Only attempt JWT decoding when the token looks like a JWT.
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ", 1)[1].strip() if " " in auth_header else ""
        if token and _looks_like_jwt(token):
            user = _extract_user_from_token(token)
    
    # Attach user to request state (or None for guests)
    request.state.user = user
    
    # For backwards compatibility, also set user_id
    if user:
        request.state.user_id = user.id
        logger.debug(f"Authenticated request: {user.email} ({user.tier})")
    else:
        request.state.user_id = _guest_user_id()
        logger.debug("Guest request (unauthenticated)")
    
    response = await call_next(request)

    # If this was a guest request and we don't yet have a stable id persisted,
    # set a cookie (and a header for easier debugging / SPA integration).
    if user is None:
        if guest_id_to_persist and _is_safe_guest_id(guest_id_to_persist):
            try:
                secure = (request.url.scheme or "").lower() == "https"
                response.set_cookie(
                    key=_GUEST_ID_COOKIE_NAME,
                    value=guest_id_to_persist,
                    max_age=60 * 60 * 24 * 365,  # 1 year
                    httponly=True,
                    samesite="lax",
                    secure=secure,
                    path="/",
                )
                response.headers["X-Stratax-Guest-Id"] = guest_id_to_persist
            except Exception:
                # Never fail the request just because we couldn't set a cookie.
                pass
        else:
            # If we didn't persist a legacy id (e.g., client already supplied one),
            # still echo the resolved id for easier client-side adoption.
            resolved = getattr(request.state, "user_id", None)
            if isinstance(resolved, str) and _is_safe_guest_id(resolved):
                response.headers["X-Stratax-Guest-Id"] = resolved

    return response


def get_user_id_from_request(request: Request) -> Optional[str]:
    """
    Helper function to get user_id from request state
    
    Args:
        request: FastAPI request object
        
    Returns:
        user_id if authenticated, None otherwise
    """
    return getattr(request.state, "user_id", None)

