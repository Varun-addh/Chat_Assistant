"""
Rate limiting middleware and utilities
"""
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Tuple
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from collections import defaultdict
import asyncio
import logging
from contextlib import suppress
import uuid
import hashlib
from app.models import User, UserTier, TIER_QUOTAS
from app.utils.demo_mode import (
    DEMO_ERROR_PAYLOAD,
    demo_session_id,
    extract_user_provided_api_key,
    infer_user_type,
)
from app.config import settings

logger = logging.getLogger(__name__)


# ------------------------------ Demo mode quotas ------------------------------
# Demo is intended to show value quickly while being cost-capped.
# All demo limits apply within a short rolling window (session duration).
DEMO_WINDOW_MINUTES = 30

# Feature quotas within the demo window
DEMO_LIMIT_QUESTIONS = 5  # spec: 5–10
DEMO_LIMIT_SYSTEM_DESIGNS = 1
DEMO_LIMIT_PRACTICE_ROUNDS = 1
DEMO_LIMIT_MOCK_INTERVIEWS = 1


def _demo_feature_limit_for_path(path: str) -> tuple[Optional[str], Optional[int]]:
    """Return (feature_name, limit) for a metered path in demo mode."""
    p = path or ""
    if p.startswith("/api/question"):
        return "questions", DEMO_LIMIT_QUESTIONS
    if p.startswith("/api/generate_architecture") or p.startswith("/api/diagrams"):
        return "designs", DEMO_LIMIT_SYSTEM_DESIGNS
    # Practice mode: only gate starting a round/session; let in-session calls proceed
    if p.startswith("/api/practice/interview/start-round") or p.startswith("/api/practice/interview/start"):
        return "practice_rounds", DEMO_LIMIT_PRACTICE_ROUNDS
    # Code execution is expensive; keep demo conservative.
    if p.startswith("/api/code/execute"):
        return "code_exec", 10
    if p.startswith("/api/mock-interview"):
        return "mock_interviews", DEMO_LIMIT_MOCK_INTERVIEWS
    # Default demo limit for other metered endpoints (keep conservative)
    return "requests", 10


async def _compute_demo_remaining(demo_user_id: str) -> dict:
    usage = await rate_limiter.get_usage(demo_user_id)

    questions_used = int(usage.get("demo:questions", 0))
    designs_used = int(usage.get("demo:designs", 0))
    practice_used = int(usage.get("demo:practice_rounds", 0))

    return {
        "questions": max(0, DEMO_LIMIT_QUESTIONS - questions_used),
        "designs": max(0, DEMO_LIMIT_SYSTEM_DESIGNS - designs_used),
        "practice_rounds": max(0, DEMO_LIMIT_PRACTICE_ROUNDS - practice_used),
    }


class InMemoryRateLimiter:
    """
    In-memory rate limiter (use Redis in production for distributed systems)
    
    Tracks requests per user per endpoint with sliding window
    """
    
    def __init__(self):
        # Structure: {user_id: {endpoint: [(timestamp, count), ...]}}
        self._records: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
        self._lock = asyncio.Lock()

        # Cleanup task is started when an event loop is running (e.g. app startup)
        self._cleanup_task: asyncio.Task | None = None

    def start(self) -> None:
        """Start background cleanup loop if running in an event loop."""
        if self._cleanup_task is not None and not self._cleanup_task.done():
            return

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop (e.g., module import time). We'll start later.
            return

        self._cleanup_task = loop.create_task(self._cleanup_loop())

    async def shutdown(self) -> None:
        """Stop background cleanup loop."""
        if self._cleanup_task is None:
            return

        self._cleanup_task.cancel()
        with suppress(asyncio.CancelledError):
            await self._cleanup_task
        self._cleanup_task = None
    
    async def _cleanup_loop(self):
        """Remove old records periodically"""
        while True:
            await asyncio.sleep(300)  # 5 minutes
            await self._cleanup_old_records()
    
    async def _cleanup_old_records(self):
        """Remove records older than 24 hours"""
        async with self._lock:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
            
            for user_id in list(self._records.keys()):
                for endpoint in list(self._records[user_id].keys()):
                    # Filter out old timestamps
                    self._records[user_id][endpoint] = [
                        (ts, count) for ts, count in self._records[user_id][endpoint]
                        if ts > cutoff
                    ]
                    
                    # Remove empty endpoints
                    if not self._records[user_id][endpoint]:
                        del self._records[user_id][endpoint]
                
                # Remove empty users
                if not self._records[user_id]:
                    del self._records[user_id]
    
    async def check_rate_limit(
        self,
        user_id: str,
        endpoint: str,
        tier: UserTier,
        window_minutes: int = 1440,  # 24 hours
        limit_override: Optional[int] = None,
    ) -> Tuple[bool, int, int]:
        """
        Check if user has exceeded rate limit
        
        Returns:
            (allowed, current_count, limit)
        """
        async with self._lock:
            now = datetime.now(timezone.utc)
            window_start = now - timedelta(minutes=window_minutes)
            
            # Get tier quota
            tier_quota = TIER_QUOTAS.get(tier, TIER_QUOTAS[UserTier.FREE])
            limit = limit_override if limit_override is not None else tier_quota["daily_api_calls"]
            
            # Unlimited tier
            if limit == -1:
                return True, 0, -1
            
            # Get user's request history for this endpoint
            records = self._records[user_id][endpoint]
            
            # Remove old records outside window
            records = [(ts, count) for ts, count in records if ts > window_start]
            
            # Count requests in window
            current_count = sum(count for _, count in records)
            
            # Check if limit exceeded
            allowed = current_count < limit
            
            # Add current request
            if allowed:
                records.append((now, 1))
                self._records[user_id][endpoint] = records
                current_count += 1
            
            return allowed, current_count, limit
    
    async def get_usage(self, user_id: str, endpoint: str = None) -> Dict[str, int]:
        """Get current usage stats for user"""
        async with self._lock:
            if endpoint:
                records = self._records[user_id].get(endpoint, [])
                return {endpoint: sum(count for _, count in records)}
            else:
                # All endpoints
                usage = {}
                for ep, records in self._records[user_id].items():
                    usage[ep] = sum(count for _, count in records)
                return usage


class RedisRateLimiter:
    """Redis-backed fixed-window rate limiter.

    Why fixed-window? It's fast and atomic with a small Lua script.
    This is meant to make multi-worker and multi-instance deployments safe.

    Behavior notes:
    - We mimic InMemoryRateLimiter's public API.
    - On Redis errors, we default to fail-open (configurable).
    """

    def __init__(self, *, redis_url: str, key_prefix: str = "stratax", fail_open: bool = True):
        self.redis_url = (redis_url or "").strip()
        self.key_prefix = (key_prefix or "stratax").strip() or "stratax"
        self.fail_open = bool(fail_open)
        self._redis = None
        self._import_ok = False

        try:
            import redis.asyncio as redis_async  # type: ignore

            self._redis = redis_async.from_url(self.redis_url, decode_responses=True)
            self._import_ok = True
        except Exception as e:
            logger.warning("Redis rate limiter unavailable (falling back): %s", e)
            self._redis = None
            self._import_ok = False

        # Atomic conditional increment script.
        # - If current >= limit: return current (no increment)
        # - Else: INCR and set EXPIRE on first hit
        self._lua = (
            "local current = redis.call('GET', KEYS[1]) "
            "if not current then current = 0 end "
            "if tonumber(current) >= tonumber(ARGV[1]) then return tonumber(current) end "
            "local newv = redis.call('INCR', KEYS[1]) "
            "if newv == 1 then redis.call('EXPIRE', KEYS[1], tonumber(ARGV[2])) end "
            "return tonumber(newv)"
        )

    def start(self) -> None:
        # No cleanup loop needed; Redis TTL handles expiry.
        return

    async def shutdown(self) -> None:
        if not self._redis:
            return
        try:
            await self._redis.close()
        except Exception:
            pass

    def _endpoint_hash(self, endpoint: str) -> str:
        ep = (endpoint or "").encode("utf-8", errors="ignore")
        return hashlib.sha1(ep).hexdigest()

    def _count_key(self, *, user_id: str, endpoint: str, window_seconds: int) -> str:
        h = self._endpoint_hash(endpoint)
        return f"{self.key_prefix}:rl:cnt:{user_id}:{h}:{int(window_seconds)}"

    def _endpoints_set_key(self, *, user_id: str, window_seconds: int) -> str:
        return f"{self.key_prefix}:rl:eps:{user_id}:{int(window_seconds)}"

    async def check_rate_limit(
        self,
        user_id: str,
        endpoint: str,
        tier: UserTier,
        window_minutes: int = 1440,
        limit_override: Optional[int] = None,
    ) -> Tuple[bool, int, int]:
        """Check + conditionally consume one unit."""

        # Get tier quota
        tier_quota = TIER_QUOTAS.get(tier, TIER_QUOTAS[UserTier.FREE])
        limit = limit_override if limit_override is not None else tier_quota["daily_api_calls"]

        # Unlimited tier
        if limit == -1:
            return True, 0, -1

        if not self._redis or not self._import_ok:
            # If Redis isn't usable, fail-open by default.
            if self.fail_open:
                return True, 0, limit
            return False, limit, limit

        window_seconds = int(max(1, window_minutes * 60))
        key = self._count_key(user_id=user_id, endpoint=endpoint, window_seconds=window_seconds)
        set_key = self._endpoints_set_key(user_id=user_id, window_seconds=window_seconds)

        try:
            # Run atomic check+incr
            new_count = await self._redis.eval(self._lua, 1, key, int(limit), int(window_seconds))

            # Track endpoints used (for usage reporting) with the same TTL.
            # This is best-effort; it does not affect the allow/deny decision.
            try:
                pipe = self._redis.pipeline()
                pipe.sadd(set_key, endpoint)
                pipe.expire(set_key, window_seconds)
                await pipe.execute()
            except Exception:
                pass

            allowed = int(new_count) <= int(limit)
            return allowed, int(new_count), int(limit)
        except Exception as e:
            logger.warning("Redis rate limit error (fail_open=%s): %s", self.fail_open, e)
            if self.fail_open:
                return True, 0, int(limit)
            return False, int(limit), int(limit)

    async def get_usage(self, user_id: str, endpoint: str = None) -> Dict[str, int]:
        """Get usage for endpoints tracked in the current known windows.

        We avoid SCAN for safety/perf and instead maintain a set of endpoints
        per user per window.
        """

        if not self._redis or not self._import_ok:
            return {}

        # We only use a small number of windows in this app.
        windows = [DEMO_WINDOW_MINUTES * 60, 86400]

        async def _get_for_window(ws: int) -> Dict[str, int]:
            set_key = self._endpoints_set_key(user_id=user_id, window_seconds=ws)
            eps = []
            try:
                eps = await self._redis.smembers(set_key)
            except Exception:
                eps = []

            if endpoint:
                eps = [endpoint] if endpoint in eps else [endpoint]

            if not eps:
                return {}

            pipe = self._redis.pipeline()
            for ep in eps:
                pipe.get(self._count_key(user_id=user_id, endpoint=ep, window_seconds=ws))
            vals = await pipe.execute()

            out: Dict[str, int] = {}
            for ep, v in zip(eps, vals):
                try:
                    if v is None:
                        continue
                    out[str(ep)] = int(v)
                except Exception:
                    continue
            return out

        merged: Dict[str, int] = {}
        for ws in windows:
            try:
                part = await _get_for_window(int(ws))
            except Exception:
                part = {}
            for k, v in part.items():
                # Prefer the larger count if the same endpoint appears in multiple windows
                merged[k] = max(int(merged.get(k, 0)), int(v))

        return merged


# Global rate limiter instance
_redis_url = (getattr(settings, "redis_url", None) or "").strip()
if _redis_url:
    rate_limiter = RedisRateLimiter(
        redis_url=_redis_url,
        key_prefix=getattr(settings, "redis_key_prefix", "stratax"),
        fail_open=bool(getattr(settings, "redis_fail_open", True)),
    )
else:
    rate_limiter = InMemoryRateLimiter()


async def rate_limit_middleware(request: Request, call_next):
    """
    Middleware to enforce rate limits on API endpoints
    
    Extracts user from request.state (set by auth middleware)
    and checks rate limits based on user tier
    """
    path = request.url.path

    # Skip rate limiting for certain paths.
    #
    # Rationale:
    # - Health/docs/auth flows should never be blocked.
    # - Mermaid rendering is a cheap UI helper; rate-limiting it breaks UX.
    skip_paths = [
        "/health",
        "/docs",
        "/openapi.json",
        "/auth/register",
        "/auth/login",
        "/auth/google",
        "/auth/google/callback",
        "/api/render_mermaid",
        "/api/sessions",
        "/api/session",
        # Mock interview history reads are cheap and are frequently polled by UI.
        # Do not meter these under demo/mock-interview quotas.
        "/api/mock-interview/history",
        # Practice Mode dashboard endpoints are cheap DB reads and are frequently polled by UI.
        # Never rate-limit these, especially for demo users.
        "/api/practice/progress",
        "/api/practice/insights",
    ]

    if any(path.startswith(p) for p in skip_paths):
        return await call_next(request)

    # Allowlist model: only meter expensive, LLM-backed routes.
    # Everything else is intentionally unmetered to keep UX stable.
    metered_prefixes = [
        "/api/question",  # chat + system design generation
        "/api/evaluate",  # evaluation uses LLM
        "/api/generate_architecture",  # multi-view architecture package generation
        "/api/practice",  # practice mode uses LLM
        "/api/mock-interview",  # mock interview uses LLM
    ]

    if not any(path.startswith(p) for p in metered_prefixes):
        return await call_next(request)

    # Ensure the cleanup loop is running (safe no-op if already started)
    rate_limiter.start()

    # Auth middleware normally sets a stable per-client id (guest_<hash>) even
    # when unauthenticated. Some unit tests mount only this middleware, so we
    # create a stable per-app fallback id to avoid cross-test coupling.
    if getattr(request.state, "user_id", None) is None:
        fallback = getattr(request.app.state, "_fallback_guest_id", None)
        if not fallback:
            fallback = f"guest_{uuid.uuid4().hex[:16]}"
            setattr(request.app.state, "_fallback_guest_id", fallback)
        setattr(request.state, "user_id", fallback)
    
    # Get user from request state (set by auth middleware)
    user: Optional[User] = getattr(request.state, "user", None)

    # Demo vs registered inference (demo is unauth + no API key headers)
    provided_key = extract_user_provided_api_key(
        request,
        x_api_key=request.headers.get("X-API-Key"),
        x_gemini_key=request.headers.get("X-Gemini-Key"),
        authorization=request.headers.get("Authorization"),
    )
    user_type = infer_user_type(request, user=user, user_provided_key=provided_key)
    has_user_key = bool(provided_key)
    demo_key_pool_enabled = settings.is_demo_key_pool_enabled()

    # Debug log for confirming demo-mode behavior (no secrets logged)
    if (path or "").startswith("/api/question"):
        logger.info(
            "[DEMO_MODE] path=%s user_type=%s auth=%s user_key_header=%s demo_key_pool_enabled=%s",
            path,
            user_type,
            bool(user),
            has_user_key,
            demo_key_pool_enabled,
        )

    limit_override: Optional[int] = None
    window_minutes: int = 1440

    if user_type == "demo":
        user_id = demo_session_id(request)
        tier = UserTier.FREE
        window_minutes = DEMO_WINDOW_MINUTES

        if (path or "").startswith("/api/question"):
            logger.info(
                "[DEMO_MODE] demo_session_id=%s demo_window_minutes=%s",
                user_id,
                DEMO_WINDOW_MINUTES,
            )

        # Hard global demo budget guard (daily): protect platform spend.
        if int(getattr(settings, "demo_global_daily_request_limit", 0) or 0) > 0:
            g_allowed, g_count, g_limit = await rate_limiter.check_rate_limit(
                user_id="demo_global",
                endpoint="demo:global",
                tier=UserTier.FREE,
                window_minutes=1440,
                limit_override=int(settings.demo_global_daily_request_limit),
            )
            if not g_allowed:
                payload = {
                    "error": "DEMO_UNAVAILABLE",
                    "message": "Demo is currently at capacity. Please sign up and connect your own AI engine to continue.",
                    "current_usage": g_count,
                    "limit": g_limit,
                }
                reset_at = datetime.now(timezone.utc) + timedelta(hours=24)
                headers = {
                    "X-RateLimit-Limit": str(g_limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(reset_at.timestamp())),
                }
                return JSONResponse(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    content={"detail": payload},
                    headers=headers,
                )

        feature_name, feature_limit = _demo_feature_limit_for_path(path)
        endpoint = f"demo:{feature_name or 'requests'}"
        limit_override = feature_limit
    elif not user:
        # Unauthenticated but treated as "registered" (has user-provided key)
        # Keep basic abuse protection, but do not enforce demo session limits.
        user_id = getattr(request.state, "user_id", None) or "guest"
        tier = UserTier.FREE
        limit_override = 100
        endpoint = path
    else:
        user_id = user.id
        tier = user.tier
        endpoint = path

    allowed, current_count, max_limit = await rate_limiter.check_rate_limit(
        user_id=user_id,
        endpoint=endpoint,
        tier=tier,
        window_minutes=window_minutes,
        limit_override=limit_override,
    )
    
    if not allowed:
        # Rate limit exceeded.
        # IMPORTANT: do not raise HTTPException from middleware.
        # In some Starlette/anyio execution paths this can surface as an
        # ExceptionGroup and appear as a 500 to clients.
        now = datetime.now(timezone.utc)

        if user_type == "demo":
            remaining = await _compute_demo_remaining(user_id)
            payload = {
                **DEMO_ERROR_PAYLOAD,
                "user_type": "demo",
                "demo_remaining": remaining,
            }
            reset_at = now + timedelta(minutes=DEMO_WINDOW_MINUTES)
        else:
            payload = {
                "error": "Rate limit exceeded",
                "current_usage": current_count,
                "limit": max_limit,
                "tier": tier,
                "message": f"You've made {current_count}/{max_limit} requests today. Upgrade to PRO for higher limits.",
            }
            reset_at = now + timedelta(hours=24)

        headers = {
            "X-RateLimit-Limit": str(max_limit),
            "X-RateLimit-Remaining": "0" if max_limit != -1 else "-1",
            "X-RateLimit-Reset": str(int(reset_at.timestamp())),
        }
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={"detail": payload},
            headers=headers,
        )
    
    # Add rate limit headers to response
    response = await call_next(request)
    # Always expose user_type for client-side debugging / telemetry (non-secret).
    response.headers["X-Stratax-User-Type"] = user_type
    response.headers["X-Stratax-Demo-Key-Pool-Enabled"] = "1" if demo_key_pool_enabled else "0"
    response.headers["X-RateLimit-Limit"] = str(max_limit)
    if max_limit == -1:
        response.headers["X-RateLimit-Remaining"] = "-1"
    else:
        response.headers["X-RateLimit-Remaining"] = str(max(0, max_limit - current_count))
    reset_at = datetime.now(timezone.utc) + (
        timedelta(minutes=DEMO_WINDOW_MINUTES) if user_type == "demo" else timedelta(hours=24)
    )
    response.headers["X-RateLimit-Reset"] = str(int(reset_at.timestamp()))
    
    return response


async def check_feature_access(user: User, feature: str) -> bool:
    """
    Check if user's tier has access to a specific feature
    
    Usage:
        if not await check_feature_access(user, "practice_mode"):
            raise HTTPException(403, "Upgrade to PRO for Practice Mode")
    """
    tier_quota = TIER_QUOTAS.get(user.tier, TIER_QUOTAS[UserTier.FREE])
    allowed_features = tier_quota.get("features", [])
    
    # Enterprise has access to all
    if "all" in allowed_features:
        return True
    
    return feature in allowed_features
