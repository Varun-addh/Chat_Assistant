"""
Rate limiting middleware and utilities
"""
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Tuple
from fastapi import Request, HTTPException, status
from collections import defaultdict
import asyncio
import logging
from contextlib import suppress
from app.models import User, UserTier, TIER_QUOTAS

logger = logging.getLogger(__name__)


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


# Global rate limiter instance
rate_limiter = InMemoryRateLimiter()


async def rate_limit_middleware(request: Request, call_next):
    """
    Middleware to enforce rate limits on API endpoints
    
    Extracts user from request.state (set by auth middleware)
    and checks rate limits based on user tier
    """
    # Skip rate limiting for certain paths
    skip_paths = [
        "/health",
        "/docs",
        "/openapi.json",
        "/auth/register",
        "/auth/login",
        "/auth/google",
        "/auth/google/callback",
    ]
    
    if any(request.url.path.startswith(path) for path in skip_paths):
        return await call_next(request)

    # Ensure the cleanup loop is running (safe no-op if already started)
    rate_limiter.start()
    
    # Get user from request state (set by auth middleware)
    user: Optional[User] = getattr(request.state, "user", None)
    
    # If no user, apply strict guest rate limiting
    limit_override: Optional[int] = None
    if not user:
        user_id = "guest"
        tier = UserTier.FREE
        # Apply stricter limits for guests
        limit_override = 10  # 10 requests per day for unauthenticated users
    else:
        user_id = user.id
        tier = user.tier
    
    # Check rate limit
    endpoint = request.url.path
    allowed, current_count, max_limit = await rate_limiter.check_rate_limit(
        user_id=user_id,
        endpoint=endpoint,
        tier=tier,
        limit_override=limit_override,
    )
    
    if not allowed:
        # Rate limit exceeded
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": "Rate limit exceeded",
                "current_usage": current_count,
                "limit": max_limit,
                "tier": tier,
                "message": f"You've made {current_count}/{max_limit} requests today. Upgrade to PRO for higher limits.",
            },
        )
    
    # Add rate limit headers to response
    response = await call_next(request)
    response.headers["X-RateLimit-Limit"] = str(max_limit)
    if max_limit == -1:
        response.headers["X-RateLimit-Remaining"] = "-1"
    else:
        response.headers["X-RateLimit-Remaining"] = str(max(0, max_limit - current_count))
    response.headers["X-RateLimit-Reset"] = str(int((datetime.now(timezone.utc) + timedelta(hours=24)).timestamp()))
    
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
