"""Redis client helper.

We keep this small and defensive:
- If REDIS_URL isn't set, Redis is disabled.
- If Redis is down, callers can choose to fail-open.

Uses redis-py asyncio client (redis.asyncio).
"""

from __future__ import annotations

import logging
from typing import Optional

from app.config import settings

logger = logging.getLogger(__name__)

try:
    import redis.asyncio as redis_async
    from redis.asyncio import Redis
    from redis.exceptions import RedisError

    _REDIS_IMPORT_OK = True
except Exception:  # pragma: no cover
    redis_async = None  # type: ignore[assignment]
    Redis = object  # type: ignore[misc,assignment]

    class RedisError(Exception):
        pass

    _REDIS_IMPORT_OK = False


_redis: Optional["Redis"] = None


def redis_enabled() -> bool:
    return bool(getattr(settings, "redis_url", None)) and _REDIS_IMPORT_OK


async def get_redis() -> Optional["Redis"]:
    """Return a singleton Redis client if enabled, else None.

    Note: We do not ping on every call; failures are handled by callers.
    """

    global _redis

    url = (getattr(settings, "redis_url", None) or "").strip()
    if not url or not _REDIS_IMPORT_OK:
        return None

    if _redis is None:
        try:
            _redis = redis_async.from_url(url, decode_responses=True)
        except Exception as e:
            logger.warning("Redis init failed: %s", e)
            _redis = None
            return None

    return _redis


async def close_redis() -> None:
    global _redis
    if _redis is None:
        return
    try:
        await _redis.close()
    except Exception:
        pass
    _redis = None
