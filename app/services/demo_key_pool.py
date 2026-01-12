"""Demo key pooling & rotation.

This is intentionally *in-memory* and best-effort.

Production note:
- If you run multiple backend instances, move this state to Redis.
- Key exhaustion signals come from provider errors (429/quota).

Goal:
- Developers should not babysit STRATAX demo keys.
- Backend automatically rotates through a pool, and temporarily cools down
  keys that appear rate-limited / quota-exhausted.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
import itertools
import logging

from app.config import settings

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class DemoKeyStatus:
    exhausted_until: Optional[datetime] = None

    def is_available(self, now: Optional[datetime] = None) -> bool:
        now = now or _utcnow()
        return self.exhausted_until is None or self.exhausted_until <= now


class DemoKeyPool:
    """A minimal rotating pool with cooldown support."""

    def __init__(self) -> None:
        self._idx = itertools.count()
        self._status: Dict[str, DemoKeyStatus] = {}

    def keys(self) -> List[str]:
        # Backward-compatible: support both STRATAX_DEMO_API_KEYS and STRATAX_DEMO_API_KEY
        keys = list(getattr(settings, "stratax_demo_api_keys", []) or [])
        if not keys:
            single = (getattr(settings, "stratax_demo_api_key", None) or "").strip()
            if single:
                keys = [single]
        # Normalize
        return [k.strip() for k in keys if (k or "").strip()]

    def get_key(self, *, preferred: Optional[str] = None) -> Optional[str]:
        """Return an available key.

        - If preferred is available, it is returned.
        - Otherwise, rotate through the pool and pick the first available.
        """
        keys = self.keys()
        if not keys:
            return None

        now = _utcnow()

        if preferred and preferred in keys:
            st = self._status.get(preferred) or DemoKeyStatus()
            if st.is_available(now):
                return preferred

        # Round-robin selection
        start = next(self._idx)
        for offset in range(len(keys)):
            k = keys[(start + offset) % len(keys)]
            st = self._status.get(k) or DemoKeyStatus()
            if st.is_available(now):
                return k

        # All exhausted
        return None

    def mark_exhausted(self, key: str, *, cooldown_seconds: int = 600, reason: str = "") -> None:
        """Put a key into cooldown.

        cooldown_seconds defaults to 10 minutes to avoid hammering a key that is
        currently rate limited.
        """
        if not key:
            return
        until = _utcnow() + timedelta(seconds=max(60, int(cooldown_seconds)))
        st = self._status.get(key) or DemoKeyStatus()
        st.exhausted_until = until
        self._status[key] = st

        r = (reason or "").strip()
        logger.warning(
            "[DEMO_KEY_POOL] Marked demo key exhausted until %s (%s...%s)%s",
            until.isoformat(),
            key[:4],
            key[-4:] if len(key) > 8 else "",
            f" reason={r}" if r else "",
        )


demo_key_pool = DemoKeyPool()
