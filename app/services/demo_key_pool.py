"""Simple in-memory demo key pool shim.

This provides a minimal implementation of the `demo_key_pool` used across
the codebase. It sources keys from `settings.stratax_demo_api_keys` and
implements a small rotation + exhaustion mechanism.

The real project may use a more robust implementation (Redis, persistent
state, cooldowns). This shim is intentionally lightweight and safe for
local development and tests.
"""
from __future__ import annotations

import threading
import time
import logging
from typing import List, Optional

from app.config import settings

logger = logging.getLogger(__name__)


class DemoKeyPool:
    def __init__(self) -> None:
        # initialize from settings; copy to avoid mutating original
        self._lock = threading.Lock()
        self._keys: List[str] = list(getattr(settings, "stratax_demo_api_keys", []) or [])
        # exhausted metadata: key -> (timestamp, reason)
        self._exhausted: dict[str, tuple[float, str]] = {}
        # simple round-robin index
        self._idx = 0

    def keys(self) -> List[str]:
        """Return the list of configured demo keys (excluding exhausted ones)."""
        with self._lock:
            return [k for k in self._keys if k and k not in self._exhausted]

    def get_key(self, *, preferred: Optional[str] = None) -> Optional[str]:
        """Return a demo key.

        If `preferred` is provided and known/unexhausted, return it.
        Otherwise perform simple round-robin among available keys.
        Returns None if no keys configured or all are exhausted.
        """
        with self._lock:
            avail = [k for k in self._keys if k and k not in self._exhausted]
            if not avail:
                return None
            if preferred and preferred in avail:
                return preferred
            # round-robin
            if self._idx >= len(avail):
                self._idx = 0
            key = avail[self._idx]
            self._idx = (self._idx + 1) % len(avail)
            return key

    def mark_exhausted(self, key: str, *, reason: str = "") -> None:
        """Mark a key as exhausted/unavailable. Records timestamp + reason."""
        if not key:
            return
        with self._lock:
            if key in self._exhausted:
                return
            self._exhausted[key] = (time.time(), reason or "exhausted")
            logger.warning("[DEMO_KEY_POOL] Marked exhausted: %s (%s)", key, reason)

    # Small convenience for introspection in logs/tests
    def status(self) -> dict:
        with self._lock:
            return {
                "configured": list(self._keys),
                "available": [k for k in self._keys if k and k not in self._exhausted],
                "exhausted": {k: v for k, v in self._exhausted.items()},
            }


# Singleton instance used by the rest of the codebase
demo_key_pool = DemoKeyPool()
