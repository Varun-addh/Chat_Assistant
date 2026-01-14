"""Time helpers.

We standardize on timezone-aware UTC datetimes to avoid `datetime.utcnow()`
(naive datetime) deprecation warnings and subtle timezone bugs.
"""

from __future__ import annotations

from datetime import datetime, timezone


def utcnow() -> datetime:
	"""Return a timezone-aware UTC timestamp."""
	return datetime.now(timezone.utc)
