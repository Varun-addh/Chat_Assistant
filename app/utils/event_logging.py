"""Event logging utilities.

Goal: create a minimal, stable event stream that can power learning loops.

Design principles:
- Safe-by-default: do not store raw text unless explicitly enabled.
- Stable IDs: use HMAC-SHA256 when analytics_hmac_key is configured.
- Non-blocking for product flows: swallow DB errors (missing table, locks, etc.).
"""

from __future__ import annotations

import hmac
import hashlib
import re
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy.orm import Session

from app.config import settings
from app.models import EventRecord


_WS_RE = re.compile(r"\s+")


def _normalize_text(text: str) -> str:
    # Keep normalization conservative to avoid surprising collisions.
    return _WS_RE.sub(" ", (text or "").strip())


def stable_hash(text: str) -> str:
    """Return a stable, anonymized hash for a text string.

    If settings.analytics_hmac_key is set, we use HMAC-SHA256 so the hash is not
    reversible even if the input space is small.

    Otherwise we fall back to plain SHA256.
    """
    normalized = _normalize_text(text)
    key = (settings.analytics_hmac_key or "").strip()
    if key:
        digest = hmac.new(key.encode("utf-8"), normalized.encode("utf-8"), hashlib.sha256).hexdigest()
    else:
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return digest


def stable_question_id(question_text: str) -> str:
    """Stable ID for a user-entered question.

    For now this is the same as stable_hash(normalized_question). Later we can
    expand to include domain/difficulty/company once those are explicit.
    """
    return stable_hash(question_text)


def _maybe_text_preview(text: str) -> Optional[str]:
    if not settings.analytics_store_raw_text:
        return None
    preview_len = int(getattr(settings, "analytics_text_preview_len", 120) or 120)
    preview_len = max(0, min(2000, preview_len))
    cleaned = (text or "").strip()
    if preview_len == 0:
        return cleaned
    return cleaned[:preview_len]


def track_event(
    db: Session,
    *,
    user_id: str,
    session_id: Optional[str],
    event_type: str,
    question_text: Optional[str] = None,
    extra: Optional[dict[str, Any]] = None,
) -> Optional[int]:
    """Persist a structured event to the database.

    Returns inserted row id if successful, else None.

    This function must never raise in request handlers.
    """
    try:
        qid = stable_question_id(question_text) if question_text else None
        chash = stable_hash(question_text) if question_text else None

        payload: dict[str, Any] = dict(extra or {})
        if question_text is not None:
            payload.setdefault("question_preview", _maybe_text_preview(question_text))

        rec = EventRecord(
            user_id=user_id or "unknown",
            session_id=session_id,
            event_type=event_type,
            question_id=qid,
            content_hash=chash,
            extra_data=payload,
            timestamp=datetime.now(timezone.utc),
        )
        db.add(rec)
        db.commit()
        try:
            return int(rec.id)  # type: ignore[arg-type]
        except Exception:
            return None
    except Exception:
        try:
            db.rollback()
        except Exception:
            pass
        return None
