from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from sqlalchemy.orm import Session

from app.models import ChatSession


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ChatSessionRepository:
    """Repository for persisted chat sessions.

    This persists the same shape as `SessionState` so we can swap storage backends
    without changing API contracts.
    """

    def get(self, db: Session, *, user_id: str, session_id: str) -> Optional[Dict[str, Any]]:
        row = (
            db.query(ChatSession)
            .filter(ChatSession.user_id == user_id)
            .filter(ChatSession.id == session_id)
            .first()
        )
        if not row:
            return None

        last_update = row.last_update
        if last_update is None:
            last_update = _utcnow()

        qna = list(row.qna or [])
        mirror_history: List[Dict[str, Any]] = []
        for it in qna:
            try:
                if not isinstance(it, dict):
                    continue
                meta = it.get("meta")
                if not isinstance(meta, dict):
                    continue
                if (meta.get("mode") or "").strip().lower() != "mirror":
                    continue
                report = meta.get("mirror_report")
                if not isinstance(report, dict):
                    continue
                mirror_history.append(
                    {
                        "question": it.get("question") or "",
                        "user_answer": meta.get("mirror_user_answer") or "",
                        "report": report,
                        "created_at": it.get("created_at") or "",
                    }
                )
            except Exception:
                continue

        return {
            "session_id": row.id,
            "qna": qna,
            "mirror_history": mirror_history,
            "partial_transcript": row.partial_transcript or "",
            "last_update": last_update.isoformat(),
            "profile_text": row.profile_text or "",
            "custom_title": row.custom_title,
        }

    def upsert(self, db: Session, *, user_id: str, state: Dict[str, Any]) -> None:
        row = (
            db.query(ChatSession)
            .filter(ChatSession.user_id == user_id)
            .filter(ChatSession.id == state["session_id"])
            .first()
        )
        if row is None:
            row = ChatSession(id=state["session_id"], user_id=user_id)
            db.add(row)

        row.qna = list(state.get("qna") or [])
        row.partial_transcript = state.get("partial_transcript") or ""
        row.profile_text = state.get("profile_text") or ""
        row.custom_title = state.get("custom_title")
        row.last_update = state.get("last_update") or _utcnow()

        # Avoid clobbering created_at on updates.
        if row.created_at is None:
            row.created_at = _utcnow()

        db.commit()

    def delete(self, db: Session, *, user_id: str, session_id: str) -> bool:
        row = (
            db.query(ChatSession)
            .filter(ChatSession.user_id == user_id)
            .filter(ChatSession.id == session_id)
            .first()
        )
        if not row:
            return False
        db.delete(row)
        db.commit()
        return True

    def list_summaries(self, db: Session, *, user_id: str) -> List[Dict[str, Any]]:
        rows = (
            db.query(ChatSession)
            .filter(ChatSession.user_id == user_id)
            .order_by(ChatSession.last_update.desc())
            .all()
        )

        items: List[Dict[str, Any]] = []
        for row in rows:
            qna = list(row.qna or [])

            title = row.custom_title
            if not title:
                title = "New Chat"
                if qna:
                    first_q = (qna[0] or {}).get("question", "")
                    title = (first_q[:50] + "...") if len(first_q) > 50 else first_q

            last_update = row.last_update or row.created_at or _utcnow()
            items.append(
                {
                    "session_id": row.id,
                    "title": title,
                    "last_update": last_update.isoformat(),
                    "qna_count": len(qna),
                }
            )

        return items
