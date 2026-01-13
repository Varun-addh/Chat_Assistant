from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from sqlalchemy.orm import Session

from app.models import HistoryTabRecord


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class HistoryTabRepository:
    """Repository for Search Intelligence history tabs."""

    def list_tabs(self, db: Session, *, user_id: str) -> List[HistoryTabRecord]:
        return (
            db.query(HistoryTabRecord)
            .filter(HistoryTabRecord.user_id == user_id)
            .order_by(HistoryTabRecord.created_at.desc())
            .all()
        )

    def get_tab(self, db: Session, *, user_id: str, tab_id: str) -> Optional[HistoryTabRecord]:
        return (
            db.query(HistoryTabRecord)
            .filter(HistoryTabRecord.user_id == user_id)
            .filter(HistoryTabRecord.tab_id == tab_id)
            .first()
        )

    def insert(self, db: Session, *, user_id: str, tab_id: str, query: str, questions: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]], created_at: Optional[datetime] = None) -> None:
        row = HistoryTabRecord(
            tab_id=tab_id,
            user_id=user_id,
            query=query,
            questions=list(questions or []),
            extra_data=dict(metadata or {}),
            created_at=created_at or _utcnow(),
        )
        db.add(row)
        db.commit()

    def update(self, db: Session, *, user_id: str, tab_id: str, query: Optional[str], questions: Optional[List[Dict[str, Any]]], metadata: Optional[Dict[str, Any]]) -> bool:
        row = self.get_tab(db, user_id=user_id, tab_id=tab_id)
        if not row:
            return False
        if query is not None:
            row.query = query
        if questions is not None:
            row.questions = list(questions)
        if metadata is not None:
            merged = dict(row.extra_data or {})
            merged.update(metadata)
            row.extra_data = merged
        db.commit()
        return True

    def delete(self, db: Session, *, user_id: str, tab_id: str) -> bool:
        row = self.get_tab(db, user_id=user_id, tab_id=tab_id)
        if not row:
            return False
        db.delete(row)
        db.commit()
        return True

    def delete_all(self, db: Session, *, user_id: str) -> int:
        rows = (
            db.query(HistoryTabRecord)
            .filter(HistoryTabRecord.user_id == user_id)
            .all()
        )
        count = len(rows)
        for r in rows:
            db.delete(r)
        db.commit()
        return count
