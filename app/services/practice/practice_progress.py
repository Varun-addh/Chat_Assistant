"""Practice Mode progress persistence + queries.

This module powers Issue #5: the flagship loop.

- Save a completed practice attempt with a stable rubric score.
- Query summaries and heatmap-friendly trend data.

Implementation notes:
- Uses SQLAlchemy models in `app.models`.
- Designed to work with both SQLite (dev/tests) and Postgres (prod).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy.orm import Session

from app.models import PracticeAttemptDimensionRecord, PracticeAttemptRecord
from app.schemas import PracticeSession
from app.services.practice.practice_scoring import (
    PracticeScoreResult,
    evaluation_report_to_json,
    score_session,
)


@dataclass(frozen=True)
class PracticeProgressSummary:
    attempts: int
    average_overall_score: Optional[float]
    last_completed_at: Optional[datetime]
    best_dimension: Optional[str]
    worst_dimension: Optional[str]


def save_completed_attempt(
    db: Session,
    *,
    user_id: str,
    session: PracticeSession,
) -> Optional[int]:
    """Persist a completed PracticeSession.

    Returns attempt_id, or None if not persisted (e.g., not complete).

    Best practice: call this once per completed session.
    """
    import logging
    _logger = logging.getLogger(__name__)

    if not getattr(session, "is_complete", False):
        _logger.warning(f"save_completed_attempt: session {session.session_id} is_complete=False, skip")
        return None

    try:
        score: PracticeScoreResult = score_session(session=session)
    except Exception as e:
        _logger.error(f"score_session failed for {session.session_id}: {e}", exc_info=True)
        # Still save a minimal record even if scoring fails completely
        score = PracticeScoreResult(
            overall_score=0.0,
            dimension_scores={},
            why=[f"Scoring failed: {e}"],
            improvement_plan=["Retry session"],
            next_session_plan={"focus": ["retry"], "question_count": 5},
        )

    attempt = PracticeAttemptRecord(
        user_id=user_id,
        session_id=session.session_id,
        domain=getattr(getattr(session, "user_profile", None), "domain", None),
        round_type=getattr(getattr(session, "round_type", None), "value", None),
        difficulty=getattr(getattr(session, "difficulty", None), "value", None),
        company=getattr(session, "company_name", None),
        question_count=len(getattr(session, "questions", []) or []),
        overall_score=float(score.overall_score),
        dimension_scores=score.dimension_scores,
        why=score.why,
        improvement_plan=score.improvement_plan,
        next_session_plan=score.next_session_plan,
        evaluation_report=evaluation_report_to_json(getattr(session, "evaluation_report", None)),
        started_at=_to_utc(getattr(session, "started_at", None)),
        completed_at=_to_utc(getattr(session, "completed_at", None)),
    )

    db.add(attempt)
    db.flush()  # assigns PK

    for dim, val in (score.dimension_scores or {}).items():
        db.add(
            PracticeAttemptDimensionRecord(
                attempt_id=attempt.id,
                dimension=str(dim),
                score=float(val),
            )
        )

    db.commit()
    _logger.info(f"✅ Saved practice attempt id={attempt.id} for user={user_id}, session={session.session_id}, score={score.overall_score:.1f}")
    return int(attempt.id)


def _to_utc(dt: Any) -> Optional[datetime]:
    if not dt:
        return None
    if isinstance(dt, datetime):
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    return None


def get_progress_summary(
    db: Session,
    *,
    user_id: str,
    lookback_days: int = 30,
    domain: Optional[str] = None,
) -> PracticeProgressSummary:
    since = datetime.now(timezone.utc) - timedelta(days=lookback_days)

    q = db.query(PracticeAttemptRecord).filter(PracticeAttemptRecord.user_id == user_id)
    q = q.filter(PracticeAttemptRecord.created_at >= since)
    if domain:
        q = q.filter(PracticeAttemptRecord.domain == domain)

    attempts = q.order_by(PracticeAttemptRecord.created_at.desc()).all()

    if not attempts:
        return PracticeProgressSummary(
            attempts=0,
            average_overall_score=None,
            last_completed_at=None,
            best_dimension=None,
            worst_dimension=None,
        )

    scores = [a.overall_score for a in attempts if a.overall_score is not None]
    avg = (sum(scores) / len(scores)) if scores else None

    # dimension best/worst based on averages across attempts
    dim_totals: dict[str, list[float]] = {}
    for a in attempts:
        ds = a.dimension_scores or {}
        if isinstance(ds, dict):
            for k, v in ds.items():
                try:
                    dim_totals.setdefault(str(k), []).append(float(v))
                except Exception:
                    continue

    best_dim = None
    worst_dim = None
    if dim_totals:
        dim_avgs = {k: (sum(vs) / len(vs)) for k, vs in dim_totals.items() if vs}
        if dim_avgs:
            best_dim = max(dim_avgs.items(), key=lambda kv: kv[1])[0]
            worst_dim = min(dim_avgs.items(), key=lambda kv: kv[1])[0]

    last_completed = attempts[0].completed_at or attempts[0].created_at

    return PracticeProgressSummary(
        attempts=len(attempts),
        average_overall_score=avg,
        last_completed_at=last_completed,
        best_dimension=best_dim,
        worst_dimension=worst_dim,
    )


def get_dimension_heatmap(
    db: Session,
    *,
    user_id: str,
    lookback_days: int = 90,
    domain: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Return heatmap points: weekly buckets x dimension average."""

    since = datetime.now(timezone.utc) - timedelta(days=lookback_days)

    q = db.query(PracticeAttemptRecord).filter(PracticeAttemptRecord.user_id == user_id)
    q = q.filter(PracticeAttemptRecord.created_at >= since)
    if domain:
        q = q.filter(PracticeAttemptRecord.domain == domain)

    attempts = q.order_by(PracticeAttemptRecord.created_at.asc()).all()

    # bucket: (week_start_iso, dimension) -> list[score]
    buckets: dict[tuple[str, str], list[float]] = {}

    for a in attempts:
        when = a.completed_at or a.created_at
        if when is None:
            continue
        when = _to_utc(when) or when.replace(tzinfo=timezone.utc)

        # week start (Monday) in UTC
        week_start = (when - timedelta(days=when.weekday())).date().isoformat()

        ds = a.dimension_scores or {}
        if not isinstance(ds, dict):
            continue
        for dim, val in ds.items():
            try:
                buckets.setdefault((week_start, str(dim)), []).append(float(val))
            except Exception:
                continue

    points: list[dict[str, Any]] = []
    for (week_start, dim), vals in sorted(buckets.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        if not vals:
            continue
        points.append(
            {
                "week_start": week_start,
                "dimension": dim,
                "avg_score": sum(vals) / len(vals),
                "attempts": len(vals),
            }
        )

    return points


def get_latest_next_session_plan(
    db: Session,
    *,
    user_id: str,
    domain: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    q = db.query(PracticeAttemptRecord).filter(PracticeAttemptRecord.user_id == user_id)
    if domain:
        q = q.filter(PracticeAttemptRecord.domain == domain)
    latest = q.order_by(PracticeAttemptRecord.created_at.desc()).first()
    if not latest:
        return None
    plan = latest.next_session_plan
    if isinstance(plan, dict):
        return plan
    return None
