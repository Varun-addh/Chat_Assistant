"""Practice Mode progress persistence + queries.

This module powers Issue #5: the flagship loop.

- Save a completed practice attempt with a stable rubric score.
- Query summaries and heatmap-friendly trend data.

Implementation notes:
- Uses SQLAlchemy models in `app.models`.
- Designed to work with both SQLite (dev/tests) and Postgres (prod).
- Heatmap/summary queries use the normalised ``PracticeAttemptDimensionRecord``
  table via SQL aggregation (no full table scan).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models import PracticeAttemptDimensionRecord, PracticeAttemptRecord
from app.schemas import PracticeSession
from app.services.practice.practice_scoring import (
    PracticeScoreResult,
    evaluation_report_to_json,
    score_session,
)

logger = logging.getLogger(__name__)


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

    Returns attempt_id, or None if not persisted (e.g., not complete or
    already saved for this session_id).
    """

    if not getattr(session, "is_complete", False):
        logger.warning("save_completed_attempt: session %s is_complete=False, skip", session.session_id)
        return None

    # Dedup guard: refuse to double-write for the same session
    existing = (
        db.query(PracticeAttemptRecord.id)
        .filter(PracticeAttemptRecord.session_id == session.session_id)
        .first()
    )
    if existing is not None:
        logger.info("save_completed_attempt: session %s already persisted (id=%s), skip", session.session_id, existing[0])
        return int(existing[0])

    try:
        score: PracticeScoreResult = score_session(session=session)
    except Exception as e:
        logger.error("score_session failed for %s: %s", session.session_id, e, exc_info=True)
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
    logger.info(
        "Saved practice attempt id=%s for user=%s, session=%s, score=%.1f",
        attempt.id, user_id, session.session_id, score.overall_score,
    )
    return int(attempt.id)


def _to_utc(dt: Any) -> Optional[datetime]:
    if not dt:
        return None
    if isinstance(dt, datetime):
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    return None


def _best_worst_from_json(
    attempts: list[PracticeAttemptRecord],
) -> tuple[Optional[str], Optional[str]]:
    """Fallback: compute best/worst dimension from the JSON column."""
    dim_totals: dict[str, list[float]] = {}
    for a in attempts:
        ds = a.dimension_scores
        if not isinstance(ds, dict):
            continue
        for k, v in ds.items():
            try:
                dim_totals.setdefault(str(k), []).append(float(v))
            except Exception:
                continue
    if not dim_totals:
        return None, None
    dim_avgs = {k: sum(vs) / len(vs) for k, vs in dim_totals.items() if vs}
    if not dim_avgs:
        return None, None
    best = max(dim_avgs.items(), key=lambda kv: kv[1])[0]
    worst = min(dim_avgs.items(), key=lambda kv: kv[1])[0]
    return best, worst


def _heatmap_from_json(
    attempts: list[PracticeAttemptRecord],
    max_points: int = 500,
) -> list[dict[str, Any]]:
    """Fallback: build heatmap from the JSON dimension_scores column."""
    buckets: dict[tuple[str, str], list[float]] = {}
    for a in attempts:
        when = a.completed_at or a.created_at
        if when is None:
            continue
        when = _to_utc(when) or when.replace(tzinfo=timezone.utc)
        week_start = (when - timedelta(days=when.weekday())).date().isoformat()
        ds = a.dimension_scores
        if not isinstance(ds, dict):
            continue
        for dim, val in ds.items():
            try:
                buckets.setdefault((week_start, str(dim)), []).append(float(val))
            except Exception:
                continue
    points: list[dict[str, Any]] = []
    for (ws, dim), vals in sorted(buckets.items()):
        if not vals:
            continue
        points.append({
            "week_start": ws,
            "dimension": dim,
            "avg_score": round(sum(vals) / len(vals), 2),
            "attempts": len(vals),
        })
        if len(points) >= max_points:
            break
    return points


# ---------------------------------------------------------------------------
# Progress summary — SQL aggregation (Fix #6)
# ---------------------------------------------------------------------------

def get_progress_summary(
    db: Session,
    *,
    user_id: str,
    lookback_days: int = 30,
    domain: Optional[str] = None,
) -> PracticeProgressSummary:
    since = datetime.now(timezone.utc) - timedelta(days=lookback_days)

    # Base filter
    base = db.query(PracticeAttemptRecord).filter(
        PracticeAttemptRecord.user_id == user_id,
        PracticeAttemptRecord.created_at >= since,
    )
    if domain:
        base = base.filter(PracticeAttemptRecord.domain == domain)

    # Scalar aggregates in a single query
    agg = base.with_entities(
        func.count(PracticeAttemptRecord.id),
        func.avg(PracticeAttemptRecord.overall_score),
        func.max(PracticeAttemptRecord.completed_at),
    ).first()

    count = int(agg[0] or 0)
    if count == 0:
        return PracticeProgressSummary(
            attempts=0,
            average_overall_score=None,
            last_completed_at=None,
            best_dimension=None,
            worst_dimension=None,
        )

    avg_score = round(float(agg[1]), 2) if agg[1] is not None else None
    last_completed = agg[2]

    # Best/worst dimension via the normalised dimension table (Fix #1)
    attempt_ids_subq = (
        base.with_entities(PracticeAttemptRecord.id).subquery()
    )
    dim_avgs = (
        db.query(
            PracticeAttemptDimensionRecord.dimension,
            func.avg(PracticeAttemptDimensionRecord.score),
        )
        .filter(PracticeAttemptDimensionRecord.attempt_id.in_(
            db.query(attempt_ids_subq.c.id)
        ))
        .group_by(PracticeAttemptDimensionRecord.dimension)
        .all()
    )

    best_dim = None
    worst_dim = None
    if dim_avgs:
        best_dim = max(dim_avgs, key=lambda r: float(r[1] or 0))[0]
        worst_dim = min(dim_avgs, key=lambda r: float(r[1] or 0))[0]
    else:
        # Fallback: compute from JSON column when no normalised rows exist
        best_dim, worst_dim = _best_worst_from_json(base.all())

    return PracticeProgressSummary(
        attempts=count,
        average_overall_score=avg_score,
        last_completed_at=last_completed,
        best_dimension=best_dim,
        worst_dimension=worst_dim,
    )


# ---------------------------------------------------------------------------
# Heatmap — SQL aggregation on normalised dimension table (Fix #1 + #2)
# ---------------------------------------------------------------------------

def get_dimension_heatmap(
    db: Session,
    *,
    user_id: str,
    lookback_days: int = 90,
    domain: Optional[str] = None,
    max_points: int = 500,
) -> list[dict[str, Any]]:
    """Return heatmap points: weekly buckets x dimension average.

    Uses ``PracticeAttemptDimensionRecord`` with SQL GROUP BY instead of
    loading all parent rows into Python.
    """

    since = datetime.now(timezone.utc) - timedelta(days=lookback_days)

    # Sub-query: matching attempt ids
    base = db.query(PracticeAttemptRecord).filter(
        PracticeAttemptRecord.user_id == user_id,
        PracticeAttemptRecord.created_at >= since,
    )
    if domain:
        base = base.filter(PracticeAttemptRecord.domain == domain)

    attempt_ids_subq = base.with_entities(PracticeAttemptRecord.id).subquery()

    # Join dimensions → attempts for the timestamp, aggregate in SQL
    rows = (
        db.query(
            PracticeAttemptDimensionRecord.dimension,
            func.avg(PracticeAttemptDimensionRecord.score),
            func.count(PracticeAttemptDimensionRecord.id),
            PracticeAttemptRecord.completed_at,
            PracticeAttemptRecord.created_at,
        )
        .join(PracticeAttemptRecord, PracticeAttemptRecord.id == PracticeAttemptDimensionRecord.attempt_id)
        .filter(PracticeAttemptDimensionRecord.attempt_id.in_(
            db.query(attempt_ids_subq.c.id)
        ))
        .group_by(
            PracticeAttemptDimensionRecord.dimension,
            PracticeAttemptRecord.completed_at,
            PracticeAttemptRecord.created_at,
        )
        .all()
    )

    if not rows:
        # Fallback: no normalised dimension rows, use JSON column
        return _heatmap_from_json(base.all(), max_points=max_points)

    # Bucket in Python by (week_start, dimension) — but only scalar rows
    buckets: dict[tuple[str, str], tuple[float, int]] = {}

    for dim, avg_score, cnt, completed_at, created_at in rows:
        when = completed_at or created_at
        if when is None:
            continue
        when = _to_utc(when) or when.replace(tzinfo=timezone.utc)
        week_start = (when - timedelta(days=when.weekday())).date().isoformat()
        key = (week_start, str(dim))
        prev_total, prev_cnt = buckets.get(key, (0.0, 0))
        buckets[key] = (prev_total + float(avg_score or 0) * int(cnt), prev_cnt + int(cnt))

    points: list[dict[str, Any]] = []
    for (week_start, dim), (total, cnt) in sorted(buckets.items()):
        if cnt == 0:
            continue
        points.append({
            "week_start": week_start,
            "dimension": dim,
            "avg_score": round(total / cnt, 2),
            "attempts": cnt,
        })
        if len(points) >= max_points:
            break

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
