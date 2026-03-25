from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from sqlalchemy.orm import Session

from app.models import PracticeSessionMetrics, PracticeSessionOutcome
from app.schemas import MetricsSummary, PracticeSession


@dataclass(frozen=True)
class PeerBucket:
    wpm_bucket: str
    filler_bucket: str


def _bucket_wpm(avg_wpm: float) -> str:
    if avg_wpm <= 0:
        return "unknown"
    if avg_wpm < 120:
        return "slow"
    if avg_wpm <= 160:
        return "balanced"
    if avg_wpm <= 190:
        return "fast"
    return "very_fast"


def _bucket_fillers(filler_per_minute: float) -> str:
    if filler_per_minute < 0:
        return "unknown"
    if filler_per_minute <= 1.0:
        return "low"
    if filler_per_minute <= 3.0:
        return "medium"
    return "high"


def derive_peer_bucket(metrics_summary: MetricsSummary) -> PeerBucket:
    total_minutes = float(metrics_summary.total_duration) / 60.0 if metrics_summary.total_duration else 0.0
    filler_per_minute = float(metrics_summary.total_fillers) / total_minutes if total_minutes > 0 else 0.0
    return PeerBucket(
        wpm_bucket=_bucket_wpm(float(metrics_summary.avg_wpm or 0.0)),
        filler_bucket=_bucket_fillers(float(filler_per_minute)),
    )


def compute_metrics_summary_from_session(session: PracticeSession) -> MetricsSummary:
    answers = list(getattr(session, "answers", []) or [])
    if not answers:
        return MetricsSummary(
            total_fillers=0,
            avg_wpm=0.0,
            longest_pause=0.0,
            avg_confidence=0.0,
            total_duration=0.0,
            overtalked_count=0,
        )

    total_fillers = sum(int(a.metrics.filler_count) for a in answers)
    avg_wpm = sum(float(a.metrics.wpm) for a in answers) / len(answers)
    longest_pause = max((float(a.metrics.longest_silence) for a in answers), default=0.0)
    avg_confidence = sum(float(a.metrics.confidence_score) for a in answers) / len(answers)
    total_duration = sum(float(a.metrics.duration) for a in answers)
    overtalked_count = sum(1 for a in answers if bool(a.metrics.overtalked))

    return MetricsSummary(
        total_fillers=int(total_fillers),
        avg_wpm=round(float(avg_wpm), 1),
        longest_pause=round(float(longest_pause), 2),
        avg_confidence=round(float(avg_confidence), 1),
        total_duration=round(float(total_duration), 2),
        overtalked_count=int(overtalked_count),
    )


def _compute_filler_per_minute(metrics_summary: MetricsSummary) -> float:
    total_minutes = float(metrics_summary.total_duration) / 60.0 if metrics_summary.total_duration else 0.0
    if total_minutes <= 0:
        return 0.0
    return float(metrics_summary.total_fillers) / total_minutes


def upsert_practice_session_metrics(
    db: Session,
    *,
    user_id: str,
    session_id: str,
    session: PracticeSession,
) -> PracticeSessionMetrics:
    summary = compute_metrics_summary_from_session(session)
    bucket = derive_peer_bucket(summary)
    filler_per_minute = _compute_filler_per_minute(summary)

    row = (
        db.query(PracticeSessionMetrics)
        .filter(PracticeSessionMetrics.user_id == user_id)
        .filter(PracticeSessionMetrics.session_id == session_id)
        .first()
    )

    if row is None:
        row = PracticeSessionMetrics(user_id=user_id, session_id=session_id)
        db.add(row)

    row.answers_count = int(len(getattr(session, "answers", []) or []))
    row.avg_wpm = float(summary.avg_wpm)
    row.total_fillers = int(summary.total_fillers)
    row.filler_per_minute = float(round(filler_per_minute, 3))
    row.total_pause_count = int(sum(int(a.metrics.pause_count or 0) for a in (session.answers or [])))
    row.longest_pause_seconds = float(summary.longest_pause)
    row.total_duration_seconds = float(summary.total_duration)
    row.avg_voice_confidence_0_10 = float(summary.avg_confidence)
    row.overtalked_count = int(summary.overtalked_count)
    row.wpm_bucket = bucket.wpm_bucket
    row.filler_bucket = bucket.filler_bucket

    db.commit()
    db.refresh(row)
    return row


def upsert_practice_session_outcome_confidence(
    db: Session,
    *,
    user_id: str,
    session_id: str,
    confidence_1_5: int,
) -> PracticeSessionOutcome:
    row = (
        db.query(PracticeSessionOutcome)
        .filter(PracticeSessionOutcome.user_id == user_id)
        .filter(PracticeSessionOutcome.session_id == session_id)
        .first()
    )

    if row is None:
        row = PracticeSessionOutcome(user_id=user_id, session_id=session_id, confidence_1_5=int(confidence_1_5))
        db.add(row)
    else:
        row.confidence_1_5 = int(confidence_1_5)

    db.commit()
    db.refresh(row)
    return row


def compute_peer_confidence_benchmark(
    db: Session,
    *,
    bucket: PeerBucket,
    min_samples: int = 3,
) -> Tuple[Optional[float], int]:
    # Exact match: both wpm_bucket AND filler_bucket
    q = (
        db.query(PracticeSessionOutcome.confidence_1_5)
        .join(
            PracticeSessionMetrics,
            (PracticeSessionMetrics.user_id == PracticeSessionOutcome.user_id)
            & (PracticeSessionMetrics.session_id == PracticeSessionOutcome.session_id),
        )
        .filter(PracticeSessionMetrics.wpm_bucket == bucket.wpm_bucket)
        .filter(PracticeSessionMetrics.filler_bucket == bucket.filler_bucket)
    )

    rows = [int(r[0]) for r in q.all() if r and r[0] is not None]
    n = len(rows)
    if n >= int(min_samples):
        avg = sum(rows) / n
        return round(float(avg), 2), n

    # Fallback: match on wpm_bucket only (wider peer group)
    q_fallback = (
        db.query(PracticeSessionOutcome.confidence_1_5)
        .join(
            PracticeSessionMetrics,
            (PracticeSessionMetrics.user_id == PracticeSessionOutcome.user_id)
            & (PracticeSessionMetrics.session_id == PracticeSessionOutcome.session_id),
        )
        .filter(PracticeSessionMetrics.wpm_bucket == bucket.wpm_bucket)
    )

    rows_fb = [int(r[0]) for r in q_fallback.all() if r and r[0] is not None]
    n_fb = len(rows_fb)
    if n_fb >= int(min_samples):
        avg_fb = sum(rows_fb) / n_fb
        return round(float(avg_fb), 2), n_fb

    return None, n_fb


def format_peer_confidence_insight(*, avg_confidence_1_5: float, n: int) -> str:
    return f"Peer benchmark (n={n}): Candidates with similar delivery patterns typically self-report confidence of {avg_confidence_1_5:.2f}/5."
