from __future__ import annotations

import uuid

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import settings
from app.database import init_db, get_db_context
from app.models import PracticeAttemptRecord, PracticeSessionOutcome, PracticeSessionMetrics
from app.routers.practice_mode import router as practice_router
from app.schemas import AnswerSubmission, MicroFeedback, PracticeSession, SpeechMetrics
from app.services.practice.practice_learning import (
    compute_metrics_summary_from_session,
    derive_peer_bucket,
    compute_peer_confidence_benchmark,
    upsert_practice_session_metrics,
)


def _make_answer(*, question_id: int, wpm: float, fillers: int, duration: float, longest_silence: float = 1.0) -> AnswerSubmission:
    metrics = SpeechMetrics(
        filler_count=int(fillers),
        wpm=float(wpm),
        longest_silence=float(longest_silence),
        confidence_score=5.0,
        overtalked=False,
        duration=float(duration),
        filler_words=[],
        pause_count=1,
        pitch_variance=0.0,
        silence_removed=0.0,
    )
    feedback = MicroFeedback(
        delivery_tips=["ok", "ok"],
        pace_feedback="ok",
        overall_note="ok",
    )
    return AnswerSubmission(
        question_id=int(question_id),
        transcript="",
        metrics=metrics,
        micro_feedback=feedback,
        audio_duration=float(duration),
    )


def test_submit_confidence_outcome_upserts(monkeypatch):
    init_db()

    app = FastAPI()
    app.include_router(practice_router)
    client = TestClient(app)

    user_id = f"test_user_{uuid.uuid4()}"
    session_id = str(uuid.uuid4())

    import app.routers.practice_mode as practice_mod

    monkeypatch.setattr(practice_mod, "get_user_id_from_request", lambda _req: user_id)
    monkeypatch.setattr(settings, "enable_practice_learning", True, raising=False)

    with get_db_context() as db:
        db.add(PracticeAttemptRecord(user_id=user_id, session_id=session_id))
        db.commit()

    resp = client.post(
        f"/api/practice/session/{session_id}/outcome/confidence",
        json={"confidence_1_5": 4},
    )
    assert resp.status_code == 200
    assert resp.json()["confidence_1_5"] == 4

    with get_db_context() as db:
        row = (
            db.query(PracticeSessionOutcome)
            .filter(PracticeSessionOutcome.user_id == user_id)
            .filter(PracticeSessionOutcome.session_id == session_id)
            .first()
        )
        assert row is not None
        assert int(row.confidence_1_5) == 4

    # Upsert update
    resp2 = client.post(
        f"/api/practice/session/{session_id}/outcome/confidence",
        json={"confidence_1_5": 2},
    )
    assert resp2.status_code == 200
    assert resp2.json()["confidence_1_5"] == 2

    with get_db_context() as db:
        row2 = (
            db.query(PracticeSessionOutcome)
            .filter(PracticeSessionOutcome.user_id == user_id)
            .filter(PracticeSessionOutcome.session_id == session_id)
            .first()
        )
        assert row2 is not None
        assert int(row2.confidence_1_5) == 2


def test_metrics_upsert_and_peer_benchmark():
    init_db()

    user_id = f"test_metrics_user_{uuid.uuid4()}"

    # Seed 2 sessions first (below min_samples)
    with get_db_context() as db:
        for conf in [3, 4]:
            sid = str(uuid.uuid4())
            sess = PracticeSession(session_id=sid)
            sess.answers = [
                _make_answer(question_id=1, wpm=150, fillers=2, duration=60),
                _make_answer(question_id=2, wpm=155, fillers=1, duration=60),
            ]
            upsert_practice_session_metrics(db, user_id=user_id, session_id=sid, session=sess)
            db.add(PracticeSessionOutcome(user_id=user_id, session_id=sid, confidence_1_5=int(conf)))
            db.commit()

        # New session bucket derived from similar metrics
        probe_session = PracticeSession(session_id=str(uuid.uuid4()))
        probe_session.answers = [
            _make_answer(question_id=1, wpm=152, fillers=1, duration=60),
            _make_answer(question_id=2, wpm=151, fillers=2, duration=60),
        ]
        probe_metrics = upsert_practice_session_metrics(
            db,
            user_id=user_id,
            session_id=probe_session.session_id,
            session=probe_session,
        )

        assert probe_metrics.wpm_bucket in {"balanced", "fast", "slow", "unknown"}
        assert probe_metrics.filler_bucket in {"low", "medium", "high", "unknown"}

        summary = compute_metrics_summary_from_session(probe_session)
        bucket = derive_peer_bucket(summary)

        avg, n = compute_peer_confidence_benchmark(db, bucket=bucket, min_samples=3)
        assert avg is None
        assert n == 2

        # Add a 3rd session in same bucket
        sid3 = str(uuid.uuid4())
        sess3 = PracticeSession(session_id=sid3)
        sess3.answers = [
            _make_answer(question_id=1, wpm=150, fillers=2, duration=60),
            _make_answer(question_id=2, wpm=150, fillers=2, duration=60),
        ]
        upsert_practice_session_metrics(db, user_id=user_id, session_id=sid3, session=sess3)
        db.add(PracticeSessionOutcome(user_id=user_id, session_id=sid3, confidence_1_5=5))
        db.commit()

        avg2, n2 = compute_peer_confidence_benchmark(db, bucket=bucket, min_samples=3)
        assert n2 >= 3
        assert avg2 is not None

        # Sanity check that metrics table has rows
        assert (
            db.query(PracticeSessionMetrics)
            .filter(PracticeSessionMetrics.user_id == user_id)
            .count()
            >= 3
        )
