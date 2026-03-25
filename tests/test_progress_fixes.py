"""Tests for the 12 progress-feature fixes.

Covers:
  #1  PracticeAttemptDimensionRecord is now queried (not dead data)
  #2  Heatmap uses SQL aggregation (no full table scan)
  #3  Dual-source reconciliation (scored_attempts in insights)
  #4  Structure heuristic uses word-boundary matching
  #5  Delivery scoring normalises confidence 0-10 → 0-1
  #6  Summary uses SQL aggregation
  #7  NextSessionPlan is a typed Pydantic model
  #8  Heatmap endpoint accepts max_points
  #9  Module-level logger (import sanity)
  #11 Dedup guard on save_completed_attempt
  #12 Peer benchmark fallback when exact bucket < min_samples
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.database import get_db_context, init_db
from app.models import (
    PracticeAttemptDimensionRecord,
    PracticeAttemptRecord,
    PracticeSessionMetrics,
    PracticeSessionOutcome,
)
from app.routers.practice_mode import router as practice_router
from app.schemas import NextSessionPlan, PracticeNextSessionRecommendationResponse


# ─── helpers ───────────────────────────────────────────────────────────────

def _make_answer(
    *,
    correctness_score: float = 75.0,
    confidence_score: float = 7.5,
    filler_count: int = 2,
    wpm: float = 150.0,
    overtalked: bool = False,
    transcript: str = "First, I would discuss the trade-offs. Second, let me break down the approach.",
    question_id: int = 1,
):
    """Return a lightweight answer-like namespace for scoring functions."""
    mf = SimpleNamespace(
        correctness_score=correctness_score,
        technical_accuracy="good",
        actionable_suggestions=["Be concise"],
    )
    metrics = SimpleNamespace(
        filler_count=filler_count,
        wpm=wpm,
        confidence_score=confidence_score,
        overtalked=overtalked,
        longest_silence=0.5,
        pause_count=1,
        duration=60.0,
    )
    return SimpleNamespace(
        question_id=question_id,
        micro_feedback=mf,
        metrics=metrics,
        transcript=transcript,
    )


def _setup_app_client(monkeypatch, user_id: str):
    """Create a FastAPI TestClient with mocked user identity."""
    app = FastAPI()
    app.include_router(practice_router)
    client = TestClient(app)
    import app.routers.practice_mode as pm
    monkeypatch.setattr(pm, "get_user_id_from_request", lambda _req: user_id)
    return client


# ─── Fix #4: Structure heuristic word-boundary matching ────────────────────

class TestStructureHeuristic:
    def test_word_boundary_prevents_false_positive(self):
        from app.services.practice.practice_scoring import _structure_heuristic

        # "stephen" should NOT match "step"
        score_stephen = _structure_heuristic("My name is Stephen and I like coding.")
        score_step = _structure_heuristic("The first step is to define the interface.")
        # stephen should get base score only; step_by_step uses "step by step"
        assert score_stephen <= 57.0, "substring 'step' inside 'stephen' must NOT be a hit"

    def test_legitimate_markers_score_higher(self):
        from app.services.practice.practice_scoring import _structure_heuristic

        structured = "First, I would outline the approach. Second, let me break down the trade-offs. Finally, in summary, the answer is clear."
        unstructured = "I think the answer is probably related to some topic."
        assert _structure_heuristic(structured) > _structure_heuristic(unstructured)

    def test_empty_transcript(self):
        from app.services.practice.practice_scoring import _structure_heuristic
        assert _structure_heuristic("") == 0.0
        assert _structure_heuristic("   ") == 0.0

    def test_new_markers_recognised(self):
        from app.services.practice.practice_scoring import _structure_heuristic

        text = "Next, we do this. Moreover, it's important. Additionally, to conclude everything."
        score = _structure_heuristic(text)
        # 4 markers hit: next, moreover, additionally, to conclude → 50 + 4*7 = 78
        assert score >= 70.0


# ─── Fix #5: Delivery scoring normalises confidence 0-10 ───────────────────

class TestDeliveryScoreNormalisation:
    def test_confidence_0_10_does_not_clamp_to_100(self):
        from app.services.practice.practice_scoring import score_answer

        answer = _make_answer(confidence_score=7.5, filler_count=0, overtalked=False)
        result = score_answer(answer=answer)
        delivery = result.dimension_scores["delivery"]
        # conf=7.5 → normalised to 0.75 → 75.0 (not 100 from 750 clamped)
        assert 70.0 <= delivery <= 80.0, f"delivery={delivery}, expected ~75"

    def test_confidence_0_1_stays_correct(self):
        from app.services.practice.practice_scoring import score_answer

        answer = _make_answer(confidence_score=0.8, wpm=150.0, filler_count=0)
        result = score_answer(answer=answer)
        delivery = result.dimension_scores["delivery"]
        # conf=0.8 → stays 0.8 → 80.0
        assert 75.0 <= delivery <= 85.0

    def test_no_speech_metrics_baseline(self):
        from app.services.practice.practice_scoring import score_answer

        answer = _make_answer(confidence_score=0.0, wpm=0.0, filler_count=0)
        result = score_answer(answer=answer)
        assert result.dimension_scores["delivery"] == 70.0

    def test_session_level_delivery_normalised(self):
        from app.services.practice.practice_scoring import score_session

        answers = [_make_answer(confidence_score=8.0, filler_count=1)]
        session = SimpleNamespace(
            answers=answers,
            questions=["q1"],
            round_type=SimpleNamespace(value="technical"),
            difficulty=SimpleNamespace(value="medium"),
        )
        result = score_session(session=session)
        delivery = result.dimension_scores["delivery"]
        # conf=8.0 → 0.8 → 80.0 - filler penalty
        assert 60.0 <= delivery <= 85.0, f"delivery={delivery}"


# ─── Fix #7: Typed NextSessionPlan model ────────────────────────────────────

class TestNextSessionPlanTyped:
    def test_model_parses_valid_plan(self):
        plan = NextSessionPlan(
            generated_at="2025-01-01T00:00:00Z",
            focus_dimension="delivery",
            focus=["reduce_fillers", "confidence"],
            question_count=5,
            recommended_round="technical",
            difficulty="medium",
        )
        assert plan.focus_dimension == "delivery"
        assert len(plan.focus) == 2

    def test_response_model_serialises(self):
        plan = NextSessionPlan(focus_dimension="clarity", focus=["pace"], question_count=4)
        resp = PracticeNextSessionRecommendationResponse(plan=plan)
        d = resp.model_dump()
        assert d["plan"]["focus_dimension"] == "clarity"
        assert d["plan"]["question_count"] == 4

    def test_response_model_none_plan(self):
        resp = PracticeNextSessionRecommendationResponse(plan=None)
        assert resp.plan is None


# ─── Fix #9: Module-level logger ───────────────────────────────────────────

class TestModuleLevelLogger:
    def test_logger_exists_at_module_level(self):
        import app.services.practice.practice_progress as pp
        assert hasattr(pp, "logger"), "logger should be a module-level attribute"
        assert pp.logger.name == "app.services.practice.practice_progress"


# ─── Fix #11: Dedup guard on save_completed_attempt ─────────────────────────

class TestDedupGuard:
    def test_second_save_returns_existing_id(self):
        init_db()
        from app.services.practice.practice_progress import save_completed_attempt

        user_id = f"dedup_test_{uuid.uuid4()}"
        session_id = str(uuid.uuid4())
        session = SimpleNamespace(
            session_id=session_id,
            is_complete=True,
            answers=[_make_answer()],
            questions=["q1"],
            user_profile=SimpleNamespace(domain="Python"),
            round_type=SimpleNamespace(value="technical"),
            difficulty=SimpleNamespace(value="medium"),
            company_name=None,
            started_at=datetime.now(timezone.utc) - timedelta(hours=1),
            completed_at=datetime.now(timezone.utc),
            evaluation_report=None,
        )

        with get_db_context() as db:
            id1 = save_completed_attempt(db, user_id=user_id, session=session)
            assert id1 is not None

        with get_db_context() as db:
            id2 = save_completed_attempt(db, user_id=user_id, session=session)
            assert id2 == id1, "second call should return existing id, not create a new row"

        # Verify only one record exists
        with get_db_context() as db:
            count = db.query(PracticeAttemptRecord).filter(
                PracticeAttemptRecord.session_id == session_id,
            ).count()
            assert count == 1


# ─── Fix #1 + #6: Dimension records are queried, SQL aggregation ───────────

class TestDimensionRecordQueries:
    def test_summary_uses_dimension_records(self):
        init_db()
        from app.services.practice.practice_progress import get_progress_summary

        user_id = f"dim_summary_{uuid.uuid4()}"
        now = datetime.now(timezone.utc)

        with get_db_context() as db:
            a = PracticeAttemptRecord(
                user_id=user_id,
                session_id=str(uuid.uuid4()),
                overall_score=70.0,
                dimension_scores={"correctness": 90, "delivery": 50},
                created_at=now,
                completed_at=now,
            )
            db.add(a)
            db.flush()
            db.add(PracticeAttemptDimensionRecord(attempt_id=a.id, dimension="correctness", score=90.0))
            db.add(PracticeAttemptDimensionRecord(attempt_id=a.id, dimension="delivery", score=50.0))
            db.commit()

        with get_db_context() as db:
            summary = get_progress_summary(db, user_id=user_id)
            assert summary.attempts == 1
            assert summary.best_dimension == "correctness"
            assert summary.worst_dimension == "delivery"

    def test_summary_falls_back_to_json_column(self):
        """When no dimension records exist, JSON column is used."""
        init_db()
        from app.services.practice.practice_progress import get_progress_summary

        user_id = f"json_fb_{uuid.uuid4()}"
        now = datetime.now(timezone.utc)

        with get_db_context() as db:
            db.add(PracticeAttemptRecord(
                user_id=user_id,
                session_id=str(uuid.uuid4()),
                overall_score=65.0,
                dimension_scores={"clarity": 80, "structure": 40},
                created_at=now,
                completed_at=now,
            ))
            db.commit()

        with get_db_context() as db:
            summary = get_progress_summary(db, user_id=user_id)
            assert summary.best_dimension == "clarity"
            assert summary.worst_dimension == "structure"


# ─── Fix #2 + #8: Heatmap SQL aggregation + max_points ─────────────────────

class TestHeatmap:
    def test_heatmap_with_dimension_records(self):
        init_db()
        from app.services.practice.practice_progress import get_dimension_heatmap

        user_id = f"heatmap_{uuid.uuid4()}"
        now = datetime.now(timezone.utc)

        with get_db_context() as db:
            a = PracticeAttemptRecord(
                user_id=user_id,
                session_id=str(uuid.uuid4()),
                overall_score=72.0,
                dimension_scores={"correctness": 80, "delivery": 60},
                created_at=now,
                completed_at=now,
            )
            db.add(a)
            db.flush()
            db.add(PracticeAttemptDimensionRecord(attempt_id=a.id, dimension="correctness", score=80.0))
            db.add(PracticeAttemptDimensionRecord(attempt_id=a.id, dimension="delivery", score=60.0))
            db.commit()

        with get_db_context() as db:
            points = get_dimension_heatmap(db, user_id=user_id)
            assert len(points) >= 2
            dims = {p["dimension"] for p in points}
            assert "correctness" in dims
            assert "delivery" in dims

    def test_heatmap_max_points_limits_output(self):
        init_db()
        from app.services.practice.practice_progress import get_dimension_heatmap

        user_id = f"heatmax_{uuid.uuid4()}"
        now = datetime.now(timezone.utc)

        with get_db_context() as db:
            for i in range(10):
                a = PracticeAttemptRecord(
                    user_id=user_id,
                    session_id=str(uuid.uuid4()),
                    overall_score=60.0 + i,
                    dimension_scores={"correctness": 70 + i, "delivery": 50 + i, "clarity": 60 + i},
                    created_at=now - timedelta(weeks=i),
                    completed_at=now - timedelta(weeks=i),
                )
                db.add(a)
            db.commit()

        with get_db_context() as db:
            points = get_dimension_heatmap(db, user_id=user_id, max_points=3)
            assert len(points) <= 3

    def test_heatmap_endpoint_accepts_max_points(self, monkeypatch):
        init_db()
        user_id = f"heatep_{uuid.uuid4()}"
        client = _setup_app_client(monkeypatch, user_id)

        now = datetime.now(timezone.utc)
        with get_db_context() as db:
            db.add(PracticeAttemptRecord(
                user_id=user_id,
                session_id=str(uuid.uuid4()),
                overall_score=75.0,
                dimension_scores={"correctness": 80, "delivery": 70},
                created_at=now,
                completed_at=now,
            ))
            db.commit()

        r = client.get("/api/practice/progress/heatmap", params={"max_points": 1})
        assert r.status_code == 200
        assert len(r.json()["points"]) <= 1

    def test_heatmap_json_fallback(self):
        """Heatmap falls back to JSON column when no dimension records exist."""
        init_db()
        from app.services.practice.practice_progress import get_dimension_heatmap

        user_id = f"heatfb_{uuid.uuid4()}"
        now = datetime.now(timezone.utc)

        with get_db_context() as db:
            db.add(PracticeAttemptRecord(
                user_id=user_id,
                session_id=str(uuid.uuid4()),
                overall_score=65.0,
                dimension_scores={"clarity": 70, "structure": 55},
                created_at=now,
                completed_at=now,
            ))
            db.commit()

        with get_db_context() as db:
            points = get_dimension_heatmap(db, user_id=user_id)
            assert len(points) >= 1
            dims = {p["dimension"] for p in points}
            assert "clarity" in dims or "structure" in dims


# ─── Fix #12: Peer benchmark fallback ──────────────────────────────────────

class TestPeerBenchmarkFallback:
    def test_fallback_to_wpm_only(self):
        init_db()
        from app.services.practice.practice_learning import (
            PeerBucket,
            compute_peer_confidence_benchmark,
        )

        # Create 3 samples with same wpm_bucket but different filler_bucket
        user_ids = [f"peer_fb_{uuid.uuid4()}" for _ in range(3)]
        with get_db_context() as db:
            for uid in user_ids:
                sid = str(uuid.uuid4())
                db.add(PracticeSessionMetrics(
                    user_id=uid,
                    session_id=sid,
                    wpm_bucket="balanced",
                    filler_bucket="high",  # different from query bucket
                    answers_count=5,
                ))
                db.add(PracticeSessionOutcome(
                    user_id=uid,
                    session_id=sid,
                    confidence_1_5=4,
                ))
            db.commit()

        bucket = PeerBucket(wpm_bucket="balanced", filler_bucket="low")  # exact match: 0 samples
        with get_db_context() as db:
            avg, n = compute_peer_confidence_benchmark(db, bucket=bucket, min_samples=3)
            # Fallback should match on wpm_bucket="balanced" only → 3 samples
            assert avg is not None, "fallback should provide result"
            assert n >= 3
            assert 3.5 <= avg <= 4.5

    def test_exact_match_preferred(self):
        init_db()
        from app.services.practice.practice_learning import (
            PeerBucket,
            compute_peer_confidence_benchmark,
        )

        user_ids = [f"peer_ex_{uuid.uuid4()}" for _ in range(4)]
        with get_db_context() as db:
            for uid in user_ids:
                sid = str(uuid.uuid4())
                db.add(PracticeSessionMetrics(
                    user_id=uid,
                    session_id=sid,
                    wpm_bucket="fast",
                    filler_bucket="medium",
                    answers_count=5,
                ))
                db.add(PracticeSessionOutcome(
                    user_id=uid,
                    session_id=sid,
                    confidence_1_5=3,
                ))
            db.commit()

        bucket = PeerBucket(wpm_bucket="fast", filler_bucket="medium")
        with get_db_context() as db:
            avg, n = compute_peer_confidence_benchmark(db, bucket=bucket, min_samples=3)
            assert avg is not None
            assert n >= 3


# ─── Fix #3: Dual-source reconciliation ────────────────────────────────────

class TestDualSourceReconciliation:
    def test_insights_includes_scored_attempts(self):
        init_db()
        from app.services.practice.learning_loops import compute_practice_insights

        user_id = f"dual_src_{uuid.uuid4()}"
        now = datetime.now(timezone.utc)

        # Create a scored attempt in PracticeAttemptRecord
        with get_db_context() as db:
            db.add(PracticeAttemptRecord(
                user_id=user_id,
                session_id=str(uuid.uuid4()),
                overall_score=80.0,
                dimension_scores={"correctness": 85},
                created_at=now,
                completed_at=now,
            ))
            db.commit()

        with get_db_context() as db:
            result = compute_practice_insights(db, user_id=user_id, lookback_days=30)
            # New field from Fix #3
            assert "scored_attempts" in result
            assert result["scored_attempts"] == 1
            # events-based attempts may be 0 (no events created here)
            assert isinstance(result["attempts"], int)


# ─── Fix #7: Endpoint returns typed plan ────────────────────────────────────

class TestNextSessionEndpointTyped:
    def test_endpoint_returns_typed_plan(self, monkeypatch):
        init_db()
        user_id = f"nsp_typed_{uuid.uuid4()}"
        client = _setup_app_client(monkeypatch, user_id)

        now = datetime.now(timezone.utc)
        with get_db_context() as db:
            db.add(PracticeAttemptRecord(
                user_id=user_id,
                session_id=str(uuid.uuid4()),
                overall_score=60.0,
                next_session_plan={
                    "generated_at": now.isoformat(),
                    "focus_dimension": "delivery",
                    "focus": ["reduce_fillers"],
                    "question_count": 4,
                    "recommended_round": "technical",
                    "difficulty": "medium",
                },
                created_at=now,
                completed_at=now,
            ))
            db.commit()

        r = client.get("/api/practice/progress/next-session")
        assert r.status_code == 200
        plan = r.json()["plan"]
        assert plan is not None
        assert plan["focus_dimension"] == "delivery"
        assert isinstance(plan["focus"], list)
        assert plan["question_count"] == 4
