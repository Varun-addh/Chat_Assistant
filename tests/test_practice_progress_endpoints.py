from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.database import get_db_context, init_db
from app.models import PracticeAttemptRecord
from app.routers.practice_mode import router as practice_router


def test_practice_progress_endpoints(monkeypatch):
    init_db()

    app = FastAPI()
    app.include_router(practice_router)
    client = TestClient(app)

    user_id = f"test_practice_progress_{uuid.uuid4()}"

    import app.routers.practice_mode as pm

    monkeypatch.setattr(pm, "get_user_id_from_request", lambda _req: user_id)

    now = datetime.now(timezone.utc)

    with get_db_context() as db:
        a1 = PracticeAttemptRecord(
            user_id=user_id,
            session_id=str(uuid.uuid4()),
            domain="Python",
            round_type="technical_round_1",
            difficulty="medium",
            question_count=5,
            overall_score=62.0,
            dimension_scores={
                "correctness": 70.0,
                "delivery": 55.0,
                "clarity": 60.0,
                "structure": 63.0,
            },
            why=["Correctness avg: 70/100"],
            improvement_plan=["Use explicit structure"],
            next_session_plan={"focus_dimension": "delivery", "focus": ["reduce_fillers"], "question_count": 4},
            created_at=now - timedelta(days=10),
            completed_at=now - timedelta(days=10),
        )

        a2 = PracticeAttemptRecord(
            user_id=user_id,
            session_id=str(uuid.uuid4()),
            domain="Python",
            round_type="technical_round_1",
            difficulty="medium",
            question_count=5,
            overall_score=78.0,
            dimension_scores={
                "correctness": 82.0,
                "delivery": 70.0,
                "clarity": 75.0,
                "structure": 80.0,
            },
            why=["Correctness avg: 82/100"],
            improvement_plan=["Restate the problem first"],
            next_session_plan={"focus_dimension": "clarity", "focus": ["conciseness"], "question_count": 5},
            created_at=now - timedelta(days=2),
            completed_at=now - timedelta(days=2),
        )

        db.add(a1)
        db.add(a2)
        db.commit()

    # Summary
    r = client.get("/api/practice/progress/summary", params={"lookback_days": 30, "domain": "Python"})
    assert r.status_code == 200
    payload = r.json()
    assert payload["attempts"] == 2
    assert 60.0 <= payload["average_overall_score"] <= 80.0
    assert payload["best_dimension"]
    assert payload["worst_dimension"]

    # Heatmap
    r = client.get("/api/practice/progress/heatmap", params={"lookback_days": 30, "domain": "Python"})
    assert r.status_code == 200
    payload = r.json()
    assert "points" in payload
    assert len(payload["points"]) >= 1
    assert {"week_start", "dimension", "avg_score", "attempts"}.issubset(payload["points"][0].keys())

    # Next-session plan
    r = client.get("/api/practice/progress/next-session", params={"domain": "Python"})
    assert r.status_code == 200
    payload = r.json()
    assert payload["plan"] is not None
    assert payload["plan"]["focus_dimension"] in {"correctness", "delivery", "clarity", "structure"}
