"""Mock Interview end-session payload regression test.

Why:
- The UI summary card reads timing and score stats.
- Some clients expect summary fields (e.g. total_time_seconds) at the top level.

Contract:
- POST /api/mock-interview/sessions/{session_id}/end returns:
  - status == "completed"
  - total_time_seconds present at top level
  - score_range present at top level (may be 0.0)
  - ended_early / questions_skipped metadata
  - evaluations[] each contain model_answer and user_answer
  - skipped_questions[] list unanswered questions
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient


def _make_app():
    from app.routers.mock_interview import router as mock_router, get_mock_service

    class _DummyService:
        async def end_session(self, session_id: str):
            return self._summaries.get(session_id, {})

        _summaries: dict = {}

    svc = _DummyService()

    app = FastAPI()
    app.include_router(mock_router, prefix="/api/mock-interview")
    app.dependency_overrides[get_mock_service] = lambda: svc
    return app, svc


def test_end_session_flattens_summary_fields(monkeypatch):
    app, svc = _make_app()
    svc._summaries["s1"] = {
        "session_id": "s1",
        "total_time_seconds": 148,
        "individual_scores": [8.0, 8.0],
        "best_score": 8.0,
        "lowest_score": 8.0,
        "score_range": 0.0,
        "ended_early": False,
        "questions_skipped": 0,
    }

    client = TestClient(app)
    resp = client.post("/api/mock-interview/sessions/s1/end")

    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "completed"
    assert data["total_time_seconds"] == 148
    assert data["score_range"] == 0.0


def test_end_session_early_termination_metadata():
    """When user ends mid-interview, response carries early-termination fields."""
    app, svc = _make_app()
    svc._summaries["early1"] = {
        "session_id": "early1",
        "total_questions": 5,
        "questions_answered": 2,
        "ended_early": True,
        "questions_skipped": 3,
        "total_time_seconds": 60,
        "individual_scores": [7.0, 6.5],
        "best_score": 7.0,
        "lowest_score": 6.5,
        "score_range": 0.5,
        "average_score": 6.8,
        "evaluations": [
            {
                "question_number": 1,
                "question": "What is REST?",
                "user_answer": "REST stands for...",
                "model_answer": "REST (Representational State Transfer) is an architectural style...",
                "score": 7.0,
            },
            {
                "question_number": 2,
                "question": "Explain CAP theorem.",
                "user_answer": "CAP is about consistency...",
                "model_answer": "The CAP theorem states that a distributed system...",
                "score": 6.5,
            },
        ],
        "skipped_questions": [
            {"question_number": 3, "question": "Design a URL shortener."},
            {"question_number": 4, "question": "What is sharding?"},
            {"question_number": 5, "question": "Explain event sourcing."},
        ],
    }

    client = TestClient(app)
    resp = client.post("/api/mock-interview/sessions/early1/end")

    assert resp.status_code == 200
    data = resp.json()

    # Core early-end fields
    assert data["status"] == "completed"
    assert data["ended_early"] is True
    assert data["questions_skipped"] == 3
    assert data["questions_answered"] == 2
    assert data["total_questions"] == 5

    # Per-question evaluation includes model_answer and user_answer
    evals = data["evaluations"]
    assert len(evals) == 2
    assert evals[0]["model_answer"] != ""
    assert evals[0]["user_answer"] == "REST stands for..."
    assert evals[1]["model_answer"] != ""
    assert evals[1]["user_answer"] == "CAP is about consistency..."

    # Skipped questions listed
    skipped = data["skipped_questions"]
    assert len(skipped) == 3
    assert skipped[0]["question"] == "Design a URL shortener."


def test_end_session_model_answer_in_evaluations():
    """Every per-question evaluation carries model_answer so users can compare."""
    app, svc = _make_app()
    svc._summaries["ma1"] = {
        "session_id": "ma1",
        "total_questions": 2,
        "questions_answered": 2,
        "ended_early": False,
        "questions_skipped": 0,
        "total_time_seconds": 180,
        "individual_scores": [9.0, 8.5],
        "best_score": 9.0,
        "lowest_score": 8.5,
        "score_range": 0.5,
        "evaluations": [
            {
                "question_number": 1,
                "question": "Explain polymorphism.",
                "user_answer": "Polymorphism means many forms...",
                "model_answer": "Polymorphism is a core OOP concept that allows objects..."
                                " to take multiple forms through method overriding/overloading.",
                "score": 9.0,
            },
            {
                "question_number": 2,
                "question": "What are design patterns?",
                "user_answer": "Design patterns are reusable solutions...",
                "model_answer": "Design patterns are well-proven, reusable solutions"
                                " to common software design problems, categorized as"
                                " creational, structural, and behavioral.",
                "score": 8.5,
            },
        ],
        "skipped_questions": [],
    }

    client = TestClient(app)
    resp = client.post("/api/mock-interview/sessions/ma1/end")

    assert resp.status_code == 200
    data = resp.json()

    for ev in data["evaluations"]:
        assert "model_answer" in ev, f"Missing model_answer in question {ev['question_number']}"
        assert len(ev["model_answer"]) > 10, "model_answer should be a substantive answer"
        assert "user_answer" in ev, f"Missing user_answer in question {ev['question_number']}"
        assert len(ev["user_answer"]) > 0
