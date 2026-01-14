"""Regression: frontend expects /api/practice/session/{id}/score.

We return deterministic scoring based on runtime session when available.
"""

from __future__ import annotations

import uuid

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers.practice_mode import router as practice_router
from app.schemas import (
    MicroFeedback,
    PracticeInterviewQuestion,
    PracticeSession,
    QuestionDifficulty,
    SpeechMetrics,
)


class _DummyPracticeService:
    def __init__(self, session: PracticeSession):
        self._session = session

    def get_session(self, session_id: str):
        if session_id == self._session.session_id:
            return self._session
        return None


def test_session_score_endpoint_returns_200(monkeypatch):
    app = FastAPI()
    app.include_router(practice_router)
    client = TestClient(app)

    session_id = str(uuid.uuid4())

    q1 = PracticeInterviewQuestion(
        id=1,
        text="Explain bias-variance tradeoff.",
        difficulty=QuestionDifficulty.EASY,
        category="technical",
    )

    metrics = SpeechMetrics(
        filler_count=1,
        wpm=155.0,
        longest_silence=0.5,
        confidence_score=7.0,
        overtalked=False,
        duration=40.0,
        filler_words=["um"],
        pause_count=1,
        pitch_variance=0.2,
        silence_removed=0.0,
    )
    micro = MicroFeedback(
        delivery_tips=["Good pace"],
        pace_feedback="Good",
        overall_note="Solid",
        correctness_score=75,
        technical_accuracy="Good",
        is_correct=True,
    )

    # Reuse schema from app.schemas
    from app.schemas import AnswerSubmission

    sess = PracticeSession(
        session_id=session_id,
        questions=[q1],
        answers=[
            AnswerSubmission(
                question_id=1,
                transcript="Bias-variance tradeoff is...",
                metrics=metrics,
                micro_feedback=micro,
                audio_duration=metrics.duration,
            )
        ],
        current_question_index=0,
        is_complete=True,
    )

    import app.routers.practice_mode as pm

    pm.practice_service = _DummyPracticeService(sess)
    monkeypatch.setattr(pm, "get_user_id_from_request", lambda _req: "guest_unknown")

    resp = client.get(f"/api/practice/session/{session_id}/score")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["session_id"] == session_id
    assert body["source"] == "runtime"
    assert "overall_score" in body
    assert "dimension_scores" in body
