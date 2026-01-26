"""Regression tests: Practice Mode coding submission endpoint.

This verifies that the frontend-facing endpoint exists (no 404) and that it
advances session state without requiring external sandbox calls.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.database import init_db
from app.routers.practice_mode import router as practice_router
from app.schemas import PracticeInterviewQuestion, PracticeSession, QuestionDifficulty, QuestionType


class _DummyPracticeService:
    def __init__(self):
        self.audio_dir = Path("data/practice_audio")
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.sessions: dict[str, PracticeSession] = {}

    async def start_interview(self, difficulty, user_profile=None, question_count=5, round_type=None, api_key=None):
        session_id = str(uuid.uuid4())
        q1 = PracticeInterviewQuestion(
            id=1,
            text="Write a function to reverse a string.",
            difficulty=QuestionDifficulty.MEDIUM,
            category="coding",
            question_type=QuestionType.CODING,
            programming_language="python",
            test_cases=[{"input": "abc", "expected_output": "cba"}],
        )
        session = PracticeSession(session_id=session_id, questions=[q1])
        self.sessions[session_id] = session
        return session_id, q1, f"{session_id}_q1.mp3"

    def get_session(self, session_id: str):
        return self.sessions.get(session_id)

    async def submit_code(self, **kwargs):
        # The real service method is exercised elsewhere; here we only need a stable response shape.
        return {"complete": False, "progress": "1/1", "requires_acknowledgment": True, "evaluation_report": None}


def test_practice_mode_submit_code_endpoint_exists(monkeypatch):
    init_db()

    app = FastAPI()
    app.include_router(practice_router)
    client = TestClient(app)

    import app.routers.practice_mode as pm

    pm.practice_service = _DummyPracticeService()
    monkeypatch.setattr(pm, "get_user_id_from_request", lambda _req: "test_user")

    start = client.post(
        "/api/practice/interview/start",
        headers={"X-API-Key": "test_key"},
        json={"difficulty": "medium", "category": "coding", "question_count": 1},
    )
    assert start.status_code == 200
    session_id = start.json()["session_id"]

    submit = client.post(
        "/api/practice/interview/submit-code",
        json={
            "session_id": session_id,
            "question_id": 1,
            "code": "def rev(s):\n    return s[::-1]\n",
            "programming_language": "python",
            "time_taken": 10,
        },
    )

    # Key assertion: endpoint exists and responds (not 404)
    assert submit.status_code == 200
    body = submit.json()
    assert "code_feedback" in body
    assert "test_results" in body
    assert body.get("requires_acknowledgment") is True

    # Alias spelling supported for older clients
    submit_alias = client.post(
        "/api/practice/interview/submit_code",
        json={
            "session_id": session_id,
            "question_id": 1,
            "code": "def rev(s):\n    return s[::-1]\n",
            "programming_language": "python",
            "time_taken": 10,
        },
    )

    assert submit_alias.status_code == 200
