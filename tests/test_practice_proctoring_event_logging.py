"""Regression tests: Practice Mode proctoring event ingest.

We keep this test lightweight and deterministic by stubbing PracticeModeService.
The backend is expected to:
- validate the practice session exists
- log a structured EventRecord with a practice_proctoring_* event_type
- store severity + metadata (no media)
"""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.database import init_db, get_db_context
from app.models import EventRecord
from app.routers.practice_mode import router as practice_router
from app.schemas import PracticeInterviewQuestion, PracticeSession, QuestionDifficulty


class _DummyPracticeService:
    def __init__(self):
        self.audio_dir = Path("data/practice_audio")
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.sessions: dict[str, PracticeSession] = {}

    async def start_interview(self, difficulty, user_profile=None, question_count=5, round_type=None, api_key=None):
        session_id = str(uuid.uuid4())
        q1 = PracticeInterviewQuestion(
            id=1,
            text="Tell me about a time you disagreed with a teammate.",
            difficulty=QuestionDifficulty.MEDIUM,
            category="behavioral",
        )
        session = PracticeSession(session_id=session_id, questions=[q1])
        self.sessions[session_id] = session
        return session_id, q1, f"{session_id}_q1.mp3"

    def get_session(self, session_id: str):
        return self.sessions.get(session_id)


def test_practice_mode_proctoring_event_logged(monkeypatch):
    init_db()

    app = FastAPI()
    app.include_router(practice_router)
    client = TestClient(app)

    user_id = f"test_practice_proctoring_{uuid.uuid4()}"

    import app.routers.practice_mode as pm

    pm.practice_service = _DummyPracticeService()
    monkeypatch.setattr(pm, "get_user_id_from_request", lambda _req: user_id)

    # Start interview to mint a valid practice session
    start = client.post(
        "/api/practice/interview/start",
        headers={"X-API-Key": "test_key"},
        json={
            "screen_shared": True,
            "camera_enabled": True,
            "difficulty": "medium",
            "category": "behavioral",
            "question_count": 1,
        },
    )
    assert start.status_code == 200
    session_id = start.json()["session_id"]

    # Ingest a proctoring event
    ingest = client.post(
        "/api/practice/proctoring/event",
        json={
            "session_id": session_id,
            "event_type": "tab_switch",
            "severity": "violation",
            "metadata": {"reason": "visibilitychange", "hidden": True},
        },
    )
    assert ingest.status_code == 200
    assert ingest.json().get("ok") is True

    with get_db_context() as db:
        rows = (
            db.query(EventRecord)
            .filter(EventRecord.user_id == user_id)
            .filter(EventRecord.session_id == session_id)
            .filter(EventRecord.event_type == "practice_proctoring_tab_switch")
            .all()
        )

    assert len(rows) == 1
    assert (rows[0].extra_data or {}).get("severity") == "violation"
    assert ((rows[0].extra_data or {}).get("metadata") or {}).get("hidden") is True
