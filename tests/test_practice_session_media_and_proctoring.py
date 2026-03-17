"""Regression tests: Live Practice proctoring + recording (backend).

This covers the MVP backend responsibilities:
- Gate interview start on required media permissions.
- Accept media uploads (store file + DB row; no processing).
- Accept proctoring events (DB audit trail).
- Expose aggregation in the session score/final report payload.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.database import init_db, get_db_context
from app.models import PracticeProctoringEvent, PracticeSessionMedia
from app.routers.practice_mode import router as practice_router
from app.schemas import PracticeInterviewQuestion, PracticeSession, QuestionDifficulty


class _DummyPracticeService:
    def __init__(self):
        self.audio_dir = Path("data/practice_audio")
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.sessions: dict[str, PracticeSession] = {}

    async def start_interview(self, difficulty, user_profile=None, question_count=5, round_type=None, api_key=None, **kwargs):
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


def test_live_practice_start_requires_media(monkeypatch):
    init_db()

    app = FastAPI()
    app.include_router(practice_router)
    client = TestClient(app)

    import app.routers.practice_mode as pm

    pm.practice_service = _DummyPracticeService()
    monkeypatch.setattr(pm, "get_user_id_from_request", lambda _req: "test_user")

    # Enforced gate: screen_shared + camera_enabled must be true.
    resp = client.post(
        "/api/practice/interview/start",
        headers={"X-API-Key": "test_key"},
        json={
            "screen_shared": True,
            "camera_enabled": False,
            "difficulty": "medium",
            "category": "behavioral",
            "question_count": 1,
        },
    )
    assert resp.status_code == 403


def test_media_upload_and_proctoring_event_aggregate_into_score(monkeypatch):
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

    # DB-backed proctoring event
    pe = client.post(
        f"/api/practice/session/{session_id}/proctoring/event",
        json={"event_type": "SCREEN_STOPPED"},
    )
    assert pe.status_code == 200

    # Media upload
    up = client.post(
        f"/api/practice/session/{session_id}/media",
        files={"file": ("combined.webm", b"WEBM", "video/webm")},
        data={"media_type": "combined", "duration_seconds": "10"},
    )
    assert up.status_code == 200
    media = up.json()
    assert media["storage_url"].startswith(f"/api/practice/session/{session_id}/media/")

    # Score endpoint should surface aggregation
    score = client.get(f"/api/practice/session/{session_id}/score")
    assert score.status_code == 200
    body = score.json()
    assert "media" in body
    assert "proctoring_summary" in body
    assert body["media"]["screen_recording_url"] == media["storage_url"]
    assert body["proctoring_summary"]["violation_count"] == 1
    events = body["proctoring_summary"]["events"]
    assert any((e or {}).get("event_type") == "SCREEN_STOPPED" for e in (events or []))

    # Verify DB rows exist
    with get_db_context() as db:
        assert db.query(PracticeSessionMedia).filter(PracticeSessionMedia.session_id == session_id).count() == 1
        assert db.query(PracticeProctoringEvent).filter(PracticeProctoringEvent.session_id == session_id).count() >= 1
