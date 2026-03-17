"""Regression tests for backend-authoritative practice proctoring enforcement."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import uuid

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.database import get_db_context, init_db
from app.models import PracticeProctoringSession
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
            text="Describe a time you had to recover from an outage.",
            difficulty=QuestionDifficulty.MEDIUM,
            category="behavioral",
        )
        session = PracticeSession(session_id=session_id, questions=[q1])
        self.sessions[session_id] = session
        return session_id, q1, f"{session_id}_q1.mp3"

    def get_session(self, session_id: str):
        return self.sessions.get(session_id)


def _client(monkeypatch):
    init_db()

    app = FastAPI()
    app.include_router(practice_router)
    client = TestClient(app)

    import app.routers.practice_mode as pm

    pm.practice_service = _DummyPracticeService()
    monkeypatch.setattr(pm, "get_user_id_from_request", lambda _req: "test_user")
    return client, pm


def _start_session(client: TestClient) -> str:
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
    return start.json()["session_id"]


def test_legacy_proctoring_event_route_updates_backend_status(monkeypatch):
    client, _pm = _client(monkeypatch)
    session_id = _start_session(client)

    status_before = client.get(f"/api/practice/session/{session_id}/proctoring/status")
    assert status_before.status_code == 200
    assert status_before.json()["status"] == "ACTIVE"
    assert status_before.json()["total_violations"] == 0

    event = client.post(
        "/api/practice/proctoring/event",
        json={
            "session_id": session_id,
            "event_type": "tab_switch",
            "severity": "warning",
            "metadata": {"reason": "visibilitychange"},
        },
    )
    assert event.status_code == 200
    body = event.json()
    assert body["status"] == "WARNING"
    assert body["action"] == "warn"
    assert body["total_violations"] == 1
    assert body["serious_violations"] == 0

    status_after = client.get(f"/api/practice/session/{session_id}/proctoring/status")
    assert status_after.status_code == 200
    assert status_after.json()["status"] == "ACTIVE"
    assert status_after.json()["action"] == "none"
    assert status_after.json()["total_violations"] == 1
    assert status_after.json()["remaining_total_before_termination"] == 4


def test_proctoring_heartbeat_and_serious_violations_can_terminate(monkeypatch):
    client, _pm = _client(monkeypatch)
    session_id = _start_session(client)

    heartbeat = client.post(
        f"/api/practice/session/{session_id}/proctoring/heartbeat",
        json={
            "camera_active": False,
            "screen_active": True,
            "tab_active": True,
            "window_focused": True,
            "detection_active": True,
            "display_surface": "monitor",
        },
    )
    assert heartbeat.status_code == 200
    hb_body = heartbeat.json()
    assert hb_body["status"] == "WARNING"
    assert hb_body["serious_violations"] == 1
    assert hb_body["total_violations"] == 1

    event_one = client.post(
        f"/api/practice/session/{session_id}/proctoring/event",
        json={"event_type": "MULTIPLE_FACES"},
    )
    assert event_one.status_code == 200
    assert event_one.json()["serious_violations"] == 2
    assert event_one.json()["status"] == "WARNING"

    event_two = client.post(
        f"/api/practice/session/{session_id}/proctoring/event",
        json={"event_type": "SCREEN_STOPPED"},
    )
    assert event_two.status_code == 200
    termination = event_two.json()
    assert termination["status"] == "TERMINATED"
    assert termination["action"] == "terminate"
    assert termination["serious_violations"] == 3
    assert termination["total_violations"] == 3
    assert termination["terminated_reason"]

    status = client.get(f"/api/practice/session/{session_id}/proctoring/status")
    assert status.status_code == 200
    assert status.json()["status"] == "TERMINATED"

    score = client.get(f"/api/practice/session/{session_id}/score")
    assert score.status_code == 200
    proctoring_summary = score.json()["proctoring_summary"]
    assert proctoring_summary["status"] == "TERMINATED"
    assert proctoring_summary["total_violation_count"] == 3
    assert proctoring_summary["serious_violation_count"] == 3
    assert proctoring_summary["event_counts"]["CAMERA_STOPPED"] == 1
    assert proctoring_summary["event_counts"]["MULTIPLE_FACES"] == 1
    assert proctoring_summary["event_counts"]["SCREEN_STOPPED"] == 1


def test_status_marks_heartbeat_stale(monkeypatch):
    client, _pm = _client(monkeypatch)
    session_id = _start_session(client)

    heartbeat = client.post(
        f"/api/practice/session/{session_id}/proctoring/heartbeat",
        json={
            "camera_active": True,
            "screen_active": True,
            "tab_active": True,
            "window_focused": True,
            "detection_active": True,
            "display_surface": "monitor",
        },
    )
    assert heartbeat.status_code == 200
    assert heartbeat.json()["status"] == "ACTIVE"

    with get_db_context() as db:
        state = (
            db.query(PracticeProctoringSession)
            .filter(PracticeProctoringSession.session_id == session_id)
            .first()
        )
        assert state is not None
        state.last_heartbeat_at = datetime.now(timezone.utc) - timedelta(seconds=30)
        db.add(state)
        db.commit()

    status = client.get(f"/api/practice/session/{session_id}/proctoring/status")
    assert status.status_code == 200
    body = status.json()
    assert body["heartbeat_stale"] is True
    assert body["status"] == "WARNING"
    assert body["risk_level"] == "MEDIUM"


def test_repeated_heartbeat_respects_cooldown_without_datetime_crash(monkeypatch):
    client, _pm = _client(monkeypatch)
    session_id = _start_session(client)

    first = client.post(
        f"/api/practice/session/{session_id}/proctoring/heartbeat",
        json={
            "camera_active": False,
            "screen_active": True,
            "tab_active": True,
            "window_focused": True,
            "detection_active": True,
            "display_surface": "monitor",
        },
    )
    assert first.status_code == 200
    assert first.json()["total_violations"] == 1
    assert first.json()["serious_violations"] == 1

    second = client.post(
        f"/api/practice/session/{session_id}/proctoring/heartbeat",
        json={
            "camera_active": False,
            "screen_active": True,
            "tab_active": True,
            "window_focused": True,
            "detection_active": True,
            "display_surface": "monitor",
        },
    )
    assert second.status_code == 200
    assert second.json()["status"] == "ACTIVE"
    assert second.json()["action"] == "none"
    assert second.json()["total_violations"] == 1
    assert second.json()["serious_violations"] == 1

    status = client.get(f"/api/practice/session/{session_id}/proctoring/status")
    assert status.status_code == 200
    assert status.json()["status"] == "ACTIVE"
    assert status.json()["action"] == "none"
    assert status.json()["total_violations"] == 1


def test_repeated_window_blur_heartbeat_does_not_accumulate_false_terminations(monkeypatch):
    client, _pm = _client(monkeypatch)
    session_id = _start_session(client)

    first = client.post(
        f"/api/practice/session/{session_id}/proctoring/heartbeat",
        json={
            "camera_active": True,
            "screen_active": True,
            "tab_active": True,
            "window_focused": False,
            "detection_active": True,
            "display_surface": "monitor",
        },
    )
    assert first.status_code == 200
    assert first.json()["total_violations"] == 0
    assert first.json()["status"] == "ACTIVE"

    second = client.post(
        f"/api/practice/session/{session_id}/proctoring/heartbeat",
        json={
            "camera_active": True,
            "screen_active": True,
            "tab_active": True,
            "window_focused": False,
            "detection_active": True,
            "display_surface": "monitor",
        },
    )
    assert second.status_code == 200
    assert second.json()["total_violations"] == 1
    assert second.json()["status"] == "WARNING"

    third = client.post(
        f"/api/practice/session/{session_id}/proctoring/heartbeat",
        json={
            "camera_active": True,
            "screen_active": True,
            "tab_active": True,
            "window_focused": False,
            "detection_active": False,
            "display_surface": "monitor",
        },
    )
    assert third.status_code == 200
    assert third.json()["total_violations"] == 1
    assert third.json()["serious_violations"] == 0

    fourth = client.post(
        f"/api/practice/session/{session_id}/proctoring/heartbeat",
        json={
            "camera_active": True,
            "screen_active": True,
            "tab_active": True,
            "window_focused": False,
            "detection_active": False,
            "display_surface": "monitor",
        },
    )
    assert fourth.status_code == 200
    assert fourth.json()["total_violations"] == 2
    assert fourth.json()["serious_violations"] == 1
    assert fourth.json()["status"] == "WARNING"

    score = client.get(f"/api/practice/session/{session_id}/score")
    assert score.status_code == 200
    summary = score.json()["proctoring_summary"]
    assert summary["status"] == "WARNING"
    assert summary["total_violation_count"] == 2
    assert summary["serious_violation_count"] == 1
    assert summary["event_counts"]["WINDOW_BLUR"] == 1
    assert summary["event_counts"]["MONITORING_INTERRUPTED"] == 1