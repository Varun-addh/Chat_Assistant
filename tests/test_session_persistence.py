"""Regression tests for session persistence across refresh.

Bug scenario:
- Frontend creates a new session (no QnA yet)
- User refreshes the tab
- UI tries GET /api/session/{id}/chat and gets 404, then falls back to a different session
  (which looks like questions are "jumbling" between chats).

We persist a stub for newly created sessions so refresh can reload history consistently.
"""

from __future__ import annotations

import asyncio
import shutil
import uuid
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers.questions import router as questions_router
from app.services.core.session_manager import SessionManager


def _cleanup_user_sessions(user_id: str) -> None:
    shutil.rmtree(Path("data/sessions") / user_id, ignore_errors=True)


@pytest.mark.asyncio
async def test_create_session_persists_stub_to_disk() -> None:
    user_id = f"test_session_persist_{uuid.uuid4()}"
    data_dir = Path("data/sessions") / user_id

    try:
        manager = SessionManager(user_id=user_id)
        state = await manager.create_session()

        session_file = data_dir / f"{state.session_id}.json"
        assert session_file.exists(), "create_session should persist a stub JSON file"

        # Simulate a process restart / new worker by creating a new manager instance
        manager2 = SessionManager(user_id=user_id)
        loaded = await manager2.get(state.session_id)
        assert loaded is not None
        assert loaded.session_id == state.session_id
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)


def test_api_get_chat_after_refresh_does_not_404_for_new_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end-ish: create session, then get chat should succeed."""

    app = FastAPI()
    app.include_router(questions_router, prefix="/api")
    client = TestClient(app)

    user_id = f"test_session_chat_{uuid.uuid4()}"

    # Force the router to use an isolated per-test user bucket.
    import app.routers.questions as questions_mod

    monkeypatch.setattr(questions_mod, "get_user_id_from_request", lambda _req: user_id)

    try:
        manager = SessionManager(user_id=user_id)
        state = asyncio.run(manager.create_session())

        # Now fetch chat history for that session
        resp = client.get(f"/api/session/{state.session_id}/chat")
        assert resp.status_code == 200
        body = resp.json()
        assert body["session_id"] == state.session_id
        assert body["items"] == []
    finally:
        _cleanup_user_sessions(user_id)


def test_question_endpoint_sets_recovery_headers_for_missing_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """If a session_id is missing (e.g., after refresh), we should recover and
    tell the client the effective session id via headers.

    We use an identity question to avoid any external LLM calls.
    """

    app = FastAPI()
    app.include_router(questions_router, prefix="/api")
    client = TestClient(app)

    user_id = f"test_session_recovery_{uuid.uuid4()}"
    import app.routers.questions as questions_mod

    monkeypatch.setattr(questions_mod, "get_user_id_from_request", lambda _req: user_id)

    missing_session_id = str(uuid.uuid4())
    resp = client.post(
        "/api/question",
        headers={"X-API-Key": "test_key"},
        json={
            "session_id": missing_session_id,
            "question": "who are you?",
            "stream": False,
            "save_to_history": True,
        },
    )

    assert resp.status_code == 200
    assert resp.headers.get("X-Stratax-Session-Id"), "effective session id should be exposed"
    assert resp.headers.get("X-Stratax-Session-Recovered") == "1"
    assert resp.headers.get("X-Stratax-Old-Session-Id") == missing_session_id

    _cleanup_user_sessions(user_id)


def test_question_with_invalid_session_id_reuses_most_recent_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the frontend sends session_id like 'undefined', the backend should reuse the
    most recent session instead of creating a new session for every message.
	
    We use an identity question to avoid external LLM calls.
    """
    app = FastAPI()
    app.include_router(questions_router, prefix="/api")
    client = TestClient(app)

    user_id = f"test_invalid_sid_reuse_{uuid.uuid4()}"
    import app.routers.questions as questions_mod
    monkeypatch.setattr(questions_mod, "get_user_id_from_request", lambda _req: user_id)

    try:
        # Create an initial session and add one QnA so it's clearly "most recent"
        manager = SessionManager(user_id=user_id)
        state = asyncio.run(manager.create_session())
        asyncio.run(manager.append_qna(state.session_id, "hi", "hello"))

        # Now submit a question with an invalid session id
        resp = client.post(
            "/api/question",
            headers={"X-API-Key": "test_key"},
            json={
                "session_id": "undefined",
                "question": "who are you?",
                "stream": False,
                "save_to_history": True,
            },
        )
        assert resp.status_code == 200
        assert resp.headers.get("X-Stratax-Session-Id") == state.session_id
        assert resp.headers.get("X-Stratax-Session-Reused") == "1"
    finally:
        _cleanup_user_sessions(user_id)


def test_get_chat_migrates_legacy_guest_unknown_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """Legacy compatibility: sessions previously stored under data/sessions/guest_unknown
    should still load for newer guest_<hash> identities.

    Expected behavior:
    - GET /api/session/{id}/chat returns 200
    - Response includes migration header
    - Session file is moved to the new guest bucket
    """
    app = FastAPI()
    app.include_router(questions_router, prefix="/api")
    client = TestClient(app)

    new_guest_id = f"guest_test_{uuid.uuid4().hex[:12]}"
    import app.routers.questions as questions_mod
    monkeypatch.setattr(questions_mod, "get_user_id_from_request", lambda _req: new_guest_id)

    legacy_mgr = SessionManager(user_id="guest_unknown")
    new_mgr = SessionManager(user_id=new_guest_id)
    state = None
    try:
        state = asyncio.run(legacy_mgr.create_session())
        asyncio.run(legacy_mgr.append_qna(state.session_id, "hi", "hello"))

        # Sanity: legacy file exists
        from pathlib import Path
        legacy_path = Path("data/sessions") / "guest_unknown" / f"{state.session_id}.json"
        assert legacy_path.exists()

        resp = client.get(f"/api/session/{state.session_id}/chat")
        assert resp.status_code == 200
        assert resp.headers.get("X-Stratax-Session-Legacy-Migrated") == "1"
        body = resp.json()
        assert body["session_id"] == state.session_id
        assert len(body["items"]) == 1

        # Legacy should be removed and new should exist
        new_path = Path("data/sessions") / new_guest_id / f"{state.session_id}.json"
        assert new_path.exists()
        assert not legacy_path.exists()
    finally:
        # Cleanup both buckets (best effort)
        _cleanup_user_sessions(new_guest_id)
        # Avoid nuking developer guest_unknown bucket: delete only the test session file if present
        if state is not None:
            try:
                from pathlib import Path
                p = Path("data/sessions") / "guest_unknown" / f"{state.session_id}.json"
                if p.exists():
                    p.unlink()
            except Exception:
                pass
