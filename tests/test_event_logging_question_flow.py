"""Regression test: /api/question writes structured events.

We use an identity question to avoid external LLM calls.
"""

from __future__ import annotations

import uuid

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.database import init_db, get_db_context
from app.models import EventRecord
from app.routers.questions import router as questions_router


def test_question_flow_writes_event_records(monkeypatch):
    # Ensure DB tables exist (the lightweight app in this test has no lifespan).
    init_db()

    app = FastAPI()
    app.include_router(questions_router, prefix="/api")
    client = TestClient(app)

    user_id = f"test_event_user_{uuid.uuid4()}"

    import app.routers.questions as questions_mod

    monkeypatch.setattr(questions_mod, "get_user_id_from_request", lambda _req: user_id)

    resp = client.post(
        "/api/question",
        headers={"X-API-Key": "test_key"},
        json={
            "session_id": str(uuid.uuid4()),
            "question": "who are you?",
            "stream": False,
            "save_to_history": False,
        },
    )
    assert resp.status_code == 200

    effective_session_id = resp.headers.get("X-Stratax-Session-Id")
    assert effective_session_id

    with get_db_context() as db:
        rows = (
            db.query(EventRecord)
            .filter(EventRecord.user_id == user_id)
            .filter(EventRecord.session_id == effective_session_id)
            .all()
        )

    event_types = {r.event_type for r in rows}
    assert "chat_prompt_received" in event_types
    assert "chat_identity_guard" in event_types
