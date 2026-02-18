"""Regression tests for abuse/defamation boundary handling.

Goal:
- After 1-2 abusive turns: boundary + redirect (no LLM call)
- After 3 abusive turns: disengage/lockout message
- Once the user resumes normal input: strikes reset and normal guards (identity) work
"""

from __future__ import annotations

import uuid

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers.questions import router as questions_router


@pytest.mark.fast
def test_abuse_guard_strikes_and_resets(monkeypatch):
    app = FastAPI()
    app.include_router(questions_router, prefix="/api")
    client = TestClient(app)

    user_id = f"test_abuse_user_{uuid.uuid4()}"

    import app.routers.questions as questions_mod

    monkeypatch.setattr(questions_mod, "get_user_id_from_request", lambda _req: user_id)

    session_id = str(uuid.uuid4())

    def ask(q: str):
        return client.post(
            "/api/question",
            headers={"X-API-Key": "test_key"},
            json={
                "session_id": session_id,
                "question": q,
                "stream": False,
                "save_to_history": False,
            },
        )

    r1 = ask("you are an idiot")
    assert r1.status_code == 200
    assert r1.headers.get("X-Stratax-Guard") == "abuse"
    assert "can’t engage" in r1.json()["answer"].lower() or "can't engage" in r1.json()["answer"].lower()

    r2 = ask("you are a fraud")
    assert r2.status_code == 200
    assert r2.headers.get("X-Stratax-Guard") == "abuse"

    r3 = ask("you stole my money")
    assert r3.status_code == 200
    assert r3.headers.get("X-Stratax-Guard") == "abuse_lockout"
    ans3 = r3.json()["answer"].lower()
    assert "can’t continue" in ans3 or "can't continue" in ans3

    # Normal question should reset strikes and allow identity guard.
    r4 = ask("who are you?")
    assert r4.status_code == 200
    assert r4.headers.get("X-Stratax-Guard") == "identity"
