from __future__ import annotations

import uuid

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers.questions import router as questions_router


@pytest.mark.fast
def test_design_roadmap_does_not_trigger_architecture_mode_chooser(monkeypatch):
    app = FastAPI()
    app.include_router(questions_router, prefix="/api")
    client = TestClient(app)

    import app.routers.questions as questions_mod

    monkeypatch.setattr(questions_mod, "get_user_id_from_request", lambda _req: "test_user")

    class _DummyLLM:
        def _is_identity_question(self, _q: str) -> bool:
            return False

        def _identity_response_text(self, _q: str) -> str:
            return ""

        async def generate_answer(self, *args, **kwargs):
            return ("Here is a roadmap...", False)

    monkeypatch.setattr(questions_mod, "llm_service", _DummyLLM())

    resp = client.post(
        "/api/question",
        headers={"X-API-Key": "test_key"},
        json={
            "session_id": str(uuid.uuid4()),
            "question": "Design a roadmap for a data engineer",
            "mode": "answer",
            "stream": False,
            "save_to_history": False,
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body.get("ui_action") != "choose_architecture_mode"
    assert (body.get("answer") or "").strip() != ""


@pytest.mark.fast
def test_system_design_still_triggers_architecture_mode_chooser(monkeypatch):
    app = FastAPI()
    app.include_router(questions_router, prefix="/api")
    client = TestClient(app)

    import app.routers.questions as questions_mod

    monkeypatch.setattr(questions_mod, "get_user_id_from_request", lambda _req: "test_user")

    class _DummyLLM:
        def _is_identity_question(self, _q: str) -> bool:
            return False

        def _identity_response_text(self, _q: str) -> str:
            return ""

        async def generate_answer(self, *args, **kwargs):
            return ("irrelevant", False)

    monkeypatch.setattr(questions_mod, "llm_service", _DummyLLM())

    resp = client.post(
        "/api/question",
        headers={"X-API-Key": "test_key"},
        json={
            "session_id": str(uuid.uuid4()),
            "question": "Design a system for video streaming like Netflix",
            "mode": "answer",
            "stream": False,
            "save_to_history": False,
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body.get("ui_action") == "choose_architecture_mode"
