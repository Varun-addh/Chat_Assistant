"""Regression tests for preventing empty (0-question) history entries.

These tests are intentionally lightweight and do not require external services.
"""

import asyncio

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers.history import router as history_router
from app.services.history_manager import HistoryManager


def test_history_manager_refuses_empty_questions():
    async def _run():
        manager = HistoryManager(user_id="test_empty_questions_guard")
        await manager.initialize()

        with pytest.raises(ValueError):
            await manager.save_search(query="test", questions=[], metadata={})

    asyncio.run(_run())


def test_history_api_rejects_empty_questions_payload():
    app = FastAPI()
    app.include_router(history_router, prefix="/api/history")

    client = TestClient(app)
    resp = client.post(
        "/api/history/",
        json={"query": "test", "questions": [], "metadata": {}},
    )

    assert resp.status_code == 422
