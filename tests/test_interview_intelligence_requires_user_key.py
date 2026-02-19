"""Regression test: Interview Intelligence should return 401 when user API key is required.

When REQUIRE_USER_API_KEY=true, the /api/intelligence/search endpoint must not silently
return empty results; it should tell the client to provide an API key.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import settings
from app.routers.interview_intelligence import router as intelligence_router


def test_intelligence_search_requires_user_key(monkeypatch):
    app = FastAPI()
    app.include_router(intelligence_router, prefix="/api/intelligence")
    client = TestClient(app)

    # Force require_user_api_key for this test
    monkeypatch.setattr(settings, "require_user_api_key", True, raising=False)

    resp = client.post(
        "/api/intelligence/search?save_to_history=false",
        json={"query": "diffusion models", "limit": 2, "refresh": True, "save_to_history": False},
    )

    assert resp.status_code == 401
    detail = (resp.json() or {}).get("detail", "")
    assert "API key required" in detail
