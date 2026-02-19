"""Regression tests for rate-limiting behavior.

Goals:
- `/api/render_mermaid` must not be rate-limited (UI helper endpoint).
- When rate limits are exceeded, middleware must return HTTP 429 (not 500).

These tests are lightweight and do not require external services.
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.middleware.rate_limit import DEMO_LIMIT_QUESTIONS, rate_limit_middleware


def test_render_mermaid_not_rate_limited_for_guest():
    app = FastAPI()

    @app.middleware("http")
    async def _rl(request, call_next):
        return await rate_limit_middleware(request, call_next)

    @app.post("/api/render_mermaid")
    async def render_mermaid():
        return {"ok": True}

    client = TestClient(app)

    # Guest limit is 10/day on other endpoints, but render_mermaid must be exempt.
    for _ in range(25):
        resp = client.post("/api/render_mermaid", json={"code": "flowchart TD\n  A-->B"})
        assert resp.status_code == 200


def test_rate_limit_exceeded_returns_429_not_500():
    app = FastAPI()

    @app.middleware("http")
    async def _rl(request, call_next):
        return await rate_limit_middleware(request, call_next)

    @app.get("/api/question")
    async def question():
        return {"ok": True}

    client = TestClient(app)

    # This endpoint is metered.
    # For unauthenticated requests without any user-provided LLM key, the
    # middleware treats the client as a demo user and applies DEMO_LIMIT_QUESTIONS.
    for i in range(DEMO_LIMIT_QUESTIONS):
        resp = client.get("/api/question")
        assert resp.status_code == 200, f"unexpected failure at call {i+1}: {resp.text}"

    resp = client.get("/api/question")
    assert resp.status_code == 429
    body = resp.json()
    assert "detail" in body
    err = body["detail"].get("error")
    assert err == "DEMO_LIMIT_REACHED"
    assert body["detail"].get("user_type") == "demo"


def test_unmetered_endpoints_not_rate_limited():
    app = FastAPI()

    @app.middleware("http")
    async def _rl(request, call_next):
        return await rate_limit_middleware(request, call_next)

    @app.get("/api/ping")
    async def ping():
        return {"pong": True}

    client = TestClient(app)

    # Should never hit the guest daily limit because /api/ping is unmetered.
    for _ in range(50):
        resp = client.get("/api/ping")
        assert resp.status_code == 200
