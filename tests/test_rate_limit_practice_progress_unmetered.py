"""Regression test: Practice Mode progress endpoints must not be rate-limited.

The UI polls these endpoints (summary/heatmap/next-session). They are DB reads
(no LLM cost) and should remain available even for demo users.
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.database import init_db
from app.middleware.rate_limit import rate_limit_middleware
from app.routers.practice_mode import router as practice_router


def test_practice_progress_endpoints_not_rate_limited_for_demo(monkeypatch):
    init_db()

    app = FastAPI()

    @app.middleware("http")
    async def _rl(request, call_next):
        return await rate_limit_middleware(request, call_next)

    app.include_router(practice_router)

    # Ensure stable user id resolution inside the practice router.
    import app.routers.practice_mode as pm

    monkeypatch.setattr(pm, "get_user_id_from_request", lambda _req: "guest_demo_user")

    client = TestClient(app)

    # These endpoints should always be 200, even when called many times.
    for _ in range(50):
        r = client.get("/api/practice/progress/summary", params={"lookback_days": 30})
        assert r.status_code == 200

        r = client.get("/api/practice/progress/heatmap", params={"lookback_days": 90})
        assert r.status_code == 200

        r = client.get("/api/practice/progress/next-session")
        assert r.status_code == 200
