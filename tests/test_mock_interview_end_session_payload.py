"""Mock Interview end-session payload regression test.

Why:
- The UI summary card reads timing and score stats.
- Some clients expect summary fields (e.g. total_time_seconds) at the top level.

Contract:
- POST /api/mock-interview/sessions/{session_id}/end returns:
  - status == "completed"
  - total_time_seconds present at top level
  - score_range present at top level (may be 0.0)
  - summary object still present for backward compatibility
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient


def test_end_session_flattens_summary_fields(monkeypatch):
    from app.routers.mock_interview import router as mock_router, get_mock_service

    class _DummyService:
        async def end_session(self, session_id: str):
            assert session_id == "s1"
            return {
                "session_id": "s1",
                "total_time_seconds": 148,
                "individual_scores": [8.0, 8.0],
                "best_score": 8.0,
                "lowest_score": 8.0,
                "score_range": 0.0,
            }

    app = FastAPI()
    app.include_router(mock_router, prefix="/api/mock-interview")
    app.dependency_overrides[get_mock_service] = lambda: _DummyService()

    client = TestClient(app)
    resp = client.post("/api/mock-interview/sessions/s1/end")

    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "completed"
    assert data["total_time_seconds"] == 148
    assert data["score_range"] == 0.0
    assert isinstance(data.get("summary"), dict)
    assert data["summary"]["total_time_seconds"] == 148
