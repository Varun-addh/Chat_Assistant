from __future__ import annotations

from contextlib import contextmanager

import pytest
from fastapi import Body, FastAPI, File, UploadFile
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.models import Base, MockInterviewSessionRecord


def test_settings_auto_enable_email_in_production_when_smtp_is_configured(monkeypatch):
    from app.config import Settings

    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("JWT_SECRET_KEY", "j" * 48)
    monkeypatch.setenv("COOKIE_SECRET", "c" * 48)
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://user:pass@localhost:5432/stratax")
    monkeypatch.setenv("STRATAX_DATABASE_URL", "postgresql+psycopg://user:pass@localhost:5432/stratax")
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("EMAIL_FROM", "noreply@example.com")
    monkeypatch.delenv("EMAIL_ENABLED", raising=False)
    monkeypatch.delenv("SMTP_USERNAME", raising=False)
    monkeypatch.delenv("EMAIL_SMTP_USERNAME", raising=False)
    monkeypatch.delenv("SMTP_PASSWORD", raising=False)
    monkeypatch.delenv("EMAIL_SMTP_PASSWORD", raising=False)

    cfg = Settings()

    assert cfg.email_enabled is True


def _make_request_size_app() -> FastAPI:
    from app.middleware.request_size import RequestSizeLimitMiddleware

    app = FastAPI()
    app.add_middleware(
        RequestSizeLimitMiddleware,
        default_limit_bytes=64,
        practice_media_limit_bytes=4096,
    )

    @app.post("/echo")
    async def echo(payload: bytes = Body(...)):
        return {"size": len(payload)}

    @app.post("/api/practice/session/test-session/media")
    async def practice_media(file: UploadFile = File(...)):
        data = await file.read()
        return {"size": len(data)}

    return app


def test_request_size_limit_rejects_large_generic_body():
    client = TestClient(_make_request_size_app())

    resp = client.post(
        "/echo",
        content=b"x" * 128,
        headers={"content-type": "application/octet-stream", "content-length": "128"},
    )

    assert resp.status_code == 413
    assert "Request body too large" in resp.json()["detail"]


def test_request_size_limit_allows_practice_media_override():
    client = TestClient(_make_request_size_app())

    resp = client.post(
        "/api/practice/session/test-session/media",
        files={"file": ("clip.webm", b"x" * 512, "audio/webm")},
    )

    assert resp.status_code == 200
    assert resp.json()["size"] == 512


class _FakeLLM:
    enabled = False

    class _Settings:
        llm_provider = "test"

    _settings = _Settings()


class _FakeInterviewIntelligenceService:
    async def search_questions(self, query, limit, force_refresh, api_key=None):
        return [
            {
                "question": "Explain database indexing and when to use it.",
                "topic": "databases",
                "key_concepts": ["B-tree", "tradeoffs"],
                "question_type": "technical",
                "is_coding_question": False,
            }
        ]


@pytest.mark.asyncio
async def test_mock_interview_sessions_persist_in_database(monkeypatch, tmp_path):
    import app.services.interview.mock_interview_service as mock_module

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)

    @contextmanager
    def override_get_db_context():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(mock_module, "get_db_context", override_get_db_context)

    service = mock_module.MockInterviewService(_FakeLLM(), _FakeInterviewIntelligenceService())
    session = await service.start_session(
        user_id="user-123",
        interview_type=mock_module.InterviewType.TECHNICAL,
        difficulty=mock_module.DifficultyLevel.EASY,
        num_questions=1,
    )

    reloaded_service = mock_module.MockInterviewService(_FakeLLM(), _FakeInterviewIntelligenceService())
    loaded = reloaded_service.active_sessions.get(session.session_id)

    assert loaded is not None
    assert loaded.session_id == session.session_id
    assert loaded.user_id == "user-123"
    assert loaded.questions[0].question_text == "Explain database indexing and when to use it."

    with TestingSessionLocal() as db:
        rows = db.query(MockInterviewSessionRecord).all()
        assert len(rows) == 1
        assert rows[0].session_id == session.session_id
        assert rows[0].status == "active"


@pytest.mark.asyncio
async def test_stt_service_rejects_disabled_provider(monkeypatch):
    from app.config import settings
    from app.services.practice.stt_service import STTService

    async def audio_stream():
        yield b"fake-audio"

    monkeypatch.setattr(settings, "stt_provider", "none")
    service = STTService()

    with pytest.raises(RuntimeError, match="disabled"):
        async for _ in service.stream_transcribe(audio_stream()):
            pass