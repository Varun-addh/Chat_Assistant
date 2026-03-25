from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers.practice_mode import router as practice_router
from app.schemas import (
    InterviewRound,
    PracticeInterviewQuestion,
    PracticeSession,
    QuestionDifficulty,
    UserProfile,
)
from app.services.chat.llm_service import LLMAuthenticationError, llm_service
from app.services.practice.adaptive_interviewer_agent import AdaptiveInterviewerAgent
from app.services.practice.conversational_agent import ConversationalAgent


class _QuickStartAuthFailService:
    async def quick_start_conversational(self, *args, **kwargs):
        raise LLMAuthenticationError("Error code: 401 - invalid_api_key")


class _StartInterviewAuthFailService:
    async def start_interview(self, *args, **kwargs):
        raise LLMAuthenticationError("Error code: 401 - invalid_api_key")


class _RoundStartCaptureService:
    def __init__(self):
        self.last_api_key = None
        self.sessions: dict[str, PracticeSession] = {}

    async def start_interview(
        self,
        difficulty,
        user_profile=None,
        question_count=5,
        round_type=None,
        api_key=None,
        **kwargs,
    ):
        self.last_api_key = api_key
        session_id = "round-session"
        first_question = PracticeInterviewQuestion(
            id=1,
            text="Explain how you would paginate an API.",
            difficulty=difficulty,
            category="technical",
        )
        self.sessions[session_id] = PracticeSession(
            session_id=session_id,
            difficulty=difficulty,
            round_type=round_type,
            user_profile=user_profile,
            questions=[first_question],
        )
        return session_id, first_question, None

    def get_session(self, session_id: str):
        return self.sessions.get(session_id)


@pytest.mark.asyncio
async def test_conversational_agent_re_raises_auth_error(monkeypatch):
    async def _raise_auth_error(*args, **kwargs):
        raise LLMAuthenticationError("Error code: 401 - invalid_api_key")

    monkeypatch.setattr(llm_service, "generate_text", _raise_auth_error)

    agent = ConversationalAgent("test-key")
    with pytest.raises(LLMAuthenticationError):
        await agent.infer_profile_from_conversation(
            user_input="I am preparing for backend engineer interviews",
            api_key="gsk_test_invalid",
        )


@pytest.mark.asyncio
async def test_adaptive_interviewer_re_raises_auth_error(monkeypatch):
    async def _raise_auth_error(*args, **kwargs):
        raise LLMAuthenticationError("Error code: 401 - invalid_api_key")

    agent = AdaptiveInterviewerAgent("test-key", "models/gemini-test")
    monkeypatch.setattr(agent, "_call_llm", _raise_auth_error)

    profile = UserProfile(
        domain="Backend Engineer",
        experience_years=4,
        skills=["Python", "APIs"],
        job_role="Backend Engineer",
        company_preference="any",
        interview_focus=["technical"],
    )

    with pytest.raises(LLMAuthenticationError):
        await agent.generate_adaptive_questions(
            user_profile=profile,
            difficulty=QuestionDifficulty.MEDIUM,
            count=3,
            api_key="gsk_test_invalid",
        )


def test_quick_start_invalid_server_key_returns_503(monkeypatch):
    app = FastAPI()
    app.include_router(practice_router)
    client = TestClient(app)

    import app.routers.practice_mode as pm

    pm.practice_service = _QuickStartAuthFailService()
    monkeypatch.setattr(pm.settings, "require_user_api_key", False)
    monkeypatch.setattr(pm, "get_user_id_from_request", lambda _req: "guest_unknown")

    response = client.post(
        "/api/practice/interview/quick-start",
        json={
            "voice_input": "I am preparing for software engineer interviews",
            "question_count": 4,
        },
    )

    assert response.status_code == 503
    assert "Server AI provider credentials are invalid" in response.json()["detail"]


def test_start_interview_invalid_user_key_returns_401(monkeypatch):
    app = FastAPI()
    app.include_router(practice_router)
    client = TestClient(app)

    import app.routers.practice_mode as pm

    pm.practice_service = _StartInterviewAuthFailService()
    pm.practice_graph = None
    monkeypatch.setattr(pm, "get_user_id_from_request", lambda _req: "guest_unknown")

    response = client.post(
        "/api/practice/interview/start",
        headers={"X-API-Key": "gsk_test_invalid"},
        json={
            "screen_shared": True,
            "camera_enabled": True,
            "difficulty": "medium",
            "category": "behavioral",
            "question_count": 1,
            "user_profile": {
                "domain": "Backend Engineer",
                "experience_years": 4,
                "skills": ["Python"],
                "job_role": "Backend Engineer",
                "company_preference": "any",
                "interview_focus": ["technical"],
            },
        },
    )

    assert response.status_code == 401
    assert "Invalid API key" in response.json()["detail"]


def test_start_round_authenticated_jwt_uses_server_key_when_require_user_api_key(monkeypatch):
    app = FastAPI()

    @app.middleware("http")
    async def _inject_user(request, call_next):
        if request.headers.get("X-Test-User"):
            request.state.user = object()
        return await call_next(request)

    app.include_router(practice_router)
    client = TestClient(app)

    import app.routers.practice_mode as pm

    service = _RoundStartCaptureService()
    pm.practice_service = service
    monkeypatch.setattr(pm, "get_user_id_from_request", lambda _req: "user_123")
    monkeypatch.setattr(pm, "_maybe_enrich_profile_focus", lambda **kwargs: (kwargs["profile"], []))
    monkeypatch.setattr(pm, "_insert_practice_proctoring_event", lambda **kwargs: None)
    monkeypatch.setattr(pm, "_track_practice_event", lambda **kwargs: None)
    monkeypatch.setattr(pm.settings, "require_user_api_key", True)
    monkeypatch.setattr(pm.settings, "llm_provider", "groq")
    monkeypatch.setattr(pm.settings, "groq_api_key", "server_groq_key")
    monkeypatch.setattr(pm.settings, "gemini_api_key", None)

    response = client.post(
        "/api/practice/interview/start-round",
        headers={
            "Authorization": "Bearer header.payload.signature",
            "X-Test-User": "1",
        },
        json={
            "screen_shared": True,
            "camera_enabled": True,
            "round_type": InterviewRound.TECHNICAL_ROUND_1.value,
            "domain": "Backend Engineer",
            "experience_years": 4,
            "question_count": 1,
        },
    )

    assert response.status_code == 200
    assert service.last_api_key == "server_groq_key"


def test_submit_answer_missing_bridge_key_with_jwt_returns_401_not_500(monkeypatch):
    app = FastAPI()
    app.include_router(practice_router)
    client = TestClient(app)

    import app.routers.practice_mode as pm

    pm.practice_service = object()
    monkeypatch.setattr(pm, "get_user_id_from_request", lambda _req: "guest_unknown")
    monkeypatch.setattr(pm.settings, "require_user_api_key", True)

    response = client.post(
        "/api/practice/interview/submit-answer",
        headers={"Authorization": "Bearer header.payload.signature"},
        files={"audio": ("a.wav", b"RIFF", "audio/wav")},
        data={"session_id": "session-1", "question_id": "1"},
    )

    assert response.status_code == 401
    assert "No active API key" in response.json()["detail"]


def test_acknowledge_feedback_missing_bridge_key_with_jwt_returns_401_not_500(monkeypatch):
    app = FastAPI()
    app.include_router(practice_router)
    client = TestClient(app)

    import app.routers.practice_mode as pm

    pm.practice_service = object()
    monkeypatch.setattr(pm, "get_user_id_from_request", lambda _req: "guest_unknown")
    monkeypatch.setattr(pm.settings, "require_user_api_key", True)

    response = client.post(
        "/api/practice/interview/acknowledge-feedback",
        headers={"Authorization": "Bearer header.payload.signature"},
        json={"session_id": "session-1", "question_id": 1, "feedback_read": True},
    )

    assert response.status_code == 401
    assert "No active API key" in response.json()["detail"]