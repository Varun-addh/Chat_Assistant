from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers.practice_mode import router as practice_router
from app.schemas import QuestionDifficulty, UserProfile
from app.services.chat.llm_service import LLMAuthenticationError, llm_service
from app.services.practice.adaptive_interviewer_agent import AdaptiveInterviewerAgent
from app.services.practice.conversational_agent import ConversationalAgent


class _QuickStartAuthFailService:
    async def quick_start_conversational(self, *args, **kwargs):
        raise LLMAuthenticationError("Error code: 401 - invalid_api_key")


class _StartInterviewAuthFailService:
    async def start_interview(self, *args, **kwargs):
        raise LLMAuthenticationError("Error code: 401 - invalid_api_key")


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