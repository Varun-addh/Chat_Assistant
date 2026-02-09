"""Demo-mode question caps regression tests.

Product contract:
- Demo mode Interview Intelligence returns at most 2 questions.
- Demo mode Mock Interview starts sessions with exactly 1 question.

We infer demo mode as: unauthenticated request AND no user-provided LLM API key headers.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import settings


def test_demo_intelligence_search_is_capped_to_two(monkeypatch):
    from app.routers import interview_intelligence as ii
    from app.schemas import InterviewQuestion, SearchQuestionsResponse

    app = FastAPI()
    app.include_router(ii.router, prefix="/api/intelligence")
    client = TestClient(app)

    monkeypatch.setattr(settings, "require_user_api_key", False, raising=False)
    # CI runs without secrets; demo endpoints require Groq to be configured.
    monkeypatch.setattr(settings, "groq_api_key", "gsk_test_dummy", raising=False)

    captured: dict[str, int] = {}

    async def _fake_search_and_build_response(q, limit, refresh, api_key=None, save_to_history=True, request=None):
        captured["limit"] = int(limit)
        questions = [
            InterviewQuestion(
                question=f"q{i}",
                answer="a",
                source="llm_generated",
                updated_at="2020-01-01T00:00:00Z",
            )
            for i in range(int(limit))
        ]
        return SearchQuestionsResponse(query=q, questions=questions, count=len(questions))

    monkeypatch.setattr(ii, "_search_and_build_response", _fake_search_and_build_response, raising=True)

    # No auth, no key headers => demo
    resp = client.get(
        "/api/intelligence/search",
        params={"q": "diffusion models", "limit": 10, "refresh": False, "save_to_history": False},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 2
    assert captured["limit"] == 2


def test_demo_mock_interview_start_is_capped_to_one_question(monkeypatch):
    from app.routers.mock_interview import router as mock_router, get_mock_service

    class _Session:
        def __init__(self, session_id: str, total_questions: int, interview_type: str, difficulty: str):
            self.session_id = session_id
            self.total_questions = total_questions
            self.interview_type = interview_type
            self.difficulty = difficulty

    class _Question:
        def __init__(self):
            self.question_id = "q1"
            self.question_text = "Tell me about yourself"
            self.interview_type = "behavioral"
            self.difficulty = "easy"
            self.topic = "general"

    class _DummyService:
        def __init__(self):
            self.last_num_questions: int | None = None

        async def start_session(self, user_id, interview_type, difficulty, num_questions=5, topic=None, api_key=None):
            self.last_num_questions = int(num_questions)
            return _Session("s1", total_questions=int(num_questions), interview_type=str(interview_type), difficulty=str(difficulty))

        async def get_current_question(self, session_id: str):
            return _Question()

    dummy = _DummyService()

    app = FastAPI()
    app.include_router(mock_router, prefix="/api/mock-interview")
    app.dependency_overrides[get_mock_service] = lambda: dummy

    # CI runs without secrets; demo endpoints require Groq to be configured.
    monkeypatch.setattr(settings, "groq_api_key", "gsk_test_dummy", raising=False)

    client = TestClient(app)

    resp = client.post(
        "/api/mock-interview/sessions/start",
        json={
            "user_id": "u1",
            "interview_type": "coding",
            "difficulty": "easy",
            "num_questions": 5,
            "topic": "arrays",
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["total_questions"] == 1
    assert dummy.last_num_questions == 1
