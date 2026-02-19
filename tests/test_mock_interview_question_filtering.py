import pytest

from app.services.interview.mock_interview_service import (
    MockInterviewService,
    InterviewType,
    DifficultyLevel,
)


class _FakeInterviewIntelligenceService:
    def __init__(self, results):
        self._results = results

    async def search_questions(self, query, limit, force_refresh, api_key=None):
        return list(self._results)


@pytest.mark.asyncio
async def test_mock_interview_technical_does_not_drop_questions_with_empty_code_solution():
    # Regression: results may include a code_solution key even for conceptual questions.
    # Empty/placeholder code_solution should NOT cause TECHNICAL filtering to drop it.
    fake_results = [
        {
            "question": "Explain what a REST API is.",
            "topic": "APIs",
            "key_concepts": ["stateless", "resources"],
            "code_solution": "",  # placeholder
            "question_type": "technical",
            "is_coding_question": False,
        },
        {
            "question": "Explain database indexing and when to use it.",
            "topic": "Databases",
            "key_concepts": ["B-tree", "tradeoffs"],
            "code_solution": None,
            "question_type": "technical",
        },
    ]

    svc = MockInterviewService.__new__(MockInterviewService)
    svc.interview_service = _FakeInterviewIntelligenceService(fake_results)

    questions = await MockInterviewService._generate_session_questions(
        svc,
        interview_type=InterviewType.TECHNICAL,
        difficulty=DifficultyLevel.EASY,
        num_questions=2,
        topic="general",
        api_key=None,
    )

    assert len(questions) == 2
    assert all(q.interview_type == InterviewType.TECHNICAL for q in questions)
    assert all(q.question_text.strip() for q in questions)
