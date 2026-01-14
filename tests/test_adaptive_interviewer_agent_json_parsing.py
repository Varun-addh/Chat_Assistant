import pytest

from app.schemas import QuestionDifficulty, UserProfile
from app.services.adaptive_interviewer_agent import AdaptiveInterviewerAgent


def test_parse_questions_handles_trailing_garbage_after_json_array():
    agent = AdaptiveInterviewerAgent(gemini_api_key="dummy", gemini_model="dummy")

    profile = UserProfile(domain="Backend", experience_years=3, skills=["Python"])

    response = (
        '[\n'
        '  {"text": "Explain caching.", "category": "technical", "question_type": "voice", '
        '   "time_limit": 90, "difficulty": "medium", '
        '   "key_points": ["types", "tradeoffs"], "expected_answer_template": "Discuss LRU etc."}\n'
        ']\n'
        'Sure! Let me know if you want more.\n'
        '{"note": "this should be ignored"}'
    )

    questions = agent._parse_questions(
        response=response,
        profile=profile,
        difficulty=QuestionDifficulty.MEDIUM,
        round_type=None,
    )

    assert len(questions) == 1
    assert questions[0].text == "Explain caching."


def test_parse_questions_handles_markdown_code_fence_and_trailing_text():
    agent = AdaptiveInterviewerAgent(gemini_api_key="dummy", gemini_model="dummy")

    profile = UserProfile(domain="Backend", experience_years=3, skills=["Python"])

    response = (
        "```json\n"
        "[{\"text\":\"Write a function\",\"category\":\"technical\",\"question_type\":\"coding\","
        "\"programming_language\":\"Python\",\"time_limit\":600,\"difficulty\":\"medium\","
        "\"key_points\":[\"signature\"],\"expected_answer_template\":\"Implement it\"}]\n"
        "```\n"
        "Extra trailing help text"
    )

    questions = agent._parse_questions(
        response=response,
        profile=profile,
        difficulty=QuestionDifficulty.MEDIUM,
        round_type=None,
    )

    assert len(questions) == 1
    assert questions[0].question_type.value == "coding"
    assert questions[0].programming_language == "Python"
