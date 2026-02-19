from app.schemas import QuestionDifficulty, UserProfile
from app.services.practice.adaptive_interviewer_agent import AdaptiveInterviewerAgent


def _profile() -> UserProfile:
    return UserProfile(
        domain="Data Science",
        experience_years=2,
        skills=["python", "pandas"],
        job_role=None,
        company_preference=None,
        interview_focus=None,
        target_round=None,
    )


def test_parse_questions_accepts_question_field_and_wrapper_object():
    agent = AdaptiveInterviewerAgent(gemini_api_key="x", gemini_model="y")

    response = '{"questions": [{"question": "What is overfitting?", "category": "technical", "question_type": "voice", "time_limit": "90", "difficulty": "easy", "key_points": "bias-variance, generalization"}]}'

    out = agent._parse_questions(
        response=response,
        profile=_profile(),
        difficulty=QuestionDifficulty.EASY,
        round_type=None,
    )

    assert len(out) == 1
    assert out[0].text == "What is overfitting?"
    assert out[0].time_limit == 90
    assert out[0].key_points == ["bias-variance", "generalization"]


def test_parse_questions_accepts_string_items_list():
    agent = AdaptiveInterviewerAgent(gemini_api_key="x", gemini_model="y")

    response = '["Explain precision vs recall."]'

    out = agent._parse_questions(
        response=response,
        profile=_profile(),
        difficulty=QuestionDifficulty.EASY,
        round_type=None,
    )

    assert len(out) == 1
    assert "precision" in out[0].text.lower()
