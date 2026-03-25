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


def test_parse_questions_rejects_fragment_titles():
    agent = AdaptiveInterviewerAgent(gemini_api_key="x", gemini_model="y")

    response = '{"questions": [' \
        '{"text": "CI/CD pipeline", "category": "technical", "question_type": "voice", "time_limit": 90},' \
        '{"text": "Explain overfitting in machine learning.", "category": "technical", "question_type": "voice", "time_limit": 90}' \
    ']}'

    out = agent._parse_questions(
        response=response,
        profile=_profile(),
        difficulty=QuestionDifficulty.EASY,
        round_type=None,
    )

    assert len(out) == 2
    # First question salvaged from noun-phrase fragment (varied templates)
    assert "CI/CD pipeline" in out[0].text
    assert out[0].text[-1] in ("?", ".")  # Salvage templates may end with ? or .
    assert out[1].text == "Explain overfitting in machine learning."


def test_parse_single_question_salvages_fragment_title():
    agent = AdaptiveInterviewerAgent(gemini_api_key="x", gemini_model="y")

    out = agent._parse_single_question(
        response='{"text": "Classification and regression", "category": "technical", "question_type": "voice", "time_limit": 90}',
        difficulty=QuestionDifficulty.MEDIUM,
        round_type=None,
        target_question_id=2,
    )

    assert out is not None
    assert "Classification and regression" in out.text
    assert out.text[-1] in ("?", ".")


def test_parse_single_question_salvages_noun_phrase_with_preposition():
    agent = AdaptiveInterviewerAgent(gemini_api_key="x", gemini_model="y")

    out = agent._parse_single_question(
        response='{"text": "definition of data partitioning", "category": "technical", "question_type": "voice", "time_limit": 90}',
        difficulty=QuestionDifficulty.MEDIUM,
        round_type=None,
        target_question_id=2,
    )

    assert out is not None
    assert "definition of data partitioning" in out.text
    assert out.text[-1] in ("?", ".")


def test_parse_questions_accepts_structured_prompt_without_prefix_match():
    agent = AdaptiveInterviewerAgent(gemini_api_key="x", gemini_model="y")

    response = '{"questions": [' \
        '{"text": "Given an imbalanced dataset how would you evaluate the model", "category": "technical", "question_type": "voice", "time_limit": 90}' \
    ']}'

    out = agent._parse_questions(
        response=response,
        profile=_profile(),
        difficulty=QuestionDifficulty.MEDIUM,
        round_type=None,
    )

    assert len(out) == 1
    assert out[0].text == "Given an imbalanced dataset how would you evaluate the model"
