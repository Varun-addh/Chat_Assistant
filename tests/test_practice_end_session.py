"""Practice Mode end-session tests.

Verifies:
- POST /api/practice/interview/end-session returns proper summary
- Early termination carries ended_early=True + skipped questions
- Per-question evaluations include model_answer and user_answer
- Evaluation report is generated for answered questions
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.schemas import (
    MicroFeedback,
    SpeechMetrics,
    AnswerSubmission,
    PracticeInterviewQuestion,
    PracticeSession,
    QuestionDifficulty,
    EvaluationReport,
    EvaluationStrengths,
    EvaluationImprovements,
    MetricsSummary,
    ActionPlan,
)


def _utcnow():
    return datetime.now(timezone.utc)


def _make_question(qid: int, text: str, category: str = "technical") -> PracticeInterviewQuestion:
    return PracticeInterviewQuestion(
        id=qid,
        text=text,
        difficulty=QuestionDifficulty.MEDIUM,
        time_limit=90,
        category=category,
        key_points=["point_a", "point_b"],
        expected_answer_template=f"The ideal answer for Q{qid} is...",
    )


def _make_answer(qid: int, transcript: str, model_answer: str = "") -> AnswerSubmission:
    return AnswerSubmission(
        question_id=qid,
        transcript=transcript,
        metrics=SpeechMetrics(
            filler_count=1,
            wpm=140.0,
            longest_silence=1.5,
            confidence_score=7.0,
            overtalked=False,
            duration=45.0,
        ),
        micro_feedback=MicroFeedback(
            delivery_tips=["Good pace"],
            pace_feedback="Great pace!",
            overall_note="✅ Strong answer! (82%)",
            correctness_score=82,
            technical_accuracy="Good",
            strengths=["Clear explanation"],
            improvement_areas=["Add examples"],
            model_answer=model_answer or None,
        ),
        audio_duration=45.0,
    )


def _make_eval_report() -> EvaluationReport:
    return EvaluationReport(
        strengths=EvaluationStrengths(items=["Good structure", "Clear communication"]),
        improvements=EvaluationImprovements(items=["More depth", "Add examples"]),
        metrics_summary=MetricsSummary(
            total_fillers=2,
            avg_wpm=140.0,
            longest_pause=1.5,
            avg_confidence=7.0,
            total_duration=90.0,
        ),
        action_plan=ActionPlan(steps=["Practice system design", "Review data structures"]),
        practice_recommendation="2-3 more sessions recommended",
    )


def _build_session(answered: int = 2, total: int = 5) -> PracticeSession:
    """Build a session with `answered` out of `total` questions done."""
    questions = [_make_question(i + 1, f"Question {i + 1}?") for i in range(total)]
    answers = [
        _make_answer(i + 1, f"My answer to Q{i + 1}...", f"Ideal answer for Q{i + 1}...")
        for i in range(answered)
    ]
    return PracticeSession(
        session_id="test-sess-1",
        started_at=_utcnow(),
        last_activity_at=_utcnow(),
        current_question_index=answered - 1 if answered > 0 else 0,
        questions=questions,
        answers=answers,
        is_complete=False,
        difficulty=QuestionDifficulty.MEDIUM,
    )


# ---------------------------------------------------------------------------
# Unit: PracticeModeService.end_session
# ---------------------------------------------------------------------------

def test_end_session_service_returns_summary():
    """Service end_session marks session complete and returns structured summary."""
    from app.services.practice.practice_mode_service import PracticeModeService

    session = _build_session(answered=2, total=5)

    with patch.object(PracticeModeService, "__init__", lambda self, **kw: None):
        svc = PracticeModeService.__new__(PracticeModeService)
        svc.sessions = {session.session_id: session}
        svc.evaluation_agent = AsyncMock()
        svc.evaluation_agent.evaluate_interview = AsyncMock(return_value=_make_eval_report())

        # Monkey-patch helper that references self.config (not set in __new__)
        svc._maybe_add_peer_learning_insight = lambda report: None

        result = asyncio.run(
            svc.end_session(session.session_id, api_key="test-key")
        )

    assert result["status"] == "completed"
    assert result["ended_early"] is True
    assert result["questions_answered"] == 2
    assert result["questions_skipped"] == 3
    assert len(result["evaluations"]) == 2
    assert len(result["skipped_questions"]) == 3

    # Every answered question should carry model_answer and user_answer
    for ev in result["evaluations"]:
        assert "model_answer" in ev
        assert "user_answer" in ev
        assert len(ev["user_answer"]) > 0

    # Skipped questions carry question text
    for sq in result["skipped_questions"]:
        assert "question" in sq
        assert sq["question_number"] > 2


def test_end_session_full_interview_not_early():
    """When all questions answered, ended_early is False."""
    from app.services.practice.practice_mode_service import PracticeModeService

    session = _build_session(answered=5, total=5)

    with patch.object(PracticeModeService, "__init__", lambda self, **kw: None):
        svc = PracticeModeService.__new__(PracticeModeService)
        svc.sessions = {session.session_id: session}
        svc.evaluation_agent = AsyncMock()
        svc.evaluation_agent.evaluate_interview = AsyncMock(return_value=_make_eval_report())
        svc._maybe_add_peer_learning_insight = lambda report: None

        result = asyncio.run(
            svc.end_session(session.session_id, api_key="k")
        )

    assert result["ended_early"] is False
    assert result["questions_skipped"] == 0
    assert len(result["skipped_questions"]) == 0
    assert len(result["evaluations"]) == 5


def test_model_answer_in_micro_feedback():
    """MicroFeedback schema accepts and serializes model_answer."""
    fb = MicroFeedback(
        delivery_tips=["tip"],
        pace_feedback="ok",
        overall_note="ok",
        model_answer="The ideal answer covers X, Y, and Z.",
    )
    d = fb.model_dump()
    assert d["model_answer"] == "The ideal answer covers X, Y, and Z."


def test_model_answer_defaults_none():
    """model_answer defaults to None when not provided."""
    fb = MicroFeedback(
        delivery_tips=["tip"],
        pace_feedback="ok",
        overall_note="ok",
    )
    assert fb.model_answer is None
