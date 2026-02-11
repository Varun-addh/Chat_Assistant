from __future__ import annotations

from datetime import datetime

from app.services.interview.mock_interview_analytics import (
    build_mock_evaluation_trace,
    compute_mock_session_trajectory,
)
from app.services.interview.mock_interview_service import (
    DifficultyLevel,
    EvaluationCriteria,
    EvaluationResult,
    InterviewSession,
    InterviewType,
    UserAnswer,
)


def _eval(*, overall: float, correctness: float, completeness: float, clarity: float, confidence: float, depth: float) -> EvaluationResult:
    return EvaluationResult(
        overall_score=overall,
        criteria_scores=EvaluationCriteria(
            correctness=correctness,
            completeness=completeness,
            clarity=clarity,
            confidence=confidence,
            technical_depth=depth,
        ),
        strengths=["s"],
        weaknesses=["w"],
        missing_points=[],
        improvement_suggestions=[],
        performance_summary="summary",
        detailed_feedback="details",
        rating_category="Good",
        follow_up_questions=[],
        model_answer="",
        recommended_resources=[],
        key_takeaways=[],
    )


def test_mock_trajectory_points_and_deltas():
    s = InterviewSession(
        session_id="s1",
        user_id="u1",
        started_at=datetime.now(),
        questions=[],
        answers=[UserAnswer(answer_text="a1"), UserAnswer(answer_text="a2")],
        evaluations=[
            _eval(overall=6.0, correctness=6, completeness=5, clarity=6, confidence=6, depth=5),
            _eval(overall=7.5, correctness=8, completeness=7, clarity=7, confidence=7, depth=7),
        ],
        current_question_index=2,
        total_questions=2,
        interview_type=InterviewType.TECHNICAL,
        difficulty=DifficultyLevel.MEDIUM,
    )

    traj = compute_mock_session_trajectory(session=s)
    assert len(traj["points"]) == 2
    assert traj["overall"]["delta"] > 0
    assert "correctness" in traj["criteria"]


def test_mock_evaluation_trace_contains_aggregation_and_averages():
    s = InterviewSession(
        session_id="s2",
        user_id="u2",
        started_at=datetime.now(),
        questions=[],
        answers=[UserAnswer(answer_text="a1")],
        evaluations=[_eval(overall=8.0, correctness=8, completeness=8, clarity=7, confidence=8, depth=7)],
        current_question_index=1,
        total_questions=1,
        interview_type=InterviewType.BEHAVIORAL,
        difficulty=DifficultyLevel.EASY,
    )

    trace = build_mock_evaluation_trace(session=s)
    assert trace["aggregation"] == "average_over_questions"
    assert "criteria_averages" in trace
    assert "why" in trace and trace["why"]
    assert "trajectory" in trace
