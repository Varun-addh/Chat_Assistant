from __future__ import annotations

from datetime import datetime, timezone

from app.schemas import AnswerSubmission, MicroFeedback, PracticeSession, QuestionDifficulty, SpeechMetrics
from app.services.practice.adaptive_pressure import adjust_difficulty, compute_pressure_state
from app.services.practice.practice_scoring import build_evaluation_trace, compute_session_trajectory, score_answer


def _metrics(*, filler_count: int, wpm: float, confidence_score: float, overtalked: bool = False) -> SpeechMetrics:
    return SpeechMetrics(
        filler_count=filler_count,
        wpm=wpm,
        longest_silence=0.0,
        confidence_score=confidence_score,
        overtalked=overtalked,
        duration=30.0,
        filler_words=[],
        pause_count=0,
        pitch_variance=0.0,
        silence_removed=0.0,
    )


def _micro(*, correctness_score: int) -> MicroFeedback:
    return MicroFeedback(
        delivery_tips=["Tip 1", "Tip 2"],
        pace_feedback="OK",
        overall_note="Note",
        correctness_score=correctness_score,
        technical_accuracy=None,
        is_correct=bool(correctness_score >= 70),
    )


def test_score_answer_deterministic_and_bounded():
    a = AnswerSubmission(
        question_id=1,
        transcript="First I would do X. Finally I would do Y.",
        metrics=_metrics(filler_count=2, wpm=150.0, confidence_score=7.0, overtalked=False),
        micro_feedback=_micro(correctness_score=80),
        audio_duration=30.0,
        submitted_at=datetime.now(timezone.utc),
    )

    r = score_answer(answer=a)
    assert r.question_id == 1
    assert 0.0 <= r.overall_score <= 100.0
    for v in r.dimension_scores.values():
        assert 0.0 <= float(v) <= 100.0
    assert isinstance(r.why, list) and r.why
    assert r.signals["filler_count"] == 2


def test_compute_session_trajectory_points_and_deltas():
    s = PracticeSession(
        session_id="s1",
        questions=[],
        answers=[
            AnswerSubmission(
                question_id=1,
                transcript="I would do X.",
                metrics=_metrics(filler_count=6, wpm=190.0, confidence_score=5.0, overtalked=True),
                micro_feedback=_micro(correctness_score=55),
                audio_duration=30.0,
                submitted_at=datetime.now(timezone.utc),
            ),
            AnswerSubmission(
                question_id=2,
                transcript="First I would do X. Second do Y. Finally Z.",
                metrics=_metrics(filler_count=1, wpm=155.0, confidence_score=8.0, overtalked=False),
                micro_feedback=_micro(correctness_score=85),
                audio_duration=30.0,
                submitted_at=datetime.now(timezone.utc),
            ),
        ],
    )

    traj = compute_session_trajectory(session=s)
    assert len(traj["points"]) == 2
    assert traj["overall"]["start"] != traj["overall"]["end"]
    assert "correctness" in traj["dimensions"]


def test_build_evaluation_trace_includes_formulas_and_trajectory():
    s = PracticeSession(
        session_id="s2",
        questions=[],
        answers=[
            AnswerSubmission(
                question_id=1,
                transcript="First do X.",
                metrics=_metrics(filler_count=1, wpm=150.0, confidence_score=8.0, overtalked=False),
                micro_feedback=_micro(correctness_score=90),
                audio_duration=30.0,
                submitted_at=datetime.now(timezone.utc),
            )
        ],
    )

    trace = build_evaluation_trace(session=s)
    assert "formulas" in trace
    assert "weights" in trace
    assert "trajectory" in trace
    assert 0.0 <= float(trace["overall_score"]) <= 100.0


def test_adaptive_pressure_state_and_difficulty_adjustment():
    s = PracticeSession(
        session_id="s3",
        difficulty=QuestionDifficulty.MEDIUM,
        questions=[],
        answers=[
            AnswerSubmission(
                question_id=1,
                transcript="I would do X.",
                metrics=_metrics(filler_count=1, wpm=150.0, confidence_score=8.0, overtalked=False),
                micro_feedback=_micro(correctness_score=88),
                audio_duration=30.0,
                submitted_at=datetime.now(timezone.utc),
            )
        ],
    )

    pressure = compute_pressure_state(session=s)
    assert pressure["mode"] in {"supportive", "balanced", "challenging"}
    assert pressure["level"] in {0, 1, 2}

    eff = adjust_difficulty(base=QuestionDifficulty.MEDIUM, mode=pressure["mode"])
    assert eff in {QuestionDifficulty.EASY, QuestionDifficulty.MEDIUM, QuestionDifficulty.HARD}
