"""Practice Mode follow-up drilling tests.

Goal:
- After a user answers Q1 and acknowledges feedback, the next question served
  should be a drill-down follow-up framed from the transcript/micro_feedback.

This test avoids external LLM calls by stubbing AdaptiveInterviewerAgent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.schemas import (
    PracticeInterviewQuestion,
    PracticeSession,
    QuestionDifficulty,
    MicroFeedback,
    SessionCoachingStyle,
    SessionFollowUpDepth,
    SessionStrategyAction,
    SessionStrategyDecision,
    SpeechMetrics,
    AnswerSubmission,
)
from app.services.practice.practice_mode_service import PracticeModeService


class _DummyTTS:
    async def synthesize_async(self, _text: str, audio_path: str):
        return audio_path


class _DummyInterviewer:
    def is_interview_complete(self, current_index: int, total: int) -> bool:
        return (current_index + 1) >= total

    def get_next_question(self, questions, current_index: int):
        nxt = current_index + 1
        return questions[nxt] if nxt < len(questions) else None

    def get_progress_indicator(self, question_id: int, total: int) -> str:
        return f"{question_id}/{total}"

    def format_tts_text(self, *_args, **_kwargs):
        return ""

    def get_questions(self, difficulty=QuestionDifficulty.MEDIUM, count: int = 5):
        return [
            PracticeInterviewQuestion(
                id=1,
                text=f"Fresh {difficulty.value} question",
                difficulty=difficulty,
                category="technical",
            )
        ]


class _DummyAdaptive:
    async def generate_follow_up_question(
        self,
        *,
        user_profile,
        difficulty,
        round_type,
        previous_question,
        transcript,
        micro_feedback=None,
        already_asked=None,
        target_question_id=None,
        follow_up_depth=None,
        coaching_style=None,
        api_key=None,
    ):
        # Prove we used the transcript/micro_feedback by encoding it in the question.
        missed = []
        if micro_feedback is not None:
            missed = micro_feedback.key_points_missed or []
        suffix = missed[0] if missed else "details"
        return PracticeInterviewQuestion(
            id=int(target_question_id or 2),
            text=f"Follow-up on your answer about '{suffix}': can you explain it deeper?",
            difficulty=difficulty,
            time_limit=120,
            category=getattr(previous_question, "category", "technical"),
        )


@pytest.mark.asyncio
async def test_ack_serves_adaptive_followup_question(tmp_path: Path):
    svc = PracticeModeService.__new__(PracticeModeService)
    svc.sessions = {}
    svc.audio_dir = tmp_path
    svc.tts_service = _DummyTTS()
    svc.interviewer_agent = _DummyInterviewer()
    svc.adaptive_interviewer = _DummyAdaptive()

    q1 = PracticeInterviewQuestion(id=1, text="Explain caching.", difficulty=QuestionDifficulty.MEDIUM, category="technical")
    q2 = PracticeInterviewQuestion(id=2, text="(placeholder)", difficulty=QuestionDifficulty.MEDIUM, category="technical")

    sess = PracticeSession(
        session_id="s1",
        user_profile=None,
        difficulty=QuestionDifficulty.MEDIUM,
        questions=[q1, q2],
        current_question_index=0,  # last answered = Q1
        pending_next_question=True,
        audio_files=[],
    )

    metrics = SpeechMetrics(
        filler_count=0,
        wpm=150,
        longest_silence=0.5,
        confidence_score=5.0,
        overtalked=False,
        duration=30.0,
        filler_words=[],
        pause_count=0,
        pitch_variance=0.0,
        silence_removed=None,
    )
    mf = MicroFeedback(
        delivery_tips=["Good"],
        pace_feedback="Great pace!",
        overall_note="OK",
        correctness_score=68,
        key_points_missed=["cache invalidation"],
    )
    ans = AnswerSubmission(
        question_id=1,
        transcript="Caching keeps hot data in memory to reduce database load, but I did not explain invalidation well.",
        metrics=metrics,
        micro_feedback=mf,
        audio_duration=30.0,
    )
    sess.answers.append(ans)

    svc.sessions[sess.session_id] = sess

    res = await svc.get_next_question_after_acknowledgment(session_id="s1", question_id=1, api_key=None)
    assert res["complete"] is False
    assert res["strategy"].action == SessionStrategyAction.FOLLOW_UP
    assert res["strategy"].decision_trace["guardrail"] is None
    assert res["strategy"].decision_trace["follow_up_budget"]["remaining"] == 1
    nxt = res["next_question"]
    assert "cache invalidation" in nxt.text
    assert nxt.id == 2


@pytest.mark.asyncio
async def test_ack_decreases_difficulty_for_very_weak_answer(tmp_path: Path):
    svc = PracticeModeService.__new__(PracticeModeService)
    svc.sessions = {}
    svc.audio_dir = tmp_path
    svc.tts_service = _DummyTTS()
    svc.interviewer_agent = _DummyInterviewer()
    svc.adaptive_interviewer = _DummyAdaptive()

    q1 = PracticeInterviewQuestion(id=1, text="Explain consensus.", difficulty=QuestionDifficulty.HARD, category="technical")
    q2 = PracticeInterviewQuestion(id=2, text="(placeholder)", difficulty=QuestionDifficulty.HARD, category="technical")

    sess = PracticeSession(
        session_id="s2",
        user_profile=None,
        difficulty=QuestionDifficulty.HARD,
        questions=[q1, q2],
        current_question_index=0,
        pending_next_question=True,
        audio_files=[],
    )

    metrics = SpeechMetrics(
        filler_count=7,
        wpm=110,
        longest_silence=2.0,
        confidence_score=2.5,
        overtalked=False,
        duration=8.0,
        filler_words=["um"],
        pause_count=2,
        pitch_variance=0.0,
        silence_removed=None,
    )
    mf = MicroFeedback(
        delivery_tips=["Slow down"],
        pace_feedback="Too hesitant",
        overall_note="Needs support",
        correctness_score=20,
        technical_accuracy="Poor",
        key_points_missed=["leader election"],
    )
    ans = AnswerSubmission(question_id=1, transcript="idk", metrics=metrics, micro_feedback=mf, audio_duration=8.0)
    sess.answers.append(ans)

    svc.sessions[sess.session_id] = sess

    res = await svc.get_next_question_after_acknowledgment(session_id="s2", question_id=1, api_key=None)

    assert res["complete"] is False
    assert res["strategy"].action == SessionStrategyAction.DECREASE_DIFFICULTY
    assert res["strategy"].decision_trace["final_action"] == SessionStrategyAction.DECREASE_DIFFICULTY.value
    assert res["next_question"].difficulty == QuestionDifficulty.MEDIUM


@pytest.mark.asyncio
async def test_ack_caps_total_followup_budget(tmp_path: Path):
    svc = PracticeModeService.__new__(PracticeModeService)
    svc.sessions = {}
    svc.audio_dir = tmp_path
    svc.tts_service = _DummyTTS()
    svc.interviewer_agent = _DummyInterviewer()
    svc.adaptive_interviewer = _DummyAdaptive()

    questions = [
        PracticeInterviewQuestion(id=1, text="Explain leader election.", difficulty=QuestionDifficulty.MEDIUM, category="technical"),
        PracticeInterviewQuestion(id=2, text="Explain replication lag.", difficulty=QuestionDifficulty.MEDIUM, category="technical"),
        PracticeInterviewQuestion(id=3, text="Explain read consistency.", difficulty=QuestionDifficulty.MEDIUM, category="technical"),
        PracticeInterviewQuestion(id=4, text="Explain quorum writes.", difficulty=QuestionDifficulty.MEDIUM, category="technical"),
        PracticeInterviewQuestion(id=5, text="Explain split brain handling.", difficulty=QuestionDifficulty.MEDIUM, category="technical"),
    ]

    metrics = SpeechMetrics(
        filler_count=1,
        wpm=145,
        longest_silence=0.6,
        confidence_score=4.8,
        overtalked=False,
        duration=35.0,
        filler_words=["um"],
        pause_count=1,
        pitch_variance=0.0,
        silence_removed=None,
    )

    sess = PracticeSession(
        session_id="s3",
        user_profile=None,
        difficulty=QuestionDifficulty.MEDIUM,
        questions=questions,
        current_question_index=2,
        pending_next_question=True,
        audio_files=[],
        strategy_history=[
            SessionStrategyDecision(
                action=SessionStrategyAction.FOLLOW_UP,
                reason="First follow-up",
                coaching_style=SessionCoachingStyle.BALANCED,
                follow_up_depth=SessionFollowUpDepth.LIGHT,
                target_difficulty=QuestionDifficulty.MEDIUM,
            ),
            SessionStrategyDecision(
                action=SessionStrategyAction.ASK_QUESTION,
                reason="Move on",
                coaching_style=SessionCoachingStyle.BALANCED,
                follow_up_depth=SessionFollowUpDepth.NONE,
                target_difficulty=QuestionDifficulty.MEDIUM,
            ),
            SessionStrategyDecision(
                action=SessionStrategyAction.FOLLOW_UP,
                reason="Second follow-up",
                coaching_style=SessionCoachingStyle.BALANCED,
                follow_up_depth=SessionFollowUpDepth.LIGHT,
                target_difficulty=QuestionDifficulty.MEDIUM,
            ),
            SessionStrategyDecision(
                action=SessionStrategyAction.ASK_QUESTION,
                reason="Move on again",
                coaching_style=SessionCoachingStyle.BALANCED,
                follow_up_depth=SessionFollowUpDepth.NONE,
                target_difficulty=QuestionDifficulty.MEDIUM,
            ),
        ],
    )

    sess.answers.extend(
        [
            AnswerSubmission(
                question_id=1,
                transcript="Leader election chooses a primary node.",
                metrics=metrics,
                micro_feedback=MicroFeedback(delivery_tips=["Good"], pace_feedback="Good", overall_note="Fine", correctness_score=75),
                audio_duration=20.0,
            ),
            AnswerSubmission(
                question_id=2,
                transcript="Replication lag measures follower delay.",
                metrics=metrics,
                micro_feedback=MicroFeedback(delivery_tips=["Good"], pace_feedback="Good", overall_note="Fine", correctness_score=74),
                audio_duration=20.0,
            ),
            AnswerSubmission(
                question_id=3,
                transcript="Read consistency defines what replicas can serve reads, but I did not cover quorum reads or write coordination in enough detail.",
                metrics=metrics,
                micro_feedback=MicroFeedback(
                    delivery_tips=["Keep the structure tight"],
                    pace_feedback="Good pace",
                    overall_note="Strong base answer",
                    correctness_score=72,
                    key_points_missed=["quorum reads"],
                ),
                audio_duration=35.0,
            ),
        ]
    )

    svc.sessions[sess.session_id] = sess

    res = await svc.get_next_question_after_acknowledgment(session_id="s3", question_id=3, api_key=None)

    assert res["complete"] is False
    assert res["strategy"].action == SessionStrategyAction.ASK_QUESTION
    assert res["strategy"].decision_trace["guardrail"] == "follow_up_budget_exhausted"
    assert res["strategy"].decision_trace["follow_up_budget"]["max"] == 2
    assert res["next_question"].id == 4
