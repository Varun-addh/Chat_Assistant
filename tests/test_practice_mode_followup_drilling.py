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
    SpeechMetrics,
    AnswerSubmission,
)
from app.services.practice.practice_mode_service import PracticeModeService


class _DummyTTS:
    async def synthesize_async(self, *_args, **_kwargs):
        return True


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
        key_points_missed=["cache invalidation"],
    )
    ans = AnswerSubmission(question_id=1, transcript="I said caching is...", metrics=metrics, micro_feedback=mf, audio_duration=30.0)
    sess.answers.append(ans)

    svc.sessions[sess.session_id] = sess

    res = await svc.get_next_question_after_acknowledgment(session_id="s1", question_id=1, api_key=None)
    assert res["complete"] is False
    nxt = res["next_question"]
    assert "cache invalidation" in nxt.text
    assert nxt.id == 2
