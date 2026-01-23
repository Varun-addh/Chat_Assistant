"""Regression: Practice Mode ack should not skip questions.

Bug:
- PracticeModeService.get_next_question_after_acknowledgment() incremented
  session.current_question_index and then called InterviewerAgent.get_next_question(),
  which *also* advances by +1. Net effect: Q1 -> ack -> Q3.

This test exercises the service method directly with lightweight stubs so we don't
run real STT/TTS.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.schemas import PracticeInterviewQuestion, PracticeSession, QuestionDifficulty
from app.services.practice.practice_mode_service import PracticeModeService


class _DummyTTS:
    async def synthesize_async(self, text: str, audio_path: str):
        # Pretend synthesis succeeded.
        return audio_path


class _DummyInterviewer:
    def is_interview_complete(self, current_index: int, total_questions: int) -> bool:
        return current_index >= total_questions - 1

    def get_next_question(self, questions, current_index: int):
        next_index = current_index + 1
        if next_index < len(questions):
            return questions[next_index]
        return None

    def format_tts_text(self, question, total_questions: int, company_name=None) -> str:
        return f"Q{question.id}/{total_questions}: {question.text}"

    def get_progress_indicator(self, question_id: int, total_questions: int) -> str:
        return f"{question_id}/{total_questions}"


@pytest.mark.asyncio
async def test_ack_returns_q2_not_q3(tmp_path: Path):
    svc = PracticeModeService.__new__(PracticeModeService)
    svc.sessions = {}
    svc.audio_dir = tmp_path
    svc.tts_service = _DummyTTS()
    svc.interviewer_agent = _DummyInterviewer()

    q1 = PracticeInterviewQuestion(id=1, text="Q1", difficulty=QuestionDifficulty.EASY, category="technical")
    q2 = PracticeInterviewQuestion(id=2, text="Q2", difficulty=QuestionDifficulty.EASY, category="technical")
    q3 = PracticeInterviewQuestion(id=3, text="Q3", difficulty=QuestionDifficulty.EASY, category="technical")

    session_id = "s"
    sess = PracticeSession(
        session_id=session_id,
        questions=[q1, q2, q3],
        current_question_index=0,  # last answered = Q1
        pending_next_question=True,
        audio_files=[],
    )
    svc.sessions[session_id] = sess

    res = await svc.get_next_question_after_acknowledgment(session_id=session_id, question_id=1, api_key=None)

    assert res["complete"] is False
    assert res["next_question"].id == 2
    assert sess.pending_next_question is False
    # current_question_index should remain "last answered" until the next submit_answer.
    assert sess.current_question_index == 0


@pytest.mark.asyncio
async def test_ack_after_last_question_marks_complete(tmp_path: Path):
    svc = PracticeModeService.__new__(PracticeModeService)
    svc.sessions = {}
    svc.audio_dir = tmp_path
    svc.tts_service = _DummyTTS()
    svc.interviewer_agent = _DummyInterviewer()

    q1 = PracticeInterviewQuestion(id=1, text="Q1", difficulty=QuestionDifficulty.EASY, category="technical")
    q2 = PracticeInterviewQuestion(id=2, text="Q2", difficulty=QuestionDifficulty.EASY, category="technical")

    session_id = "s2"
    sess = PracticeSession(
        session_id=session_id,
        questions=[q1, q2],
        current_question_index=1,  # last answered = Q2
        pending_next_question=True,
        audio_files=[],
    )
    svc.sessions[session_id] = sess

    res = await svc.get_next_question_after_acknowledgment(session_id=session_id, question_id=2, api_key=None)

    assert res["complete"] is True
