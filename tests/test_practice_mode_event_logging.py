"""Regression tests: Practice Mode spine emits structured events.

We do NOT run real STT/TTS/LLM work here.
We stub PracticeModeService so the router code path is exercised (file upload,
response building) while keeping the test fast and deterministic.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.database import init_db, get_db_context
from app.models import EventRecord
from app.routers.practice_mode import router as practice_router
from app.schemas import (
    PracticeInterviewQuestion,
    PracticeSession,
    QuestionDifficulty,
    SessionCoachingStyle,
    SessionFollowUpDepth,
    SessionStrategyAction,
    SessionStrategyDecision,
    SpeechMetrics,
    MicroFeedback,
)


class _DummyPracticeService:
    def __init__(self):
        self.audio_dir = Path("data/practice_audio")
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.sessions: dict[str, PracticeSession] = {}

    async def start_interview(self, difficulty, user_profile=None, question_count=5, round_type=None, api_key=None, **kwargs):
        session_id = str(uuid.uuid4())
        q1 = PracticeInterviewQuestion(
            id=1,
            text="Tell me about a time you disagreed with a teammate.",
            difficulty=QuestionDifficulty.MEDIUM,
            category="behavioral",
        )
        q2 = PracticeInterviewQuestion(
            id=2,
            text="Explain how you would debug a memory leak in a web service.",
            difficulty=QuestionDifficulty.MEDIUM,
            category="technical",
        )
        session = PracticeSession(session_id=session_id, questions=[q1, q2])
        self.sessions[session_id] = session
        return session_id, q1, f"{session_id}_q1.mp3"

    def get_session(self, session_id: str):
        return self.sessions.get(session_id)

    async def submit_answer(self, session_id: str, question_id: int, audio_file_path: str, api_key=None):
        # Keep it minimal, deterministic.
        metrics = SpeechMetrics(
            filler_count=2,
            wpm=150.0,
            longest_silence=0.8,
            confidence_score=7.5,
            overtalked=False,
            duration=40.0,
            filler_words=["um", "like"],
            pause_count=1,
            pitch_variance=0.1,
            silence_removed=0.0,
        )
        micro = MicroFeedback(
            delivery_tips=["Slow down slightly", "Use clearer structure"],
            pace_feedback="Good pace",
            overall_note="Solid answer",
            correctness_score=80,
            technical_accuracy="Good",
            is_correct=True,
        )
        strategy = SessionStrategyDecision(
            action=SessionStrategyAction.FOLLOW_UP,
            reason="Drill one missed concept before moving on.",
            coaching_style=SessionCoachingStyle.BALANCED,
            follow_up_depth=SessionFollowUpDepth.LIGHT,
            target_difficulty=QuestionDifficulty.MEDIUM,
            source="fallback_rules",
            learning_focus=["core technical depth and correctness"],
            decision_trace={
                "guardrail": None,
                "proposed_action": "FOLLOW_UP",
                "proposed_source": "fallback_rules",
                "overall_score": 80.0,
                "correctness_score": 80,
                "remaining_questions": 1,
                "last_strategy_action": None,
                "follow_up_budget": {"used": 0, "max": 1, "remaining": 1},
            },
        )
        return {
            "transcript": "I would start by capturing heap snapshots and comparing growth over time...",
            "metrics": metrics,
            "micro_feedback": micro,
            "strategy": strategy,
            "complete": False,
            "progress": "1/2",
            "requires_acknowledgment": True,
            "current_question_id": question_id,
        }

    async def get_next_question_after_acknowledgment(self, session_id: str, question_id: int, api_key=None):
        session = self.sessions[session_id]
        next_q = session.questions[1]
        return {
            "complete": False,
            "progress": "2/2",
            "next_question": next_q,
            "tts_audio_url": f"{session_id}_q2.mp3",
            "strategy": SessionStrategyDecision(
                action=SessionStrategyAction.FOLLOW_UP,
                reason="Drill one missed concept before moving on.",
                coaching_style=SessionCoachingStyle.BALANCED,
                follow_up_depth=SessionFollowUpDepth.LIGHT,
                target_difficulty=QuestionDifficulty.MEDIUM,
                source="fallback_rules",
                learning_focus=["core technical depth and correctness"],
                decision_trace={
                    "guardrail": None,
                    "proposed_action": "FOLLOW_UP",
                    "proposed_source": "fallback_rules",
                    "overall_score": 80.0,
                    "correctness_score": 80,
                    "remaining_questions": 1,
                    "last_strategy_action": None,
                    "follow_up_budget": {"used": 0, "max": 1, "remaining": 1},
                },
            ),
        }


def test_practice_mode_emits_events(monkeypatch):
    init_db()

    app = FastAPI()
    app.include_router(practice_router)
    client = TestClient(app)

    user_id = f"test_practice_events_{uuid.uuid4()}"

    import app.routers.practice_mode as pm

    pm.practice_service = _DummyPracticeService()
    monkeypatch.setattr(pm, "get_user_id_from_request", lambda _req: user_id)

    # Start interview
    start = client.post(
        "/api/practice/interview/start",
        headers={"X-API-Key": "test_key"},
        json={
            "screen_shared": True,
            "camera_enabled": True,
            "difficulty": "medium",
            "category": "behavioral",
            "question_count": 2,
        },
    )
    assert start.status_code == 200
    assert start.json().get("auto_start_recording") is True
    session_id = start.json()["session_id"]

    # Submit answer (multipart)
    audio_bytes = b"RIFF----WAVEfmt "
    submit = client.post(
        "/api/practice/interview/submit-answer",
        headers={"X-API-Key": "test_key"},
        files={"audio": ("a.wav", audio_bytes, "audio/wav")},
        data={"session_id": session_id, "question_id": 1},
    )
    assert submit.status_code == 200
    assert submit.json().get("strategy", {}).get("action") == "FOLLOW_UP"

    # Acknowledge feedback
    ack = client.post(
        "/api/practice/interview/acknowledge-feedback",
        headers={"X-API-Key": "test_key"},
        json={"session_id": session_id, "question_id": 1, "feedback_read": True},
    )
    assert ack.status_code == 200
    assert ack.json().get("auto_start_recording") is True
    assert ack.json().get("strategy", {}).get("action") == "FOLLOW_UP"

    # Phase 3: Rate feedback usefulness (human label)
    rated = client.post(
        "/api/practice/interview/rate-feedback",
        headers={"X-API-Key": "test_key"},
        json={
            "session_id": session_id,
            "question_id": 1,
            "usefulness_rating": 5,
            "perceived_difficulty": "medium",
            "comment": "Very actionable feedback.",
        },
    )
    assert rated.status_code == 200
    assert rated.json().get("ok") is True

    with get_db_context() as db:
        rows = (
            db.query(EventRecord)
            .filter(EventRecord.user_id == user_id)
            .filter(EventRecord.session_id == session_id)
            .all()
        )

    event_types = {r.event_type for r in rows}
    assert "practice_session_started" in event_types
    assert "practice_question_served" in event_types
    assert "practice_answer_audio_received" in event_types
    assert "practice_answer_processed" in event_types
    assert "practice_feedback_acknowledged" in event_types
    assert "practice_feedback_rated" in event_types
    assert "practice_strategy_previewed" in event_types
    assert "practice_strategy_executed" in event_types

    preview_row = next(r for r in rows if r.event_type == "practice_strategy_previewed")
    execute_row = next(r for r in rows if r.event_type == "practice_strategy_executed")
    assert preview_row.extra_data.get("strategy_action") == "FOLLOW_UP"
    assert preview_row.extra_data.get("strategy_follow_up_budget_remaining") == 1
    assert execute_row.extra_data.get("strategy_stage") == "execution"
    assert execute_row.extra_data.get("strategy_guardrail") is None
