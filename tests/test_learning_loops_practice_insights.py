"""Regression test: practice learning loop recommends focus areas from event history."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from app.database import get_db_context, init_db
from app.models import EventRecord
from app.services.learning_loops import compute_practice_insights


def test_practice_insights_recommends_weak_category():
    init_db()

    user_id = f"test_focus_user_{uuid.uuid4()}"
    session_id = f"sess_{uuid.uuid4()}"

    now = datetime.now(timezone.utc)

    with get_db_context() as db:
        # Session start (for domain filtering)
        db.add(
            EventRecord(
                user_id=user_id,
                session_id=session_id,
                event_type="practice_session_started",
                extra_data={"domain": "Python Backend"},
                timestamp=now,
            )
        )

        # Two system design questions served
        db.add(
            EventRecord(
                user_id=user_id,
                session_id=session_id,
                event_type="practice_question_served",
                extra_data={
                    "question_id": 1,
                    "category": "system_design",
                    "difficulty": "hard",
                    "round_type": "system_design",
                },
                timestamp=now,
            )
        )
        db.add(
            EventRecord(
                user_id=user_id,
                session_id=session_id,
                event_type="practice_question_served",
                extra_data={
                    "question_id": 2,
                    "category": "system_design",
                    "difficulty": "hard",
                    "round_type": "system_design",
                },
                timestamp=now,
            )
        )

        # Two technical questions served
        db.add(
            EventRecord(
                user_id=user_id,
                session_id=session_id,
                event_type="practice_question_served",
                extra_data={
                    "question_id": 3,
                    "category": "technical",
                    "difficulty": "medium",
                    "round_type": "technical",
                },
                timestamp=now,
            )
        )
        db.add(
            EventRecord(
                user_id=user_id,
                session_id=session_id,
                event_type="practice_question_served",
                extra_data={
                    "question_id": 4,
                    "category": "technical",
                    "difficulty": "medium",
                    "round_type": "technical",
                },
                timestamp=now,
            )
        )

        # Answers processed: weak in system_design, strong in technical
        db.add(
            EventRecord(
                user_id=user_id,
                session_id=session_id,
                event_type="practice_answer_processed",
                extra_data={
                    "question_id": 1,
                    "correctness_score": 40,
                    "confidence_score": 0.9,
                    "filler_count": 2,
                    "wpm": 130,
                },
                timestamp=now,
            )
        )
        db.add(
            EventRecord(
                user_id=user_id,
                session_id=session_id,
                event_type="practice_answer_processed",
                extra_data={
                    "question_id": 2,
                    "correctness_score": 55,
                    "confidence_score": 0.8,
                    "filler_count": 3,
                    "wpm": 120,
                },
                timestamp=now,
            )
        )
        db.add(
            EventRecord(
                user_id=user_id,
                session_id=session_id,
                event_type="practice_answer_processed",
                extra_data={
                    "question_id": 3,
                    "correctness_score": 85,
                    "confidence_score": 0.8,
                    "filler_count": 2,
                    "wpm": 140,
                },
                timestamp=now,
            )
        )
        db.add(
            EventRecord(
                user_id=user_id,
                session_id=session_id,
                event_type="practice_answer_processed",
                extra_data={
                    "question_id": 4,
                    "correctness_score": 80,
                    "confidence_score": 0.7,
                    "filler_count": 2,
                    "wpm": 150,
                },
                timestamp=now,
            )
        )

        db.commit()

        insights = compute_practice_insights(db, user_id=user_id, domain="Python Backend")

    assert insights["attempts"] == 4

    rec = insights["recommended_focus"]
    assert any("system design" in x.lower() for x in rec)

    by_cat = insights["by_category"]
    assert "system_design" in by_cat
    assert by_cat["system_design"]["attempts"] == 2
    assert by_cat["system_design"]["avg_correctness"] is not None
