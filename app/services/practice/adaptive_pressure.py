"""Deterministic, testable "Adaptive Pressure" rules.

This module intentionally does NOT call any LLMs.
It computes a small pressure state that can optionally influence
follow-up generation (tone/difficulty) when enabled via config.
"""

from __future__ import annotations

from typing import Any, Optional

from app.schemas import PracticeSession, QuestionDifficulty
from app.services.practice.practice_scoring import score_answer


def compute_pressure_state(*, session: PracticeSession) -> dict[str, Any]:
    """Compute a deterministic pressure state from recent answers.

    Returns a JSON-serializable dict:
    - mode: supportive | balanced | challenging
    - level: 0..2
    - reasons: list[str]
    """

    answers = list(getattr(session, "answers", []) or [])
    if not answers:
        return {
            "mode": "balanced",
            "level": 1,
            "reasons": ["No answers yet."],
        }

    # Use the last up to 2 answers for responsiveness.
    recent = answers[-2:]
    scored = [score_answer(answer=a) for a in recent]

    last = scored[-1]
    prev = scored[0] if len(scored) == 2 else None

    last_overall = float(last.overall_score)
    last_correctness = float((last.dimension_scores or {}).get("correctness", 0.0))
    last_delivery = float((last.dimension_scores or {}).get("delivery", 0.0))

    delta = None
    if prev is not None:
        delta = float(last_overall - float(prev.overall_score))

    reasons: list[str] = []

    # Supportive if struggling.
    if last_overall <= 55.0 or last_correctness <= 60.0:
        reasons.append("Recent performance indicates the candidate is still stabilizing fundamentals.")
        if delta is not None and delta < -3.0:
            reasons.append("Recent trend is downward.")
        return {"mode": "supportive", "level": 0, "reasons": reasons}

    # Challenging if strong and/or improving.
    if last_overall >= 78.0 and last_correctness >= 75.0 and last_delivery >= 70.0:
        reasons.append("Recent performance is strong across correctness and delivery.")
        if delta is not None and delta >= 3.0:
            reasons.append("Recent trend is upward.")
        return {"mode": "challenging", "level": 2, "reasons": reasons}

    # Default: balanced.
    reasons.append("Maintain steady pressure to optimize learning without overload.")
    if delta is not None:
        reasons.append(f"Recent overall delta: {delta:+.1f}.")
    return {"mode": "balanced", "level": 1, "reasons": reasons}


def adjust_difficulty(*, base: Optional[QuestionDifficulty], mode: str) -> Optional[QuestionDifficulty]:
    if base is None:
        return None

    if mode == "supportive":
        if base == QuestionDifficulty.HARD:
            return QuestionDifficulty.MEDIUM
        if base == QuestionDifficulty.MEDIUM:
            return QuestionDifficulty.EASY
        return QuestionDifficulty.EASY

    if mode == "challenging":
        if base == QuestionDifficulty.EASY:
            return QuestionDifficulty.MEDIUM
        if base == QuestionDifficulty.MEDIUM:
            return QuestionDifficulty.HARD
        return QuestionDifficulty.HARD

    return base
