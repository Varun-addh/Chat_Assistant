"""Deterministic Practice Mode scoring (flagship loop rubric).

Goal:
- Provide a stable, explainable scoring contract that does NOT require an LLM.
- Produce: overall score (0-100), dimension scores (0-100), 'why' reasons,
  improvement plan, and a next-session plan.

This is intentionally conservative: it uses only signals we already have:
- MicroFeedback correctness_score / technical_accuracy / actionable_suggestions
- SpeechMetrics filler_count / wpm / confidence_score / overtalked

We can later swap/augment with model-based grading, but the contract stays.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from app.schemas import AnswerSubmission, EvaluationReport, PracticeSession


_DIMENSIONS = (
    "correctness",
    "delivery",
    "clarity",
    "structure",
)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _map_technical_accuracy(label: Optional[str]) -> Optional[float]:
    if not label:
        return None
    s = label.strip().lower()
    if s in {"excellent", "great"}:
        return 90.0
    if s in {"good"}:
        return 80.0
    if s in {"fair", "ok", "okay"}:
        return 65.0
    if s in {"poor"}:
        return 45.0
    return None


def _structure_heuristic(transcript: str) -> float:
    """Cheap, explainable heuristic: reward explicit structure markers.

    Uses word-boundary regex to avoid false positives (e.g. 'step' inside
    'stephen').
    """
    t = (transcript or "").lower()
    if not t.strip():
        return 0.0

    markers = [
        "first", "second", "third", "finally", "in summary",
        "overall", "to start", "step by step", "i would",
        "next", "then", "moreover", "to conclude", "lastly",
        "additionally", "let me break", "in conclusion",
    ]
    hits = sum(1 for m in markers if re.search(r"\b" + re.escape(m) + r"\b", t))
    # 50 base, +7 per marker, capped.
    return _clamp(50.0 + 7.0 * hits, 40.0, 95.0)


@dataclass(frozen=True)
class PracticeScoreResult:
    overall_score: float
    dimension_scores: dict[str, float]
    why: list[str]
    improvement_plan: list[str]
    next_session_plan: dict[str, Any]


@dataclass(frozen=True)
class PracticeAnswerScoreResult:
    question_id: int
    overall_score: float
    dimension_scores: dict[str, float]
    signals: dict[str, Any]
    why: list[str]


def score_answer(*, answer: AnswerSubmission) -> PracticeAnswerScoreResult:
    """Compute deterministic per-answer scores.

    This powers within-session trajectory and trace.
    """

    mf = getattr(answer, "micro_feedback", None)
    metrics = getattr(answer, "metrics", None)
    transcript = getattr(answer, "transcript", "") or ""

    # Correctness
    cs: Optional[float] = None
    if mf is not None and getattr(mf, "correctness_score", None) is not None:
        cs = float(getattr(mf, "correctness_score"))
    if cs is None and mf is not None:
        cs = _map_technical_accuracy(getattr(mf, "technical_accuracy", None))
    if cs is None:
        cs = 60.0
    correctness = _clamp(float(cs), 0.0, 100.0)

    fillers = int(getattr(metrics, "filler_count", 0) or 0)
    wpm = float(getattr(metrics, "wpm", 0.0) or 0.0)
    conf = float(getattr(metrics, "confidence_score", 0.0) or 0.0)
    overtalked = bool(getattr(metrics, "overtalked", False))

    # Delivery scoring
    # confidence_score may arrive on a 0-1 or 0-10 scale depending on
    # the speech provider.  Normalise to 0-1 before converting to 0-100.
    has_speech_metrics = conf > 0.0 or wpm > 0.0
    conf_norm = conf / 10.0 if conf > 1.0 else conf
    if has_speech_metrics:
        delivery = conf_norm * 100.0
    else:
        # No speech data — assume average-good delivery (70/100)
        delivery = 70.0
    delivery -= min(20.0, fillers * 2.5)
    if overtalked:
        delivery -= 10.0
    delivery = _clamp(delivery, 0.0, 100.0)

    # Clarity
    dist = abs(wpm - 150.0)
    clarity = 90.0 - (dist * 0.6)
    clarity -= min(20.0, fillers * 2.0)
    clarity = _clamp(clarity, 0.0, 100.0)

    # Structure
    structure = _structure_heuristic(transcript)

    dims = {
        "correctness": float(correctness),
        "delivery": float(delivery),
        "clarity": float(clarity),
        "structure": float(structure),
    }

    overall = (
        dims["correctness"] * 0.45
        + dims["delivery"] * 0.25
        + dims["clarity"] * 0.15
        + dims["structure"] * 0.15
    )
    overall = _clamp(float(overall), 0.0, 100.0)

    why = [
        f"Correctness: {dims['correctness']:.0f}/100",
        f"Delivery: {dims['delivery']:.0f}/100 (confidence={conf_norm:.2f}/1.0, fillers={fillers}, overtalked={int(overtalked)})",
        f"Clarity: {dims['clarity']:.0f}/100 (wpm={wpm:.0f})",
        f"Structure: {dims['structure']:.0f}/100",
    ]

    signals = {
        "confidence_score": conf,
        "filler_count": fillers,
        "wpm": wpm,
        "overtalked": overtalked,
        "technical_accuracy": getattr(mf, "technical_accuracy", None) if mf is not None else None,
        "correctness_score": getattr(mf, "correctness_score", None) if mf is not None else None,
    }

    return PracticeAnswerScoreResult(
        question_id=int(getattr(answer, "question_id", 0) or 0),
        overall_score=overall,
        dimension_scores=dims,
        signals=signals,
        why=why,
    )


def compute_session_trajectory(*, session: PracticeSession) -> dict[str, Any]:
    """Return a deterministic within-session trajectory summary."""

    answers = list(getattr(session, "answers", []) or [])
    if not answers:
        return {
            "points": [],
            "overall": None,
            "dimensions": {},
            "best_improvement_dimension": None,
            "note": "No answers yet.",
        }

    points: list[dict[str, Any]] = []
    per_dim_series: dict[str, list[float]] = {k: [] for k in _DIMENSIONS}
    overall_series: list[float] = []

    for a in answers:
        r = score_answer(answer=a)
        points.append(
            {
                "question_id": r.question_id,
                "overall_score": float(r.overall_score),
                "dimension_scores": r.dimension_scores,
            }
        )
        overall_series.append(float(r.overall_score))
        for dim in _DIMENSIONS:
            per_dim_series[dim].append(float(r.dimension_scores.get(dim, 0.0)))

    def _delta(xs: list[float]) -> Optional[dict[str, float]]:
        if not xs:
            return None
        start = float(xs[0])
        end = float(xs[-1])
        return {"start": start, "end": end, "delta": float(end - start)}

    overall_delta = _delta(overall_series)
    dim_deltas: dict[str, dict[str, float]] = {}
    for dim, xs in per_dim_series.items():
        d = _delta(xs)
        if d is not None:
            dim_deltas[dim] = d

    best_dim = None
    if dim_deltas:
        best_dim = max(dim_deltas.items(), key=lambda kv: kv[1].get("delta", 0.0))[0]

    note = "Stable"
    if overall_delta is not None:
        if overall_delta["delta"] >= 5.0:
            note = "Improving"
        elif overall_delta["delta"] <= -5.0:
            note = "Declining"

    return {
        "points": points,
        "overall": overall_delta,
        "dimensions": dim_deltas,
        "best_improvement_dimension": best_dim,
        "note": note,
    }


def build_evaluation_trace(*, session: PracticeSession) -> dict[str, Any]:
    """Build a deterministic, explainable scoring trace for the session."""

    score = score_session(session=session)
    trajectory = compute_session_trajectory(session=session)

    weights = {"correctness": 0.45, "delivery": 0.25, "clarity": 0.15, "structure": 0.15}
    dims = score.dimension_scores or {}

    formulas = {
        "correctness": "correctness_score (0-100) else map(technical_accuracy) else 60",
        "delivery": "confidence_score*10 - min(20, filler_count*2.5) - (10 if overtalked)",
        "clarity": "90 - abs(wpm-150)*0.6 - min(20, filler_count*2)",
        "structure": "clamp(50 + 10*structure_marker_hits, 40, 95)",
        "overall": "0.45*correctness + 0.25*delivery + 0.15*clarity + 0.15*structure",
    }

    return {
        "overall_score": float(score.overall_score),
        "dimension_scores": {k: float(v) for k, v in (dims or {}).items()},
        "weights": weights,
        "formulas": formulas,
        "why": list(score.why or []),
        "trajectory": trajectory,
    }


def score_session(*, session: PracticeSession) -> PracticeScoreResult:
    """Compute attempt-level scores from a completed session."""

    # Aggregate per-answer signals.
    if not session.answers:
        dim = {k: 0.0 for k in _DIMENSIONS}
        return PracticeScoreResult(
            overall_score=0.0,
            dimension_scores=dim,
            why=["No answers were submitted."],
            improvement_plan=["Retry the session and answer at least 2 questions."],
            next_session_plan={"focus": ["complete_answers"], "question_count": 5},
        )

    correctness_scores: list[float] = []
    delivery_scores: list[float] = []
    clarity_scores: list[float] = []
    structure_scores: list[float] = []

    fillers_total = 0
    overtalked_count = 0
    avg_wpm_vals: list[float] = []
    avg_conf_vals: list[float] = []

    suggestions: list[str] = []

    for a in session.answers:
        mf = a.micro_feedback
        metrics = a.metrics

        # Correctness (prefer explicit correctness_score, fallback to technical_accuracy)
        cs: Optional[float] = None
        if getattr(mf, "correctness_score", None) is not None:
            cs = float(mf.correctness_score)  # 0-100
        if cs is None:
            cs = _map_technical_accuracy(getattr(mf, "technical_accuracy", None))
        if cs is None:
            cs = 60.0  # conservative default
        correctness_scores.append(_clamp(cs, 0.0, 100.0))

        # Delivery: mostly confidence score (0-10), with penalties
        conf = float(getattr(metrics, "confidence_score", 0.0) or 0.0)
        fillers = int(getattr(metrics, "filler_count", 0) or 0)
        overtalked = bool(getattr(metrics, "overtalked", False))

        fillers_total += fillers
        if overtalked:
            overtalked_count += 1

        # Normalise confidence_score to 0-1 range (may arrive as 0-10)
        conf_norm = conf / 10.0 if conf > 1.0 else conf
        delivery = conf_norm * 100.0
        # Penalize fillers (small)
        delivery -= min(20.0, fillers * 2.5)
        # Penalize overtalking
        if overtalked:
            delivery -= 10.0
        delivery_scores.append(_clamp(delivery, 0.0, 100.0))

        # Clarity: WPM closeness to ideal (140-160) + filler penalty
        wpm = float(getattr(metrics, "wpm", 0.0) or 0.0)
        avg_wpm_vals.append(wpm)
        avg_conf_vals.append(conf)

        # Distance from sweet spot 150
        dist = abs(wpm - 150.0)
        clarity = 90.0 - (dist * 0.6)
        clarity -= min(20.0, fillers * 2.0)
        clarity_scores.append(_clamp(clarity, 0.0, 100.0))

        # Structure: transcript markers heuristic
        structure_scores.append(_structure_heuristic(a.transcript))

        # Suggestions (dedupe later)
        for s in (getattr(mf, "actionable_suggestions", None) or []):
            if s and isinstance(s, str):
                suggestions.append(s.strip())

    def _avg(xs: list[float]) -> float:
        return sum(xs) / max(1, len(xs))

    dims = {
        "correctness": _avg(correctness_scores),
        "delivery": _avg(delivery_scores),
        "clarity": _avg(clarity_scores),
        "structure": _avg(structure_scores),
    }

    # Weighted overall
    overall = (
        dims["correctness"] * 0.45
        + dims["delivery"] * 0.25
        + dims["clarity"] * 0.15
        + dims["structure"] * 0.15
    )
    overall = _clamp(overall, 0.0, 100.0)

    avg_wpm = _avg(avg_wpm_vals)
    avg_conf = _avg(avg_conf_vals)

    why: list[str] = []
    why.append(f"Correctness avg: {dims['correctness']:.0f}/100")
    why.append(f"Delivery avg: {dims['delivery']:.0f}/100 (fillers={fillers_total}, overtalked={overtalked_count})")
    why.append(f"Clarity avg: {dims['clarity']:.0f}/100 (avg_wpm={avg_wpm:.0f})")
    why.append(f"Structure avg: {dims['structure']:.0f}/100")

    # Improvement plan: top actionable suggestions + dimension-based defaults
    improvement_plan: list[str] = []
    seen = set()
    for s in suggestions:
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        improvement_plan.append(s)
        if len(improvement_plan) >= 3:
            break

    # Fill with deterministic coaching steps
    if len(improvement_plan) < 3:
        # pick weakest dims
        weakest = sorted(dims.items(), key=lambda kv: kv[1])
        for dim, _score in weakest:
            if len(improvement_plan) >= 3:
                break
            if dim == "correctness":
                improvement_plan.append("After each question, restate the problem and give a crisp 2–3 step solution before details.")
            elif dim == "delivery":
                improvement_plan.append("Aim for fewer filler words: pause silently for 1s instead of saying 'um/like'.")
            elif dim == "clarity":
                improvement_plan.append("Target ~150 WPM: slow down on key points, speed up on setup.")
            elif dim == "structure":
                improvement_plan.append("Use explicit structure: 'First…, Second…, Finally…' in every answer.")

    # Next session plan: focus on weakest dimension
    weakest_dim = min(dims.items(), key=lambda kv: kv[1])[0]
    focus_map = {
        "correctness": ["core_concepts", "accuracy", "edge_cases"],
        "delivery": ["confidence", "reduce_fillers", "timeboxing"],
        "clarity": ["pace", "conciseness", "signposting"],
        "structure": ["answer_framework", "step_by_step", "summarize"],
    }

    # Suggest a slightly shorter next session if score is low (reduce fatigue)
    qc = int(getattr(session, "questions", None) and len(session.questions) or 5)
    if overall < 55:
        qc = min(qc, 4)

    next_session_plan = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "focus_dimension": weakest_dim,
        "focus": focus_map.get(weakest_dim, [weakest_dim]),
        "question_count": qc,
        "recommended_round": getattr(getattr(session, "round_type", None), "value", None),
        "difficulty": getattr(getattr(session, "difficulty", None), "value", None),
    }

    return PracticeScoreResult(
        overall_score=overall,
        dimension_scores={k: float(_clamp(v, 0.0, 100.0)) for k, v in dims.items()},
        why=why,
        improvement_plan=improvement_plan[:3],
        next_session_plan=next_session_plan,
    )


def evaluation_report_to_json(report: Optional[EvaluationReport]) -> Optional[dict[str, Any]]:
    if not report:
        return None
    # Pydantic v2: mode="json" converts datetime/etc to JSON-safe types (str).
    # This prevents "Object of type datetime is not JSON serializable" when
    # psycopg serializes the dict for a JSON column.
    if hasattr(report, "model_dump"):
        return report.model_dump(mode="json")  # type: ignore[attr-defined]
    return report.dict()  # type: ignore[call-arg]
