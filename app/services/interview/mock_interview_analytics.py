from __future__ import annotations

from typing import Any, Optional


_CRITERIA = (
    "correctness",
    "completeness",
    "clarity",
    "confidence",
    "technical_depth",
)


def compute_mock_session_trajectory(*, session: Any) -> dict[str, Any]:
    """Deterministic within-session trajectory for Mock Interview.

    This is intentionally post-hoc and light: it does not influence question
    sequencing (Mock Interview should feel linear/panel-like).
    """

    evals = list(getattr(session, "evaluations", []) or [])
    if not evals:
        return {
            "points": [],
            "overall": None,
            "criteria": {},
            "best_improvement_criterion": None,
            "note": "No evaluations yet.",
        }

    points: list[dict[str, Any]] = []
    overall_series: list[float] = []
    per_crit_series: dict[str, list[float]] = {k: [] for k in _CRITERIA}

    for idx, e in enumerate(evals):
        overall = float(getattr(e, "overall_score", 0.0) or 0.0)
        overall_series.append(overall)

        cs = getattr(e, "criteria_scores", None)
        crit_scores = {k: float(getattr(cs, k, 0.0) or 0.0) for k in _CRITERIA} if cs is not None else {k: 0.0 for k in _CRITERIA}
        for k in _CRITERIA:
            per_crit_series[k].append(float(crit_scores.get(k, 0.0)))

        points.append(
            {
                "question_number": idx + 1,
                "overall_score": overall,
                "criteria_scores": crit_scores,
            }
        )

    def _delta(xs: list[float]) -> Optional[dict[str, float]]:
        if not xs:
            return None
        start = float(xs[0])
        end = float(xs[-1])
        return {"start": start, "end": end, "delta": float(end - start)}

    overall_delta = _delta(overall_series)

    crit_deltas: dict[str, dict[str, float]] = {}
    for k, xs in per_crit_series.items():
        d = _delta(xs)
        if d is not None:
            crit_deltas[k] = d

    best_crit = None
    if crit_deltas:
        best_crit = max(crit_deltas.items(), key=lambda kv: kv[1].get("delta", 0.0))[0]

    note = "Stable"
    if overall_delta is not None:
        if overall_delta["delta"] >= 0.7:
            note = "Improving"
        elif overall_delta["delta"] <= -0.7:
            note = "Declining"

    return {
        "points": points,
        "overall": overall_delta,
        "criteria": crit_deltas,
        "best_improvement_criterion": best_crit,
        "note": note,
    }


def build_mock_evaluation_trace(*, session: Any) -> dict[str, Any]:
    """Deterministic post-hoc trace for Mock Interview.

    Unlike Practice Mode, this is not a formula-based trace: the per-question
    evaluations are LLM-based. The trace explains *how we aggregate* what the
    user already received into session-level signals.
    """

    evals = list(getattr(session, "evaluations", []) or [])
    if not evals:
        return {
            "aggregation": "average_over_questions",
            "average_score": None,
            "criteria_averages": {},
            "why": ["No evaluations yet."],
            "trajectory": compute_mock_session_trajectory(session=session),
        }

    avg_score = float(getattr(session, "average_score", None) or 0.0)
    if avg_score <= 0.0:
        avg_score = sum(float(getattr(e, "overall_score", 0.0) or 0.0) for e in evals) / max(1, len(evals))

    # Criterion averages
    crit_totals = {k: 0.0 for k in _CRITERIA}
    for e in evals:
        cs = getattr(e, "criteria_scores", None)
        for k in _CRITERIA:
            crit_totals[k] += float(getattr(cs, k, 0.0) or 0.0) if cs is not None else 0.0

    crit_avgs = {k: (v / max(1, len(evals))) for k, v in crit_totals.items()}

    strongest = max(crit_avgs.items(), key=lambda kv: kv[1])[0] if crit_avgs else None
    weakest = min(crit_avgs.items(), key=lambda kv: kv[1])[0] if crit_avgs else None

    # Consistency (simple stddev on overall scores)
    scores = [float(getattr(e, "overall_score", 0.0) or 0.0) for e in evals]
    mean = sum(scores) / max(1, len(scores))
    variance = sum((x - mean) ** 2 for x in scores) / max(1, len(scores))
    std_dev = variance ** 0.5

    if len(scores) < 2:
        consistency = "N/A"
    elif std_dev < 0.6:
        consistency = "Very Consistent"
    elif std_dev < 1.0:
        consistency = "Consistent"
    else:
        consistency = "Variable"

    why = [
        f"Session score is the average of per-question overall scores ({len(evals)} questions).",
        f"Strongest criterion (avg): {strongest}",
        f"Weakest criterion (avg): {weakest}",
        f"Consistency: {consistency}",
    ]

    return {
        "aggregation": "average_over_questions",
        "average_score": round(float(avg_score), 2),
        "criteria_averages": {k: round(float(v), 2) for k, v in crit_avgs.items()},
        "why": why,
        "trajectory": compute_mock_session_trajectory(session=session),
    }
