"""Learning loops powered by structured EventRecord data.

This module turns raw product events into lightweight, deterministic signals
that can improve the next session *today* (without any model training).

Design goals:
- Safe-by-default: do not require raw text.
- Best-effort: callers can treat outputs as optional hints.
- Deterministic and explainable: simple aggregates, no magic.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from sqlalchemy.orm import Session

from app.models import EventRecord


@dataclass(frozen=True)
class PracticeAttempt:
    session_id: str
    question_id: int
    category: Optional[str]
    difficulty: Optional[str]
    round_type: Optional[str]
    correctness_score: Optional[float]
    is_correct: Optional[bool]
    wpm: Optional[float]
    filler_count: Optional[float]
    confidence_score: Optional[float]


def _as_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _as_int(v: Any) -> Optional[int]:
    try:
        if v is None:
            return None
        return int(v)
    except Exception:
        return None


def _avg(values: Iterable[Optional[float]]) -> Optional[float]:
    xs = [v for v in values if v is not None]
    if not xs:
        return None
    return sum(xs) / len(xs)


def _collect_practice_attempts(
    db: Session,
    *,
    user_id: str,
    lookback_days: int = 30,
    max_events: int = 750,
    domain: Optional[str] = None,
) -> List[PracticeAttempt]:
    """Join practice_question_served + practice_answer_processed into attempts.

    If domain is provided, attempts are filtered to sessions whose
    practice_session_started.extra_data["domain"] matches (case-insensitive).
    """

    since = datetime.now(timezone.utc) - timedelta(days=lookback_days)

    event_types = [
        "practice_session_started",
        "practice_question_served",
        "practice_answer_processed",
    ]

    rows: List[EventRecord] = (
        db.query(EventRecord)
        .filter(EventRecord.user_id == user_id)
        .filter(EventRecord.timestamp >= since)
        .filter(EventRecord.event_type.in_(event_types))
        .order_by(EventRecord.timestamp.desc())
        .limit(max_events)
        .all()
    )

    # Map session_id -> domain (if present)
    session_domain: Dict[str, str] = {}

    served: Dict[Tuple[str, int], Dict[str, Any]] = {}
    processed: Dict[Tuple[str, int], Dict[str, Any]] = {}

    for r in rows:
        if not r.session_id:
            continue
        extra = r.extra_data or {}

        if r.event_type == "practice_session_started":
            dom = extra.get("domain")
            if isinstance(dom, str) and dom.strip():
                session_domain[r.session_id] = dom.strip()
            continue

        qid = _as_int(extra.get("question_id"))
        if qid is None:
            continue
        key = (r.session_id, qid)

        if r.event_type == "practice_question_served":
            served[key] = extra
        elif r.event_type == "practice_answer_processed":
            processed[key] = extra

    attempts: List[PracticeAttempt] = []

    for (sess_id, qid), proc in processed.items():
        if domain:
            sd = session_domain.get(sess_id)
            if not sd or sd.strip().lower() != domain.strip().lower():
                continue

        meta = served.get((sess_id, qid), {})

        attempts.append(
            PracticeAttempt(
                session_id=sess_id,
                question_id=qid,
                category=meta.get("category"),
                difficulty=meta.get("difficulty"),
                round_type=meta.get("round_type"),
                correctness_score=_as_float(proc.get("correctness_score")),
                is_correct=proc.get("is_correct") if isinstance(proc.get("is_correct"), bool) else None,
                wpm=_as_float(proc.get("wpm")),
                filler_count=_as_float(proc.get("filler_count")),
                confidence_score=_as_float(proc.get("confidence_score")),
            )
        )

    # Sort oldest -> newest for nicer UX when displaying
    attempts.sort(key=lambda a: (a.session_id, a.question_id))
    return attempts


def compute_practice_insights(
    db: Session,
    *,
    user_id: str,
    domain: Optional[str] = None,
    lookback_days: int = 30,
    max_events: int = 750,
) -> Dict[str, Any]:
    """Compute simple, explainable per-user practice insights."""

    attempts = _collect_practice_attempts(
        db,
        user_id=user_id,
        domain=domain,
        lookback_days=lookback_days,
        max_events=max_events,
    )

    by_category: Dict[str, Dict[str, Any]] = {}
    by_difficulty: Dict[str, Dict[str, Any]] = {}

    for a in attempts:
        if a.category:
            cat = str(a.category)
            by_category.setdefault(cat, {"attempts": 0, "scores": []})
            by_category[cat]["attempts"] += 1
            by_category[cat]["scores"].append(a.correctness_score)

        if a.difficulty:
            diff = str(a.difficulty)
            by_difficulty.setdefault(diff, {"attempts": 0, "scores": []})
            by_difficulty[diff]["attempts"] += 1
            by_difficulty[diff]["scores"].append(a.correctness_score)

    def _finalize_bucket(b: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for k, v in b.items():
            out[k] = {
                "attempts": int(v.get("attempts", 0)),
                "avg_correctness": _avg(v.get("scores", [])),
            }
        return out

    by_category_out = _finalize_bucket(by_category)
    by_difficulty_out = _finalize_bucket(by_difficulty)

    overall_avg_correctness = _avg([a.correctness_score for a in attempts])
    overall_avg_confidence = _avg([a.confidence_score for a in attempts])
    overall_avg_filler = _avg([a.filler_count for a in attempts])
    overall_avg_wpm = _avg([a.wpm for a in attempts])

    recommended_focus: List[str] = []

    # 1) Content weaknesses (by category)
    weak_cats = []
    for cat, stats in by_category_out.items():
        if stats.get("attempts", 0) < 2:
            continue
        avg = stats.get("avg_correctness")
        if avg is None:
            continue
        weak_cats.append((avg, cat))

    weak_cats.sort(key=lambda t: t[0])  # lowest avg first

    for avg, cat in weak_cats[:3]:
        if avg < 70:
            # Keep it human-friendly; avoid internal enum-ish strings.
            if cat.lower() in {"system_design", "system design"}:
                recommended_focus.append("system design fundamentals and tradeoffs")
            elif cat.lower() in {"behavioral"}:
                recommended_focus.append("behavioral storytelling (STAR/impact) and clarity")
            elif cat.lower() in {"technical"}:
                recommended_focus.append("core technical depth and correctness")
            else:
                recommended_focus.append(f"improve performance in: {cat}")

    # 2) Delivery weaknesses (speech)
    if overall_avg_filler is not None and overall_avg_filler > 6:
        recommended_focus.append("reduce filler words and pause intentionally")

    if overall_avg_confidence is not None and overall_avg_confidence < 0.5:
        recommended_focus.append("improve confidence and structured delivery")

    # 3) If we have little data but poor overall score, provide a generic focus.
    if not recommended_focus and overall_avg_correctness is not None and overall_avg_correctness < 70:
        recommended_focus.append("tighten fundamentals before increasing difficulty")

    # Deduplicate but keep stable order
    dedup: List[str] = []
    for f in recommended_focus:
        if f not in dedup:
            dedup.append(f)

    return {
        "user_id": user_id,
        "domain": domain,
        "lookback_days": int(lookback_days),
        "attempts": len(attempts),
        "overall": {
            "avg_correctness": overall_avg_correctness,
            "avg_confidence": overall_avg_confidence,
            "avg_filler_count": overall_avg_filler,
            "avg_wpm": overall_avg_wpm,
        },
        "by_category": by_category_out,
        "by_difficulty": by_difficulty_out,
        "recommended_focus": dedup,
    }


def merge_focus_areas(existing: Optional[List[str]], recommended: List[str], *, max_items: int = 5) -> List[str]:
    """Merge recommended focus areas into an existing list, deduped."""
    out: List[str] = []
    for x in existing or []:
        if isinstance(x, str) and x.strip() and x not in out:
            out.append(x.strip())
    for x in recommended or []:
        if isinstance(x, str) and x.strip() and x not in out:
            out.append(x.strip())
    return out[:max_items]
