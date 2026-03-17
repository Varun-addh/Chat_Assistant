from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy.orm import Session

from app.models import PracticeProctoringEvent, PracticeProctoringSession


PROCTORING_TOTAL_VIOLATION_LIMIT = 5
PROCTORING_SERIOUS_VIOLATION_LIMIT = 3
PROCTORING_HEARTBEAT_STALE_SECONDS = 15


CANONICAL_EVENT_ALIASES = {
    "camera_started": "CAMERA_STARTED",
    "camera_stopped": "CAMERA_STOPPED",
    "camera_heartbeat": "CAMERA_HEARTBEAT",
    "screen_stopped": "SCREEN_STOPPED",
    "tab_switch": "TAB_SWITCH",
    "window_blur": "WINDOW_BLUR",
    "window_minimized": "WINDOW_MINIMIZED",
    "face_missing": "FACE_MISSING",
    "multiple_faces": "MULTIPLE_FACES",
    "user_left_frame": "USER_LEFT_FRAME",
    "monitoring_interrupted": "MONITORING_INTERRUPTED",
    "display_surface_mismatch": "DISPLAY_SURFACE_MISMATCH",
    "session_started_with_proctoring": "SESSION_STARTED_WITH_PROCTORING",
    "session_started_without_proctoring": "SESSION_STARTED_WITHOUT_PROCTORING",
}

LOW_VIOLATION_EVENTS = {
    "TAB_SWITCH",
    "WINDOW_BLUR",
    "WINDOW_MINIMIZED",
    "DISPLAY_SURFACE_MISMATCH",
}

SERIOUS_VIOLATION_EVENTS = {
    "SCREEN_STOPPED",
    "CAMERA_STOPPED",
    "MULTIPLE_FACES",
    "FACE_MISSING",
    "USER_LEFT_FRAME",
    "MONITORING_INTERRUPTED",
}

INFORMATIONAL_EVENTS = {
    "CAMERA_STARTED",
    "CAMERA_HEARTBEAT",
    "SESSION_STARTED_WITH_PROCTORING",
    "SESSION_STARTED_WITHOUT_PROCTORING",
}

EVENT_COOLDOWN_SECONDS = {
    "TAB_SWITCH": 3,
    "WINDOW_BLUR": 3,
    "WINDOW_MINIMIZED": 3,
    "DISPLAY_SURFACE_MISMATCH": 10,
    "SCREEN_STOPPED": 10,
    "CAMERA_STOPPED": 10,
    "MULTIPLE_FACES": 12,
    "FACE_MISSING": 12,
    "USER_LEFT_FRAME": 12,
    "MONITORING_INTERRUPTED": 12,
}

HEARTBEAT_EVENT_STREAK_THRESHOLDS = {
    "CAMERA_STOPPED": 1,
    "SCREEN_STOPPED": 1,
    "TAB_SWITCH": 1,
    "WINDOW_BLUR": 2,
    "MONITORING_INTERRUPTED": 2,
    "DISPLAY_SURFACE_MISMATCH": 1,
}

EVENT_MESSAGES = {
    "TAB_SWITCH": "Please return to the interview tab.",
    "WINDOW_BLUR": "Please keep the interview window focused.",
    "WINDOW_MINIMIZED": "Please keep the interview window open and visible.",
    "DISPLAY_SURFACE_MISMATCH": "Please share the entire screen instead of a single window.",
    "SCREEN_STOPPED": "Screen sharing stopped. Resume it to continue.",
    "CAMERA_STOPPED": "Camera stopped. Re-enable it to continue.",
    "MULTIPLE_FACES": "Multiple faces detected. Only one person may be present.",
    "FACE_MISSING": "Face not visible. Stay in frame.",
    "USER_LEFT_FRAME": "You left the camera frame. Return to continue.",
    "MONITORING_INTERRUPTED": "Proctoring monitoring was interrupted. Refresh if this persists.",
}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_aware_utc(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _public_monitoring_metadata(metadata: Optional[dict[str, Any]]) -> dict[str, Any]:
    return {
        key: value
        for key, value in dict(metadata or {}).items()
        if not str(key).startswith("_internal_")
    }


def _should_emit_heartbeat_event(
    metadata: dict[str, Any],
    *,
    event_type: str,
    condition_active: bool,
) -> tuple[bool, int]:
    canonical = canonicalize_proctoring_event_type(event_type)
    threshold = int(HEARTBEAT_EVENT_STREAK_THRESHOLDS.get(canonical, 1))
    base_key = canonical.lower()
    streak_key = f"_internal_heartbeat_{base_key}_streak"
    reported_key = f"_internal_heartbeat_{base_key}_reported"

    streak = int(metadata.get(streak_key, 0) or 0)
    reported = bool(metadata.get(reported_key, False))

    if condition_active:
        streak += 1
    else:
        streak = 0
        reported = False

    should_emit = bool(condition_active and not reported and streak >= threshold)
    if should_emit:
        reported = True

    metadata[streak_key] = streak
    metadata[reported_key] = reported
    return should_emit, streak


def canonicalize_proctoring_event_type(event_type: str) -> str:
    raw = (event_type or "").strip()
    if not raw:
        return ""
    alias = CANONICAL_EVENT_ALIASES.get(raw)
    if alias:
        return alias
    alias = CANONICAL_EVENT_ALIASES.get(raw.lower())
    if alias:
        return alias
    return raw.upper()


def default_proctoring_severity(event_type: str) -> str:
    canonical = canonicalize_proctoring_event_type(event_type)
    if canonical in SERIOUS_VIOLATION_EVENTS:
        return "violation"
    if canonical in LOW_VIOLATION_EVENTS:
        return "warning"
    return "info"


def is_violation_event(event_type: str) -> bool:
    canonical = canonicalize_proctoring_event_type(event_type)
    return canonical in LOW_VIOLATION_EVENTS or canonical in SERIOUS_VIOLATION_EVENTS


def is_serious_violation_event(event_type: str) -> bool:
    canonical = canonicalize_proctoring_event_type(event_type)
    return canonical in SERIOUS_VIOLATION_EVENTS


def default_proctoring_message(event_type: str) -> Optional[str]:
    canonical = canonicalize_proctoring_event_type(event_type)
    return EVENT_MESSAGES.get(canonical)


def heartbeat_is_stale(
    state: PracticeProctoringSession,
    *,
    reference_time: Optional[datetime] = None,
) -> bool:
    now = _as_aware_utc(reference_time) or _utcnow()
    baseline = _as_aware_utc(state.last_heartbeat_at) or _as_aware_utc(state.created_at)
    if not baseline:
        return False
    return (now - baseline).total_seconds() > PROCTORING_HEARTBEAT_STALE_SECONDS


def ensure_proctoring_session_state(
    db: Session,
    *,
    session_id: str,
    user_id: Optional[str] = None,
    reference_time: Optional[datetime] = None,
    monitoring_metadata: Optional[dict[str, Any]] = None,
) -> PracticeProctoringSession:
    now = reference_time or _utcnow()
    state = (
        db.query(PracticeProctoringSession)
        .filter(PracticeProctoringSession.session_id == session_id)
        .first()
    )
    if not state:
        state = PracticeProctoringSession(
            session_id=session_id,
            user_id=user_id,
            status="ACTIVE",
            risk_level="LOW",
            total_violations=0,
            serious_violations=0,
            monitoring_metadata=monitoring_metadata or {},
            created_at=now,
            updated_at=now,
        )
        db.add(state)
        db.flush()
        return state

    if user_id and (not state.user_id or state.user_id == "guest_unknown"):
        state.user_id = user_id
    if monitoring_metadata:
        merged = dict(state.monitoring_metadata or {})
        merged.update(monitoring_metadata)
        state.monitoring_metadata = merged
    state.updated_at = now
    db.add(state)
    db.flush()
    return state


def get_proctoring_session_state(db: Session, *, session_id: str) -> Optional[PracticeProctoringSession]:
    return (
        db.query(PracticeProctoringSession)
        .filter(PracticeProctoringSession.session_id == session_id)
        .first()
    )


def _termination_reason_for_state(state: PracticeProctoringSession) -> Optional[str]:
    if int(state.serious_violations or 0) >= PROCTORING_SERIOUS_VIOLATION_LIMIT:
        return f"Session terminated after {PROCTORING_SERIOUS_VIOLATION_LIMIT} serious proctoring violations."
    if int(state.total_violations or 0) >= PROCTORING_TOTAL_VIOLATION_LIMIT:
        return f"Session terminated after {PROCTORING_TOTAL_VIOLATION_LIMIT} total proctoring violations."
    return state.terminated_reason


def evaluate_proctoring_session_state(
    state: PracticeProctoringSession,
    *,
    reference_time: Optional[datetime] = None,
) -> tuple[PracticeProctoringSession, bool]:
    now = reference_time or _utcnow()
    total = int(state.total_violations or 0)
    serious = int(state.serious_violations or 0)
    stale = heartbeat_is_stale(state, reference_time=now)

    if state.status != "TERMINATED":
        termination_reason = _termination_reason_for_state(state)
        if termination_reason and (serious >= PROCTORING_SERIOUS_VIOLATION_LIMIT or total >= PROCTORING_TOTAL_VIOLATION_LIMIT):
            state.status = "TERMINATED"
            state.terminated_reason = termination_reason
        elif total > 0 or stale:
            state.status = "WARNING"
        else:
            state.status = "ACTIVE"

    if state.status == "TERMINATED" or serious >= 2 or total >= 4:
        state.risk_level = "HIGH"
    elif serious >= 1 or total >= 2 or stale:
        state.risk_level = "MEDIUM"
    else:
        state.risk_level = "LOW"

    state.updated_at = now
    return state, stale


def build_proctoring_status_payload(
    state: PracticeProctoringSession,
    *,
    reference_time: Optional[datetime] = None,
    action: str = "none",
    message: Optional[str] = None,
) -> dict[str, Any]:
    now = reference_time or _utcnow()
    state, stale = evaluate_proctoring_session_state(state, reference_time=now)
    return {
        "session_id": state.session_id,
        "status": state.status,
        "risk_level": state.risk_level,
        "total_violations": int(state.total_violations or 0),
        "serious_violations": int(state.serious_violations or 0),
        "remaining_total_before_termination": max(0, PROCTORING_TOTAL_VIOLATION_LIMIT - int(state.total_violations or 0)),
        "remaining_serious_before_termination": max(0, PROCTORING_SERIOUS_VIOLATION_LIMIT - int(state.serious_violations or 0)),
        "heartbeat_stale": stale,
        "last_heartbeat_at": _as_aware_utc(state.last_heartbeat_at),
        "terminated_reason": state.terminated_reason,
        "action": action,
        "message": message,
        "monitoring_metadata": _public_monitoring_metadata(state.monitoring_metadata),
    }


def _latest_event_for_type(
    db: Session,
    *,
    session_id: str,
    event_type: str,
) -> Optional[PracticeProctoringEvent]:
    return (
        db.query(PracticeProctoringEvent)
        .filter(PracticeProctoringEvent.session_id == session_id)
        .filter(PracticeProctoringEvent.event_type == event_type)
        .order_by(PracticeProctoringEvent.event_ts.desc())
        .first()
    )


def _is_suppressed_by_cooldown(
    db: Session,
    *,
    session_id: str,
    event_type: str,
    reference_time: datetime,
) -> bool:
    cooldown = int(EVENT_COOLDOWN_SECONDS.get(event_type, 0))
    if cooldown <= 0:
        return False
    latest = _latest_event_for_type(db, session_id=session_id, event_type=event_type)
    latest_event_ts = _as_aware_utc(getattr(latest, "event_ts", None)) if latest else None
    current_reference_time = _as_aware_utc(reference_time) or _utcnow()
    if not latest_event_ts:
        return False
    return (current_reference_time - latest_event_ts).total_seconds() < cooldown


def ingest_proctoring_signal(
    db: Session,
    *,
    session_id: str,
    user_id: Optional[str] = None,
    event_type: str,
    metadata: Optional[dict[str, Any]] = None,
    client_timestamp: Optional[datetime] = None,
    source: str = "api",
    severity_hint: Optional[str] = None,
    reference_time: Optional[datetime] = None,
) -> dict[str, Any]:
    now = reference_time or _utcnow()
    canonical = canonicalize_proctoring_event_type(event_type)
    state = ensure_proctoring_session_state(db, session_id=session_id, user_id=user_id, reference_time=now)

    payload = dict(metadata or {})
    payload.setdefault("source", source)
    if client_timestamp:
        payload.setdefault("client_timestamp", client_timestamp.isoformat())

    severity = severity_hint or default_proctoring_severity(canonical)
    payload.setdefault("severity", severity)

    counted = False
    suppressed = False
    action = "none"
    message = None

    if is_violation_event(canonical) and state.status != "TERMINATED":
        suppressed = _is_suppressed_by_cooldown(
            db,
            session_id=session_id,
            event_type=canonical,
            reference_time=now,
        )
        if not suppressed:
            counted = True
            state.total_violations = int(state.total_violations or 0) + 1
            if is_serious_violation_event(canonical):
                state.serious_violations = int(state.serious_violations or 0) + 1
            action = "warn"
            message = default_proctoring_message(canonical)

    payload["counted"] = counted
    if suppressed:
        payload["suppressed_by_cooldown"] = True

    row = PracticeProctoringEvent(
        session_id=session_id,
        event_type=canonical,
        event_ts=now,
        extra_data=payload,
    )
    db.add(row)

    state.last_event_at = now
    db.add(state)
    db.flush()

    snapshot = build_proctoring_status_payload(state, reference_time=now, action=action, message=message)
    if snapshot["status"] == "TERMINATED":
        snapshot["action"] = "terminate"
        snapshot["message"] = snapshot["terminated_reason"]
    return snapshot


def process_proctoring_heartbeat(
    db: Session,
    *,
    session_id: str,
    user_id: Optional[str] = None,
    camera_active: bool,
    screen_active: bool,
    tab_active: bool = True,
    window_focused: bool = True,
    detection_active: bool = True,
    display_surface: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
    client_timestamp: Optional[datetime] = None,
    reference_time: Optional[datetime] = None,
) -> dict[str, Any]:
    now = reference_time or _utcnow()
    monitoring_metadata = dict(metadata or {})
    monitoring_metadata.update(
        {
            "camera_active": bool(camera_active),
            "screen_active": bool(screen_active),
            "tab_active": bool(tab_active),
            "window_focused": bool(window_focused),
            "detection_active": bool(detection_active),
            "display_surface": display_surface,
            "last_client_timestamp": client_timestamp.isoformat() if client_timestamp else None,
        }
    )

    state = ensure_proctoring_session_state(
        db,
        session_id=session_id,
        user_id=user_id,
        reference_time=now,
        monitoring_metadata=monitoring_metadata,
    )
    state_monitoring_metadata = dict(state.monitoring_metadata or {})

    camera_stopped_emit, camera_stopped_streak = _should_emit_heartbeat_event(
        state_monitoring_metadata,
        event_type="CAMERA_STOPPED",
        condition_active=not camera_active,
    )
    screen_stopped_emit, screen_stopped_streak = _should_emit_heartbeat_event(
        state_monitoring_metadata,
        event_type="SCREEN_STOPPED",
        condition_active=not screen_active,
    )
    tab_switch_emit, tab_switch_streak = _should_emit_heartbeat_event(
        state_monitoring_metadata,
        event_type="TAB_SWITCH",
        condition_active=not tab_active,
    )
    window_blur_emit, window_blur_streak = _should_emit_heartbeat_event(
        state_monitoring_metadata,
        event_type="WINDOW_BLUR",
        condition_active=not window_focused,
    )
    monitoring_interrupted_emit, monitoring_interrupted_streak = _should_emit_heartbeat_event(
        state_monitoring_metadata,
        event_type="MONITORING_INTERRUPTED",
        condition_active=not detection_active,
    )
    display_surface_mismatch_emit, display_surface_mismatch_streak = _should_emit_heartbeat_event(
        state_monitoring_metadata,
        event_type="DISPLAY_SURFACE_MISMATCH",
        condition_active=bool(display_surface and str(display_surface).lower() != "monitor"),
    )

    state.monitoring_metadata = state_monitoring_metadata
    state.last_heartbeat_at = now
    state.updated_at = now
    db.add(state)
    db.flush()

    snapshots: list[dict[str, Any]] = []
    derived_metadata = {"heartbeat": True}
    if camera_stopped_emit:
        snapshots.append(
            ingest_proctoring_signal(
                db,
                session_id=session_id,
                user_id=user_id,
                event_type="CAMERA_STOPPED",
                metadata={**derived_metadata, "streak": camera_stopped_streak},
                client_timestamp=client_timestamp,
                source="heartbeat",
                reference_time=now,
            )
        )
    if screen_stopped_emit:
        snapshots.append(
            ingest_proctoring_signal(
                db,
                session_id=session_id,
                user_id=user_id,
                event_type="SCREEN_STOPPED",
                metadata={**derived_metadata, "streak": screen_stopped_streak},
                client_timestamp=client_timestamp,
                source="heartbeat",
                reference_time=now,
            )
        )
    if tab_switch_emit:
        snapshots.append(
            ingest_proctoring_signal(
                db,
                session_id=session_id,
                user_id=user_id,
                event_type="TAB_SWITCH",
                metadata={"heartbeat": True, "reason": "tab_inactive", "streak": tab_switch_streak},
                client_timestamp=client_timestamp,
                source="heartbeat",
                reference_time=now,
            )
        )
    if window_blur_emit:
        snapshots.append(
            ingest_proctoring_signal(
                db,
                session_id=session_id,
                user_id=user_id,
                event_type="WINDOW_BLUR",
                metadata={"heartbeat": True, "reason": "window_unfocused", "streak": window_blur_streak},
                client_timestamp=client_timestamp,
                source="heartbeat",
                reference_time=now,
            )
        )
    if monitoring_interrupted_emit:
        snapshots.append(
            ingest_proctoring_signal(
                db,
                session_id=session_id,
                user_id=user_id,
                event_type="MONITORING_INTERRUPTED",
                metadata={**derived_metadata, "streak": monitoring_interrupted_streak},
                client_timestamp=client_timestamp,
                source="heartbeat",
                reference_time=now,
            )
        )
    if display_surface_mismatch_emit:
        snapshots.append(
            ingest_proctoring_signal(
                db,
                session_id=session_id,
                user_id=user_id,
                event_type="DISPLAY_SURFACE_MISMATCH",
                metadata={"heartbeat": True, "display_surface": display_surface, "streak": display_surface_mismatch_streak},
                client_timestamp=client_timestamp,
                source="heartbeat",
                reference_time=now,
            )
        )

    if not snapshots:
        return build_proctoring_status_payload(state, reference_time=now)

    snapshots.sort(key=lambda item: {"none": 0, "warn": 1, "terminate": 2}.get(item.get("action", "none"), 0))
    return snapshots[-1]