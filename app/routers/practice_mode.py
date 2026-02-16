"""
Practice Mode API Router.
FastAPI endpoints for realistic interview practice mode.
"""

import logging
import re
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Header, Query, Request
from fastapi.responses import FileResponse
from pathlib import Path
from typing import Optional, Any
from datetime import datetime, timezone
import uuid
import aiofiles
import os

from app.schemas import (
    StartInterviewRequest,
    StartInterviewResponse,
    SubmitAnswerResponse,
    SubmitCodeRequest,
    SubmitCodeResponse,
    QuestionDifficulty,
    PracticeModeConfig,
    UserProfile,
    QuickStartRequest,
    ConversationalResponse,
    AcknowledgeFeedbackRequest,
    PracticeFeedbackRatedRequest,
    PracticeFeedbackRatedResponse,
    NextQuestionResponse,
    PracticeProgressSummaryResponse,
    PracticeHeatmapResponse,
    PracticeNextSessionRecommendationResponse,
    InterviewRound,
    RoundConfig,
    RoundSelectionRequest,
    RoundSelectionResponse,
    ProctoringEventIn,
    ProctoringEventOut,
    PracticeSessionStartIn,
    PracticeSessionStartOut,
    PracticeSessionMediaType,
    PracticeSessionMediaOut,
    PracticeSessionProctoringEventIn,
    PracticeSessionProctoringEventOut,
    CodeTestResult,
    CodeEvaluationFeedback,
    PracticeConfidenceOutcomeIn,
    PracticeConfidenceOutcomeOut,
    ResumeContext,
)
from app.services.practice.practice_mode_service import PracticeModeService
from app.services.practice.practice_mode_graph import PracticeModeGraph

from app.config import settings
from app.database import get_db_context
from app.middleware.auth import get_user_id_from_request
from app.utils.event_logging import track_event, stable_question_id, stable_hash
from app.services.practice.learning_loops import compute_practice_insights, merge_focus_areas, get_previously_asked_questions
from app.models import PracticeAttemptRecord, PracticeSessionMedia, PracticeProctoringEvent
from app.services.practice.practice_learning import upsert_practice_session_metrics
from app.services.practice.practice_learning import upsert_practice_session_outcome_confidence
from app.services.practice.practice_progress import (
    get_dimension_heatmap,
    get_latest_next_session_plan,
    get_progress_summary,
    save_completed_attempt,
)
from app.services.practice.practice_scoring import evaluation_report_to_json, score_session
from app.services.chat.ai_native_enhancements import CodeExecutionSandbox

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/practice",
    tags=["Practice Mode"]
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _media_root_dir() -> Path:
    # Keep recordings out of code directory; keep it local and simple for MVP.
    # In production, swap this for S3/HF storage.
    root = Path("data") / "practice_session_media"
    root.mkdir(parents=True, exist_ok=True)
    return root


_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_\-]+$")
_SAFE_FILENAME_RE = re.compile(r"^[A-Za-z0-9_.,\-]+$")
MAX_MEDIA_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB


def _safe_ext_from_upload(upload: UploadFile) -> str:
    name = (upload.filename or "").strip()
    _, ext = os.path.splitext(name)
    ext = (ext or "").lower()
    if ext and len(ext) <= 10:
        return ext
    # Fallback
    ctype = (upload.content_type or "").lower()
    if "webm" in ctype:
        return ".webm"
    if "mp4" in ctype:
        return ".mp4"
    if "quicktime" in ctype:
        return ".mov"
    return ".bin"


def _insert_practice_proctoring_event(
    *,
    session_id: str,
    event_type: str,
    metadata: Optional[dict[str, Any]] = None,
    event_ts: Optional[datetime] = None,
) -> None:
    """Insert a proctoring event row (best-effort, never raises)."""
    try:
        with get_db_context() as db:
            row = PracticeProctoringEvent(
                session_id=session_id,
                event_type=event_type,
                event_ts=event_ts or _utcnow(),
                extra_data=metadata or {},
            )
            db.add(row)
            db.commit()
    except Exception:
        pass


def _get_media_and_proctoring_summary(session_id: str) -> dict[str, Any]:
    """Aggregate media URLs + proctoring summary for final report."""
    media: dict[str, Optional[str]] = {
        "screen_recording_url": None,
        "camera_recording_url": None,
    }
    proctoring_summary: dict[str, Any] = {
        "violation_count": 0,
        "events": [],
    }

    violation_types = {
        "SCREEN_STOPPED",
        "CAMERA_STOPPED",
        "TAB_SWITCH",
        "WINDOW_MINIMIZED",
    }

    with get_db_context() as db:
        rows = (
            db.query(PracticeSessionMedia)
            .filter(PracticeSessionMedia.session_id == session_id)
            .order_by(PracticeSessionMedia.created_at.desc())
            .all()
        )
        # Prefer explicit screen/camera; fall back to combined for screen.
        screen_row = next((r for r in rows if r.media_type == "screen"), None)
        camera_row = next((r for r in rows if r.media_type == "camera"), None)
        combined_row = next((r for r in rows if r.media_type == "combined"), None)

        if screen_row:
            media["screen_recording_url"] = screen_row.storage_url
        elif combined_row:
            media["screen_recording_url"] = combined_row.storage_url

        if camera_row:
            media["camera_recording_url"] = camera_row.storage_url

        events = (
            db.query(PracticeProctoringEvent)
            .filter(PracticeProctoringEvent.session_id == session_id)
            .order_by(PracticeProctoringEvent.event_ts.asc())
            .all()
        )

    violation_events = [e.event_type for e in events if e.event_type in violation_types]
    proctoring_summary["violation_count"] = int(len(violation_events))
    proctoring_summary["events"] = sorted(set(violation_events))

    return {"media": media, "proctoring_summary": proctoring_summary}


@router.post("/session/{session_id}/outcome/confidence", response_model=PracticeConfidenceOutcomeOut)
async def submit_practice_confidence_outcome(
    session_id: str,
    payload: PracticeConfidenceOutcomeIn,
    http_request: Request,
) -> PracticeConfidenceOutcomeOut:
    """Submit a self-reported confidence outcome (1-5) for a completed session.

    Notes:
    - Gated behind ENABLE_PRACTICE_LEARNING.
    - Stores only the score (no raw audio/transcripts).
    """

    if not bool(getattr(settings, "enable_practice_learning", False)):
        raise HTTPException(status_code=404, detail="Not found")

    user_id = get_user_id_from_request(http_request) or "guest_unknown"

    # Ensure the session is complete (runtime session OR persisted attempt).
    runtime_ok = False
    if practice_service:
        sess = practice_service.get_session(session_id)
        runtime_ok = bool(sess and getattr(sess, "is_complete", False))

    with get_db_context() as db:
        attempt_ok = (
            db.query(PracticeAttemptRecord)
            .filter(PracticeAttemptRecord.session_id == session_id)
            .filter(PracticeAttemptRecord.user_id == user_id)
            .first()
            is not None
        )

        if not (runtime_ok or attempt_ok):
            raise HTTPException(status_code=400, detail="Session is not complete")

        upsert_practice_session_outcome_confidence(
            db,
            user_id=user_id,
            session_id=session_id,
            confidence_1_5=int(payload.confidence_1_5),
        )

    _track_practice_event(
        user_id=user_id,
        session_id=session_id,
        event_type="practice_outcome_confidence_submitted",
        question_text=None,
        extra={"confidence_1_5": int(payload.confidence_1_5)},
    )

    return PracticeConfidenceOutcomeOut(
        ok=True,
        session_id=session_id,
        confidence_1_5=int(payload.confidence_1_5),
    )


@router.post("/proctoring/event", response_model=ProctoringEventOut)
async def ingest_proctoring_event(payload: ProctoringEventIn, http_request: Request) -> ProctoringEventOut:
    """Ingest privacy-safe proctoring events for Practice Mode.

    Notes:
    - The backend cannot enable the camera; clients must use getUserMedia().
    - This endpoint stores event metadata only (no frames/audio).
    - Events are tied to an existing practice session_id.
    """

    user_id = get_user_id_from_request(http_request) or "guest_unknown"

    if not practice_service:
        raise HTTPException(status_code=503, detail="Practice Mode is not initialized")

    sess = practice_service.get_session(payload.session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Practice session not found")

    event_type = f"practice_proctoring_{payload.event_type.value}"
    _track_practice_event(
        user_id=user_id,
        session_id=payload.session_id,
        event_type=event_type,
        extra={
            "severity": payload.severity,
            "metadata": payload.metadata,
            "client_timestamp": payload.client_timestamp.isoformat() if payload.client_timestamp else None,
        },
    )

    return ProctoringEventOut(ok=True)


@router.post("/session/{session_id}/start", response_model=PracticeSessionStartOut)
async def start_session_with_proctoring_gate(
    session_id: str,
    payload: PracticeSessionStartIn,
    http_request: Request,
) -> PracticeSessionStartOut:
    """Enforce required proctoring permissions before a live practice session proceeds."""
    user_id = get_user_id_from_request(http_request) or "guest_unknown"

    if not practice_service:
        raise HTTPException(status_code=503, detail="Practice Mode is not initialized")

    sess = practice_service.get_session(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Practice session not found")

    if (not payload.screen_shared) or (not payload.camera_enabled):
        _insert_practice_proctoring_event(
            session_id=session_id,
            event_type="SESSION_STARTED_WITHOUT_PROCTORING",
            metadata={"screen_shared": bool(payload.screen_shared), "camera_enabled": bool(payload.camera_enabled)},
        )
        raise HTTPException(status_code=403, detail="Screen share + camera are required to start")

    _insert_practice_proctoring_event(
        session_id=session_id,
        event_type="SESSION_STARTED_WITH_PROCTORING",
        metadata={"screen_shared": True, "camera_enabled": True},
    )
    _track_practice_event(
        user_id=user_id,
        session_id=session_id,
        event_type="practice_session_started_with_proctoring",
        extra={"screen_shared": True, "camera_enabled": True},
    )

    return PracticeSessionStartOut(ok=True)


@router.post("/session/{session_id}/media", response_model=PracticeSessionMediaOut)
async def upload_practice_session_media(
    session_id: str,
    file: UploadFile = File(..., description="Recording file"),
    media_type: PracticeSessionMediaType = Form(...),
    duration_seconds: Optional[int] = Form(default=None),
):
    """Upload a practice session recording and store a DB row.

    This endpoint stores the file only; it does not process video.
    """
    if not practice_service:
        raise HTTPException(status_code=503, detail="Practice Mode is not initialized")

    # Validate session_id to prevent path traversal
    if not _SAFE_ID_RE.fullmatch(session_id):
        raise HTTPException(status_code=400, detail="Invalid session_id")

    sess = practice_service.get_session(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Practice session not found")

    ext = _safe_ext_from_upload(file)
    media_dir = _media_root_dir() / session_id
    media_dir.mkdir(parents=True, exist_ok=True)
    file_id = str(uuid.uuid4())
    temp_disk_path = media_dir / f"tmp_{file_id}{ext}"

    async with aiofiles.open(temp_disk_path, "wb") as f:
        content = await file.read()
        if len(content) > MAX_MEDIA_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"File too large (max {MAX_MEDIA_UPLOAD_BYTES // (1024*1024)}MB)")
        await f.write(content)

    duration_int: Optional[int] = int(duration_seconds) if duration_seconds is not None else None

    row_id: int
    storage_url: str

    with get_db_context() as db:
        row = PracticeSessionMedia(
            session_id=session_id,
            media_type=media_type.value,
            storage_url="",
            duration_seconds=duration_int,
        )
        db.add(row)
        db.flush()  # assign id
        row_id = int(row.id)
        storage_url = f"/api/practice/session/{session_id}/media/{row_id}"
        row.storage_url = storage_url
        db.commit()

    # Rename to include the DB id so the GET endpoint can locate the exact file.
    final_disk_path = media_dir / f"{row_id}_{file_id}{ext}"
    try:
        os.replace(str(temp_disk_path), str(final_disk_path))
    except Exception:
        # Best-effort: keep temp file; GET may still fail but DB row exists.
        pass

    return PracticeSessionMediaOut(
        media_id=row_id,
        session_id=session_id,
        media_type=media_type,
        storage_url=storage_url,
        duration_seconds=duration_int,
    )


@router.get("/session/{session_id}/media/{media_id}")
async def fetch_practice_session_media(session_id: str, media_id: int):
    """Serve a previously uploaded recording file."""
    if not _SAFE_ID_RE.fullmatch(session_id):
        raise HTTPException(status_code=400, detail="Invalid session_id")
    with get_db_context() as db:
        row = (
            db.query(PracticeSessionMedia)
            .filter(PracticeSessionMedia.id == int(media_id))
            .filter(PracticeSessionMedia.session_id == session_id)
            .first()
        )
    if not row:
        raise HTTPException(status_code=404, detail="Media not found")

    media_dir = _media_root_dir() / session_id
    if not media_dir.exists():
        raise HTTPException(status_code=404, detail="Media file not found")

    candidates = sorted(media_dir.glob(f"{int(media_id)}_*"))
    if not candidates:
        raise HTTPException(status_code=404, detail="Media file not found")

    return FileResponse(path=str(candidates[0]), media_type="application/octet-stream")


@router.post("/session/{session_id}/proctoring/event", response_model=PracticeSessionProctoringEventOut)
async def ingest_session_proctoring_event(
    session_id: str,
    payload: PracticeSessionProctoringEventIn,
    http_request: Request,
) -> PracticeSessionProctoringEventOut:
    """Insert a DB-backed proctoring event for audit + reporting."""
    user_id = get_user_id_from_request(http_request) or "guest_unknown"

    if not practice_service:
        raise HTTPException(status_code=503, detail="Practice Mode is not initialized")

    sess = practice_service.get_session(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Practice session not found")

    _insert_practice_proctoring_event(
        session_id=session_id,
        event_type=payload.event_type.value,
        metadata={
            "metadata": payload.metadata,
            "client_timestamp": payload.client_timestamp.isoformat() if payload.client_timestamp else None,
        },
    )
    _track_practice_event(
        user_id=user_id,
        session_id=session_id,
        event_type=f"practice_proctoring_{payload.event_type.value.lower()}",
        extra={"metadata": payload.metadata},
    )
    return PracticeSessionProctoringEventOut(ok=True)

# Service instance (will be initialized in main.py)
practice_service: Optional[PracticeModeService] = None
practice_graph: Optional[PracticeModeGraph] = None


def _copy_user_profile(profile: UserProfile, update: dict[str, Any]) -> UserProfile:
    """Compatibility helper for Pydantic v1/v2."""
    if hasattr(profile, "model_copy"):
        return profile.model_copy(update=update)  # type: ignore[attr-defined]
    return profile.copy(update=update)  # type: ignore[call-arg]


def _maybe_enrich_profile_focus(*, user_id: str, profile: Optional[UserProfile]) -> tuple[Optional[UserProfile], list[str]]:
    """Enrich UserProfile.interview_focus from recent practice events (best-effort)."""
    if not profile:
        return None, []

    try:
        with get_db_context() as db:
            insights = compute_practice_insights(db, user_id=user_id, domain=profile.domain)
        recommended: list[str] = list(insights.get("recommended_focus") or [])
    except Exception:
        recommended = []

    if not recommended:
        return profile, []

    existing = getattr(profile, "interview_focus", None)
    merged = merge_focus_areas(existing, recommended, max_items=5)
    if merged == (existing or []):
        return profile, recommended

    return _copy_user_profile(profile, {"interview_focus": merged}), recommended


def _safe_enum_value(v: Any) -> Any:
    try:
        return v.value  # type: ignore[attr-defined]
    except Exception:
        return v


def _track_practice_event(
    *,
    user_id: str,
    session_id: Optional[str],
    event_type: str,
    question_text: Optional[str] = None,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    """Best-effort structured telemetry for Practice Mode.

    Never raises; never blocks product flows.
    """
    if not getattr(settings, "enable_event_logging", True):
        return
    try:
        # Always persist question text for practice_question_served events
        # (LLM-generated text, not user PII) — needed for cross-session dedup.
        if question_text and event_type == "practice_question_served":
            extra = dict(extra or {})
            extra.setdefault("question_text", question_text)
        with get_db_context() as db:
            track_event(
                db,
                user_id=user_id,
                session_id=session_id,
                event_type=event_type,
                question_text=question_text,
                extra=extra or {},
            )
    except Exception:
        pass


def _persist_completed_practice_attempt(*, user_id: str, session_id: str) -> None:
    """Persist completed attempt to DB (best-effort, background-safe)."""

    try:
        if not practice_service:
            logger.warning(f"[persist] practice_service is None, cannot persist attempt for session {session_id}")
            return

        sess = practice_service.get_session(session_id)
        if not sess:
            logger.warning(f"[persist] Session {session_id} not found in memory, cannot persist")
            return
        if not getattr(sess, "is_complete", False):
            logger.warning(f"[persist] Session {session_id} is_complete=False, skipping persist")
            return
        if not sess.answers:
            logger.warning(f"[persist] Session {session_id} has no answers, skipping persist")
            return

        logger.info(f"[persist] Persisting attempt for user={user_id}, session={session_id}, "
                     f"questions={len(sess.questions or [])}, answers={len(sess.answers or [])}")

        with get_db_context() as db:
            existing = (
                db.query(PracticeAttemptRecord)
                .filter(PracticeAttemptRecord.session_id == session_id)
                .first()
            )
            if existing:
                logger.info(f"[persist] Attempt already exists for session {session_id}, skipping")
            else:
                attempt_id = save_completed_attempt(db, user_id=user_id, session=sess)
                logger.info(f"[persist] ✅ Saved attempt id={attempt_id} for session {session_id}")

            # Privacy-safe learning: store aggregate metrics only.
            if bool(getattr(settings, "enable_practice_learning", False)):
                upsert_practice_session_metrics(db, user_id=user_id, session_id=session_id, session=sess)
    except Exception as e:
        logger.error(f"[persist] ❌ Failed to persist attempt for session {session_id}: {e}", exc_info=True)
        return


def init_practice_mode(
    gemini_api_key: str, 
    gemini_model: str = "models/gemini-3-flash-preview",
    config: Optional[PracticeModeConfig] = None
):
    """
    Initialize the practice mode service.
    Call this from main.py on startup.
    
    Args:
        gemini_api_key: Gemini API key
        gemini_model: Gemini model name
        config: Optional custom configuration
    """
    global practice_service, practice_graph
    
    if config is None:
        config = PracticeModeConfig()
    
    practice_service = PracticeModeService(
        config=config,
        gemini_api_key=gemini_api_key,
        gemini_model=gemini_model
    )

    # Optional: LangGraph orchestration layer (soft dependency).
    practice_graph = None
    if bool(getattr(settings, "enable_practice_mode_langgraph", False)):
        try:
            g = PracticeModeGraph(practice_service)
            practice_graph = g if g.available else None
            if practice_graph:
                logger.info("Practice Mode LangGraph enabled")
            else:
                logger.info("Practice Mode LangGraph requested but unavailable; falling back")
        except Exception as e:
            practice_graph = None
            logger.warning(f"Practice Mode LangGraph init failed; falling back: {e}")
    
    logger.info("Practice Mode initialized")


def cleanup_practice_mode():
    """
    Cleanup practice mode resources on shutdown.
    Call this from main.py lifespan shutdown.
    """
    global practice_service, practice_graph
    
    try:
        if practice_service:
            # Cleanup TTS resources
            if hasattr(practice_service, 'tts_service'):
                practice_service.tts_service.cleanup()
            logger.info("Practice Mode cleaned up successfully")
        practice_graph = None
    except Exception as e:
        logger.error(f"Error cleaning up Practice Mode: {e}")


# ── Resume Upload for Practice Mode ─────────────────────────────────────

@router.post("/upload-resume")
async def upload_resume_for_practice(
    file: UploadFile = File(...),
    http_request: Request = None,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    x_gemini_key: Optional[str] = Header(None, alias="X-Gemini-Key"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
):
    """
    📄 Upload resume for resume-based interview practice.
    
    Accepts: .txt, .md, .pdf, .docx
    
    Pipeline: Parse → Structure → Return structured JSON (raw file is NOT stored).
    
    The returned `resume_context` can be passed in `RoundSelectionRequest.resume_context`
    or `QuickStartRequest.resume_context` to enable claim-based probing.
    
    Returns:
        Structured resume context with skills, projects, achievements, etc.
    """
    try:
        from app.services.core.resume_parser import parse_resume

        # Validate file type
        allowed = {".txt", ".md", ".pdf", ".docx"}
        ext = Path(file.filename).suffix.lower() if file.filename else ""
        if ext not in allowed:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(allowed))}"
            )

        # Read file content
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        # Size guard: 5MB max
        if len(content) > 5 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large (max 5MB)")

        # Resolve API key for LLM parsing
        groq_key = x_api_key
        gemini_key = x_gemini_key
        if not groq_key and authorization and authorization.startswith("Bearer "):
            bearer_value = authorization.split(" ", 1)[1].strip()
            if bearer_value.count(".") != 2:
                groq_key = bearer_value
        api_key = gemini_key if gemini_key else groq_key

        if not api_key:
            from app.config import settings
            if settings.require_user_api_key:
                raise HTTPException(
                    status_code=401,
                    detail="No active API key. Please add your Groq or Gemini API key to parse resume.",
                )
            if settings.llm_provider == "gemini":
                api_key = settings.gemini_api_key or settings.groq_api_key
            else:
                api_key = settings.groq_api_key or settings.gemini_api_key

        # Parse resume → structured JSON (raw bytes are discarded after this call)
        result = await parse_resume(content, file.filename, api_key=api_key)

        logger.info(
            f"✅ Resume parsed for practice: {len(result.skills)} skills, "
            f"{len(result.projects)} projects, {len(result.achievements)} achievements"
        )

        return {
            "status": "ok",
            "resume_context": result.to_dict(),
            "summary": {
                "skills_count": len(result.skills),
                "projects_count": len(result.projects),
                "achievements_count": len(result.achievements),
                "years_of_experience": result.years_of_experience,
                "primary_domain": result.primary_domain,
            },
            "message": "Resume parsed successfully. Pass the resume_context in your interview start request to enable claim-based probing.",
        }

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Resume upload failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Resume parsing failed: {str(e)}")


@router.get("/rounds/available", response_model=RoundSelectionResponse)
async def get_available_rounds(
    experience_years: Optional[int] = None,
    domain: Optional[str] = None
):
    """
    🎯 Get available interview rounds with recommendations.
    
    Returns all interview rounds (HR, Technical 1/2, System Design, etc.)
    with optional personalized recommendations based on user profile.
    
    Args:
        experience_years: User's years of experience (for recommendations)
        domain: User's domain (for relevant rounds)
        
    Returns:
        - List of all available rounds with configurations
        - Recommended starting round (if experience provided)
        - Suggested round sequence (if experience provided)
    """
    try:
        if not practice_service:
            raise HTTPException(status_code=503, detail="Practice mode not initialized")
        
        from app.services.practice.round_config_service import RoundConfigService
        
        all_rounds = RoundConfigService.get_all_rounds()
        
        # Get recommendations if profile data provided
        recommended_round = None
        recommended_sequence = None
        
        if experience_years is not None:
            recommended_round = RoundConfigService.get_recommended_round(experience_years)
            recommended_sequence = RoundConfigService.get_recommended_sequence(experience_years)
        
        # Filter by domain if provided
        if domain:
            domain_rounds_enum = RoundConfigService.get_rounds_for_domain(domain)
            # Filter to only show relevant rounds
            relevant_round_types = set(domain_rounds_enum)
            all_rounds = [r for r in all_rounds if r.round_type in relevant_round_types]
        
        return RoundSelectionResponse(
            rounds=all_rounds,
            recommended_round=recommended_round,
            recommended_sequence=recommended_sequence
        )
        
    except Exception as e:
        logger.error(f"Error getting available rounds: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/insights")
async def get_practice_insights(
    http_request: Request,
    domain: Optional[str] = Query(None, description="Optional domain filter (matches session start domain)"),
    lookback_days: int = Query(30, ge=1, le=365),
):
    """Get lightweight, explainable practice insights for the current user.

    This endpoint is intentionally simple: it aggregates recent practice events
    into a recommended focus list and a few summary stats.
    """
    user_id = get_user_id_from_request(http_request) or "guest_unknown"
    with get_db_context() as db:
        return compute_practice_insights(
            db,
            user_id=user_id,
            domain=domain,
            lookback_days=lookback_days,
        )


@router.get("/progress/summary", response_model=PracticeProgressSummaryResponse)
async def get_progress_summary_endpoint(
    http_request: Request,
    domain: Optional[str] = Query(None, description="Optional domain filter"),
    lookback_days: int = Query(30, ge=1, le=365),
):
    """Flagship loop: a tiny, fast summary suitable for a dashboard card."""

    user_id = get_user_id_from_request(http_request) or "guest_unknown"
    with get_db_context() as db:
        s = get_progress_summary(db, user_id=user_id, lookback_days=lookback_days, domain=domain)
        return PracticeProgressSummaryResponse(
            attempts=s.attempts,
            average_overall_score=s.average_overall_score,
            last_completed_at=s.last_completed_at,
            best_dimension=s.best_dimension,
            worst_dimension=s.worst_dimension,
        )


@router.get("/progress/heatmap", response_model=PracticeHeatmapResponse)
async def get_progress_heatmap_endpoint(
    http_request: Request,
    domain: Optional[str] = Query(None, description="Optional domain filter"),
    lookback_days: int = Query(90, ge=1, le=365),
):
    """Flagship loop: weekly x dimension trend points."""

    user_id = get_user_id_from_request(http_request) or "guest_unknown"
    with get_db_context() as db:
        points = get_dimension_heatmap(db, user_id=user_id, lookback_days=lookback_days, domain=domain)
        return PracticeHeatmapResponse(points=points)


@router.get("/progress/next-session", response_model=PracticeNextSessionRecommendationResponse)
async def get_next_session_plan_endpoint(
    http_request: Request,
    domain: Optional[str] = Query(None, description="Optional domain filter"),
):
    """Flagship loop: latest recommended settings for the next targeted session."""

    user_id = get_user_id_from_request(http_request) or "guest_unknown"
    with get_db_context() as db:
        plan = get_latest_next_session_plan(db, user_id=user_id, domain=domain)
        return PracticeNextSessionRecommendationResponse(plan=plan)


@router.post("/interview/start-round", response_model=StartInterviewResponse)
async def start_round_based_interview(
    payload: RoundSelectionRequest,
    http_request: Request,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    x_gemini_key: Optional[str] = Header(None, alias="X-Gemini-Key"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
):
    """
    🎯 Start a round-based interview (NEW FEATURE!)
    
    **What's different:**
    - Focus on ONE specific interview round (e.g., "System Design Round")
    - Questions tailored to that round type
    - Realistic round durations and question counts
    - Company-specific adaptations (optional)
    
    **Example rounds:**
    - HR Screening: 4 questions, 20 minutes
    - Technical Round 1: 6 questions, 45 minutes
    - System Design: 2-3 questions, 60 minutes
    - Behavioral: 5 questions, 40 minutes
    
    Args:
        round_type: The specific round to practice (REQUIRED)
        domain: Primary domain like "Python", "Data Engineering" (REQUIRED)
        experience_years: Years of experience 0-30 (default: 2)
        company_specific: Target company (optional)
        
    Returns:
        Session with questions specific to the chosen round
    """
    try:
        logger.info(f"📥 Received request: round_type={payload.round_type}, domain={payload.domain}, exp={payload.experience_years}")

        if (not payload.screen_shared) or (not payload.camera_enabled):
            raise HTTPException(status_code=403, detail="Screen share + camera are required to start")

        user_id = get_user_id_from_request(http_request) or "guest_unknown"
        
        if not practice_service:
            raise HTTPException(status_code=503, detail="Practice mode not initialized")
        
        from app.services.practice.round_config_service import RoundConfigService
        
        # Get configuration for the selected round
        round_config = RoundConfigService.get_round_config(payload.round_type)
        
        # Use user-provided question count if available, otherwise use round default
        final_question_count = payload.question_count or round_config.question_count
        logger.info(f"🎯 Starting {round_config.name} with {final_question_count} questions")
        
        # Build user profile from domain/experience or use provided profile
        if payload.user_profile:
            profile = payload.user_profile
        else:
            # Create profile from domain and experience
            # Infer basic skills from domain
            basic_skills = [payload.domain] if payload.domain else []
            
            profile = UserProfile(
                domain=payload.domain,
                experience_years=payload.experience_years,
                skills=basic_skills,  # Required field
                company_preference=payload.company_specific
            )
        
        # Wire resume context from request into profile (if provided)
        if payload.resume_context and not profile.resume_context:
            profile.resume_context = payload.resume_context
            logger.info(f"📄 Resume context attached: {len(payload.resume_context.skills)} skills, {len(payload.resume_context.projects)} projects")
        
        # Override company preference if specified
        if payload.company_specific:
            profile.company_preference = payload.company_specific
        
        # Calculate difficulty based on experience (NOT round's default difficulty)
        experience_based_difficulty = RoundConfigService.get_difficulty_for_experience(
            profile.experience_years
        )

        # Learning loop: enrich focus areas from prior attempts (best-effort)
        profile, recommended_focus = _maybe_enrich_profile_focus(user_id=user_id, profile=profile)
        
        logger.info(f"📋 Profile: {profile.domain} with {profile.experience_years} years | Difficulty: {experience_based_difficulty.value.upper()} | Company: {profile.company_preference or 'Generic'}")
        
        # API Key selection (Bridge Settings)
        groq_key = x_api_key
        gemini_key = x_gemini_key
        if not groq_key and authorization and authorization.startswith("Bearer "):
            groq_key = authorization.split(" ")[1]
            
        api_key = gemini_key if gemini_key else groq_key
        
        # Fallback to dev keys in local mode if headers are empty
        if not api_key:
            from app.config import settings
            if settings.require_user_api_key:
                raise HTTPException(
                    status_code=401,
                    detail="No active API key. Please add your Groq or Gemini API key in Bridge Settings to continue.",
                )
            if settings.llm_provider == "gemini":
                api_key = settings.gemini_api_key or settings.groq_api_key
            else:
                api_key = settings.groq_api_key or settings.gemini_api_key
 
        # Cross-session dedup: fetch previously asked questions for this user+domain
        previously_asked: list[str] = []
        try:
            with get_db_context() as db:
                previously_asked = get_previously_asked_questions(
                    db, user_id=user_id, domain=payload.domain
                )
            if previously_asked:
                logger.info(f"📋 Cross-session dedup: {len(previously_asked)} previously asked questions for {payload.domain}")
        except Exception:
            pass

        # Start interview with EXPERIENCE-BASED difficulty (not round's default)
        session_id, first_question, audio_filename = await practice_service.start_interview(
            difficulty=experience_based_difficulty,  # ✅ Use experience-based difficulty
            user_profile=profile,
            question_count=final_question_count,  # ✅ Now dynamic
            round_type=payload.round_type,
            api_key=api_key,
            previously_asked=previously_asked or None,
        )
        
        # Get session to retrieve total questions
        session = practice_service.get_session(session_id)
        total_questions = len(session.questions) if session else round_config.question_count

        # Stamp user_id for cleanup-time persist
        if session:
            session.user_id = user_id
        
        tts_audio_url = f"/api/practice/audio/{audio_filename}" if audio_filename else ""

        _insert_practice_proctoring_event(
            session_id=session_id,
            event_type="SESSION_STARTED_WITH_PROCTORING",
            metadata={"screen_shared": True, "camera_enabled": True, "flow": "round_based"},
        )

        _track_practice_event(
            user_id=user_id,
            session_id=session_id,
            event_type="practice_session_started",
            question_text=getattr(first_question, "text", None),
            extra={
                "flow": "round_based",
                "round_type": _safe_enum_value(payload.round_type),
                "domain": payload.domain,
                "experience_years": payload.experience_years,
                "difficulty": _safe_enum_value(experience_based_difficulty),
                "question_count": int(final_question_count),
                "company": profile.company_preference,
                "recommended_focus": recommended_focus,
            },
        )
        _track_practice_event(
            user_id=user_id,
            session_id=session_id,
            event_type="practice_question_served",
            question_text=getattr(first_question, "text", None),
            extra={
                "question_num": int(getattr(first_question, "id", 1) or 1),
                "question_id": int(getattr(first_question, "id", 1) or 1),
                "question_hash": stable_question_id(getattr(first_question, "text", "") or ""),
                "difficulty": _safe_enum_value(getattr(first_question, "difficulty", None)),
                "category": getattr(first_question, "category", None),
                "round_type": _safe_enum_value(getattr(first_question, "round_type", None)),
                "tts": bool(audio_filename),
            },
        )
        
        logger.info(f"🎤 TTS Audio: {'✅ ' + tts_audio_url if tts_audio_url else '❌ No audio generated'}")
        
        return StartInterviewResponse(
            session_id=session_id,
            first_question=first_question,
            tts_audio_url=tts_audio_url,
            total_questions=total_questions,
            progress=f"1/{total_questions}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting round-based interview: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/difficulty-preview")
async def get_difficulty_preview(experience_years: int):
    """
    Get difficulty level preview based on years of experience.
    
    This helps the frontend display difficulty badges on round cards:
    - 0-2 years → EASY
    - 3-6 years → MEDIUM
    - 7+ years → HARD
    
    Args:
        experience_years: Years of experience (0-30)
        
    Returns:
        {"difficulty": "easy|medium|hard", "label": "EASY|MEDIUM|HARD"}
    """
    try:
        from app.services.practice.round_config_service import RoundConfigService
        
        difficulty = RoundConfigService.get_difficulty_for_experience(experience_years)
        
        return {
            "difficulty": difficulty.value,
            "label": difficulty.value.upper(),
            "experience_years": experience_years,
            "description": f"Based on {experience_years} years of experience"
        }
    except Exception as e:
        logger.error(f"Error getting difficulty preview: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/interview/quick-start", response_model=ConversationalResponse)
async def quick_start_interview(
    request: QuickStartRequest,
    http_request: Request,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    x_gemini_key: Optional[str] = Header(None, alias="X-Gemini-Key"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
):
    """
    🚀 AI QUICK START - Zero-click conversational interview setup.
    """
    try:
        if not practice_service:
            raise HTTPException(status_code=503, detail="Practice mode not initialized")
        
        user_id = get_user_id_from_request(http_request) or "guest_unknown"
        logger.info("🚀 Quick Start AI Mode initiated")
        
        # API Key selection (Bridge Settings)
        groq_key = x_api_key
        gemini_key = x_gemini_key
        # Only treat Authorization as an API key if it does NOT look like a JWT.
        if not groq_key and authorization and authorization.startswith("Bearer "):
            bearer_value = authorization.split(" ", 1)[1].strip()
            if bearer_value.count(".") != 2:  # Not a JWT
                groq_key = bearer_value
            
        api_key = gemini_key if gemini_key else groq_key
        
        # Authenticated users (logged in via JWT) can use server keys even when
        # REQUIRE_USER_API_KEY is true — the requirement is about *demo/guest* traffic.
        is_authenticated = getattr(http_request.state, "user", None) is not None
        
        logger.info(
            f"Quick-start auth debug: api_key={'yes' if api_key else 'NO'}, "
            f"is_authenticated={is_authenticated}, "
            f"has_auth_header={'yes' if authorization else 'NO'}, "
            f"has_x_api_key={'yes' if x_api_key else 'NO'}, "
            f"has_gemini_key={'yes' if x_gemini_key else 'NO'}"
        )
        
        # Fallback to dev keys
        if not api_key:
            from app.config import settings
            if settings.require_user_api_key and not is_authenticated:
                logger.warning(
                    "Quick-start 401: REQUIRE_USER_API_KEY=true, user not authenticated, "
                    "no API key provided. Client must either login (JWT) or supply an API key."
                )
                raise HTTPException(
                    status_code=401,
                    detail="No active API key. Please log in or add your Groq/Gemini API key in Bridge Settings.",
                )
            if settings.llm_provider == "gemini":
                api_key = settings.gemini_api_key or settings.groq_api_key
            else:
                api_key = settings.groq_api_key or settings.gemini_api_key

        # Use AI to build profile from conversational input
        result = await practice_service.quick_start_conversational(
            voice_input=request.voice_input,
            context=request.context,
            auto_mode=request.auto_mode,
            use_memory=request.session_memory,
            question_count=request.question_count,  # User can override AI decision
            target_company=request.target_company,   # User can specify exact company
            api_key=api_key,
            user_id=user_id,
        )
        
        # Track events for quick-start sessions (other entry points track in their handlers)
        if result.session_id and result.first_question:
            # Stamp user_id for cleanup-time persist
            qs_session = practice_service.get_session(result.session_id) if practice_service else None
            if qs_session:
                qs_session.user_id = user_id

            profile = result.suggested_profile
            _track_practice_event(
                user_id=user_id,
                session_id=result.session_id,
                event_type="practice_session_started",
                question_text=getattr(result.first_question, "text", None),
                extra={
                    "flow": "quick_start",
                    "domain": getattr(profile, "domain", None) if profile else None,
                    "experience_years": getattr(profile, "experience_years", None) if profile else None,
                    "company": getattr(profile, "company_preference", None) if profile else None,
                    "question_count": result.total_questions,
                },
            )
            _track_practice_event(
                user_id=user_id,
                session_id=result.session_id,
                event_type="practice_question_served",
                question_text=getattr(result.first_question, "text", None),
                extra={
                    "question_num": 1,
                    "question_id": int(getattr(result.first_question, "id", 1) or 1),
                    "question_hash": stable_question_id(getattr(result.first_question, "text", "") or ""),
                    "difficulty": _safe_enum_value(getattr(result.first_question, "difficulty", None)),
                    "category": getattr(result.first_question, "category", None),
                    "tts": bool(result.tts_audio_url),
                },
            )

        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in Quick Start: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/interview/start", response_model=StartInterviewResponse)
async def start_interview(
    payload: StartInterviewRequest,
    http_request: Request,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    x_gemini_key: Optional[str] = Header(None, alias="X-Gemini-Key"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
):
    """
    Start a new practice interview session.
    """
    try:
        if not practice_service:
            raise HTTPException(status_code=503, detail="Practice mode not initialized")

        if (not payload.screen_shared) or (not payload.camera_enabled):
            raise HTTPException(status_code=403, detail="Screen share + camera are required to start")
        
        logger.info(f"Starting interview with difficulty: {payload.difficulty}")

        user_id = get_user_id_from_request(http_request) or "guest_unknown"
        
        # API Key selection (Bridge Settings)
        groq_key = x_api_key
        gemini_key = x_gemini_key
        if not groq_key and authorization and authorization.startswith("Bearer "):
            groq_key = authorization.split(" ")[1]
            
        api_key = gemini_key if gemini_key else groq_key
        
        # Fallback to dev keys respecting provider preference
        if not api_key:
            from app.config import settings
            if settings.require_user_api_key:
                raise HTTPException(
                    status_code=401,
                    detail="No active API key. Please add your Groq or Gemini API key in Bridge Settings to continue.",
                )
            if settings.llm_provider == "gemini":
                api_key = settings.gemini_api_key or settings.groq_api_key
            else:
                api_key = settings.groq_api_key or settings.gemini_api_key

        # Learning loop: enrich focus areas from prior attempts (best-effort)
        enriched_profile, recommended_focus = _maybe_enrich_profile_focus(
            user_id=user_id,
            profile=payload.user_profile,
        )

        # Cross-session dedup: fetch previously asked questions for this user+domain
        previously_asked: list[str] = []
        try:
            domain = getattr(enriched_profile, "domain", None) if enriched_profile else None
            if domain:
                with get_db_context() as db:
                    previously_asked = get_previously_asked_questions(
                        db, user_id=user_id, domain=domain
                    )
                if previously_asked:
                    logger.info(f"📋 Cross-session dedup: {len(previously_asked)} previously asked questions for {domain}")
        except Exception:
            pass

        # Start interview with user profile
        runner = practice_graph if (practice_graph and practice_graph.available) else None
        if runner:
            session_id, first_question, audio_filename = await runner.start_interview(
                difficulty=payload.difficulty,
                user_profile=enriched_profile,
                question_count=payload.question_count,
                api_key=api_key,
            )
        else:
            session_id, first_question, audio_filename = await practice_service.start_interview(
            difficulty=payload.difficulty,
            user_profile=enriched_profile,
            question_count=payload.question_count,
            api_key=api_key,
            previously_asked=previously_asked or None,
            )
        
        # Get session to retrieve total questions count
        session = practice_service.get_session(session_id)
        total_questions = len(session.questions) if session else payload.question_count

        # Stamp user_id for cleanup-time persist
        if session:
            session.user_id = user_id
        
        # Build audio URL
        tts_audio_url = f"/api/practice/audio/{audio_filename}" if audio_filename else ""

        _insert_practice_proctoring_event(
            session_id=session_id,
            event_type="SESSION_STARTED_WITH_PROCTORING",
            metadata={"screen_shared": True, "camera_enabled": True, "flow": "standard"},
        )

        _track_practice_event(
            user_id=user_id,
            session_id=session_id,
            event_type="practice_session_started",
            question_text=getattr(first_question, "text", None),
            extra={
                "flow": "standard",
                "difficulty": _safe_enum_value(payload.difficulty),
                "question_count": int(payload.question_count),
                "round_type": _safe_enum_value(payload.round_type),
                "category": payload.category,
                "company": getattr(enriched_profile, "company_preference", None) if enriched_profile else None,
                "recommended_focus": recommended_focus,
            },
        )
        _track_practice_event(
            user_id=user_id,
            session_id=session_id,
            event_type="practice_question_served",
            question_text=getattr(first_question, "text", None),
            extra={
                "question_num": int(getattr(first_question, "id", 1) or 1),
                "question_id": int(getattr(first_question, "id", 1) or 1),
                "question_hash": stable_question_id(getattr(first_question, "text", "") or ""),
                "difficulty": _safe_enum_value(getattr(first_question, "difficulty", None)),
                "category": getattr(first_question, "category", None),
                "round_type": _safe_enum_value(getattr(first_question, "round_type", None)),
                "tts": bool(audio_filename),
            },
        )
        
        return StartInterviewResponse(
            session_id=session_id,
            first_question=first_question,
            tts_audio_url=tts_audio_url,
            total_questions=total_questions,
            progress=f"1/{total_questions}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting interview: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/interview/submit-answer", response_model=SubmitAnswerResponse)
async def submit_answer(
    background_tasks: BackgroundTasks,
    http_request: Request,
    audio: UploadFile = File(..., description="Audio file of the answer"),
    session_id: str = Form(..., description="Session ID"),
    question_id: int = Form(..., ge=1, le=15, description="Question ID (1-15)"),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    x_gemini_key: Optional[str] = Header(None, alias="X-Gemini-Key"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
):
    """
    Submit an answer to a question.
    """
    try:
        if not practice_service:
            raise HTTPException(status_code=503, detail="Practice mode not initialized")
        
        logger.info(f"Receiving answer for session {session_id}, question {question_id}")

        user_id = get_user_id_from_request(http_request) or "guest_unknown"
        
        # API Key selection (Bridge Settings)
        groq_key = x_api_key
        gemini_key = x_gemini_key
        if not groq_key and authorization and authorization.startswith("Bearer "):
            groq_key = authorization.split(" ")[1]
            
        api_key = gemini_key if gemini_key else groq_key
        
        # Fallback to dev keys respecting provider preference
        if not api_key:
            from app.config import settings
            if settings.require_user_api_key:
                raise HTTPException(
                    status_code=401,
                    detail="No active API key. Please add your Groq or Gemini API key in Bridge Settings to continue.",
                )
            if settings.llm_provider == "gemini":
                api_key = settings.gemini_api_key or settings.groq_api_key
            else:
                api_key = settings.groq_api_key or settings.gemini_api_key

        # Validate audio file
        if not audio.content_type or not audio.content_type.startswith("audio/"):
            raise HTTPException(status_code=400, detail="Invalid audio file")
        
        # Save uploaded audio temporarily
        temp_audio_path = practice_service.audio_dir / f"temp_{session_id}_q{question_id}.wav"
        
        async with aiofiles.open(temp_audio_path, 'wb') as f:
            content = await audio.read()
            await f.write(content)

        _track_practice_event(
            user_id=user_id,
            session_id=session_id,
            event_type="practice_answer_audio_received",
            question_text=None,
            extra={
                "question_id": int(question_id),
                "audio_bytes": len(content or b""),
                "content_type": audio.content_type,
            },
        )

        # Process answer
        runner = practice_graph if (practice_graph and practice_graph.available) else None
        if runner:
            result = await runner.submit_answer(
                session_id=session_id,
                question_id=question_id,
                audio_file_path=str(temp_audio_path),
                api_key=api_key,
            )
        else:
            result = await practice_service.submit_answer(
                session_id=session_id,
                question_id=question_id,
                audio_file_path=str(temp_audio_path),
                api_key=api_key
            )
        
        # Clean up temp file
        background_tasks.add_task(temp_audio_path.unlink, missing_ok=True)
        
        # Build response
        response = SubmitAnswerResponse(
            transcript=result["transcript"],
            metrics=result["metrics"],
            micro_feedback=result["micro_feedback"],
            complete=result["complete"],
            progress=result["progress"],
            requires_acknowledgment=result.get("requires_acknowledgment", True),
            current_question_id=result.get("current_question_id", question_id)
        )

        # Premium: deterministic trace + trajectory + pressure (best-effort)
        try:
            from app.services.practice.practice_scoring import build_evaluation_trace, compute_session_trajectory
            sess = practice_service.get_session(session_id) if practice_service else None
            if sess is not None:
                response.evaluation_trace = build_evaluation_trace(session=sess)
                response.trajectory = compute_session_trajectory(session=sess)
                try:
                    from app.services.practice.adaptive_pressure import compute_pressure_state
                    response.pressure = compute_pressure_state(session=sess)
                except Exception:
                    pass
        except Exception:
            pass

        # Telemetry: processed answer summary (avoid storing transcript by default)
        metrics_obj = result.get("metrics")
        fb_obj = result.get("micro_feedback")
        _track_practice_event(
            user_id=user_id,
            session_id=session_id,
            event_type="practice_answer_processed",
            question_text=None,
            extra={
                "question_id": int(question_id),
                "transcript_len": len((result.get("transcript") or "")),
                "wpm": getattr(metrics_obj, "wpm", None),
                "filler_count": getattr(metrics_obj, "filler_count", None),
                "confidence_score": getattr(metrics_obj, "confidence_score", None),
                "correctness_score": getattr(fb_obj, "correctness_score", None),
                "technical_accuracy": getattr(fb_obj, "technical_accuracy", None),
                "is_correct": getattr(fb_obj, "is_correct", None),
                "complete": bool(result.get("complete")),
                "requires_acknowledgment": bool(result.get("requires_acknowledgment", True)),
            },
        )
        
        # Don't add next_question - user must acknowledge feedback first
        # Next question will be fetched via /acknowledge-feedback endpoint
        
        # Add evaluation report if complete
        if "evaluation_report" in result:
            response.evaluation_report = result["evaluation_report"]

        # Flagship loop: persist completed attempt for progress/trends (best-effort)
        # Persist on completion regardless of whether evaluation_report succeeded
        if bool(result.get("complete")):
            background_tasks.add_task(
                _persist_completed_practice_attempt,
                user_id=user_id,
                session_id=session_id,
            )
        
        logger.info(f"Answer processed successfully for session {session_id}")
        return response
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error submitting answer: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/interview/submit_code", response_model=SubmitCodeResponse)
@router.post("/interview/submit-code", response_model=SubmitCodeResponse)
async def submit_code(
    payload: SubmitCodeRequest,
    background_tasks: BackgroundTasks,
    http_request: Request,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    x_gemini_key: Optional[str] = Header(None, alias="X-Gemini-Key"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
):
    """Submit a coding answer for a Practice Mode question.

    This endpoint exists primarily to support a coding-question UI.
    For safety and determinism:
    - We do NOT execute user code by default.
    - If code execution is explicitly configured (Judge0 API key), we can run it in a sandbox.
    """
    try:
        if not practice_service:
            raise HTTPException(status_code=503, detail="Practice mode not initialized")

        user_id = get_user_id_from_request(http_request) or "guest_unknown"

        # API Key selection (Bridge Settings)
        groq_key = x_api_key
        gemini_key = x_gemini_key
        if not groq_key and authorization and authorization.startswith("Bearer "):
            groq_key = authorization[7:]
        api_key = gemini_key if gemini_key else groq_key

        session = practice_service.get_session(payload.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Practice session not found")

        # Best-effort: locate the question to determine test cases (if any)
        question = None
        try:
            question = session.questions[int(payload.question_id) - 1]
        except Exception:
            question = None

        test_cases = getattr(question, "test_cases", None) if question else None

        # Only run sandbox execution when explicitly configured.
        should_execute = bool(getattr(settings, "enable_code_execution", False)) and bool(getattr(settings, "judge0_api_key", None))

        execution_success = False
        execution_output: str = ""
        execution_error: str = ""
        if should_execute:
            try:
                sandbox = CodeExecutionSandbox(judge0_api_key=getattr(settings, "judge0_api_key", None))
                result = await sandbox.execute_code(
                    code=payload.code,
                    language=(payload.programming_language or "python").lower(),
                    test_cases=test_cases,
                )
                execution_success = bool(result.get("success"))
                execution_output = str(result.get("output") or "")
                execution_error = str(result.get("error") or "")
            except Exception as e:
                execution_success = False
                execution_error = str(e)

        # Map test cases into API response format (best-effort; may be empty)
        results: list[CodeTestResult] = []
        if isinstance(test_cases, list) and test_cases:
            for i, tc in enumerate(test_cases, start=1):
                tc_in = str((tc or {}).get("input", ""))
                tc_expected = str((tc or {}).get("expected_output", (tc or {}).get("expected", "")))
                results.append(
                    CodeTestResult(
                        test_case_id=i,
                        input_data=tc_in,
                        expected_output=tc_expected,
                        actual_output=(execution_output.strip() or None) if should_execute else None,
                        passed=bool(execution_success) if should_execute else False,
                        error=(execution_error.strip() or None) if execution_error else None,
                    )
                )

        # Minimal, deterministic scoring (avoid LLM in this path)
        if should_execute:
            correctness = 90 if execution_success else 0
            approach_quality = "good" if execution_success else "needs_improvement"
            improvements = [] if execution_success else ["Fix runtime/compile errors", "Re-check edge cases"]
            best_practices = ["Add input validation", "Write small helper functions"] if execution_success else ["Start with a minimal working version"]
        else:
            correctness = 0
            approach_quality = "needs_improvement"
            improvements = ["Code execution not configured on server (set JUDGE0_API_KEY to enable sandbox execution)"]
            best_practices = ["Add tests", "Handle edge cases"]

        code_feedback = CodeEvaluationFeedback(
            correctness_score=int(correctness),
            approach_quality=str(approach_quality),
            time_complexity=None,
            space_complexity=None,
            strengths=["Submitted a complete solution"] if payload.code.strip() else [],
            improvements=improvements,
            best_practices=best_practices,
            alternative_approaches=None,
        )

        svc_resp = await practice_service.submit_code(
            session_id=payload.session_id,
            question_id=int(payload.question_id),
            code=payload.code,
            programming_language=payload.programming_language,
            time_taken=int(payload.time_taken),
            correctness_score=int(correctness),
            summary=(execution_error.strip() if execution_error else "Coding submission received."),
            api_key=api_key,
        )

        _track_practice_event(
            user_id=user_id,
            session_id=payload.session_id,
            event_type="practice_code_submitted",
            question_text=getattr(question, "text", None) if question else None,
            extra={
                "question_id": int(payload.question_id),
                "language": payload.programming_language,
                "code_len": len(payload.code or ""),
                "time_taken": int(payload.time_taken),
                "executed": bool(should_execute),
                "execution_success": bool(execution_success) if should_execute else None,
            },
        )

        # Flagship loop: persist completed attempt for progress/trends (best-effort)
        if bool(svc_resp.get("complete")):
            background_tasks.add_task(
                _persist_completed_practice_attempt,
                user_id=user_id,
                session_id=payload.session_id,
            )

        return SubmitCodeResponse(
            test_results=results,
            all_tests_passed=bool(execution_success) if should_execute else False,
            code_feedback=code_feedback,
            complete=bool(svc_resp.get("complete")),
            next_question=None,
            evaluation_report=svc_resp.get("evaluation_report"),
            progress=str(svc_resp.get("progress")),
            requires_acknowledgment=bool(svc_resp.get("requires_acknowledgment", True)),
        )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error submitting code: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/interview/end-session")
async def end_practice_session(
    session_id: str = Form(..., description="Session ID to end"),
    background_tasks: BackgroundTasks = None,
    http_request: Request = None,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    x_gemini_key: Optional[str] = Header(None, alias="X-Gemini-Key"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
):
    """
    🛑 End a practice interview session early.

    Users can call this at any point to gracefully stop and get results
    for the questions they've already answered.

    **Response includes:**
    - `status`: always `"completed"`
    - `ended_early`: `true` when user ends before answering every question
    - `questions_answered` / `questions_skipped`: counts
    - `evaluations[]`: per-question feedback **with `model_answer`** and `user_answer`
    - `skipped_questions[]`: questions that were not reached
    - `evaluation_report`: full session evaluation (strengths, improvements, action plan)
    """
    try:
        if not practice_service:
            raise HTTPException(status_code=503, detail="Practice mode not initialized")

        user_id = get_user_id_from_request(http_request) or "guest_unknown" if http_request else "guest_unknown"

        # API Key selection (Bridge Settings)
        groq_key = x_api_key
        gemini_key = x_gemini_key
        if not groq_key and authorization and authorization.startswith("Bearer "):
            bearer_value = authorization.split(" ", 1)[1].strip()
            if bearer_value.count(".") != 2:  # Not a JWT
                groq_key = bearer_value

        api_key = gemini_key if gemini_key else groq_key

        # Authenticated users can use server keys
        is_authenticated = getattr(http_request.state, "user", None) is not None if http_request else False

        if not api_key:
            if settings.require_user_api_key and not is_authenticated:
                raise HTTPException(
                    status_code=401,
                    detail="No active API key. Please add your Groq or Gemini API key in Bridge Settings to continue.",
                )
            if settings.llm_provider == "gemini":
                api_key = settings.gemini_api_key or settings.groq_api_key
            else:
                api_key = settings.groq_api_key or settings.gemini_api_key

        logger.info(f"🛑 End session requested for {session_id} by {user_id}")

        result = await practice_service.end_session(session_id, api_key=api_key)

        # Persist completed attempt (best-effort, background)
        # Always persist on end-session, even if evaluation_report is absent
        if background_tasks:
            background_tasks.add_task(
                _persist_completed_practice_attempt,
                user_id=user_id,
                session_id=session_id,
            )

        _track_practice_event(
            user_id=user_id,
            session_id=session_id,
            event_type="practice_session_ended_early",
            question_text=None,
            extra={
                "questions_answered": result.get("questions_answered", 0),
                "questions_skipped": result.get("questions_skipped", 0),
                "ended_early": result.get("ended_early", False),
            },
        )

        return result

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ending practice session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/interview/acknowledge-feedback", response_model=NextQuestionResponse)
async def acknowledge_feedback(
    payload: AcknowledgeFeedbackRequest,
    background_tasks: BackgroundTasks,
    http_request: Request,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    x_gemini_key: Optional[str] = Header(None, alias="X-Gemini-Key"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
):
    """
    🎯 NEW FLOW: Acknowledge feedback and get next question.
    """
    try:
        if not practice_service:
            raise HTTPException(status_code=503, detail="Practice mode not initialized")
        
        logger.info(f"📋 Feedback acknowledged for session {payload.session_id}, question {payload.question_id}")

        user_id = get_user_id_from_request(http_request) or "guest_unknown"
        
        # API Key selection (Bridge Settings)
        groq_key = x_api_key
        gemini_key = x_gemini_key
        if not groq_key and authorization and authorization.startswith("Bearer "):
            groq_key = authorization.split(" ")[1]
            
        api_key = gemini_key if gemini_key else groq_key
        
        # Fallback to dev keys
        if not api_key:
            from app.config import settings
            if settings.require_user_api_key:
                raise HTTPException(
                    status_code=401,
                    detail="No active API key. Please add your Groq or Gemini API key in Bridge Settings to continue.",
                )
            if settings.llm_provider == "gemini":
                api_key = settings.gemini_api_key or settings.groq_api_key
            else:
                api_key = settings.groq_api_key or settings.gemini_api_key

        # Get next question after acknowledgment
        runner = practice_graph if (practice_graph and practice_graph.available) else None
        if runner:
            result = await runner.get_next_question_after_acknowledgment(
                session_id=payload.session_id,
                question_id=payload.question_id,
                api_key=api_key,
            )
        else:
            result = await practice_service.get_next_question_after_acknowledgment(
                session_id=payload.session_id,
                question_id=payload.question_id,
                api_key=api_key
            )

        _track_practice_event(
            user_id=user_id,
            session_id=payload.session_id,
            event_type="practice_feedback_acknowledged",
            question_text=None,
            extra={
                "question_id": int(payload.question_id),
                "feedback_read": bool(getattr(payload, "feedback_read", True)),
                "complete": bool(result.get("complete")),
            },
        )
        
        # Build response
        response = NextQuestionResponse(
            complete=result["complete"],
            progress=result["progress"]
        )

        if result.get("pressure") is not None:
            response.pressure = result.get("pressure")
        
        if result["complete"]:
            # Interview finished
            response.evaluation_report = result.get("evaluation_report")
            logger.info(f"✅ Interview complete for session {payload.session_id}")

            # Flagship loop: persist completed attempt for progress/trends (best-effort)
            # Persist on completion regardless of whether evaluation_report succeeded
            background_tasks.add_task(
                _persist_completed_practice_attempt,
                user_id=user_id,
                session_id=payload.session_id,
            )
        else:
            # Next question available
            response.next_question = result["next_question"]
            if result.get("tts_audio_url"):
                response.tts_audio_url = f"/api/practice/audio/{result['tts_audio_url']}"
            logger.info(f"➡️ Next question ready: Q{result['next_question'].id}")

            next_q = result.get("next_question")
            _track_practice_event(
                user_id=user_id,
                session_id=payload.session_id,
                event_type="practice_question_served",
                question_text=getattr(next_q, "text", None),
                extra={
                    "question_num": int(getattr(next_q, "id", 0) or 0),
                    "question_id": int(getattr(next_q, "id", 0) or 0),
                    "question_hash": stable_question_id(getattr(next_q, "text", "") or ""),
                    "difficulty": _safe_enum_value(getattr(next_q, "difficulty", None)),
                    "category": getattr(next_q, "category", None),
                    "round_type": _safe_enum_value(getattr(next_q, "round_type", None)),
                    "tts": bool(result.get("tts_audio_url")),
                },
            )
        
        return response
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error acknowledging feedback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/interview/rate-feedback", response_model=PracticeFeedbackRatedResponse)
async def rate_feedback(
    payload: PracticeFeedbackRatedRequest,
    http_request: Request,
):
    """Phase 3: collect ground-truth human labels for feedback usefulness.

    This endpoint is intentionally lightweight and best-effort: it should never
    block the main practice flow.
    """

    user_id = get_user_id_from_request(http_request) or "guest_unknown"

    # Best-effort: attach stable question hash if we can resolve the question text.
    q_hash: Optional[str] = None
    try:
        if practice_service and payload.session_id:
            sess = practice_service.get_session(payload.session_id)
            if sess and getattr(sess, "questions", None):
                for q in sess.questions:  # type: ignore[attr-defined]
                    if int(getattr(q, "id", -1)) == int(payload.question_id):
                        q_text = str(getattr(q, "text", "") or "")
                        if q_text.strip():
                            q_hash = stable_question_id(q_text)
                        break
    except Exception:
        q_hash = None

    # Optional comment: store only if raw text analytics are enabled.
    comment = (payload.comment or "").strip() or None
    comment_hash = stable_hash(comment) if comment else None

    extra: dict[str, Any] = {
        "question_id": int(payload.question_id),
        "question_hash": q_hash,
        "usefulness_rating": int(payload.usefulness_rating),
        "perceived_difficulty": _safe_enum_value(payload.perceived_difficulty) if payload.perceived_difficulty else None,
        "comment_len": len(comment or ""),
        "comment_hash": comment_hash,
    }

    if comment and getattr(settings, "analytics_store_raw_text", False):
        # Keep it bounded even when enabled.
        extra["comment_preview"] = comment[:200]

    _track_practice_event(
        user_id=user_id,
        session_id=payload.session_id,
        event_type="practice_feedback_rated",
        question_text=None,
        extra=extra,
    )

    return PracticeFeedbackRatedResponse(ok=True)


@router.get("/conversational-response", response_model=ConversationalResponse)
async def get_conversational_response(
    q: str = Query(...), 
    context: Optional[str] = Query(None),
    http_request: Request = None,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    x_gemini_key: Optional[str] = Header(None, alias="X-Gemini-Key"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
):
    """Get a conversational AI response for onboarding."""
    try:
        if not practice_service:
            raise HTTPException(status_code=503, detail="Practice mode not initialized")

        user_id = get_user_id_from_request(http_request) if http_request else "guest_unknown"
        user_id = user_id or "guest_unknown"

        # API Key selection (Bridge Settings)
        groq_key = x_api_key
        gemini_key = x_gemini_key
        if not groq_key and authorization and authorization.startswith("Bearer "):
            groq_key = authorization.split(" ")[1]
            
        api_key = gemini_key if gemini_key else groq_key
        
        # Fallback to dev keys
        if not api_key:
            from app.config import settings
            if settings.require_user_api_key:
                raise HTTPException(
                    status_code=401,
                    detail="No active API key. Please add your Groq or Gemini API key in Bridge Settings to continue.",
                )
            if settings.llm_provider == "gemini":
                api_key = settings.gemini_api_key or settings.groq_api_key
            else:
                api_key = settings.groq_api_key or settings.gemini_api_key

        result = await practice_service.get_conversational_response(
            voice_input=q,
            context=context,
            api_key=api_key,
            user_id=user_id,
        )
        return result
        
    except Exception as e:
        logger.error(f"Error getting conversational response: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/audio/{filename}")
async def get_audio(filename: str):
    """
    Serve TTS audio files.
    
    Args:
        filename: Audio filename
        
    Returns:
        Audio file
    """
    try:
        if not practice_service:
            raise HTTPException(status_code=503, detail="Practice mode not initialized")
        
        # Get audio path — validate filename to prevent path traversal
        if not _SAFE_FILENAME_RE.fullmatch(filename):
            raise HTTPException(status_code=400, detail="Invalid filename")
        audio_path = practice_service.get_audio_path(filename)
        
        # Check if file exists
        if not audio_path.exists():
            raise HTTPException(status_code=404, detail="Audio file not found")

        suffix = audio_path.suffix.lower()
        if suffix == ".mp3":
            media_type = "audio/mpeg"
        else:
            media_type = "audio/wav"
        
        # Serve file
        return FileResponse(
            path=audio_path,
            media_type=media_type,
            filename=filename
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving audio: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/session/{session_id}")
async def get_session(session_id: str):
    """
    Get session details.
    
    Args:
        session_id: Session identifier
        
    Returns:
        Session information
    """
    try:
        if not practice_service:
            raise HTTPException(status_code=503, detail="Practice mode not initialized")
        
        session = practice_service.get_session(session_id)
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return session
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """
    Delete a session and clean up files.
    
    Args:
        session_id: Session identifier
    """
    try:
        if not practice_service:
            raise HTTPException(status_code=503, detail="Practice mode not initialized")
        
        await practice_service.cleanup_session(session_id)
        
        return {"message": "Session deleted successfully"}
        
    except Exception as e:
        logger.error(f"Error deleting session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/status")
async def get_status():
    """
    Get practice mode service status.
    
    Returns:
        Service status information
    """
    try:
        if not practice_service:
            return {
                "status": "not_initialized",
                "message": "Practice mode service not initialized"
            }
        
        status = practice_service.get_service_status()
        status["status"] = "running"
        
        return status
        
    except Exception as e:
        logger.error(f"Error getting status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/session/{session_id}/evaluation")
async def get_evaluation(
    session_id: str,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    x_gemini_key: Optional[str] = Header(None, alias="X-Gemini-Key"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
):
    """
    Get evaluation report for a completed session.
    Useful for debugging when frontend doesn't receive evaluation.
    
    Args:
        session_id: Session identifier
        
    Returns:
        Evaluation report or error details
    """
    try:
        if not practice_service:
            raise HTTPException(status_code=503, detail="Practice mode not initialized")
        
        session = practice_service.get_session(session_id)
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        if not session.is_complete:
            return {
                "status": "incomplete",
                "message": "Interview not yet complete",
                "progress": f"{len(session.answers)}/{len(session.questions)} questions answered"
            }
        
        if session.evaluation_report:
            return {
                "status": "success",
                "evaluation": session.evaluation_report,
                "generated_at": session.evaluation_report.generated_at
            }
        else:
            # Try to regenerate evaluation if missing
            logger.warning(f"Evaluation missing for completed session {session_id}, regenerating...")
            try:
                # API Key selection
                groq_key = x_api_key
                gemini_key = x_gemini_key
                if not groq_key and authorization and authorization.startswith("Bearer "):
                    groq_key = authorization.split(" ")[1]
                api_key = gemini_key if gemini_key else groq_key
                
                # Fallback to dev keys
                if not api_key:
                    from app.config import settings
                    if settings.require_user_api_key:
                        raise HTTPException(
                            status_code=401,
                            detail="No active API key. Please add your Groq or Gemini API key in Bridge Settings to continue.",
                        )
                    if settings.llm_provider == "gemini":
                        api_key = settings.gemini_api_key or settings.groq_api_key
                    else:
                        api_key = settings.groq_api_key or settings.gemini_api_key

                evaluation_report = await practice_service.evaluation_agent.evaluate_interview(
                    session.answers,
                    session_id,
                    api_key=api_key
                )
                session.evaluation_report = evaluation_report
                return {
                    "status": "regenerated",
                    "evaluation": evaluation_report,
                    "message": "Evaluation was missing and has been regenerated"
                }
            except Exception as regen_error:
                logger.error(f"Failed to regenerate evaluation: {regen_error}", exc_info=True)
                return {
                    "status": "error",
                    "message": f"Evaluation generation failed: {str(regen_error)}",
                    "answers_count": len(session.answers),
                    "metrics_available": all(hasattr(a, 'metrics') for a in session.answers)
                }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting evaluation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/session/{session_id}/score")
async def get_session_score(
    session_id: str,
    http_request: Request,
):
    """Get deterministic scoring for a Practice Mode session.

    The frontend expects this endpoint after an interview completes.

    Behavior:
    - If the session is still in memory, compute score from the runtime session.
    - Otherwise, fall back to the persisted PracticeAttemptRecord (if available).
    """

    try:
        if not practice_service:
            raise HTTPException(status_code=503, detail="Practice mode not initialized")

        # Prefer runtime session (freshly completed sessions will be here).
        session = practice_service.get_session(session_id)
        if session:
            score = score_session(session=session)
            agg = _get_media_and_proctoring_summary(session_id)

            trace = None
            trajectory = None
            try:
                from app.services.practice.practice_scoring import build_evaluation_trace, compute_session_trajectory

                trace = build_evaluation_trace(session=session)
                trajectory = compute_session_trajectory(session=session)
            except Exception:
                trace = None
                trajectory = None
            return {
                "status": "success",
                "source": "runtime",
                "session_id": session_id,
                "complete": bool(getattr(session, "is_complete", False)),
                "overall_score": float(score.overall_score),
                "dimension_scores": score.dimension_scores,
                "why": score.why,
                "improvement_plan": score.improvement_plan,
                "next_session_plan": score.next_session_plan,
                "evaluation_trace": trace,
                "trajectory": trajectory,
                "evaluation_report": evaluation_report_to_json(getattr(session, "evaluation_report", None)),
                **agg,
            }

        # Fall back to persisted attempt (useful after refresh or session cleanup).
        user_id = get_user_id_from_request(http_request) or "guest_unknown"
        with get_db_context() as db:
            q = db.query(PracticeAttemptRecord).filter(PracticeAttemptRecord.session_id == session_id)
            # Avoid leaking across users if we have a stable user_id.
            if user_id and user_id != "guest_unknown":
                q = q.filter(PracticeAttemptRecord.user_id == user_id)
            rec = q.first()

        if not rec:
            raise HTTPException(status_code=404, detail="Session not found")

        return {
            "status": "success",
            "source": "db",
            "session_id": session_id,
            "complete": True,
            "overall_score": rec.overall_score,
            "dimension_scores": rec.dimension_scores,
            "why": rec.why,
            "improvement_plan": rec.improvement_plan,
            "next_session_plan": rec.next_session_plan,
            "evaluation_report": rec.evaluation_report,
            "started_at": rec.started_at,
            "completed_at": rec.completed_at,
            "created_at": rec.created_at,
            **_get_media_and_proctoring_summary(session_id),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session score: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/cleanup")
async def cleanup_old_sessions(max_age_minutes: int = 60):
    """
    Clean up old inactive sessions.
    
    Args:
        max_age_minutes: Maximum session age in minutes
    """
    try:
        if not practice_service:
            raise HTTPException(status_code=503, detail="Practice mode not initialized")
        
        await practice_service.cleanup_old_sessions(max_age_minutes)
        
        return {"message": "Cleanup completed"}
        
    except Exception as e:
        logger.error(f"Error during cleanup: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
