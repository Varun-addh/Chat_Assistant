"""
Practice Mode API Router.
FastAPI endpoints for realistic interview practice mode.
"""

import logging
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Header, Query, Request
from fastapi.responses import FileResponse
from pathlib import Path
from typing import Optional, Any
import aiofiles
import os

from app.schemas import (
    StartInterviewRequest,
    StartInterviewResponse,
    SubmitAnswerResponse,
    QuestionDifficulty,
    PracticeModeConfig,
    UserProfile,
    QuickStartRequest,
    ConversationalResponse,
    AcknowledgeFeedbackRequest,
    PracticeFeedbackRatedRequest,
    PracticeFeedbackRatedResponse,
    NextQuestionResponse,
    InterviewRound,
    RoundConfig,
    RoundSelectionRequest,
    RoundSelectionResponse
)
from app.services.practice_mode_service import PracticeModeService

from app.config import settings
from app.database import get_db_context
from app.middleware.auth import get_user_id_from_request
from app.utils.event_logging import track_event, stable_question_id, stable_hash
from app.services.learning_loops import compute_practice_insights, merge_focus_areas

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/practice",
    tags=["Practice Mode"]
)

# Service instance (will be initialized in main.py)
practice_service: Optional[PracticeModeService] = None


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
    global practice_service
    
    if config is None:
        config = PracticeModeConfig()
    
    practice_service = PracticeModeService(
        config=config,
        gemini_api_key=gemini_api_key,
        gemini_model=gemini_model
    )
    
    logger.info("Practice Mode initialized")


def cleanup_practice_mode():
    """
    Cleanup practice mode resources on shutdown.
    Call this from main.py lifespan shutdown.
    """
    global practice_service
    
    try:
        if practice_service:
            # Cleanup TTS resources
            if hasattr(practice_service, 'tts_service'):
                practice_service.tts_service.cleanup()
            logger.info("Practice Mode cleaned up successfully")
    except Exception as e:
        logger.error(f"Error cleaning up Practice Mode: {e}")


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
        
        from app.services.round_config_service import RoundConfigService
        
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
        raise HTTPException(status_code=500, detail=str(e))


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

        user_id = get_user_id_from_request(http_request) or "guest_unknown"
        
        if not practice_service:
            raise HTTPException(status_code=503, detail="Practice mode not initialized")
        
        from app.services.round_config_service import RoundConfigService
        
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
            api_key = settings.gemini_api_key or settings.groq_api_key
 
        # Start interview with EXPERIENCE-BASED difficulty (not round's default)
        session_id, first_question, audio_filename = await practice_service.start_interview(
            difficulty=experience_based_difficulty,  # ✅ Use experience-based difficulty
            user_profile=profile,
            question_count=final_question_count,  # ✅ Now dynamic
            round_type=payload.round_type,
            api_key=api_key
        )
        
        # Get session to retrieve total questions
        session = practice_service.get_session(session_id)
        total_questions = len(session.questions) if session else round_config.question_count
        
        tts_audio_url = f"/api/practice/audio/{audio_filename}" if audio_filename else ""

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
        
    except Exception as e:
        logger.error(f"Error starting round-based interview: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


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
        from app.services.round_config_service import RoundConfigService
        
        difficulty = RoundConfigService.get_difficulty_for_experience(experience_years)
        
        return {
            "difficulty": difficulty.value,
            "label": difficulty.value.upper(),
            "experience_years": experience_years,
            "description": f"Based on {experience_years} years of experience"
        }
    except Exception as e:
        logger.error(f"Error getting difficulty preview: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/interview/quick-start", response_model=ConversationalResponse)
async def quick_start_interview(
    request: QuickStartRequest,
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
        
        logger.info("🚀 Quick Start AI Mode initiated")
        
        # API Key selection (Bridge Settings)
        groq_key = x_api_key
        gemini_key = x_gemini_key
        if not groq_key and authorization and authorization.startswith("Bearer "):
            groq_key = authorization.split(" ")[1]
            
        api_key = gemini_key if gemini_key else groq_key
        
        # Fallback to dev keys
        if not api_key:
            from app.config import settings
            api_key = settings.gemini_api_key or settings.groq_api_key

        # Use AI to build profile from conversational input
        result = await practice_service.quick_start_conversational(
            voice_input=request.voice_input,
            context=request.context,
            auto_mode=request.auto_mode,
            use_memory=request.session_memory,
            question_count=request.question_count,  # User can override AI decision
            target_company=request.target_company,   # User can specify exact company
            api_key=api_key
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error in Quick Start: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


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
            if settings.llm_provider == "gemini":
                api_key = settings.gemini_api_key or settings.groq_api_key
            else:
                api_key = settings.groq_api_key or settings.gemini_api_key

        # Learning loop: enrich focus areas from prior attempts (best-effort)
        enriched_profile, recommended_focus = _maybe_enrich_profile_focus(
            user_id=user_id,
            profile=payload.user_profile,
        )

        # Start interview with user profile
        session_id, first_question, audio_filename = await practice_service.start_interview(
            difficulty=payload.difficulty,
            user_profile=enriched_profile,
            question_count=payload.question_count,
            api_key=api_key
        )
        
        # Get session to retrieve total questions count
        session = practice_service.get_session(session_id)
        total_questions = len(session.questions) if session else payload.question_count
        
        # Build audio URL
        tts_audio_url = f"/api/practice/audio/{audio_filename}" if audio_filename else ""

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
        
    except Exception as e:
        logger.error(f"Error starting interview: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/interview/submit-answer", response_model=SubmitAnswerResponse)
async def submit_answer(
    background_tasks: BackgroundTasks,
    http_request: Request,
    audio: UploadFile = File(..., description="Audio file of the answer"),
    session_id: str = Form(..., description="Session ID"),
    question_id: int = Form(..., ge=1, le=5, description="Question ID (1-5)"),
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
        
        logger.info(f"Answer processed successfully for session {session_id}")
        return response
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error submitting answer: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/interview/acknowledge-feedback", response_model=NextQuestionResponse)
async def acknowledge_feedback(
    payload: AcknowledgeFeedbackRequest,
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
            api_key = settings.gemini_api_key or settings.groq_api_key

        # Get next question after acknowledgment
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
        
        if result["complete"]:
            # Interview finished
            response.evaluation_report = result.get("evaluation_report")
            logger.info(f"✅ Interview complete for session {payload.session_id}")
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
        raise HTTPException(status_code=500, detail=str(e))


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
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    x_gemini_key: Optional[str] = Header(None, alias="X-Gemini-Key"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
):
    """Get a conversational AI response for onboarding."""
    try:
        if not practice_service:
            raise HTTPException(status_code=503, detail="Practice mode not initialized")
        
        # API Key selection (Bridge Settings)
        groq_key = x_api_key
        gemini_key = x_gemini_key
        if not groq_key and authorization and authorization.startswith("Bearer "):
            groq_key = authorization.split(" ")[1]
            
        api_key = gemini_key if gemini_key else groq_key
        
        # Fallback to dev keys
        if not api_key:
            from app.config import settings
            api_key = settings.gemini_api_key or settings.groq_api_key

        result = await practice_service.get_conversational_response(
            voice_input=q,
            context=context,
            api_key=api_key
        )
        return result
        
    except Exception as e:
        logger.error(f"Error getting conversational response: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
        
        # Get audio path
        audio_path = practice_service.get_audio_path(filename)
        
        # Check if file exists
        if not audio_path.exists():
            raise HTTPException(status_code=404, detail="Audio file not found")
        
        # Serve file
        return FileResponse(
            path=audio_path,
            media_type="audio/wav",
            filename=filename
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving audio: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


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
        raise HTTPException(status_code=500, detail=str(e))


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
        raise HTTPException(status_code=500, detail=str(e))


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
        raise HTTPException(status_code=500, detail=str(e))


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
                    api_key = settings.gemini_api_key or settings.groq_api_key

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
        raise HTTPException(status_code=500, detail=str(e))


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
        raise HTTPException(status_code=500, detail=str(e))
