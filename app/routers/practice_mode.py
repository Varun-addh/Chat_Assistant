"""
Practice Mode API Router.
FastAPI endpoints for realistic interview practice mode.
"""

import logging
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Header
from fastapi.responses import FileResponse
from pathlib import Path
from typing import Optional
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
    NextQuestionResponse,
    InterviewRound,
    RoundConfig,
    RoundSelectionRequest,
    RoundSelectionResponse
)
from app.services.practice_mode_service import PracticeModeService

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/practice",
    tags=["Practice Mode"]
)

# Service instance (will be initialized in main.py)
practice_service: Optional[PracticeModeService] = None


def init_practice_mode(
    gemini_api_key: str, 
    gemini_model: str = "models/gemini-flash-latest",
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


@router.post("/interview/start-round", response_model=StartInterviewResponse)
async def start_round_based_interview(
    request: RoundSelectionRequest,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
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
        logger.info(f"📥 Received request: round_type={request.round_type}, domain={request.domain}, exp={request.experience_years}")
        
        if not practice_service:
            raise HTTPException(status_code=503, detail="Practice mode not initialized")
        
        from app.services.round_config_service import RoundConfigService
        
        # Get configuration for the selected round
        round_config = RoundConfigService.get_round_config(request.round_type)
        
        logger.info(f"🎯 Starting {round_config.name} with {round_config.question_count} questions")
        
        # Build user profile from domain/experience or use provided profile
        if request.user_profile:
            profile = request.user_profile
        else:
            # Create profile from domain and experience
            # Infer basic skills from domain
            basic_skills = [request.domain] if request.domain else []
            
            profile = UserProfile(
                domain=request.domain,
                experience_years=request.experience_years,
                skills=basic_skills,  # Required field
                company_preference=request.company_specific
            )
        
        # Override company preference if specified
        if request.company_specific:
            profile.company_preference = request.company_specific
        
        # Calculate difficulty based on experience (NOT round's default difficulty)
        experience_based_difficulty = RoundConfigService.get_difficulty_for_experience(
            profile.experience_years
        )
        
        logger.info(f"📋 Profile: {profile.domain} with {profile.experience_years} years | Difficulty: {experience_based_difficulty.value.upper()} | Company: {profile.company_preference or 'Generic'}")
        
        api_key = x_api_key
        if not api_key and authorization:
            if authorization.startswith("Bearer "):
                api_key = authorization.split(" ")[1]

        # Start interview with EXPERIENCE-BASED difficulty (not round's default)
        session_id, first_question, audio_filename = await practice_service.start_interview(
            difficulty=experience_based_difficulty,  # ✅ Use experience-based difficulty
            user_profile=profile,
            question_count=round_config.question_count,
            round_type=request.round_type,
            api_key=api_key
        )
        
        # Get session to retrieve total questions
        session = practice_service.get_session(session_id)
        total_questions = len(session.questions) if session else round_config.question_count
        
        tts_audio_url = f"/api/practice/audio/{audio_filename}" if audio_filename else ""
        
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
    authorization: Optional[str] = Header(None, alias="Authorization"),
):
    """
    🚀 AI QUICK START - Zero-click conversational interview setup.
    
    **Simple Mode (Default):**
    - User: "I'm preparing for Senior SWE at Google"
    - AI: Infers EVERYTHING (role, company, difficulty, question count)
    - Returns: Immediate start with first question
    
    **Advanced Mode (Optional overrides):**
    - question_count: Force specific number (3-10)
    - target_company: Override AI's company extraction
    
    **Difference from Traditional Setup:**
    - Traditional: User fills ALL fields explicitly
    - Quick Start: User describes goal, AI infers details
    
    Examples:
    - "Senior Backend Engineer at Meta" → AI infers Meta, 5-7 hard questions
    - "Junior data scientist, 1 year" → AI infers easy, 3-5 questions
    - "I'm preparing for Google interviews" + target_company="Amazon" → Amazon questions
    """
    try:
        if not practice_service:
            raise HTTPException(status_code=503, detail="Practice mode not initialized")
        
        logger.info("🚀 Quick Start AI Mode initiated")
        
        api_key = x_api_key
        if not api_key and authorization:
            if authorization.startswith("Bearer "):
                api_key = authorization.split(" ")[1]

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
    request: StartInterviewRequest,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
):
    """
    Start a new practice interview session.
    
    Returns:
    - session_id: Unique session identifier
    - first_question: First interview question
    - tts_audio_url: URL to download TTS audio
    """
    try:
        if not practice_service:
            raise HTTPException(status_code=503, detail="Practice mode not initialized")
        
        logger.info(f"Starting interview with difficulty: {request.difficulty}")
        
        # Log user profile if provided
        if request.user_profile:
            logger.info(f"Adaptive interview for {request.user_profile.domain}, {request.user_profile.experience_years}yrs exp, skills: {request.user_profile.skills}")
        
        api_key = x_api_key
        if not api_key and authorization:
            if authorization.startswith("Bearer "):
                api_key = authorization.split(" ")[1]

        # Start interview with user profile
        session_id, first_question, audio_filename = await practice_service.start_interview(
            difficulty=request.difficulty,
            user_profile=request.user_profile,
            question_count=request.question_count,
            api_key=api_key
        )
        
        # Get session to retrieve total questions count
        session = practice_service.get_session(session_id)
        total_questions = len(session.questions) if session else request.question_count
        
        # Build audio URL
        tts_audio_url = f"/api/practice/audio/{audio_filename}"
        
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
    audio: UploadFile = File(..., description="Audio file of the answer"),
    session_id: str = Form(..., description="Session ID"),
    question_id: int = Form(..., ge=1, le=5, description="Question ID (1-5)"),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
):
    """
    Submit an answer to a question.
    
    Returns:
    - transcript: Transcribed answer
    - metrics: Speech analytics metrics
    - micro_feedback: Immediate delivery feedback
    - next_question: Next question (if not complete)
    - tts_audio_url: TTS audio for next question (if any)
    - complete: Whether interview is complete
    - evaluation_report: Final report (if complete)
    - progress: Progress indicator (e.g., "3/5")
    """
    try:
        if not practice_service:
            raise HTTPException(status_code=503, detail="Practice mode not initialized")
        
        logger.info(f"Receiving answer for session {session_id}, question {question_id}")
        
        # Validate audio file
        if not audio.content_type or not audio.content_type.startswith("audio/"):
            raise HTTPException(status_code=400, detail="Invalid audio file")
        
        # Save uploaded audio temporarily
        temp_audio_path = practice_service.audio_dir / f"temp_{session_id}_q{question_id}.wav"
        
        async with aiofiles.open(temp_audio_path, 'wb') as f:
            content = await audio.read()
            await f.write(content)
        
        api_key = x_api_key
        if not api_key and authorization:
            if authorization.startswith("Bearer "):
                api_key = authorization.split(" ")[1]

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
    request: AcknowledgeFeedbackRequest,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
):
    """
    🎯 NEW FLOW: Acknowledge feedback and get next question.
    
    **User Flow:**
    1. Submit answer → Get feedback
    2. **Review feedback** (user reads correctness score, tips, etc.)
    3. **Click "Next" button** → Calls this endpoint
    4. Get next question
    
    **Why this matters:**
    - Prevents auto-progression to next question
    - Ensures user sees their correctness evaluation
    - Better learning experience
    
    Args:
        request: Acknowledgment request with session_id and question_id
        
    Returns:
        Next question with audio OR completion status
    """
    try:
        if not practice_service:
            raise HTTPException(status_code=503, detail="Practice mode not initialized")
        
        logger.info(f"📋 Feedback acknowledged for session {request.session_id}, question {request.question_id}")
        
        api_key = x_api_key
        if not api_key and authorization:
            if authorization.startswith("Bearer "):
                api_key = authorization.split(" ")[1]

        # Get next question after acknowledgment
        result = await practice_service.get_next_question_after_acknowledgment(
            session_id=request.session_id,
            question_id=request.question_id,
            api_key=api_key
        )
        
        # Build response
        response = NextQuestionResponse(
            complete=result["complete"],
            progress=result["progress"]
        )
        
        if result["complete"]:
            # Interview finished
            response.evaluation_report = result.get("evaluation_report")
            logger.info(f"✅ Interview complete for session {request.session_id}")
        else:
            # Next question available
            response.next_question = result["next_question"]
            if result.get("tts_audio_url"):
                response.tts_audio_url = f"/api/practice/audio/{result['tts_audio_url']}"
            logger.info(f"➡️ Next question ready: Q{result['next_question'].id}")
        
        return response
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error acknowledging feedback: {e}", exc_info=True)
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
                api_key = x_api_key
                if not api_key and authorization:
                    if authorization.startswith("Bearer "):
                        api_key = authorization.split(" ")[1]

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
