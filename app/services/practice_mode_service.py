"""
Practice Mode Service - Main orchestrator.
Coordinates all three agents for complete interview practice flow.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Tuple
import uuid
from datetime import datetime
import asyncio

from app.schemas import (
    PracticeSession,
    PracticeModeConfig,
    PracticeInterviewQuestion,
    QuestionDifficulty,
    AnswerSubmission,
    SpeechMetrics,
    MicroFeedback,
    EvaluationReport,
    UserProfile,
    InterviewRound
)
from app.services.interviewer_agent import InterviewerAgent
from app.services.adaptive_interviewer_agent import AdaptiveInterviewerAgent
from app.services.speech_analytics_agent import SpeechAnalyticsAgent
from app.services.evaluation_agent import EvaluationAgent
from app.services.local_stt_service import LocalSTTService
from app.services.local_tts_service import LocalTTSService
from app.services.conversational_agent import ConversationalAgent

logger = logging.getLogger(__name__)


class PracticeModeService:
    """
    Main orchestrator for Practice Mode.
    Coordinates Interviewer, Speech Analytics, and Evaluation agents.
    """
    
    def __init__(
        self, 
        config: PracticeModeConfig,
        gemini_api_key: str,
        gemini_model: str = "models/gemini-3-flash-preview"
    ):
        """
        Initialize the practice mode service.
        
        Args:
            config: Practice mode configuration
            gemini_api_key: Gemini API key for evaluation
            gemini_model: Gemini model name (default: models/gemini-flash-latest)
        """
        self.config = config
        self.sessions: Dict[str, PracticeSession] = {}
        
        # Initialize audio storage
        self.audio_dir = Path(config.audio_storage_path)
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize agents
        logger.info("Initializing Practice Mode agents...")
        # Keep old agent as fallback - NOW WITH GEMINI FOR AI ANALYSIS
        self.interviewer_agent = InterviewerAgent(config.analytics, gemini_api_key=gemini_api_key)
        # NEW: Adaptive agent for intelligent question generation
        self.adaptive_interviewer = AdaptiveInterviewerAgent(gemini_api_key, gemini_model)
        # 🚀 NEW: Conversational AI agent for zero-click onboarding
        self.conversational_agent = ConversationalAgent(gemini_api_key)
        self.analytics_agent = SpeechAnalyticsAgent(config.analytics)
        self.evaluation_agent = EvaluationAgent(gemini_api_key, model_name=gemini_model)
        
        # Store gemini API key for later use
        self.gemini_api_key = gemini_api_key
        
        # Initialize STT/TTS services
        self.stt_service = LocalSTTService(config.stt)
        self.tts_service = LocalTTSService(config.tts, output_dir=str(self.audio_dir))
        
        # 🚀 NEW: Automatic Background Cleanup - Runs every 10 mins
        # DISABLED per user request: History should only delete manually
        # asyncio.create_task(self._background_cleanup_loop())
        
        # Warmup models (optional but recommended)
        asyncio.create_task(self._warmup_models())
        
        logger.info("Practice Mode Service initialized successfully with auto-cleanup")
    
    async def _background_cleanup_loop(self):
        """Perpetual background task to clean up expired sessions."""
        while True:
            try:
                # Wait for 10 minutes between cleanup runs
                await asyncio.sleep(600) 
                
                # Use the configured timeout from settings (default 30 mins)
                timeout = self.config.session_timeout_minutes
                logger.debug(f"🧹 Running background cleanup (Timeout: {timeout}m)")
                await self.cleanup_old_sessions(max_age_minutes=timeout)
                
            except Exception as e:
                logger.error(f"Error in background cleanup loop: {e}")
                await asyncio.sleep(60) # Wait a bit before retrying on error
    
    async def _warmup_models(self):
        """Warmup STT and TTS models for faster first request."""
        try:
            await asyncio.sleep(1)  # Let service start first
            logger.info("Warming up models...")
            self.stt_service.warmup()
            self.tts_service.warmup()
            logger.info("Models warmed up")
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")
    
    async def start_interview(
        self, 
        difficulty: QuestionDifficulty = QuestionDifficulty.MEDIUM,
        user_profile: Optional[UserProfile] = None,
        question_count: int = 5,
        round_type: Optional[InterviewRound] = None,
        api_key: Optional[str] = None
    ) -> Tuple[str, PracticeInterviewQuestion, str]:
        """
        Start a new practice interview session with adaptive questions.
        
        Args:
            difficulty: Question difficulty level
            user_profile: User profile for adaptive question generation
            question_count: Number of questions
            round_type: Specific interview round (NEW - for round-based practice)
            
        Returns:
            Tuple of (session_id, first_question, tts_audio_path)
        """
        try:
            # Generate session ID
            session_id = str(uuid.uuid4())
            round_info = f" ({round_type.value} round)" if round_type else ""
            logger.info(f"Starting new interview session{round_info}: {session_id}")
            
            # Special handling for FULL_INTERVIEW - mixed rounds simulation
            if round_type == InterviewRound.FULL_INTERVIEW:
                from app.services.round_config_service import RoundConfigService
                
                experience_years = user_profile.experience_years if user_profile else 3
                # Get the base breakdown (sums to 18)
                base_breakdown = RoundConfigService.get_full_interview_breakdown(experience_years)
                base_total = sum(base_breakdown.values())
                
                # Scale the breakdown to match the requested question_count
                # Ensure we have at least 1 question per sub-round if possible
                scaled_breakdown = {}
                remaining_questions = question_count
                
                # Sort by count descending to handle rounding errors better
                sorted_sub_rounds = sorted(base_breakdown.items(), key=lambda x: x[1], reverse=True)
                
                for i, (sub_round, base_count) in enumerate(sorted_sub_rounds):
                    if i == len(sorted_sub_rounds) - 1:
                        # Last round gets all remaining questions
                        scaled_count = max(1, remaining_questions)
                    else:
                        # Calculate proportional count (at least 1)
                        scaled_count = max(1, round((base_count / base_total) * question_count))
                        # Don't exceed total
                        scaled_count = min(scaled_count, remaining_questions - (len(sorted_sub_rounds) - 1 - i))
                    
                    scaled_breakdown[sub_round] = scaled_count
                    remaining_questions -= scaled_count
                
                logger.info(f"🎯 Full Interview Day breakdown (scaled to {question_count}): {scaled_breakdown}")
                
                # Generate questions for each round in sequence
                all_questions = []
                question_id_counter = 1
                
                # Use current items in order (preserving HR -> Tech -> Behavioral sequence)
                for sub_round in base_breakdown.keys():
                    sub_count = scaled_breakdown[sub_round]
                    logger.info(f"  Generating {sub_count} questions for {sub_round.value}")
                    
                    if user_profile:
                        sub_questions = await self.adaptive_interviewer.generate_adaptive_questions(
                            user_profile=user_profile,
                            difficulty=difficulty,
                            count=sub_count,
                            round_type=sub_round,
                            api_key=api_key
                        )
                    else:
                        sub_questions = self.interviewer_agent.get_questions(
                            difficulty=difficulty, 
                            count=sub_count
                        )
                    
                    # Re-number questions sequentially
                    for q in sub_questions:
                        q.id = question_id_counter
                        question_id_counter += 1
                    
                    all_questions.extend(sub_questions)
                
                questions = all_questions
                logger.info(f"✅ Full interview generated: {len(questions)} total questions across {len(base_breakdown)} rounds")
            
            # Regular single-round interview
            else:
                # Get questions - USE ADAPTIVE AGENT if profile provided
                if user_profile:
                    logger.info(f"Using adaptive interviewer for {user_profile.domain} with {user_profile.experience_years}yrs experience - {question_count} questions")
                    questions = await self.adaptive_interviewer.generate_adaptive_questions(
                        user_profile=user_profile,
                        difficulty=difficulty,
                        count=question_count,
                        round_type=round_type,  # NEW - pass round type
                        api_key=api_key
                    )
                else:
                    logger.info(f"No profile provided, using standard question bank - {question_count} questions")
                    questions = self.interviewer_agent.get_questions(difficulty=difficulty, count=question_count)
            
            # Create session
            session = PracticeSession(
                session_id=session_id,
                questions=questions,
                company_name=user_profile.company_preference if user_profile else None,
                current_question_index=0
            )
            
            # Store session
            self.sessions[session_id] = session
            
            # Get first question
            first_question = questions[0]
            
            # Generate TTS audio for first question (optional - continue without it if it fails)
            audio_filename = None
            try:
                audio_filename = f"{session_id}_q1.mp3"  # Use mp3 for gTTS
                audio_path = str(self.audio_dir / audio_filename)
                
                formatted_text = self.interviewer_agent.format_tts_text(
                    first_question,
                    total_questions=len(questions),
                    company_name=user_profile.company_preference if user_profile else None
                )
                result = await asyncio.wait_for(
                    self.tts_service.synthesize_async(formatted_text, audio_path),
                    timeout=15.0  # 15 second timeout for TTS
                )
                
                # Store audio filename
                session.audio_files.append(audio_filename)
                logger.info(f"TTS audio generated for question 1: {audio_filename}")
            except asyncio.TimeoutError:
                logger.warning(f"TTS generation timed out for question 1, continuing without audio")
                audio_filename = None
            except Exception as e:
                logger.warning(f"TTS generation failed for question 1: {e}, continuing without audio")
                audio_filename = None
            
            logger.info(f"Interview started: {session_id}")
            return session_id, first_question, audio_filename
            
        except Exception as e:
            logger.error(f"Error starting interview: {e}", exc_info=True)
            raise
    
    async def quick_start_conversational(
        self,
        voice_input: Optional[str] = None,
        context: Optional[str] = None,
        auto_mode: bool = True,
        use_memory: bool = True,
        question_count: Optional[int] = None,
        target_company: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        """
        🚀 AI QUICK START - Conversational zero-click interview setup.
        
        Flow:
        1. User says: "I'm preparing for Senior SWE at Meta"
        2. AI infers: domain, level, skills, difficulty, count
        3. Generates adaptive questions immediately
        4. Returns first question + starts interview
        
        Args:
            voice_input: Natural language from user (voice or text)
            context: Additional context (resume, past performance, etc.)
            auto_mode: If True, auto-start interview. If False, ask for confirmation
            use_memory: Use previous session data for personalization
            question_count: Override AI's question count decision (3-10)
            target_company: Override AI's company inference (e.g., "Google", "Meta")
            
        Returns:
            ConversationalResponse with first question or clarification request
        """
        from app.schemas import ConversationalResponse
        from app.services.llm_service import llm_service
        
        try:
            logger.info("🚀 Quick Start: Analyzing user input...")
            
            # Default input if none provided
            if not voice_input:
                return ConversationalResponse(
                    ai_message="Hi! Tell me what role you're preparing for, and I'll set up your interview!",
                    needs_clarification=True,
                    ready_to_start=False
                )

            # Deterministic identity/developer attribution (never call an LLM)
            if llm_service._is_identity_question(voice_input):
                return ConversationalResponse(
                    ai_message=llm_service._identity_response_text(voice_input),
                    needs_clarification=False,
                    ready_to_start=False,
                )
            
            # Use AI to infer profile from conversation
            profile, ai_message, difficulty, ai_question_count = await self.conversational_agent.infer_profile_from_conversation(
                user_input=voice_input,
                context=context,
                api_key=api_key
            )
            
            # Override with user's explicit choices if provided
            if question_count:
                ai_question_count = question_count
                logger.info(f"User override: question_count={question_count}")
            
            if target_company:
                profile.company_preference = target_company
                logger.info(f"User override: target_company={target_company}")
                ai_message = f"Perfect! I'll create a {target_company}-specific interview for you."
            
            logger.info(f"✅ Profile inferred: {profile.domain}, {profile.experience_years}yrs, {difficulty.value}, company={profile.company_preference}")
            
            if not auto_mode:
                # Ask for confirmation before starting
                company_mention = f" for {profile.company_preference}" if profile.company_preference and profile.company_preference.lower() not in ["any", "general"] else ""
                return ConversationalResponse(
                    ai_message=f"{ai_message} Ready to start with {ai_question_count} {difficulty.value} questions{company_mention}?",
                    needs_clarification=True,
                    ready_to_start=False,
                    suggested_profile=profile
                )
            
            # Auto-start interview with inferred profile
            session_id, first_question, audio_filename = await self.start_interview(
                difficulty=difficulty,
                user_profile=profile,
                question_count=ai_question_count,
                round_type=profile.target_round if hasattr(profile, 'target_round') else None,  # Support round selection
                api_key=api_key
            )
            
            # Get session to retrieve total questions count
            session = self.sessions.get(session_id)
            total_questions = len(session.questions) if session else ai_question_count
            
            # Build audio URL
            tts_audio_url = f"/api/practice/audio/{audio_filename}" if audio_filename else None
            
            return ConversationalResponse(
                ai_message=f"{ai_message} Starting now...",
                needs_clarification=False,
                ready_to_start=True,
                suggested_profile=profile,
                session_id=session_id,
                first_question=first_question,
                tts_audio_url=tts_audio_url,
                total_questions=total_questions,
                progress=f"1/{total_questions}"
            )
            
        except Exception as e:
            logger.error(f"Error in quick start: {e}", exc_info=True)
            return ConversationalResponse(
                ai_message="I had trouble understanding that. Could you tell me what role you're preparing for?",
                needs_clarification=True,
                ready_to_start=False
            )

    async def get_conversational_response(
        self,
        voice_input: str,
        context: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """Backward-compatible conversational onboarding endpoint.

        The router exposes this as GET /api/practice/conversational-response.
        We keep it as a lightweight wrapper around quick_start_conversational.
        """
        return await self.quick_start_conversational(
            voice_input=voice_input,
            context=context,
            auto_mode=False,
            api_key=api_key,
        )
    
    async def submit_answer(
        self, 
        session_id: str,
        question_id: int,
        audio_file_path: str,
        api_key: Optional[str] = None
    ) -> dict:
        """
        Process submitted answer.
        
        Args:
            session_id: Session identifier
            question_id: Question ID (1-5)
            audio_file_path: Path to uploaded audio file
            
        Returns:
            Response dict with transcript, metrics, feedback, next question, etc.
        """
        try:
            logger.info(f"Processing answer for session {session_id}, question {question_id}")
            
            # Get session
            session = self.sessions.get(session_id)
            if not session:
                raise ValueError(f"Session not found: {session_id}")
            
            # Update activity timestamp
            session.last_activity_at = datetime.utcnow()
            
            # Validate question
            question = session.questions[question_id - 1]
            
            # Step 1: Transcribe audio (STT)
            transcript, stt_metadata = await self.stt_service.transcribe_async(audio_file_path)
            logger.info(f"Transcription (full): '{transcript}'")
            
            # Step 2: Analyze speech (Analytics Agent)
            metrics = self.analytics_agent.analyze_audio(
                audio_path=audio_file_path,
                transcript=transcript,
                time_limit=question.time_limit,
                stt_metadata=stt_metadata  # NEW: Pass VAD info from STT
            )
            
            # Step 3: Generate micro-feedback with AI analysis
            # Use adaptive agent with COMPREHENSIVE EVALUATION
            micro_feedback = await self.adaptive_interviewer.generate_micro_feedback(
                metrics,
                question_text=question.text,
                transcript=transcript,
                question_key_points=question.key_points if hasattr(question, 'key_points') else None,
                question_expected_answer=question.expected_answer_template if hasattr(question, 'expected_answer_template') else None,
                question_category=question.category,
                api_key=api_key
            )
            
            # Create answer submission
            answer = AnswerSubmission(
                question_id=question_id,
                transcript=transcript,
                metrics=metrics,
                micro_feedback=micro_feedback,
                audio_duration=metrics.duration
            )
            
            # Store answer
            session.answers.append(answer)
            # Update to 0-based index (question_id is 1-based, so subtract 1)
            session.current_question_index = question_id - 1
            
            # Check if interview is complete
            is_complete = self.interviewer_agent.is_interview_complete(
                session.current_question_index, 
                len(session.questions)
            )
            
            response = {
                "transcript": transcript,
                "metrics": metrics,
                "micro_feedback": micro_feedback,
                "complete": is_complete,
                "progress": self.interviewer_agent.get_progress_indicator(
                    question_id, 
                    len(session.questions)
                ),
                "requires_acknowledgment": not is_complete,  # Only require ack if not complete
                "current_question_id": question_id
            }
            
            # Store next question details in session for later retrieval (don't send yet)
            if not is_complete:
                session.pending_next_question = True
                # Don't generate TTS yet - wait for acknowledgment
                # This prevents auto-progression to next question
            
            else:
                # Interview complete - generate evaluation
                logger.info("Interview complete, generating evaluation...")
                session.is_complete = True
                session.completed_at = datetime.utcnow()
                
                try:
                    logger.info(f"Generating final evaluation for {len(session.answers)} answers...")
                    evaluation_report = await self.evaluation_agent.evaluate_interview(
                        session.answers,
                        session_id,
                        api_key=api_key
                    )
                    
                    session.evaluation_report = evaluation_report
                    response["evaluation_report"] = evaluation_report
                    logger.info("✅ Evaluation report generated successfully")
                    
                    # Log evaluation summary for debugging
                    logger.info(f"Evaluation Summary - Strengths: {len(evaluation_report.strengths.items)}, "
                              f"Improvements: {len(evaluation_report.improvements.items)}, "
                              f"Action Steps: {len(evaluation_report.action_plan.steps)}")
                    
                except Exception as eval_error:
                    logger.error(f"❌ Evaluation generation failed: {eval_error}", exc_info=True)
                    # Set a fallback message
                    response["evaluation_error"] = str(eval_error)
                    response["evaluation_report"] = None
            
            logger.info(f"Answer processed successfully for session {session_id}")
            return response
            
        except Exception as e:
            logger.error(f"Error processing answer: {e}", exc_info=True)
            raise
    
    def get_session(self, session_id: str) -> Optional[PracticeSession]:
        """Get session by ID."""
        session = self.sessions.get(session_id)
        if session:
            session.last_activity_at = datetime.utcnow()
        return session
    
    async def get_next_question_after_acknowledgment(
        self,
        session_id: str,
        question_id: int,
        api_key: Optional[str] = None
    ) -> dict:
        """
        Get next question after user acknowledges feedback.
        
        This enforces the flow: Answer → Feedback → Acknowledge → Next Question
        
        Args:
            session_id: Session identifier
            question_id: Question ID that was just answered
            
        Returns:
            Dict with next question or completion status
        """
        try:
            # Get session
            session = self.sessions.get(session_id)
            if not session:
                raise ValueError(f"Session not found: {session_id}")
            
            # Validate the acknowledged question matches current state
            if session.current_question_index != question_id - 1:
                raise ValueError(f"Invalid question_id {question_id}. Current is {session.current_question_index + 1}")
            
            # Increment to next question index
            session.current_question_index += 1
            
            # Check if already complete (after increment)
            is_complete = self.interviewer_agent.is_interview_complete(
                session.current_question_index,
                len(session.questions)
            )
            
            if is_complete:
                # Already evaluated, just return report
                return {
                    "complete": True,
                    "evaluation_report": session.evaluation_report,
                    "progress": self.interviewer_agent.get_progress_indicator(
                        question_id,
                        len(session.questions)
                    )
                }
            
            # Get next question
            next_question = self.interviewer_agent.get_next_question(
                session.questions,
                session.current_question_index
            )
            
            if not next_question:
                raise ValueError("No next question available")
            
            # Generate TTS for next question
            audio_filename = None
            try:
                audio_filename = f"{session_id}_q{next_question.id}.mp3"
                audio_path = str(self.audio_dir / audio_filename)
                
                formatted_text = self.interviewer_agent.format_tts_text(
                    next_question,
                    total_questions=len(session.questions),
                    company_name=session.company_name
                )
                
                result = await asyncio.wait_for(
                    self.tts_service.synthesize_async(formatted_text, audio_path),
                    timeout=15.0
                )
                session.audio_files.append(audio_filename)
                logger.info(f"✅ TTS generated after acknowledgment for Q{next_question.id}")
            except asyncio.TimeoutError:
                logger.warning(f"TTS timeout for Q{next_question.id}")
                audio_filename = None
            except Exception as e:
                logger.warning(f"TTS failed for Q{next_question.id}: {e}")
                audio_filename = None
            
            # Clear pending flag
            session.pending_next_question = False
            
            return {
                "next_question": next_question,
                "tts_audio_url": audio_filename,
                "complete": False,
                "progress": self.interviewer_agent.get_progress_indicator(
                    next_question.id,
                    len(session.questions)
                )
            }
            
        except Exception as e:
            logger.error(f"Error getting next question: {e}", exc_info=True)
            raise
    
    def get_audio_path(self, filename: str) -> Path:
        """Get full path for audio file."""
        return self.audio_dir / filename
    
    async def cleanup_session(self, session_id: str):
        """Clean up session and audio files."""
        try:
            session = self.sessions.get(session_id)
            if not session:
                return
            
            # Delete audio files
            for filename in session.audio_files:
                audio_path = self.audio_dir / filename
                audio_path.unlink(missing_ok=True)
            
            # Remove session
            del self.sessions[session_id]
            
            logger.info(f"Session cleaned up: {session_id}")
            
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")
    
    async def cleanup_old_sessions(self, max_age_minutes: int = 60):
        """Clean up old inactive sessions."""
        try:
            now = datetime.utcnow()
            to_remove = []
            
            for session_id, session in self.sessions.items():
                # Use last_activity_at instead of started_at to avoid timing out active users
                age_minutes = (now - session.last_activity_at).total_seconds() / 60
                
                if age_minutes > max_age_minutes:
                    to_remove.append(session_id)
            
            for session_id in to_remove:
                await self.cleanup_session(session_id)
            
            if to_remove:
                logger.info(f"Cleaned up {len(to_remove)} old sessions")
                
        except Exception as e:
            logger.warning(f"Old session cleanup error: {e}")
    
    def get_service_status(self) -> dict:
        """Get service status information."""
        return {
            "active_sessions": len(self.sessions),
            "stt_info": self.stt_service.get_model_info(),
            "tts_info": self.tts_service.get_engine_info(),
            "config": {
                "max_concurrent_sessions": self.config.max_concurrent_sessions,
                "session_timeout_minutes": self.config.session_timeout_minutes,
                "audio_storage_path": self.config.audio_storage_path
            }
        }
