"""
Practice Mode Service - Main orchestrator.
Coordinates all three agents for complete interview practice flow.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import uuid
from datetime import datetime
import asyncio

from app.utils.time import utcnow
from app.config import settings
from app.database import get_db_context

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
    InterviewRound,
    SessionStrategyDecision,
)
from app.services.practice.interviewer_agent import InterviewerAgent
from app.services.practice.adaptive_interviewer_agent import AdaptiveInterviewerAgent
from app.services.practice.speech_analytics_agent import SpeechAnalyticsAgent
from app.services.practice.evaluation_agent import EvaluationAgent
from app.services.practice.local_stt_service import LocalSTTService
from app.services.practice.local_tts_service import LocalTTSService
from app.services.practice.conversational_agent import ConversationalAgent
from app.services.practice.session_strategy_agent import SessionStrategyAgent
from app.services.chat.llm_service import LLMAuthenticationError

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
        self.session_strategy_agent = SessionStrategyAgent(self)
        
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

    def _get_session_strategy_agent(self) -> SessionStrategyAgent:
        agent = getattr(self, "session_strategy_agent", None)
        if agent is None:
            agent = SessionStrategyAgent(self)
            self.session_strategy_agent = agent
        return agent

    @staticmethod
    def _find_answer_for_question(session: PracticeSession, question_id: int) -> Optional[AnswerSubmission]:
        for answer in reversed(getattr(session, "answers", []) or []):
            if int(getattr(answer, "question_id", -1)) == int(question_id):
                return answer
        return None

    async def _preview_strategy_decision(
        self,
        *,
        session: PracticeSession,
        previous_question: PracticeInterviewQuestion,
        last_answer: AnswerSubmission,
        api_key: Optional[str],
    ) -> Optional[SessionStrategyDecision]:
        if not bool(getattr(settings, "enable_practice_session_brain", True)):
            return None

        try:
            return await self._get_session_strategy_agent().preview_next_action(
                session=session,
                previous_question=previous_question,
                last_answer=last_answer,
                api_key=api_key,
            )
        except Exception as exc:
            logger.warning("Session brain preview failed, falling back to legacy flow: %s", exc)
            return None

    def _maybe_add_peer_learning_insight(self, evaluation_report: EvaluationReport) -> None:
        if not bool(getattr(settings, "enable_practice_learning", False)):
            return
        try:
            from app.services.practice.practice_learning import (
                derive_peer_bucket,
                compute_peer_confidence_benchmark,
                format_peer_confidence_insight,
            )

            bucket = derive_peer_bucket(evaluation_report.metrics_summary)
            with get_db_context() as db:
                avg_conf, n = compute_peer_confidence_benchmark(db, bucket=bucket, min_samples=3)
            if avg_conf is not None:
                evaluation_report.learning_insight = format_peer_confidence_insight(
                    avg_confidence_1_5=avg_conf,
                    n=n,
                )
        except Exception:
            return
    
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
        api_key: Optional[str] = None,
        previously_asked: Optional[List[str]] = None,
    ) -> Tuple[str, PracticeInterviewQuestion, str]:
        """
        Start a new practice interview session with adaptive questions.
        
        Args:
            difficulty: Question difficulty level
            user_profile: User profile for adaptive question generation
            question_count: Number of questions
            round_type: Specific interview round (NEW - for round-based practice)
            previously_asked: Question texts from prior sessions (for cross-session dedup)
            
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
                from app.services.practice.round_config_service import RoundConfigService
                
                experience_years = user_profile.experience_years if user_profile else 3
                # Get the base breakdown (sums to 18)
                base_breakdown = RoundConfigService.get_full_interview_breakdown(experience_years)
                base_total = sum(base_breakdown.values()) or 1  # guard against empty/zero breakdown
                
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
                            api_key=api_key,
                            previously_asked=previously_asked,
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
                        round_type=round_type,
                        api_key=api_key,
                        previously_asked=previously_asked,
                    )
                else:
                    logger.info(f"No profile provided, using standard question bank - {question_count} questions")
                    questions = self.interviewer_agent.get_questions(difficulty=difficulty, count=question_count)
            
            # Create session
            session = PracticeSession(
                session_id=session_id,
                user_profile=user_profile,
                difficulty=difficulty,
                round_type=round_type,
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
                # Use wav as the requested container for pyttsx3 (Windows SAPI); gTTS will return mp3.
                audio_path = str(self.audio_dir / f"{session_id}_q1.wav")
                
                formatted_text = self.interviewer_agent.format_tts_text(
                    first_question,
                    total_questions=len(questions),
                    company_name=user_profile.company_preference if user_profile else None
                )
                result_path = await self.tts_service.synthesize_async(formatted_text, audio_path)
                audio_filename = Path(result_path).name
                
                # Store audio filename
                if audio_filename:
                    session.audio_files.append(audio_filename)
                logger.info(f"TTS audio generated for question 1: {audio_filename}")
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
        api_key: Optional[str] = None,
        user_id: Optional[str] = None,
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
        from app.services.chat.llm_service import llm_service
        
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
            
            # Cross-session dedup: fetch previously asked questions for this user+domain
            previously_asked: List[str] = []
            if user_id:
                try:
                    from app.database import get_db_context
                    from app.services.practice.learning_loops import get_previously_asked_questions
                    with get_db_context() as db:
                        previously_asked = get_previously_asked_questions(
                            db, user_id=user_id, domain=profile.domain
                        )
                    if previously_asked:
                        logger.info(f"📋 Cross-session dedup: {len(previously_asked)} previously asked questions for {profile.domain}")
                except Exception as e:
                    logger.warning(f"Could not fetch previously asked questions: {e}")

            # Auto-start interview with inferred profile
            session_id, first_question, audio_filename = await self.start_interview(
                difficulty=difficulty,
                user_profile=profile,
                question_count=ai_question_count,
                round_type=profile.target_round if hasattr(profile, 'target_round') else None,  # Support round selection
                api_key=api_key,
                previously_asked=previously_asked or None,
            )
            
            # Get session to retrieve total questions count
            session = self.sessions.get(session_id)
            total_questions = len(session.questions) if session else ai_question_count
            
            # Build audio URL
            tts_audio_url = f"/api/practice/audio/{audio_filename}" if audio_filename else None

            # UI hint: only auto-start recording for VOICE questions.
            auto_start_recording = True
            try:
                qt = getattr(first_question, "question_type", None)
                if qt is not None:
                    auto_start_recording = (getattr(qt, "value", qt) == "voice")
            except Exception:
                auto_start_recording = True
            
            return ConversationalResponse(
                ai_message=f"{ai_message} Starting now...",
                needs_clarification=False,
                ready_to_start=True,
                suggested_profile=profile,
                session_id=session_id,
                first_question=first_question,
                auto_start_recording=auto_start_recording,
                tts_audio_url=tts_audio_url,
                total_questions=total_questions,
                progress=f"1/{total_questions}"
            )

        except LLMAuthenticationError:
            logger.error("Error in quick start: invalid LLM credentials", exc_info=True)
            raise
            
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
        user_id: Optional[str] = None,
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
            user_id=user_id,
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
            session.last_activity_at = utcnow()
            
            # Validate question
            question = session.questions[question_id - 1]
            
            # Step 1: Transcribe audio (STT)
            transcript, stt_metadata = await self.stt_service.transcribe_async(audio_file_path)
            logger.info(f"Transcription (full): '{transcript}'")
            
            # Transcription quality gate: warn when Whisper output looks garbled
            _lang_prob = stt_metadata.get("language_probability", 1.0) if stt_metadata else 1.0
            _word_count = len(transcript.split()) if transcript else 0
            _audio_dur = stt_metadata.get("duration", 0) if stt_metadata else 0
            _transcript_quality_low = (
                _lang_prob < 0.70
                or (_audio_dur > 5 and _word_count < 4)
            )
            if _transcript_quality_low:
                logger.warning(
                    f"Low transcription quality: lang_prob={_lang_prob:.2f}, "
                    f"words={_word_count}, audio_dur={_audio_dur:.1f}s"
                )
            
            # Step 2: Analyze speech (Analytics Agent)
            metrics = self.analytics_agent.analyze_audio(
                audio_path=audio_file_path,
                transcript=transcript,
                time_limit=question.time_limit,
                stt_metadata=stt_metadata  # NEW: Pass VAD info from STT
            )
            
            # Step 3: Generate micro-feedback with AI analysis
            # Use adaptive agent with COMPREHENSIVE EVALUATION
            try:
                micro_feedback = await self.adaptive_interviewer.generate_micro_feedback(
                    metrics,
                    question_text=question.text,
                    transcript=transcript,
                    question_key_points=question.key_points if hasattr(question, 'key_points') else None,
                    question_expected_answer=question.expected_answer_template if hasattr(question, 'expected_answer_template') else None,
                    question_category=question.category,
                    api_key=api_key
                )
            except Exception as mf_err:
                logger.error(f"❌ Micro-feedback generation failed: {mf_err}", exc_info=True)
                # Fallback: create minimal feedback so the answer is still recorded
                from app.schemas import MicroFeedback
                micro_feedback = MicroFeedback(
                    delivery_tips=["Feedback unavailable due to API error"],
                    pace_feedback="Unable to analyze",
                    overall_note=f"AI feedback could not be generated: {type(mf_err).__name__}",
                    correctness_score=None,
                    technical_accuracy=None,
                )
            
            # Inject transcription quality warning into micro-feedback if garbled
            if _transcript_quality_low and micro_feedback:
                _qual_tip = (
                    f"⚠️ Low transcription confidence (lang_prob={_lang_prob:.0%}) — "
                    "evaluation may be inaccurate. Try speaking closer to the mic."
                )
                if hasattr(micro_feedback, "delivery_tips") and isinstance(micro_feedback.delivery_tips, list):
                    micro_feedback.delivery_tips.insert(0, _qual_tip)
                    micro_feedback.delivery_tips = micro_feedback.delivery_tips[:2]
            
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
            
            # Check if interview is complete (all pre-generated questions answered)
            all_questions_answered = self.interviewer_agent.is_interview_complete(
                session.current_question_index, 
                len(session.questions)
            )
            
            # If all questions answered, give the strategy brain a chance to extend
            # (e.g., follow-up drill, difficulty change) before declaring complete.
            is_complete = all_questions_answered
            strategy_preview = None
            if all_questions_answered and bool(getattr(settings, "enable_practice_session_brain", True)):
                try:
                    strategy_preview = await self._preview_strategy_decision(
                        session=session,
                        previous_question=question,
                        last_answer=answer,
                        api_key=api_key,
                    )
                    if strategy_preview is not None:
                        from app.schemas import SessionStrategyAction
                        _continuing_actions = {
                            SessionStrategyAction.FOLLOW_UP,
                            SessionStrategyAction.ASK_QUESTION,
                            SessionStrategyAction.INCREASE_DIFFICULTY,
                            SessionStrategyAction.DECREASE_DIFFICULTY,
                        }
                        if strategy_preview.action in _continuing_actions:
                            is_complete = False
                            logger.info(
                                f"Strategy brain overrides completion: action={strategy_preview.action.value}, "
                                f"reason={strategy_preview.reason}"
                            )
                except Exception as exc:
                    logger.warning("Strategy brain preview at completion failed: %s", exc)
                    strategy_preview = None
            
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
                # Re-use the preview we already computed (if any), else compute fresh
                preview = strategy_preview
                if preview is None:
                    preview = await self._preview_strategy_decision(
                        session=session,
                        previous_question=question,
                        last_answer=answer,
                        api_key=api_key,
                    )
                session.pending_strategy_decision = preview
                if preview is not None:
                    session.current_coaching_style = preview.coaching_style
                    response["strategy"] = preview
                # Don't generate TTS yet - wait for acknowledgment
                # This prevents auto-progression to next question
            
            else:
                # Interview complete - generate evaluation
                logger.info("Interview complete, generating evaluation...")
                session.is_complete = True
                session.completed_at = utcnow()
                session.pending_strategy_decision = None
                
                try:
                    logger.info(f"Generating final evaluation for {len(session.answers)} answers...")
                    evaluation_report = await self.evaluation_agent.evaluate_interview(
                        session.answers,
                        session_id,
                        api_key=api_key,
                        questions=session.questions,
                    )

                    self._maybe_add_peer_learning_insight(evaluation_report)
                    
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

    async def submit_code(
        self,
        session_id: str,
        question_id: int,
        code: str,
        programming_language: str,
        time_taken: int,
        correctness_score: int = 0,
        summary: str = "Coding submission received.",
        api_key: Optional[str] = None,
    ) -> dict:
        """Record a coding submission as an answered question.

        Practice Mode's data model is centered around AnswerSubmission.
        For coding questions we store:
        - transcript: the code itself
        - metrics: a minimal placeholder (delivery metrics are N/A)
        - micro_feedback: a minimal placeholder populated from correctness_score/summary
        """

        try:
            logger.info(f"Processing code submission for session {session_id}, question {question_id}")

            session = self.sessions.get(session_id)
            if not session:
                raise ValueError(f"Session not found: {session_id}")

            session.last_activity_at = utcnow()

            # Validate question index
            if question_id < 1 or question_id > len(session.questions):
                raise ValueError(f"Invalid question_id {question_id}")

            # Placeholder delivery metrics (coding doesn't have audio)
            metrics = SpeechMetrics(
                filler_count=0,
                wpm=0.0,
                longest_silence=0.0,
                confidence_score=0.0,
                overtalked=False,
                duration=0.0,
                filler_words=[],
                pause_count=0,
                pitch_variance=0.0,
                silence_removed=0.0,
            )

            is_correct = bool(int(correctness_score) >= 70)
            micro_feedback = MicroFeedback(
                delivery_tips=["Coding answer received.", "Review correctness + edge cases."],
                pace_feedback="N/A (coding submission)",
                overall_note=(summary or "Coding submission received."),
                correctness_score=int(correctness_score),
                technical_accuracy="Good" if is_correct else "Poor",
                is_correct=is_correct,
            )

            answer = AnswerSubmission(
                question_id=int(question_id),
                transcript=code or "",
                metrics=metrics,
                micro_feedback=micro_feedback,
                audio_duration=0.0,
            )

            session.answers.append(answer)
            session.current_question_index = question_id - 1

            is_complete = self.interviewer_agent.is_interview_complete(
                session.current_question_index,
                len(session.questions),
            )

            response = {
                "complete": is_complete,
                "progress": self.interviewer_agent.get_progress_indicator(
                    question_id,
                    len(session.questions),
                ),
                "requires_acknowledgment": not is_complete,
            }

            if not is_complete:
                session.pending_next_question = True
                preview = await self._preview_strategy_decision(
                    session=session,
                    previous_question=session.questions[question_id - 1],
                    last_answer=answer,
                    api_key=api_key,
                )
                session.pending_strategy_decision = preview
                if preview is not None:
                    session.current_coaching_style = preview.coaching_style
                    response["strategy"] = preview
            else:
                session.is_complete = True
                session.completed_at = utcnow()
                session.pending_strategy_decision = None
                try:
                    evaluation_report = await self.evaluation_agent.evaluate_interview(
                        session.answers,
                        session_id,
                        api_key=api_key,
                        questions=session.questions,
                    )

                    self._maybe_add_peer_learning_insight(evaluation_report)
                    session.evaluation_report = evaluation_report
                    response["evaluation_report"] = evaluation_report
                except Exception as eval_error:
                    logger.error(f"❌ Evaluation generation failed after coding submission: {eval_error}", exc_info=True)
                    response["evaluation_error"] = str(eval_error)
                    response["evaluation_report"] = None

            logger.info(f"Code submission processed successfully for session {session_id}")
            return response
        except Exception:
            logger.error("Error processing code submission", exc_info=True)
            raise
    
    def get_session(self, session_id: str) -> Optional[PracticeSession]:
        """Get session by ID."""
        session = self.sessions.get(session_id)
        if session:
            session.last_activity_at = utcnow()
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

            # NOTE: session.current_question_index represents the *last answered* question (0-based).
            # The InterviewerAgent.get_next_question() already advances by +1, so we must NOT
            # increment here (otherwise we skip a question: Q1 ack would jump to Q3).

            # Check if already complete (last answered was the final question)
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

            # Premium: deterministic pressure state (may influence follow-up generation)
            pressure_state = None
            try:
                from app.config import settings
                if bool(getattr(settings, "enable_adaptive_pressure", False)):
                    from app.services.practice.adaptive_pressure import adjust_difficulty, compute_pressure_state

                    pressure_state = compute_pressure_state(session=session)
                    effective = adjust_difficulty(base=getattr(session, "difficulty", None), mode=str(pressure_state.get("mode")))
                else:
                    effective = getattr(session, "difficulty", None)
            except Exception:
                effective = getattr(session, "difficulty", None)
                pressure_state = None

            strategy_result: Optional[dict] = None
            if bool(getattr(settings, "enable_practice_session_brain", True)):
                last_answer = self._find_answer_for_question(session, question_id)
                previous_question = session.questions[question_id - 1]
                decision = getattr(session, "pending_strategy_decision", None)

                if last_answer is not None:
                    if decision is None:
                        decision = await self._preview_strategy_decision(
                            session=session,
                            previous_question=previous_question,
                            last_answer=last_answer,
                            api_key=api_key,
                        )
                    if decision is not None:
                        try:
                            strategy_result = await self._get_session_strategy_agent().execute_next_action(
                                session=session,
                                previous_question=previous_question,
                                last_answer=last_answer,
                                decision=decision,
                                api_key=api_key,
                            )
                            session.pending_strategy_decision = None
                            session.current_coaching_style = decision.coaching_style
                            session.strategy_history.append(decision)
                        except Exception as exc:
                            session.pending_strategy_decision = None
                            logger.warning("Session brain execution failed, falling back to legacy next-question flow: %s", exc)
                            strategy_result = None

            if strategy_result is not None:
                if strategy_result.get("complete"):
                    session.pending_next_question = False
                    return {
                        "complete": True,
                        "evaluation_report": strategy_result.get("evaluation_report") or session.evaluation_report,
                        "progress": strategy_result.get("progress") or self.interviewer_agent.get_progress_indicator(
                            question_id,
                            len(session.questions)
                        ),
                        "pressure": pressure_state,
                        "strategy": strategy_result.get("strategy"),
                    }

                next_question = strategy_result.get("next_question")
                if next_question is None:
                    raise ValueError("No next question available")

                audio_filename = None
                try:
                    audio_path = str(self.audio_dir / f"{session_id}_q{next_question.id}.wav")
                    formatted_text = self.interviewer_agent.format_tts_text(
                        next_question,
                        total_questions=len(session.questions),
                        company_name=session.company_name
                    )
                    result_path = await self.tts_service.synthesize_async(formatted_text, audio_path)
                    audio_filename = Path(result_path).name
                    if audio_filename:
                        session.audio_files.append(audio_filename)
                    logger.info(f"✅ TTS generated after strategy execution for Q{next_question.id}")
                except Exception as e:
                    logger.warning(f"TTS failed for Q{next_question.id}: {e}")
                    audio_filename = None

                session.pending_next_question = False
                return {
                    "next_question": next_question,
                    "tts_audio_url": audio_filename,
                    "complete": False,
                    "pressure": pressure_state,
                    "strategy": strategy_result.get("strategy"),
                    "progress": strategy_result.get("progress") or self.interviewer_agent.get_progress_indicator(
                        next_question.id,
                        len(session.questions)
                    ),
                }

            # 🚀 ADAPTIVE FOLLOW-UP (skipped when the answer was very poor):
            # Replace the upcoming question with a drill-down follow-up framed from
            # the candidate's *last* answer (transcript + micro_feedback). This makes
            # the practice feel like a real interviewer (even when a topic/profile is selected).
            #
            # However, if the candidate answered completely wrong (correctness < 30),
            # drilling deeper on the same topic traps them in a loop.  Instead we
            # move them forward to a fresh topic so the interview feels progressive.
            try:
                next_index = session.current_question_index + 1
                if 0 <= next_index < len(session.questions):
                    # Find the last answer corresponding to the acknowledged question
                    last_answer = None
                    for a in reversed(getattr(session, "answers", []) or []):
                        if int(getattr(a, "question_id", -1)) == int(question_id):
                            last_answer = a
                            break

                    # Guard: skip follow-up drill if the answer was very poor —
                    # the user clearly doesn't know this topic, move on.
                    _skip_followup = False
                    if last_answer is not None:
                        _mf = getattr(last_answer, "micro_feedback", None)
                        _cs = getattr(_mf, "correctness_score", None) if _mf else None
                        _transcript = (getattr(last_answer, "transcript", "") or "").strip()

                        if _cs is not None and _cs < 30:
                            logger.info(
                                f"Skipping adaptive follow-up: correctness_score={_cs} < 30, "
                                "moving to next pre-generated topic"
                            )
                            _skip_followup = True
                        elif len(_transcript.split()) < 3:
                            logger.info(
                                f"Skipping adaptive follow-up: transcript too short ({len(_transcript.split())} words), "
                                "moving to next pre-generated topic"
                            )
                            _skip_followup = True

                    prev_q = session.questions[question_id - 1]
                    already_asked = [q.text for q in (session.questions or []) if getattr(q, "text", None)]

                    if last_answer is not None and not _skip_followup:
                        follow_up = await self.adaptive_interviewer.generate_follow_up_question(
                            user_profile=session.user_profile,
                            difficulty=effective,
                            round_type=getattr(prev_q, "round_type", None) or session.round_type,
                            previous_question=prev_q,
                            transcript=getattr(last_answer, "transcript", "") or "",
                            micro_feedback=getattr(last_answer, "micro_feedback", None),
                            already_asked=already_asked,
                            target_question_id=next_index + 1,
                            api_key=api_key,
                        )
                        if follow_up is not None and getattr(follow_up, "text", "").strip():
                            # Preserve numbering slot and keep total question count stable
                            session.questions[next_index] = follow_up
                            next_question = follow_up
            except Exception as e:
                # Never block progression; fall back to the existing pre-generated question.
                logger.warning(f"Adaptive follow-up injection failed, using fallback next question: {e}")
            
            if not next_question:
                raise ValueError("No next question available")
            
            # Generate TTS for next question
            audio_filename = None
            try:
                # Use wav as the requested container for pyttsx3 (Windows SAPI); gTTS will return mp3.
                audio_path = str(self.audio_dir / f"{session_id}_q{next_question.id}.wav")
                
                formatted_text = self.interviewer_agent.format_tts_text(
                    next_question,
                    total_questions=len(session.questions),
                    company_name=session.company_name
                )
                
                result_path = await self.tts_service.synthesize_async(formatted_text, audio_path)
                audio_filename = Path(result_path).name
                if audio_filename:
                    session.audio_files.append(audio_filename)
                logger.info(f"✅ TTS generated after acknowledgment for Q{next_question.id}")
            except Exception as e:
                logger.warning(f"TTS failed for Q{next_question.id}: {e}")
                audio_filename = None
            
            # Clear pending flag
            session.pending_next_question = False
            session.pending_strategy_decision = None
            
            return {
                "next_question": next_question,
                "tts_audio_url": audio_filename,
                "complete": False,
                "pressure": pressure_state,
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
    
    async def end_session(self, session_id: str, api_key: Optional[str] = None) -> dict:
        """End a practice interview session early and generate evaluation.

        Can be called at any point — even if not all questions have been answered.
        If the session is already complete, returns the existing evaluation.

        Returns:
            Dict with session summary, per-question feedback (including model_answer),
            skipped questions, and evaluation report.
        """
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        # If not already complete, mark as complete and generate evaluation
        if not session.is_complete:
            session.is_complete = True
            session.completed_at = utcnow()

            if session.answers:
                try:
                    logger.info(
                        f"Generating evaluation for early-ended session {session_id} "
                        f"({len(session.answers)}/{len(session.questions)} answered)"
                    )
                    evaluation_report = await self.evaluation_agent.evaluate_interview(
                        session.answers,
                        session_id,
                        api_key=api_key,
                        questions=session.questions,
                    )
                    self._maybe_add_peer_learning_insight(evaluation_report)
                    session.evaluation_report = evaluation_report
                    logger.info("✅ Early-end evaluation report generated")
                except Exception as e:
                    logger.error(f"❌ Evaluation failed on early end: {e}", exc_info=True)

        # Build per-question feedback list (answered questions only)
        answered = []
        for i, ans in enumerate(session.answers):
            q = session.questions[i] if i < len(session.questions) else None
            entry: dict = {
                "question_number": i + 1,
                "question": q.text if q else f"Question {i + 1}",
                "category": q.category if q else "unknown",
                "difficulty": q.difficulty if q else "medium",
                "user_answer": ans.transcript,
                "model_answer": (
                    getattr(ans.micro_feedback, "model_answer", None)
                    or (q.expected_answer_template if q else None)
                    or ""
                ),
                "correctness_score": getattr(ans.micro_feedback, "correctness_score", None),
                "technical_accuracy": getattr(ans.micro_feedback, "technical_accuracy", None),
                "strengths": getattr(ans.micro_feedback, "strengths", None) or [],
                "improvement_areas": getattr(ans.micro_feedback, "improvement_areas", None) or [],
                "key_points_covered": getattr(ans.micro_feedback, "key_points_covered", None) or [],
                "key_points_missed": getattr(ans.micro_feedback, "key_points_missed", None) or [],
                "speech_metrics": {
                    "wpm": ans.metrics.wpm,
                    "filler_count": ans.metrics.filler_count,
                    "confidence_score": ans.metrics.confidence_score,
                    "duration": ans.metrics.duration,
                },
            }
            answered.append(entry)

        # Skipped questions list
        skipped = []
        for i in range(len(session.answers), len(session.questions)):
            q = session.questions[i]
            skipped.append({
                "question_number": i + 1,
                "question": q.text,
                "category": q.category,
                "difficulty": q.difficulty,
                "time_limit": q.time_limit,
            })

        total_q = len(session.questions)
        answered_count = len(session.answers)

        return {
            "status": "completed",
            "session_id": session_id,
            "started_at": session.started_at.isoformat(),
            "completed_at": session.completed_at.isoformat() if session.completed_at else None,
            "total_questions": total_q,
            "questions_answered": answered_count,
            "ended_early": answered_count < total_q,
            "questions_skipped": total_q - answered_count,
            "round_type": session.round_type,
            "difficulty": session.difficulty,
            "evaluations": answered,
            "skipped_questions": skipped,
            "evaluation_report": session.evaluation_report,
        }
    
    async def cleanup_session(self, session_id: str):
        """Clean up session and audio files.

        Before deleting the session from memory, attempt to persist progress
        to the database so that answered questions are never silently lost
        (e.g. when a user closes the tab without clicking 'End Session').
        """
        try:
            session = self.sessions.get(session_id)
            if not session:
                return

            # --- Best-effort persist before eviction ---
            if session.answers:
                try:
                    # Mark as complete if not already (abandoned sessions)
                    if not session.is_complete:
                        session.is_complete = True
                        session.completed_at = utcnow()

                    owner = getattr(session, "user_id", None) or "guest_unknown"

                    from app.database import get_db_context
                    from app.models import PracticeAttemptRecord
                    from app.services.practice.practice_progress import save_completed_attempt

                    with get_db_context() as db:
                        already = (
                            db.query(PracticeAttemptRecord)
                            .filter(PracticeAttemptRecord.session_id == session_id)
                            .first()
                        )
                        if not already:
                            aid = save_completed_attempt(db, user_id=owner, session=session)
                            logger.info(f"[cleanup-persist] ✅ Saved attempt id={aid} for session {session_id}")
                        else:
                            logger.debug(f"[cleanup-persist] Already persisted session {session_id}")
                except Exception as pe:
                    logger.warning(f"[cleanup-persist] Could not persist session {session_id}: {pe}")

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
            now = utcnow()
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
