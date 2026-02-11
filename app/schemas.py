from pydantic import BaseModel, Field, AliasChoices
from pydantic import ConfigDict
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime

from app.utils.time import utcnow


class CreateSessionResponse(BaseModel):
	session_id: str


class UpdateSessionTitleRequest(BaseModel):
	title: str = Field(..., min_length=1, max_length=100)


class QuestionIn(BaseModel):
	session_id: str = Field(..., description="Session identifier")
	question: str = Field(..., min_length=1)
	system_prompt: Optional[str] = Field(default=None, description="Override default system role text")
	stream: Optional[bool] = Field(default=False, description="Hint to stream on supported endpoints")
	# Copilot mode selection
	mode: Optional[str] = Field(
		default="answer",
		description="Copilot mode: answer|mirror (mirror analyzes user's answer instead of answering directly)",
	)
	# Mirror mode input (user answers in their own words; the system analyzes gaps)
	user_answer: Optional[str] = Field(
		default=None,
		description="In mirror mode: the user's draft answer to analyze",
	)
	# Response depth (optional UI-controlled knob)
	depth: Optional[str] = Field(
		default=None,
		description="Response depth: quick|standard|deep (overrides auto inference)",
	)
	# Architecture mode selection for system design questions
	architecture_mode: Optional[str] = Field(default=None, description="Architecture generation mode: 'single' for comprehensive diagram, 'multi-view' for focused layers")
	# Style customization (optional)
	style_mode: Optional[str] = Field(default="auto", description="Response style preset: auto|varied|concise|deep-dive|mentor|executive|faq|qa|checklist|narrative")
	tone: Optional[str] = Field(default=None, description="Desired tone: neutral|friendly|mentor|executive|academic|coaching")
	layout: Optional[str] = Field(default=None, description="Preferred layout: bullets|narrative|qa|faq|checklist|pros-cons|steps")
	variability: Optional[float] = Field(default=0.5, ge=0.0, le=1.0, description="0–1; higher = more variety in tone/layout")
	seed: Optional[int] = Field(default=None, description="Optional seed to make style variation deterministic")
	save_to_history: Optional[bool] = Field(default=True, description="Whether to save this interaction to session history")


class AnswerOut(BaseModel):
	answer: str
	session_id: Optional[str] = Field(
		default=None,
		description="Session identifier (echoed for UI/session routing).",
	)
	mode: Optional[str] = Field(
		default=None,
		description="Effective chat mode for this response (e.g., 'answer' or 'mirror').",
	)
	created_at: datetime
	truncated: bool = False  # True if response was truncated by token limits
	# Optional UI hints (backward compatible)
	ui_action: Optional[str] = Field(
		default=None,
		description="Optional client UI action hint (e.g., 'choose_architecture_mode').",
	)
	ui_payload: Optional[Dict[str, Any]] = Field(
		default=None,
		description="Optional UI payload for the ui_action.",
	)


# ===== Mirror Mode Schemas =====


class MirrorReport(BaseModel):
	"""Validated schema for Interview Mirror structured output.

	This is used internally to ensure we never crash or emit malformed reports
	when the model drifts (extra keys, wrong types, bad confidence values).
	"""

	model_config = ConfigDict(extra="forbid")

	topic: str = Field(default="General", min_length=1, max_length=120)
	message: str = Field(default="", max_length=600)
	strengths: List[str] = Field(default_factory=list, max_length=3)
	gaps: List[str] = Field(default_factory=list, max_length=5)
	red_flags: List[str] = Field(default_factory=list, max_length=3)
	likely_followups: List[str] = Field(default_factory=list, max_length=3)
	upgrade_lines: List[str] = Field(default_factory=list, max_length=2)
	confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class MirrorFeedbackIn(BaseModel):
	"""User feedback on a generated Mirror report.

	This enables building a quality dataset and calibrating heuristics over time.
	"""

	session_id: str = Field(..., description="Session identifier")
	question: str = Field(..., min_length=1)
	user_answer: Optional[str] = Field(default=None, description="The user's draft answer that was analyzed")
	report: Optional[Dict[str, Any]] = Field(default=None, description="Optional structured report snapshot")
	helpful: bool = Field(..., description="Whether the report was helpful")
	flags: List[str] = Field(default_factory=list, description="Optional tags like: too_harsh, incorrect, generic, helpful_upgrade_lines")
	comment: Optional[str] = Field(default=None, max_length=500, description="Optional free-text feedback")


class MirrorFeedbackOut(BaseModel):
	ok: bool = True


class QnA(BaseModel):
	question: str
	answer: str
	created_at: datetime


class SessionHistory(BaseModel):
	session_id: str
	items: List[QnA]


class SessionSummary(BaseModel):
    session_id: str
    title: str
    last_update: datetime
    qna_count: int


class SessionList(BaseModel):
    items: List[SessionSummary]


class EvaluationIn(BaseModel):
	"""Request for one-click evaluation of a coding attempt.

	- session_id: existing session to attach evaluation context
	- problem: short problem name or prompt (optional but recommended)
	- code: candidate's solution source code
	- language: programming language hint (default: python)
	"""
	session_id: str = Field(..., description="Session identifier")
	problem: Optional[str] = Field(default=None, description="Problem title or prompt")
	code: str = Field(..., min_length=1, description="Source code to evaluate")
	language: Optional[str] = Field(default="python", description="Code language: python|js|ts|java|cpp|go ...")


class EvaluationScores(BaseModel):
	correctness: float = Field(..., ge=0.0, le=10.0)
	optimization: float = Field(..., ge=0.0, le=10.0)
	approach_explanation: float = Field(..., ge=0.0, le=10.0)
	complexity_discussion: float = Field(..., ge=0.0, le=10.0)
	edge_cases_testing: float = Field(..., ge=0.0, le=10.0)
	total: float = Field(..., ge=0.0, le=10.0)


class StaticSignals(BaseModel):
	uses_recursion: bool
	uses_memoization: bool
	uses_dynamic_programming: bool
	loop_nesting_depth: int
	uses_slicing_heavily: bool
	uses_list_or_set_comprehension: bool
	function_count: int
	comment_density: float
	estimated_time_complexity_hint: Optional[str] = None


class EvaluationOut(BaseModel):
	session_id: str
	problem: Optional[str]
	language: Optional[str]
	approach_auto_explanation: str
	feedback_summary: str
	strengths: List[str]
	weaknesses: List[str]
	scores: EvaluationScores
	static_signals: StaticSignals
	recommendations: List[str]
	created_at: datetime
	markdown: Optional[str] = None


# Interview Intelligence Schemas (ENHANCED for modern service)
class InterviewQuestion(BaseModel):
	"""A single interview question with answer and metadata."""
	question: str = Field(..., description="The interview question")
	answer: str = Field(..., description="Comprehensive answer with explanations")
	source: str = Field(..., description="Source URL or identifier")
	updated_at: str = Field(..., description="ISO timestamp of last update")
	
	# Existing optional fields (backward compatible)
	topic: Optional[str] = Field(default=None, description="Topic category (e.g., python, data-science)")
	code_solution: Optional[str] = Field(default=None, description="Code solution for coding questions")
	language: Optional[str] = Field(default=None, description="Programming language for code solution")
	is_coding_question: Optional[bool] = Field(default=False, description="Whether this is a coding question")
	
	# NEW: Enhanced metadata from modern service (all optional for backward compatibility)
	difficulty: Optional[str] = Field(default=None, description="Difficulty level: easy, medium, hard")
	question_type: Optional[str] = Field(default=None, description="Type: coding, behavioral, system-design, technical")
	key_concepts: Optional[List[str]] = Field(default=None, description="Key concepts tested in this question")
	common_mistakes: Optional[List[str]] = Field(default=None, description="Common mistakes candidates make")
	follow_up_questions: Optional[List[str]] = Field(default=None, description="Related follow-up questions")
	time_complexity: Optional[str] = Field(default=None, description="Time complexity analysis (for coding questions)")
	space_complexity: Optional[str] = Field(default=None, description="Space complexity analysis (for coding questions)")
	companies: Optional[List[str]] = Field(default=None, description="Companies known to ask this question")


class InterviewQuestionsResponse(BaseModel):
	"""Response containing a collection of interview questions for a topic."""
	topic: str = Field(..., description="The requested topic")
	questions: List[InterviewQuestion] = Field(..., description="List of interview questions")
	count: int = Field(..., description="Number of questions returned")
	message: Optional[str] = Field(default=None, description="Optional status message")


class TopicListResponse(BaseModel):
	"""Response containing list of available topics."""
	topics: List[str] = Field(..., description="List of available topic names")


class InterviewSearchRequest(BaseModel):
	"""Request payload for searching interview questions."""
	query: str = Field(..., description="Search query to find relevant interview questions")
	limit: Optional[int] = Field(default=20, ge=1, le=50, description="Maximum number of results to return")
	refresh: Optional[bool] = Field(default=False, description="If true, bypass cache and generate fresh results")
	save_to_history: Optional[bool] = Field(default=True, description="If true, save this search to history")


class SearchQuestionsResponse(BaseModel):
	"""Response containing search results for interview questions."""
	query: str = Field(..., description="The search query")
	questions: List[InterviewQuestion] = Field(..., description="Matching interview questions")
	count: int = Field(..., description="Number of results returned")


# ===== Practice Mode Schemas =====
from enum import Enum


class QuestionDifficulty(str, Enum):
	"""Question difficulty levels."""
	EASY = "easy"
	MEDIUM = "medium"
	HARD = "hard"


class QuestionType(str, Enum):
	"""Question delivery and response type."""
	VOICE = "voice"              # Voice-based Q&A (behavioral, verbal technical)
	CODING = "coding"            # Code editor with execution (write actual code)
	SYSTEM_DESIGN = "system_design"  # Whiteboard/diagram-based


class InterviewRound(str, Enum):
	"""Interview round types - mirrors real company interview processes."""
	HR_SCREENING = "hr_screening"
	TECHNICAL_ROUND_1 = "technical_round_1"
	TECHNICAL_ROUND_2 = "technical_round_2"
	SYSTEM_DESIGN = "system_design"
	BEHAVIORAL = "behavioral"
	MANAGERIAL = "managerial"
	MACHINE_LEARNING = "machine_learning"
	DATA_ENGINEERING = "data_engineering"
	FRONTEND_SPECIALIST = "frontend_specialist"
	BACKEND_SPECIALIST = "backend_specialist"
	DEVOPS = "devops"
	SECURITY = "security"
	FULL_INTERVIEW = "full_interview"


class ProctoringEventType(str, Enum):
	"""Strict, opt-in proctoring signals for Practice Mode.

	Privacy contract:
	- Event-only (no frames/audio by default)
	- Client initiates camera access; backend only logs signals
	"""
	CAMERA_STARTED = "camera_started"
	CAMERA_STOPPED = "camera_stopped"
	CAMERA_HEARTBEAT = "camera_heartbeat"
	TAB_SWITCH = "tab_switch"
	WINDOW_BLUR = "window_blur"
	FACE_MISSING = "face_missing"
	MULTIPLE_FACES = "multiple_faces"
	USER_LEFT_FRAME = "user_left_frame"


class ProctoringEventIn(BaseModel):
	"""Ingest a proctoring signal for an active practice session."""
	session_id: str = Field(..., description="Practice session identifier")
	event_type: ProctoringEventType = Field(..., description="Type of proctoring signal")
	severity: Literal["info", "warning", "violation"] = Field(
		default="info",
		description="UI/analytics severity: info|warning|violation",
	)
	metadata: Dict[str, Any] = Field(default_factory=dict, description="Small, non-sensitive structured details")
	client_timestamp: Optional[datetime] = Field(
		default=None,
		description="Optional client-side timestamp for correlation (server will still record its own timestamp)",
	)


class ProctoringEventOut(BaseModel):
	ok: bool = True


class PracticeSessionMediaType(str, Enum):
	"""Media recording type for Practice sessions."""
	SCREEN = "screen"
	CAMERA = "camera"
	COMBINED = "combined"


class PracticeProctoringEventType(str, Enum):
	"""DB-backed proctoring event types (MVP)."""
	SCREEN_STOPPED = "SCREEN_STOPPED"
	CAMERA_STOPPED = "CAMERA_STOPPED"
	TAB_SWITCH = "TAB_SWITCH"
	WINDOW_MINIMIZED = "WINDOW_MINIMIZED"
	SESSION_STARTED_WITH_PROCTORING = "SESSION_STARTED_WITH_PROCTORING"


class PracticeSessionStartIn(BaseModel):
	"""Gate that enforces required media permissions before a live practice starts."""
	screen_shared: bool = Field(..., description="Client confirmed entire screen is being shared")
	camera_enabled: bool = Field(..., description="Client confirmed camera is enabled")


class PracticeSessionStartOut(BaseModel):
	ok: bool = True


class PracticeSessionMediaOut(BaseModel):
	media_id: int
	session_id: str
	media_type: PracticeSessionMediaType
	storage_url: str
	duration_seconds: Optional[int] = None


class PracticeSessionProctoringEventIn(BaseModel):
	event_type: PracticeProctoringEventType
	metadata: Dict[str, Any] = Field(default_factory=dict)
	client_timestamp: Optional[datetime] = Field(default=None)


class PracticeSessionProctoringEventOut(BaseModel):
	ok: bool = True


class RoundConfig(BaseModel):
	"""Configuration for a specific interview round."""
	round_type: InterviewRound = Field(..., description="Type of interview round")
	name: str = Field(..., description="Display name (e.g., 'Technical Round 1')")
	description: str = Field(..., description="What this round focuses on")
	duration_minutes: int = Field(..., description="Typical duration in minutes")
	question_count: int = Field(..., description="Number of questions in this round")
	difficulty: QuestionDifficulty = Field(..., description="Base difficulty level")
	question_time_limit: int = Field(..., description="Average time per question in seconds")
	categories: List[str] = Field(..., description="Question categories for this round")


class UserProfile(BaseModel):
	"""User profile for adaptive interview customization."""
	domain: str = Field(..., description="Primary domain (e.g., 'Python Backend', 'Data Science', 'Frontend React')")
	experience_years: int = Field(..., ge=0, le=50, description="Years of experience (0-50)")
	skills: List[str] = Field(..., min_length=1, description="List of skills (e.g., ['Python', 'AWS', 'Docker'])")
	job_role: Optional[str] = Field(default=None, description="Target job role (e.g., 'Senior Engineer', 'Tech Lead')")
	company_preference: Optional[str] = Field(default=None, description="Target company (e.g., 'Google', 'Meta', 'Amazon', 'Microsoft', 'Netflix', 'Apple', 'Startup', 'Enterprise')")
	interview_focus: Optional[List[str]] = Field(default=None, description="Specific areas to focus on")
	target_round: Optional[InterviewRound] = Field(default=None, description="Specific interview round to practice")


class PracticeInterviewQuestion(BaseModel):
	"""Individual interview question for practice mode."""
	id: int = Field(..., description="Question sequence number (1-based, e.g., 1, 2, 3, ...)")
	text: str = Field(..., description="Question text")
	difficulty: QuestionDifficulty = Field(..., description="Question difficulty")
	time_limit: int = Field(default=90, description="Time limit in seconds")
	category: str = Field(..., description="Question category (e.g., behavioral, technical)")
	
	# NEW: Question type determines UI/UX
	question_type: QuestionType = Field(default=QuestionType.VOICE, description="How the question is answered (voice/coding/whiteboard)")
	
	# Coding question specific fields
	programming_language: Optional[str] = Field(default=None, description="For coding questions: Python, JavaScript, SQL, etc.")
	code_template: Optional[str] = Field(default=None, description="For coding questions: starter code template")
	test_cases: Optional[List[Dict[str, Any]]] = Field(default=None, description="For coding questions: input/output test cases")
	
	# Answer evaluation criteria
	key_points: Optional[List[str]] = Field(default=None, description="Key concepts that should be covered")
	expected_answer_template: Optional[str] = Field(default=None, description="Template/guideline for ideal answer")
	evaluation_criteria: Optional[List[str]] = Field(default=None, description="What to look for when evaluating")
	round_type: Optional[InterviewRound] = Field(default=None, description="Which interview round this question belongs to")


class SpeechMetrics(BaseModel):
	"""Speech analytics metrics extracted from audio."""
	filler_count: int = Field(..., description="Number of filler words detected")
	wpm: float = Field(..., description="Words per minute")
	longest_silence: float = Field(..., description="Longest pause duration in seconds")
	confidence_score: float = Field(..., ge=0, le=10, description="Confidence score based on pitch variance (1-10)")
	overtalked: bool = Field(..., description="Whether answer exceeded time limit by >10%")
	duration: float = Field(..., description="Total duration in seconds")
	filler_words: List[str] = Field(default_factory=list, description="List of detected filler words")
	pause_count: int = Field(default=0, description="Number of significant pauses (>2s)")
	pitch_variance: float = Field(default=0.0, description="Raw pitch variance value")
	silence_removed: Optional[float] = Field(default=None, description="Seconds of silence removed by VAD filter")


class MicroFeedback(BaseModel):
	"""Micro-feedback provided after each answer with AI-powered correctness evaluation."""
	model_config = {"protected_namespaces": ()}

	# Delivery Feedback (existing)
	delivery_tips: List[str] = Field(..., max_length=2, description="1-2 short delivery tips")
	pace_feedback: str = Field(..., description="Speaking pace feedback")
	overall_note: str = Field(..., description="Overall feedback note")
	speech_quality: Optional[str] = Field(default=None, description="Speech quality assessment")
	
	# Content Evaluation (NEW - AI-powered)
	correctness_score: Optional[int] = Field(default=None, ge=0, le=100, description="Answer correctness (0-100%)")
	technical_accuracy: Optional[str] = Field(default=None, description="Technical accuracy assessment (Excellent/Good/Fair/Poor)")
	key_points_covered: Optional[List[str]] = Field(default=None, description="Key concepts mentioned correctly")
	key_points_missed: Optional[List[str]] = Field(default=None, description="Important points not mentioned")
	strengths: Optional[List[str]] = Field(default=None, max_length=2, description="What was good in the answer")
	improvement_areas: Optional[List[str]] = Field(default=None, max_length=2, description="What could be better")
	actionable_suggestions: Optional[List[str]] = Field(default=None, max_length=2, description="Specific tips to improve")
	is_correct: Optional[bool] = Field(default=None, description="Overall correctness (true if score >= 70%)")
	
	# Model / ideal answer so the user can compare their answer
	model_answer: Optional[str] = Field(default=None, description="Ideal/model answer to compare against")
	
	# Deprecated field (keep for backward compatibility)
	content_relevance: Optional[str] = Field(default=None, description="[Deprecated] Use correctness_score instead")
	timestamp: datetime = Field(default_factory=utcnow)


class AnswerSubmission(BaseModel):
	"""Submitted answer with transcript and metrics."""
	question_id: int = Field(..., description="Question ID")
	transcript: str = Field(..., description="Transcribed answer text")
	metrics: SpeechMetrics = Field(..., description="Speech analytics")
	micro_feedback: MicroFeedback = Field(..., description="Immediate feedback")
	audio_duration: float = Field(..., description="Audio duration in seconds")
	submitted_at: datetime = Field(default_factory=utcnow)


class EvaluationStrengths(BaseModel):
	"""Strengths identified in the evaluation."""
	items: List[str] = Field(..., min_length=2, max_length=3, description="2-3 specific strengths")


class EvaluationImprovements(BaseModel):
	"""Areas to improve identified in the evaluation."""
	items: List[str] = Field(..., min_length=2, max_length=3, description="2-3 specific improvement areas")


class MetricsSummary(BaseModel):
	"""Aggregated metrics summary."""
	total_fillers: int = Field(..., description="Total filler words across all answers")
	avg_wpm: float = Field(..., description="Average words per minute")
	longest_pause: float = Field(..., description="Longest pause duration")
	avg_confidence: float = Field(..., description="Average confidence score")
	total_duration: float = Field(..., description="Total interview duration in seconds")
	overtalked_count: int = Field(default=0, description="Number of questions where overtalking occurred")


class ActionPlan(BaseModel):
	"""Action plan for improvement."""
	steps: List[str] = Field(..., min_length=2, max_length=3, description="2-3 concrete action steps")


class EvaluationReport(BaseModel):
	"""Final evaluation report generated by Gemini Pro."""
	strengths: EvaluationStrengths = Field(..., description="Identified strengths")
	improvements: EvaluationImprovements = Field(..., description="Areas to improve")
	metrics_summary: MetricsSummary = Field(..., description="Aggregated metrics")
	action_plan: ActionPlan = Field(..., description="Concrete action steps")
	practice_recommendation: str = Field(..., description="Estimated practice sessions needed")
	# Optional, privacy-safe cohort insight (only when enabled)
	learning_insight: Optional[str] = Field(default=None, description="Optional peer benchmark insight")
	generated_at: datetime = Field(default_factory=utcnow)


class PracticeConfidenceOutcomeIn(BaseModel):
	"""Self-reported confidence after a completed practice session (1-5)."""
	confidence_1_5: int = Field(..., ge=1, le=5, description="Self-reported confidence (1-5)")


class PracticeConfidenceOutcomeOut(BaseModel):
	ok: bool = True
	session_id: str
	confidence_1_5: int


class PracticeSession(BaseModel):
	"""Complete practice interview session."""
	session_id: str = Field(..., description="Unique session identifier")
	# Optional context for persistence/progress (backward-compatible)
	user_profile: Optional[UserProfile] = Field(default=None, description="User profile used to generate questions")
	difficulty: Optional[QuestionDifficulty] = Field(default=None, description="Difficulty level used for this session")
	round_type: Optional[InterviewRound] = Field(default=None, description="Interview round practiced (if any)")
	started_at: datetime = Field(default_factory=utcnow)
	last_activity_at: datetime = Field(default_factory=utcnow)
	completed_at: Optional[datetime] = None
	current_question_index: int = Field(default=0, description="Current question index (0-4)")
	questions: List[PracticeInterviewQuestion] = Field(default_factory=list)
	company_name: Optional[str] = Field(default=None, description="Target company if applicable")
	answers: List[AnswerSubmission] = Field(default_factory=list)
	evaluation_report: Optional[EvaluationReport] = None
	is_complete: bool = Field(default=False)
	audio_files: List[str] = Field(default_factory=list, description="Generated TTS audio filenames")
	pending_next_question: bool = Field(default=False, description="True if waiting for feedback acknowledgment")


# API Request/Response Models for Practice Mode
class StartInterviewRequest(BaseModel):
	"""Request to start a new practice interview."""
	# MVP: enforce required proctoring permissions at session start
	screen_shared: bool = Field(..., description="Client confirmed entire screen is being shared")
	camera_enabled: bool = Field(..., description="Client confirmed camera is enabled")
	difficulty: QuestionDifficulty = Field(default=QuestionDifficulty.MEDIUM)
	category: str = Field(default="behavioral", description="Interview category")
	question_count: int = Field(default=5, ge=1, le=10, description="Number of questions (1-10, default: 5)")
	# NEW: User profile for adaptive interviews
	user_profile: Optional[UserProfile] = Field(default=None, description="User profile for intelligent question generation")
	# NEW: Round-based interview
	round_type: Optional[InterviewRound] = Field(default=None, description="Specific interview round to practice")


class RoundSelectionRequest(BaseModel):
	"""Request to start a round-based interview."""
	# MVP: enforce required proctoring permissions at session start
	screen_shared: bool = Field(..., description="Client confirmed entire screen is being shared")
	camera_enabled: bool = Field(..., description="Client confirmed camera is enabled")
	round_type: InterviewRound = Field(..., description="Interview round to practice")
	domain: str = Field(..., description="Primary domain (REQUIRED): e.g., 'Python', 'Java', 'Data Engineering', 'Machine Learning', 'Frontend', 'DevOps'")
	experience_years: int = Field(default=2, ge=0, le=30, description="Years of experience in the domain")
	question_count: Optional[int] = Field(default=None, ge=1, le=15, description="Number of questions (1-15). If not provided, uses round default.")
	user_profile: Optional[UserProfile] = Field(default=None, description="Optional: Complete user profile (overrides domain/experience if provided)")
	company_specific: Optional[str] = Field(default=None, description="Optional: Make it company-specific (e.g., 'Google', 'Amazon')")


class RoundSelectionResponse(BaseModel):
	"""Response with available rounds and their configurations."""
	rounds: List[RoundConfig] = Field(..., description="Available interview rounds")
	recommended_round: Optional[InterviewRound] = Field(default=None, description="Recommended round based on user profile")
	recommended_sequence: Optional[List[InterviewRound]] = Field(default=None, description="Suggested round progression")


class QuickStartRequest(BaseModel):
	"""🚀 AI Quick Start - Zero-click conversational onboarding."""
	voice_input: Optional[str] = Field(default=None, description="Voice/text input from user")
	context: Optional[str] = Field(default=None, description="Additional context (resume, LinkedIn, etc.)")
	auto_mode: bool = Field(default=True, description="Let AI decide everything")
	session_memory: bool = Field(default=True, description="Use previous session data")
	question_count: Optional[int] = Field(default=None, ge=3, le=10, description="Number of questions (3-10). If not provided, AI decides.")
	target_company: Optional[str] = Field(default=None, description="Specific target company (e.g., 'Google', 'Meta', 'Amazon', 'Microsoft'). If not provided, AI infers from voice_input.")
	# NEW: Round-based selection
	target_round: Optional[InterviewRound] = Field(default=None, description="Specific interview round to practice")


class ConversationalResponse(BaseModel):
	"""Response from AI conversational agent."""
	ai_message: str = Field(..., description="AI's conversational response")
	needs_clarification: bool = Field(default=False, description="Needs more info from user")
	ready_to_start: bool = Field(default=False, description="Ready to begin interview")
	suggested_profile: Optional[UserProfile] = Field(default=None, description="Inferred user profile")
	session_id: Optional[str] = Field(default=None, description="Session ID if started")
	first_question: Optional[PracticeInterviewQuestion] = Field(default=None, description="First question if started")
	tts_audio_url: Optional[str] = Field(default=None, description="Audio URL if started")
	total_questions: Optional[int] = Field(default=None, description="Total number of questions if started")
	progress: Optional[str] = Field(default=None, description="Progress indicator if started (e.g., '1/5')")


class StartInterviewResponse(BaseModel):
	"""Response after starting an interview."""
	session_id: str
	first_question: PracticeInterviewQuestion
	tts_audio_url: str
	total_questions: int = Field(..., description="Total number of questions in this interview")
	progress: str = Field(..., description="Progress indicator (e.g., '1/5')")
	message: str = "Interview session started successfully"


class SubmitAnswerResponse(BaseModel):
	"""Response after submitting an answer."""
	transcript: str
	metrics: SpeechMetrics
	micro_feedback: MicroFeedback
	next_question: Optional[PracticeInterviewQuestion] = None
	tts_audio_url: Optional[str] = None
	complete: bool = Field(default=False)
	evaluation_report: Optional[EvaluationReport] = None
	# Premium analytics (optional, deterministic when present)
	evaluation_trace: Optional[Dict[str, Any]] = Field(
		default=None,
		description="Deterministic, explainable scoring trace for the session so far",
	)
	trajectory: Optional[Dict[str, Any]] = Field(
		default=None,
		description="Within-session score trajectory (per-answer trend + deltas)",
	)
	pressure: Optional[Dict[str, Any]] = Field(
		default=None,
		description="Adaptive pressure state (deterministic rules; may influence follow-ups when enabled)",
	)
	progress: str = Field(..., description="Progress indicator (e.g., '2/5')")
	# NEW: UI flow control
	requires_acknowledgment: bool = Field(default=True, description="User must acknowledge feedback before proceeding")
	current_question_id: int = Field(..., description="Question ID that was just answered")


class AcknowledgeFeedbackRequest(BaseModel):
	"""Request to acknowledge feedback and get next question."""
	session_id: str = Field(
		..., 
		description="Session identifier",
		validation_alias=AliasChoices("session_id", "sessionId")
	)
	question_id: int = Field(
		..., 
		description="Question ID that was just answered",
		validation_alias=AliasChoices("question_id", "questionId")
	)
	feedback_read: bool = Field(
		default=True, 
		description="Confirmation that user read the feedback",
		validation_alias=AliasChoices("feedback_read", "feedbackRead")
	)


class PracticeFeedbackRatedRequest(BaseModel):
	"""Phase 3: human usefulness/preference labels for practice feedback.

	This is the ground-truth supervision signal: the user explicitly rates how
	useful the feedback was (and optionally how hard the question felt).
	"""

	session_id: str = Field(
		...,
		description="Session identifier",
		validation_alias=AliasChoices("session_id", "sessionId"),
	)
	question_id: int = Field(
		...,
		ge=1,
		description="Question ID the feedback corresponds to",
		validation_alias=AliasChoices("question_id", "questionId"),
	)
	usefulness_rating: int = Field(
		...,
		ge=1,
		le=5,
		description="How useful was the feedback? 1 (not useful) to 5 (very useful)",
		validation_alias=AliasChoices("usefulness_rating", "usefulnessRating", "usefulness"),
	)
	perceived_difficulty: Optional[QuestionDifficulty] = Field(
		default=None,
		description="How hard did this question feel to the user (optional)",
		validation_alias=AliasChoices("perceived_difficulty", "perceivedDifficulty"),
	)
	comment: Optional[str] = Field(
		default=None,
		max_length=1000,
		description="Optional free-text: why this was (not) useful",
	)


class PracticeFeedbackRatedResponse(BaseModel):
	"""Ack response for a rating submission."""
	ok: bool = True


class NextQuestionResponse(BaseModel):
	"""Response with next question after feedback acknowledgment."""
	next_question: Optional[PracticeInterviewQuestion] = Field(None, description="Next question if interview not complete")
	tts_audio_url: Optional[str] = Field(None, description="Audio URL for next question")
	complete: bool = Field(default=False, description="True if interview is complete")
	evaluation_report: Optional[EvaluationReport] = Field(None, description="Final evaluation if complete")
	pressure: Optional[Dict[str, Any]] = Field(
		default=None,
		description="Adaptive pressure state (deterministic rules; may influence follow-ups when enabled)",
	)
	progress: str = Field(..., description="Updated progress (e.g., '3/5')")


class PracticeProgressSummaryResponse(BaseModel):
	"""High-level progress rollup for the flagship practice loop."""
	attempts: int = Field(..., ge=0)
	average_overall_score: Optional[float] = Field(default=None, ge=0.0, le=100.0)
	last_completed_at: Optional[datetime] = None
	best_dimension: Optional[str] = None
	worst_dimension: Optional[str] = None


class PracticeHeatmapPoint(BaseModel):
	"""One cell in a weekly x dimension heatmap."""
	week_start: str = Field(..., description="ISO date (YYYY-MM-DD) for the Monday of the week")
	dimension: str
	avg_score: float = Field(..., ge=0.0, le=100.0)
	attempts: int = Field(..., ge=1)


class PracticeHeatmapResponse(BaseModel):
	points: List[PracticeHeatmapPoint] = Field(default_factory=list)


class PracticeNextSessionRecommendationResponse(BaseModel):
	"""Settings the client can use to start the next targeted session."""
	plan: Optional[Dict[str, Any]] = None


# Code Submission Models (for coding questions)
class SubmitCodeRequest(BaseModel):
	"""Request to submit code for a coding question."""
	session_id: str = Field(..., description="Session identifier", validation_alias=AliasChoices("session_id", "sessionId"))
	question_id: int = Field(..., description="Question ID", validation_alias=AliasChoices("question_id", "questionId"))
	code: str = Field(..., description="User's code solution")
	programming_language: str = Field(..., description="Language: Python, JavaScript, SQL, etc.", validation_alias=AliasChoices("programming_language", "programmingLanguage"))
	time_taken: int = Field(..., description="Time taken in seconds", validation_alias=AliasChoices("time_taken", "timeTaken"))


class CodeTestResult(BaseModel):
	"""Individual test case result."""
	test_case_id: int = Field(..., description="Test case number")
	input_data: str = Field(..., description="Test input")
	expected_output: str = Field(..., description="Expected output")
	actual_output: Optional[str] = Field(None, description="User's code output")
	passed: bool = Field(..., description="Test passed or failed")
	error: Optional[str] = Field(None, description="Error message if failed")


class CodeEvaluationFeedback(BaseModel):
	"""Feedback on code quality and approach."""
	correctness_score: int = Field(..., ge=0, le=100, description="Code correctness (0-100%)")
	approach_quality: str = Field(..., description="Quality of approach: excellent/good/needs_improvement")
	time_complexity: Optional[str] = Field(None, description="Big-O time complexity analysis")
	space_complexity: Optional[str] = Field(None, description="Big-O space complexity analysis")
	strengths: List[str] = Field(default_factory=list, description="What the code does well")
	improvements: List[str] = Field(default_factory=list, description="Areas for improvement")
	best_practices: List[str] = Field(default_factory=list, description="Coding best practices followed/missed")
	alternative_approaches: Optional[str] = Field(None, description="Other ways to solve this problem")


class SubmitCodeResponse(BaseModel):
	"""Response after code submission."""
	test_results: List[CodeTestResult] = Field(..., description="Results from running test cases")
	all_tests_passed: bool = Field(..., description="True if all test cases passed")
	code_feedback: CodeEvaluationFeedback = Field(..., description="AI evaluation of code quality")
	complete: bool = Field(default=False, description="True if interview is complete")
	next_question: Optional[PracticeInterviewQuestion] = Field(None, description="Next question if not complete")
	evaluation_report: Optional[EvaluationReport] = Field(None, description="Final evaluation if complete")
	progress: str = Field(..., description="Progress indicator")
	requires_acknowledgment: bool = Field(default=True, description="User must acknowledge feedback")


# ------------------------------ Code execution (backend) ------------------------------
class CodeExecutionTestCase(BaseModel):
	input: str = Field(default="", description="stdin input")
	expected_output: Optional[str] = Field(default=None, description="expected stdout for pass/fail (optional)")


class CodeExecutionIn(BaseModel):
	language: str = Field(..., min_length=1, max_length=32)
	code: str = Field(..., min_length=1, max_length=20000)
	stdin: Optional[str] = Field(default="", max_length=8000)
	test_cases: Optional[List[CodeExecutionTestCase]] = Field(default=None, description="Optional test cases")
	trace: bool = Field(default=False, description="If true, return step-by-step trace events (Python only)")
	explain_trace: bool = Field(
		default=False,
		description="If true and trace=true, include a short explanation for each executed line",
	)
	trace_max_events: int = Field(
		default=2000,
		ge=1,
		le=10000,
		description="Max trace events to return when trace=true (Python only)",
	)
	explain_max_lines: int = Field(
		default=200,
		ge=1,
		le=2000,
		description="Max distinct line explanations to return when explain_trace=true",
	)
	store_code: bool = Field(default=False, description="If true, allow storing code in telemetry (not recommended)")


class CodeExecutionTraceEvent(BaseModel):
	step: int
	line: int
	event: str = Field(default="line")
	locals: Optional[Dict[str, str]] = None
	explanation: Optional[str] = None


class CodeExecutionTestResult(BaseModel):
	input: str
	expected_output: Optional[str] = None
	actual_output: Optional[str] = None
	passed: bool
	error: Optional[str] = None


class CodeExecutionOut(BaseModel):
	success: bool
	status: Optional[str] = None
	stdout: str = ""
	stderr: str = ""
	time_seconds: Optional[float] = None
	memory_kb: Optional[int] = None
	test_results: Optional[List[CodeExecutionTestResult]] = None
	trace_events: Optional[List[CodeExecutionTraceEvent]] = None
	line_explanations: Optional[Dict[int, str]] = None


# Configuration Models for Practice Mode
class TTSConfig(BaseModel):
	"""TTS service configuration."""
	model_config = {"protected_namespaces": ()}
	
	engine: str = Field(default="coqui", description="TTS engine: 'coqui' or 'piper'")
	tts_model_name: str = Field(default="tts_models/en/ljspeech/tacotron2-DDC", description="TTS model name")
	sample_rate: int = Field(default=22050)
	max_generation_time: float = Field(default=2.0, description="Max TTS generation time in seconds")


class STTConfig(BaseModel):
	"""STT service configuration."""
	model_config = {"protected_namespaces": ()}
	
	stt_model_size: str = Field(default="base", description="faster-whisper model size")
	device: str = Field(default="cpu", description="Device: 'cpu' or 'cuda'")
	compute_type: str = Field(default="int8", description="Compute type for faster-whisper")
	max_transcription_time: float = Field(default=3.0, description="Max STT time in seconds")
	# Real-time factor (seconds compute / seconds audio). Lower is faster.
	# Default is realistic for CPU 'base' models; tune via env if needed.
	target_rtf: float = Field(default=0.35, description="Warn if transcription is slower than this real-time factor")


class SpeechAnalyticsConfig(BaseModel):
	"""Speech analytics configuration."""
	filler_words: List[str] = Field(
		default_factory=lambda: [
			"um", "uh", "like", "you know", "basically", 
			"actually", "sort of", "kind of", "literally", "so"
		]
	)
	significant_pause_threshold: float = Field(default=2.0, description="Pause threshold in seconds")
	silence_top_db: int = Field(default=40, description="Silence detection threshold")
	overtalk_threshold: float = Field(default=1.1, description="Overtalk multiplier (110%)")
	ideal_wpm_min: int = Field(default=140, description="Ideal WPM minimum")
	ideal_wpm_max: int = Field(default=160, description="Ideal WPM maximum")


class PracticeModeConfig(BaseModel):
	"""Overall practice mode configuration."""
	tts: TTSConfig = Field(default_factory=TTSConfig)
	stt: STTConfig = Field(default_factory=STTConfig)
	analytics: SpeechAnalyticsConfig = Field(default_factory=SpeechAnalyticsConfig)
	audio_storage_path: str = Field(default="data/practice_audio")
	session_timeout_minutes: int = Field(default=30)
	max_concurrent_sessions: int = Field(default=100)


# ===== Multi-View Architecture Generation Schemas =====

class ArchitectureViewType(str, Enum):
	"""Types of architecture views - each tells one story."""
	SYSTEM_OVERVIEW = "system_overview"
	REQUEST_FLOW = "request_flow"
	ASYNC_PROCESSING = "async_processing"
	DATA_MODEL = "data_model"
	DEPLOYMENT = "deployment"
	OBSERVABILITY = "observability"
	SECURITY = "security"


class DiagramStyle(str, Enum):
	"""Visual style presets for diagrams."""
	MODERN = "modern"
	MINIMAL = "minimal"
	DETAILED = "detailed"


class ArchitectureViewOut(BaseModel):
	"""A single architectural view with its diagram and metadata."""
	view_type: ArchitectureViewType = Field(..., description="Type of architectural view")
	title: str = Field(..., description="Human-readable title")
	description: str = Field(..., description="What this view explains")
	mermaid_code: str = Field(..., description="Mermaid diagram code")
	key_insights: List[str] = Field(default_factory=list, description="3-5 key takeaways")
	complexity_level: str = Field(..., description="junior|mid|senior|architect")
	estimated_explanation_time: str = Field(..., description="How long to explain (e.g., '2 min')")
	audience: str = Field(..., description="Who this view is for")
	key_question: str = Field(..., description="The question this view answers")


class GenerateArchitectureRequest(BaseModel):
	"""Request to generate multi-view architecture."""
	system_description: str = Field(
		..., 
		min_length=10,
		description="Description of the system to design (e.g., 'Event management platform with real-time notifications')"
	)
	user_level: str = Field(
		default="mid",
		description="User expertise level: junior|mid|senior|architect (determines view complexity)"
	)
	specific_views: Optional[List[ArchitectureViewType]] = Field(
		default=None,
		description="Specific views to generate. If None, AI decides based on system description."
	)
	style: DiagramStyle = Field(
		default=DiagramStyle.MODERN,
		description="Visual style for diagrams"
	)
	include_explanations: bool = Field(
		default=True,
		description="Include key insights and explanations for each view"
	)
	session_id: Optional[str] = Field(
		default=None,
		description="Optional session ID to save to history"
	)


class ArchitecturePackageOut(BaseModel):
	"""Complete architecture with multiple coordinated views."""
	system_name: str = Field(..., description="Name of the system being designed")
	description: str = Field(..., description="Brief system description")
	views: List[ArchitectureViewOut] = Field(..., description="All architectural views")
	view_order: List[ArchitectureViewType] = Field(..., description="Recommended viewing order")
	total_views: int = Field(..., description="Number of views generated")
	generated_at: datetime = Field(default_factory=utcnow)
	metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
	
	# Helpful guidance
	how_to_use: str = Field(
		default="Review views in order. Each view answers one specific question about the system.",
		description="Guidance on how to use these views"
	)
	interview_tips: List[str] = Field(
		default_factory=lambda: [
			"Start with System Overview to set context",
			"Use Request Flow to explain user journeys",
			"Reference other views when asked about specific aspects",
			"Each view should take 2-3 minutes to explain"
		],
		description="Tips for using in interviews"
	)


class RenderViewRequest(BaseModel):
	"""Request to render a specific view to SVG."""
	mermaid_code: str = Field(..., description="Mermaid diagram code")
	theme: str = Field(default="default", description="Diagram theme")
	style: DiagramStyle = Field(default=DiagramStyle.MODERN, description="Visual style")
	add_step_numbers: bool = Field(default=True, description="Add step numbers to edges")


class ArchitectureExportRequest(BaseModel):
	"""Request to export architecture package."""
	package: ArchitecturePackageOut = Field(..., description="Architecture package to export")
	format: str = Field(default="markdown", description="Export format: markdown|pdf|html")
	include_diagrams: bool = Field(default=True, description="Include rendered diagrams")
	include_code: bool = Field(default=True, description="Include Mermaid source code")