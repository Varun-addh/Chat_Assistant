from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class CreateSessionResponse(BaseModel):
	session_id: str


class QuestionIn(BaseModel):
	session_id: str = Field(..., description="Session identifier")
	question: str = Field(..., min_length=1)
	system_prompt: Optional[str] = Field(default=None, description="Override default system role text")
	stream: Optional[bool] = Field(default=False, description="Hint to stream on supported endpoints")
	# Style customization (optional)
	style_mode: Optional[str] = Field(default="auto", description="Response style preset: auto|varied|concise|deep-dive|mentor|executive|faq|qa|checklist|narrative")
	tone: Optional[str] = Field(default=None, description="Desired tone: neutral|friendly|mentor|executive|academic|coaching")
	layout: Optional[str] = Field(default=None, description="Preferred layout: bullets|narrative|qa|faq|checklist|pros-cons|steps")
	variability: Optional[float] = Field(default=0.5, ge=0.0, le=1.0, description="0–1; higher = more variety in tone/layout")
	seed: Optional[int] = Field(default=None, description="Optional seed to make style variation deterministic")


class AnswerOut(BaseModel):
	answer: str
	created_at: datetime


class QnA(BaseModel):
	question: str
	answer: str
	created_at: datetime


class SessionHistory(BaseModel):
	session_id: str
	items: List[QnA]


class SessionSummary(BaseModel):
    session_id: str
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
	correctness: float = Field(..., ge=0.0, le=1.0)
	optimization: float = Field(..., ge=0.0, le=1.0)
	approach_explanation: float = Field(..., ge=0.0, le=1.0)
	complexity_discussion: float = Field(..., ge=0.0, le=1.0)
	edge_cases_testing: float = Field(..., ge=0.0, le=1.0)
	total: float = Field(..., ge=0.0, le=1.0)


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


class SearchQuestionsResponse(BaseModel):
	"""Response containing search results for interview questions."""
	query: str = Field(..., description="The search query")
	questions: List[InterviewQuestion] = Field(..., description="Matching interview questions")
	count: int = Field(..., description="Number of results returned")