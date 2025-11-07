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
