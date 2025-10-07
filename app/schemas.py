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
