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
