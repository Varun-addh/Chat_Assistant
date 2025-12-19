from fastapi import APIRouter, HTTPException, Query, Depends, Header
from typing import Optional
from pydantic import BaseModel
import logging

from app.services.mock_interview_service import (
    InterviewType,
    DifficultyLevel,
    UserAnswer,
    InterviewSession,
    EvaluationResult,
)

router = APIRouter()
logger = logging.getLogger(__name__)


# Dependency to get the mock interview service
def get_mock_service():
    """Dependency to ensure mock service is available"""
    from app.services.mock_interview_service import mock_interview_service
    
    if not mock_interview_service:
        logger.error("Mock interview service not available - check main.py lifespan initialization")
        raise HTTPException(
            status_code=503,
            detail="Mock interview service not initialized. Please restart the server."
        )
    
    return mock_interview_service


# Request/Response Models
class StartSessionRequest(BaseModel):
    user_id: str
    interview_type: InterviewType
    difficulty: DifficultyLevel
    num_questions: int = 5
    topic: Optional[str] = None


class StartSessionResponse(BaseModel):
    session_id: str
    first_question: dict
    total_questions: int
    interview_type: str
    difficulty: str


class SubmitAnswerRequest(BaseModel):
    session_id: str
    answer_text: str
    time_taken_seconds: Optional[int] = None
    input_method: str = "text"
    code_solution: Optional[str] = None
    language: Optional[str] = None


class SubmitAnswerResponse(BaseModel):
    evaluation: dict
    next_question: Optional[dict] = None
    is_last_question: bool
    progress: dict


@router.post("/sessions/start", response_model=StartSessionResponse)
async def start_mock_interview(
    request: StartSessionRequest,
    service = Depends(get_mock_service)
):
    """
    Start a new mock interview session
    
    **Example Request:**
    ```json
    {
        "user_id": "user123",
        "interview_type": "coding",
        "difficulty": "medium",
        "num_questions": 5,
        "topic": "python"
    }
    ```
    """
    try:
        # Start session
        session = await service.start_session(
            user_id=request.user_id,
            interview_type=request.interview_type,
            difficulty=request.difficulty,
            num_questions=request.num_questions,
            topic=request.topic
        )
        
        # Get first question
        first_question = await service.get_current_question(session.session_id)
        
        return StartSessionResponse(
            session_id=session.session_id,
            first_question={
                "question_id": first_question.question_id,
                "question_text": first_question.question_text,
                "interview_type": first_question.interview_type,
                "difficulty": first_question.difficulty,
                "topic": first_question.topic,
                "question_number": 1,
                "total_questions": session.total_questions
            },
            total_questions=session.total_questions,
            interview_type=session.interview_type,
            difficulty=session.difficulty
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions/submit-answer", response_model=SubmitAnswerResponse)
async def submit_answer(
    request: SubmitAnswerRequest,
    service = Depends(get_mock_service),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None)
):
    """
    Submit answer and get AI evaluation
    
    **Example Request:**
    ```json
    {
        "session_id": "abc123",
        "answer_text": "To reverse a string in Python...",
        "time_taken_seconds": 120,
        "input_method": "text",
        "code_solution": "def reverse(s): return s[::-1]",
        "language": "python"
    }
    ```
    
    **Response includes:**
    - Detailed evaluation with scores (0-10) for each criterion
    - Strengths and weaknesses
    - Missing points
    - Improvement suggestions
    - Next question (if available)
    """
    try:
        # Extract API key from headers
        api_key = x_api_key
        if not api_key and authorization and authorization.startswith("Bearer "):
            api_key = authorization.removeprefix("Bearer ")

        # Create UserAnswer object
        answer = UserAnswer(
            answer_text=request.answer_text,
            time_taken_seconds=request.time_taken_seconds,
            input_method=request.input_method,
            code_solution=request.code_solution,
            language=request.language
        )
        
        # Submit and evaluate
        evaluation = await service.submit_answer(
            session_id=request.session_id,
            answer=answer,
            api_key=api_key
        )
        
        # Get next question
        next_question_obj = await service.get_current_question(request.session_id)
        
        # Get session for progress
        session = service.active_sessions.get(request.session_id)
        
        next_question = None
        is_last = True
        
        if next_question_obj and session:
            is_last = False
            next_question = {
                "question_id": next_question_obj.question_id,
                "question_text": next_question_obj.question_text,
                "interview_type": next_question_obj.interview_type,
                "difficulty": next_question_obj.difficulty,
                "topic": next_question_obj.topic,
                "question_number": session.current_question_index + 1,
                "total_questions": session.total_questions
            }
        
        progress = {
            "current": session.current_question_index if session else 0,
            "total": session.total_questions if session else 0,
            "percentage": round(
                (session.current_question_index / session.total_questions * 100)
                if session else 0
            )
        }
        
        return SubmitAnswerResponse(
            evaluation={
                "overall_score": evaluation.overall_score,
                "rating_category": evaluation.rating_category,
                "criteria_scores": {
                    "correctness": evaluation.criteria_scores.correctness,
                    "completeness": evaluation.criteria_scores.completeness,
                    "clarity": evaluation.criteria_scores.clarity,
                    "confidence": evaluation.criteria_scores.confidence,
                    "technical_depth": evaluation.criteria_scores.technical_depth
                },
                "performance_summary": evaluation.performance_summary,
                "detailed_feedback": evaluation.detailed_feedback,
                "strengths": evaluation.strengths,
                "weaknesses": evaluation.weaknesses,
                "missing_points": evaluation.missing_points,
                "improvement_suggestions": evaluation.improvement_suggestions,
                "follow_up_questions": evaluation.follow_up_questions,
                "model_answer": evaluation.model_answer,
                # ENHANCED FIELDS
                "detailed_strengths": [
                    {
                        "point": s.point,
                        "explanation": s.explanation,
                        "example": s.example,
                        "impact_level": s.impact_level
                    } for s in evaluation.detailed_strengths
                ] if evaluation.detailed_strengths else [],
                "detailed_weaknesses": [
                    {
                        "point": w.point,
                        "explanation": w.explanation,
                        "example": w.example,
                        "impact_level": w.impact_level,
                        "what_to_add": w.what_to_add
                    } for w in evaluation.detailed_weaknesses
                ] if evaluation.detailed_weaknesses else [],
                "answer_comparisons": [
                    {
                        "aspect": c.aspect,
                        "user_said": c.user_said,
                        "should_say": c.should_say,
                        "gap_explanation": c.gap_explanation
                    } for c in evaluation.answer_comparisons
                ] if evaluation.answer_comparisons else [],
                "recommended_resources": evaluation.recommended_resources,
                "key_takeaways": evaluation.key_takeaways
            },
            next_question=next_question,
            is_last_question=is_last,
            progress=progress
        )
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to submit answer: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}")
async def get_session_status(
    session_id: str,
    service = Depends(get_mock_service)
):
    """Get current session status"""
    try:
        session = service.active_sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        current_question = await service.get_current_question(session_id)
        
        return {
            "session_id": session.session_id,
            "started_at": session.started_at.isoformat(),
            "interview_type": session.interview_type,
            "difficulty": session.difficulty,
            "progress": {
                "current": session.current_question_index,
                "total": session.total_questions,
                "percentage": round(
                    (session.current_question_index / session.total_questions * 100)
                )
            },
            "current_question": {
                "question_text": current_question.question_text,
                "question_number": session.current_question_index + 1
            } if current_question else None,
            "average_score": session.average_score
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}/summary")
async def get_session_summary(
    session_id: str,
    service = Depends(get_mock_service)
):
    """
    Get complete session summary with all evaluations
    
    **Returns:**
    - Overall performance metrics
    - All questions with evaluations
    - Strengths and areas for improvement
    - Recommendations for practice
    """
    try:
        summary = await service.get_session_summary(session_id)
        return summary
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get summary: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions/{session_id}/end")
async def end_session(
    session_id: str,
    service = Depends(get_mock_service)
):
    """End interview session and get final summary"""
    try:
        summary = await service.end_session(session_id)
        
        return {
            "status": "completed",
            "summary": summary
        }
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to end session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    service = Depends(get_mock_service)
):
    """
    Delete a specific interview session
    
    **Use Cases:**
    - Remove a session from history
    - Clean up incomplete sessions
    - Delete practice sessions
    
    **Note:** This action cannot be undone
    """
    try:
        # Check if session exists
        if session_id not in service.active_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Remove from active sessions
        del service.active_sessions[session_id]
        
        # Save to persist the deletion
        service._save_sessions()
        
        logger.info(f"Deleted session {session_id}")
        
        return {
            "status": "success",
            "message": "Session deleted successfully",
            "session_id": session_id
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/history/{user_id}/sessions/{session_id}")
async def delete_user_session(
    user_id: str,
    session_id: str,
    service = Depends(get_mock_service)
):
    """
    Delete a specific session for a user
    
    **Security:** Validates that the session belongs to the specified user
    """
    try:
        # Check if session exists
        session = service.active_sessions.get(session_id)
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Verify ownership
        if session.user_id != user_id:
            raise HTTPException(
                status_code=403, 
                detail="You don't have permission to delete this session"
            )
        
        # Delete the session
        del service.active_sessions[session_id]
        service._save_sessions()
        
        logger.info(f"User {user_id} deleted session {session_id}")
        
        return {
            "status": "success",
            "message": "Session deleted successfully",
            "session_id": session_id
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete user session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/history/{user_id}")
async def clear_user_history(
    user_id: str,
    service = Depends(get_mock_service)
):
    """
    Clear all interview history for a specific user
    
    **Warning:** This will delete ALL sessions for the user
    **Note:** This action cannot be undone
    """
    try:
        # Find all sessions for this user
        user_session_ids = [
            sid for sid, session in service.active_sessions.items()
            if session.user_id == user_id
        ]
        
        if not user_session_ids:
            return {
                "status": "success",
                "message": "No sessions found for this user",
                "deleted_count": 0
            }
        
        # Delete all user sessions
        for session_id in user_session_ids:
            del service.active_sessions[session_id]
        
        # Save changes
        service._save_sessions()
        
        logger.info(f"Cleared {len(user_session_ids)} sessions for user {user_id}")
        
        return {
            "status": "success",
            "message": f"Deleted {len(user_session_ids)} sessions",
            "deleted_count": len(user_session_ids),
            "user_id": user_id
        }
    
    except Exception as e:
        logger.error(f"Failed to clear user history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Check if mock interview service is available"""
    try:
        from app.services.mock_interview_service import mock_interview_service
        
        return {
            "status": "healthy" if mock_interview_service else "unavailable",
            "service": "mock_interview",
            "features": {
                "coding_interviews": True,
                "behavioral_interviews": True,
                "technical_interviews": True,
                "ai_evaluation": True,
                "voice_input": True,
                "progressive_hints": True,
                "time_tracking": True,
                "session_persistence": True
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


@router.post("/sessions/{session_id}/hint")
async def get_hint(
    session_id: str,
    hint_level: int = Query(default=1, ge=1, le=3, description="Hint level: 1=gentle, 2=specific, 3=detailed"),
    service = Depends(get_mock_service),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None)
):
    """
    Get a progressive hint for the current question
    
    **Hint Levels:**
    - 1: Gentle nudge in the right direction
    - 2: More specific guidance
    - 3: Detailed hint (almost giving away the answer)
    
    **Note:** Hint usage is tracked and may affect final evaluation
    """
    try:
        # Extract API key from headers
        api_key = x_api_key
        if not api_key and authorization and authorization.startswith("Bearer "):
            api_key = authorization.removeprefix("Bearer ")
            
        hint = await service.get_hint(session_id, hint_level, api_key=api_key)
        
        if not hint:
            raise HTTPException(status_code=404, detail="No question active or hints unavailable")
        
        # Get hint count for this question
        session = service.active_sessions.get(session_id)
        current_question = await service.get_current_question(session_id)
        hints_used = session.hints_used.get(current_question.question_id, 0) if session and current_question else 0
        
        return {
            "hint": hint,
            "level": hint_level,
            "hints_used": hints_used,
            "note": "Using hints may slightly impact your evaluation score"
        }
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get hint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}/progress")
async def get_progress(
    session_id: str,
    service = Depends(get_mock_service)
):
    """
    Get detailed progress information for the session
    
    **Returns:**
    - Current question number
    - Total questions
    - Time spent per question
    - Hints used per question
    - Scores so far
    """
    try:
        session = service.active_sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Calculate average score so far
        scores = [e.overall_score for e in session.evaluations]
        avg_score = sum(scores) / len(scores) if scores else None
        
        return {
            "session_id": session_id,
            "current_question": session.current_question_index + 1,
            "total_questions": session.total_questions,
            "questions_answered": len(session.answers),
            "average_score": round(avg_score, 1) if avg_score else None,
            "scores": scores,
            "time_per_question": session.time_per_question,
            "hints_used": session.hints_used,
            "started_at": session.started_at.isoformat(),
            "is_complete": session.current_question_index >= session.total_questions
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get progress: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{user_id}")
async def get_user_history(
    user_id: str,
    limit: int = Query(default=20, ge=1, le=100, description="Maximum number of sessions to return"),
    include_details: bool = Query(default=False, description="Include full evaluations and questions"),
    service = Depends(get_mock_service)
):
    """
    Get interview history for a specific user
    
    **Query Parameters:**
    - `limit`: Maximum sessions to return (default: 20, max: 100)
    - `include_details`: Include full evaluations and questions (default: false)
    
    **Returns:**
    
    **Without details (default):**
    - Session list with basic info (type, difficulty, scores, dates)
    - Overall statistics and performance trends
    - Lightweight for listing sessions
    
    **With details (`include_details=true`):**
    - Everything above PLUS:
    - Full question text for each question
    - Complete evaluations with feedback
    - Strengths, weaknesses, improvement suggestions
    - Time and hints per question
    - Useful for detailed review or analytics
    
    **Example Usage:**
    - List sessions: `/api/mock-interview/history/user123`
    - Get full details: `/api/mock-interview/history/user123?include_details=true`
    - Limit results: `/api/mock-interview/history/user123?limit=10`
    """
    try:
        # Get all sessions for this user
        user_sessions = []
        
        for session_id, session in service.active_sessions.items():
            if session.user_id == user_id:
                # Calculate session stats
                scores = [e.overall_score for e in session.evaluations]
                avg_score = sum(scores) / len(scores) if scores else None
                
                session_data = {
                    "session_id": session_id,
                    "interview_type": session.interview_type,
                    "difficulty": session.difficulty,
                    "total_questions": session.total_questions,
                    "questions_answered": len(session.answers),
                    "average_score": round(avg_score, 1) if avg_score else None,
                    "started_at": session.started_at.isoformat(),
                    "completed_at": session.completed_at.isoformat() if session.completed_at else None,
                    "is_complete": session.current_question_index >= session.total_questions,
                    "status": "completed" if session.completed_at else "in_progress"
                }
                
                # Include detailed evaluations if requested (flatten into main object for frontend)
                if include_details:
                    total_time = sum(session.time_per_question.values())
                    total_hints = sum(session.hints_used.values())
                    
                    # Add timing/hint summary
                    session_data["total_time_seconds"] = total_time
                    session_data["total_hints_used"] = total_hints
                    session_data["individual_scores"] = scores
                    
                    # Add evaluations array (flattened format for frontend)
                    session_data["evaluations"] = [
                        {
                            "question_number": i + 1,
                            "question": session.questions[i].question_text,
                            "user_answer": session.answers[i].answer_text if i < len(session.answers) else None,
                            "code_solution": session.answers[i].code_solution if i < len(session.answers) else None,
                            "model_answer": session.questions[i].context if hasattr(session.questions[i], 'context') else None,
                            "score": session.evaluations[i].overall_score if i < len(session.evaluations) else None,
                            "rating": session.evaluations[i].rating_category if i < len(session.evaluations) else None,
                            "feedback": session.evaluations[i].detailed_feedback if i < len(session.evaluations) else None,
                            "performance_summary": session.evaluations[i].performance_summary if i < len(session.evaluations) else None,
                            "strengths": session.evaluations[i].strengths if i < len(session.evaluations) else [],
                            "weaknesses": session.evaluations[i].weaknesses if i < len(session.evaluations) else [],
                            "missing_points": session.evaluations[i].missing_points if i < len(session.evaluations) else [],
                            "improvement_suggestions": session.evaluations[i].improvement_suggestions if i < len(session.evaluations) else [],
                            "criteria_scores": {
                                "correctness": session.evaluations[i].criteria_scores.correctness,
                                "completeness": session.evaluations[i].criteria_scores.completeness,
                                "clarity": session.evaluations[i].criteria_scores.clarity,
                                "confidence": session.evaluations[i].criteria_scores.confidence,
                                "technical_depth": session.evaluations[i].criteria_scores.technical_depth
                            } if i < len(session.evaluations) else None,
                            "time_spent_seconds": session.time_per_question.get(session.questions[i].question_id, 0),
                            "hints_used": session.hints_used.get(session.questions[i].question_id, 0),
                            "question_type": session.questions[i].interview_type,
                            "difficulty": session.questions[i].difficulty,
                            "topic": session.questions[i].topic
                        }
                        for i in range(len(session.questions))
                    ]
                
                user_sessions.append(session_data)
        
        # Sort by start time (most recent first)
        user_sessions.sort(key=lambda x: x["started_at"], reverse=True)
        
        # Apply limit
        user_sessions = user_sessions[:limit]
        
        # Calculate overall stats
        if user_sessions:
            all_scores = [s["average_score"] for s in user_sessions if s["average_score"] is not None]
            overall_avg = sum(all_scores) / len(all_scores) if all_scores else None
            
            stats = {
                "total_sessions": len(user_sessions),
                "completed_sessions": sum(1 for s in user_sessions if s["is_complete"]),
                "overall_average_score": round(overall_avg, 1) if overall_avg else None,
                "best_score": round(max(all_scores), 1) if all_scores else None,
                "interview_types": {
                    "coding": sum(1 for s in user_sessions if s["interview_type"] == "coding"),
                    "behavioral": sum(1 for s in user_sessions if s["interview_type"] == "behavioral"),
                    "technical": sum(1 for s in user_sessions if s["interview_type"] == "technical"),
                    "system_design": sum(1 for s in user_sessions if s["interview_type"] == "system_design")
                }
            }
        else:
            stats = {
                "total_sessions": 0,
                "completed_sessions": 0,
                "overall_average_score": None,
                "best_score": None,
                "interview_types": {
                    "coding": 0,
                    "behavioral": 0,
                    "technical": 0,
                    "system_design": 0
                }
            }
        
        return {
            "user_id": user_id,
            "sessions": user_sessions,
            "statistics": stats,
            "metadata": {
                "total_returned": len(user_sessions),
                "limit": limit,
                "include_details": include_details
            }
        }
    
    except Exception as e:
        logger.error(f"Failed to get user history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))