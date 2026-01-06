from fastapi import APIRouter, HTTPException, Query, Request, Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

from app.services.history_manager import HistoryManager, default_history_manager

logger = logging.getLogger(__name__)

router = APIRouter()

class SaveHistoryRequest(BaseModel):
    """Request to save search to history"""
    query: str = Field(..., description="Search query")
    questions: List[Dict[str, Any]] = Field(..., description="List of questions")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata")


class UpdateHistoryRequest(BaseModel):
    """Request to update history tab"""
    query: Optional[str] = None
    questions: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None


class HistoryTabResponse(BaseModel):
    """Single history tab"""
    tab_id: str
    query: str
    questions: List[Dict[str, Any]]
    created_at: str
    metadata: Dict[str, Any]
    question_count: int


class HistoryListResponse(BaseModel):
    """List of history tabs"""
    tabs: List[HistoryTabResponse]
    total: int
    offset: int
    limit: Optional[int]


class HistoryStatsResponse(BaseModel):
    """History statistics"""
    total_tabs: int
    total_questions: int
    avg_questions_per_tab: float
    most_common_queries: List[tuple]
    oldest_tab: Optional[str]
    newest_tab: Optional[str]


def get_history_manager(request: Request) -> HistoryManager:
    """
    Get history manager for current user
    
    Uses user_id from request state if authenticated,
    otherwise uses default manager
    """
    # Check if user is authenticated
    if hasattr(request.state, "user_id"):
        user_id = request.state.user_id
        logger.debug(f"Using history manager for user: {user_id}")
        return HistoryManager(user_id=user_id)
    
    # Use default manager (guest user)
    logger.debug("Using default history manager")
    return default_history_manager


@router.get("/", response_model=HistoryListResponse)
async def get_history(
    request: Request,
    limit: Optional[int] = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    sort_by: str = Query(default="created_at", pattern="^(created_at|query|question_count)$"),
    ascending: bool = Query(default=False)
):
    """
    Get all history tabs
    
    Query Parameters:
    - limit: Max number of tabs (default: 50)
    - offset: Skip N tabs (default: 0)
    - sort_by: Sort field (created_at, query, question_count)
    - ascending: Sort order (default: False = newest first)
    
    Returns:
        List of history tabs with pagination
    """
    try:
        history = get_history_manager(request)
        await history.initialize()
        
        tabs = await history.get_all_tabs(
            limit=limit,
            offset=offset,
            sort_by=sort_by,
            ascending=ascending
        )
        
        # Get total count
        # Get all tabs for total count calculation
        all_tabs = await history.get_all_tabs()
        
        # Helper to detect empty/placeholder sessions
        def is_placeholder_tab(tab):
            questions = tab.get('questions') or []
            question_count = tab.get('question_count')
            if question_count is None:
                question_count = len(questions)

            # Treat 0-question entries as placeholders/invalid history.
            # These typically come from failed searches or misfires.
            return (
                not tab.get('tab_id') or
                (not tab.get('query') and not questions) or
                question_count == 0
            )

        # Calculate total count of valid tabs
        valid_all_tabs = [t for t in all_tabs if not is_placeholder_tab(t)]
        total_valid = len(valid_all_tabs)
        
        # Paginated results (already fetched)
        # Filter placeholders from this page
        filtered_page_tabs = [tab for tab in tabs if not is_placeholder_tab(tab)]
        
        logger.info(f"📋 History list: total={total_valid}, returning={len(filtered_page_tabs)}")

        return HistoryListResponse(
            tabs=[HistoryTabResponse(**tab) for tab in filtered_page_tabs],
            total=total_valid,
            offset=offset,
            limit=limit
        )
    
    except Exception as e:
        logger.error(f"Failed to get history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{tab_id}", response_model=HistoryTabResponse)
async def get_history_tab(request: Request, tab_id: str):
    """
    Get specific history tab by ID
    
    Path Parameters:
    - tab_id: Unique tab identifier (session_id)
    
    Returns:
        Single history tab with all questions
        Returns 404 if tab doesn't exist (may happen on Space restart with ephemeral storage)
    """
    try:
        history = get_history_manager(request)
        await history.initialize()
        
        tab = await history.get_tab(tab_id)
        
        if not tab:
            # Tab not found - auto-recovery: instead of throwing error, return a placeholder
            # This prevents the "Tab not found" UI error on frontend
            logger.info(f"Tab {tab_id} not found. Returning recovery placeholder.")
            return HistoryTabResponse(
                tab_id=tab_id,
                query="New Search",
                questions=[],
                created_at=datetime.now().isoformat(),
                metadata={},
                question_count=0
            )

        return HistoryTabResponse(**tab.to_dict())
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get tab {tab_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))



@router.post("/", response_model=Dict[str, str])
async def save_to_history(request: Request, payload: SaveHistoryRequest):
    """
    Save search results to history
    
    Body:
    - query: Search query
    - questions: List of questions
    - metadata: Optional metadata (company, verified_only, etc.)
    
    Returns:
        {"tab_id": "uuid", "message": "Saved to history"}
    """
    try:
        if not (payload.query or "").strip():
            raise HTTPException(status_code=422, detail="Query cannot be empty")

        if not payload.questions:
            raise HTTPException(
                status_code=422,
                detail="Cannot save history entry with 0 questions"
            )

        history = get_history_manager(request)
        await history.initialize()
        
        tab_id = await history.save_search(
            query=payload.query,
            questions=payload.questions,
            metadata=payload.metadata
        )
        
        return {
            "tab_id": tab_id,
            "message": f"Saved {len(payload.questions)} questions to history"
        }

    except HTTPException:
        raise
    except ValueError as e:
        # Raised by HistoryManager guardrails
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to save to history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{tab_id}")
async def update_history_tab(
    request: Request,
    tab_id: str,
    payload: UpdateHistoryRequest
):
    """
    Update existing history tab
    
    Path Parameters:
    - tab_id: Tab to update
    
    Body:
    - query: New query (optional)
    - questions: New questions (optional)
    - metadata: New metadata (optional)
    
    Returns:
        {"message": "Tab updated"}
    """
    try:
        history = get_history_manager(request)
        await history.initialize()
        
        success = await history.update_tab(
            tab_id=tab_id,
            query=payload.query,
            questions=payload.questions,
            metadata=payload.metadata
        )
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Tab {tab_id} not found")
        
        return {"message": "Tab updated successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update tab {tab_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{tab_id}")
async def delete_history_tab(request: Request, tab_id: str):
    """
    Delete specific history tab
    
    Path Parameters:
    - tab_id: Tab to delete
    
    Returns:
        {"message": "Tab deleted"}
    """
    try:
        history = get_history_manager(request)
        await history.initialize()
        
        success = await history.delete_tab(tab_id)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Tab {tab_id} not found")
        
        return {"message": f"Tab {tab_id} deleted successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete tab {tab_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/")
async def delete_all_history(request: Request):
    """
    Delete ALL history tabs
    
    ⚠️ WARNING: This action cannot be undone!
    
    Returns:
        {"message": "X tabs deleted"}
    """
    try:
        history = get_history_manager(request)
        await history.initialize()
        
        count = await history.delete_all_tabs()
        
        return {"message": f"Deleted {count} tabs successfully"}
    
    except Exception as e:
        logger.error(f"Failed to delete all tabs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search/query")
async def search_history(
    request: Request,
    q: str = Query(..., description="Search query"),
    limit: int = Query(default=20, ge=1, le=100)
):
    """
    Search within history
    
    Searches in:
    - Tab query text
    - Question text
    - Answer text
    
    Query Parameters:
    - q: Search query
    - limit: Max results (default: 20)
    
    Returns:
        Matching history tabs
    """
    try:
        history = get_history_manager(request)
        await history.initialize()
        
        results = await history.search_history(
            search_query=q,
            limit=limit
        )
        
        return {
            "query": q,
            "results": results,
            "count": len(results)
        }
    
    except Exception as e:
        logger.error(f"History search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/overview", response_model=HistoryStatsResponse)
async def get_history_stats(request: Request):
    """
    Get history statistics
    
    Returns:
    - Total tabs
    - Total questions
    - Average questions per tab
    - Most common query words
    - Date range
    """
    try:
        history = get_history_manager(request)
        await history.initialize()
        
        stats = await history.get_stats()
        
        return HistoryStatsResponse(**stats)
    
    except Exception as e:
        logger.error(f"Failed to get stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


from fastapi import Path

@router.get("/export/{format}")
async def export_history(
    request: Request,
    format: str = Path(..., pattern="^(json|csv)$", description="Export format (json or csv)")
):
    """
    Export entire history
    
    Path Parameters:
    - format: Export format (json or csv)
    
    Returns:
        Exported data as text
    """
    try:
        history = get_history_manager(request)
        await history.initialize()
        
        exported = await history.export_history(format=format)
        
        # Set appropriate content type
        media_type = "application/json" if format == "json" else "text/csv"
        
        from fastapi.responses import Response
        
        return Response(
            content=exported,
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename=history.{format}"
            }
        )
    
    except Exception as e:
        logger.error(f"Export failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/debug/raw")
async def debug_raw_history(request: Request):
    """
    DEBUG ENDPOINT: Show raw history file contents
    """
    try:
        history = get_history_manager(request)
        await history.initialize()
        
        # Get all tabs
        all_tabs = await history.get_all_tabs()
        
        # Get stats
        stats = await history.get_stats()
        
        return {
            "total_tabs": len(all_tabs),
            "tabs": all_tabs,
            "stats": stats,
            "history_file": str(history.history_file),
            "file_exists": history.history_file.exists()
        }
    except Exception as e:
        logger.error(f"Debug endpoint failed: {e}", exc_info=True)
        return {"error": str(e)}


# Export router
__all__ = ['router']