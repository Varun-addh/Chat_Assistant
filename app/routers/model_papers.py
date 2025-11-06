from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from app.schemas import ModelPaperQuery, ModelPaperResponse, ModelQuestion
from app.services.model_papers_service import model_papers_service


router = APIRouter()


@router.post("/model_papers", response_model=ModelPaperResponse)
async def create_model_paper(payload: ModelPaperQuery) -> ModelPaperResponse:
    topic = (payload.topic or "").strip()
    if not topic:
        raise HTTPException(status_code=400, detail="Topic is required")
    items = await _get_items(topic, payload.limit, payload.include_recent_days)
    return ModelPaperResponse(topic=topic, total=len(items), items=items)


@router.get("/model_papers/{topic}", response_model=ModelPaperResponse)
async def get_model_paper(
    topic: str,
    limit: int = Query(20, ge=1, le=200),
    include_recent_days: int = Query(30, ge=0, le=365),
) -> ModelPaperResponse:
    topic = (topic or "").strip()
    if not topic:
        raise HTTPException(status_code=400, detail="Topic is required")
    items = await _get_items(topic, limit, include_recent_days)
    return ModelPaperResponse(topic=topic, total=len(items), items=items)


async def _get_items(topic: str, limit: int, include_recent_days: int) -> list[ModelQuestion]:
    # Use async method directly - it handles AI calls properly
    items = await model_papers_service.get_model_paper(
        topic, 
        limit=limit, 
        include_recent_days=include_recent_days,
        use_ai_generation=True
    )
    return items


