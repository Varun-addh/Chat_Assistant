from fastapi import APIRouter, HTTPException, Query, Header, Request, Depends
from typing import Any, List, Optional
from pydantic import BaseModel, Field
import logging
import asyncio

from app.schemas import (
    InterviewQuestion,
    InterviewQuestionsResponse,
    TopicListResponse,
    SearchQuestionsResponse,
    InterviewSearchRequest,
)
from app.utils.audit import auditor

from app.services.core.history_manager import default_history_manager
from fastapi import WebSocket, WebSocketDisconnect
import time
from app.config import settings
from app.utils.demo_mode import extract_user_provided_api_key, infer_user_type
from app.auth import get_current_user

# NOTE: app.services.interview_intelligence_service imports heavy ML dependencies.
# To keep router import fast (especially in tests), we lazily load the service singletons.
interview_intelligence_service = None
enhanced_interview_service = None
ultra_production_service = None


def _ensure_intelligence_services_loaded() -> None:
    global interview_intelligence_service, enhanced_interview_service, ultra_production_service
    if (
        interview_intelligence_service is None
        or enhanced_interview_service is None
        or ultra_production_service is None
    ):
        from app.services.interview.interview_intelligence_service import (
            interview_intelligence_service as _interview_intelligence_service,
            enhanced_interview_service as _enhanced_interview_service,
            ultra_production_service as _ultra_production_service,
        )
        interview_intelligence_service = _interview_intelligence_service
        enhanced_interview_service = _enhanced_interview_service
        ultra_production_service = _ultra_production_service

router = APIRouter()
logger = logging.getLogger(__name__)

from app.utils.intelligence_formatting import (
    auto_format_code_blocks,
    format_coding_answer_for_interview_tab,
    clean_history_metadata,
    apply_formatting_to_questions,
    _unescape_common_whitespace_sequences,
    _has_interview_structure,
    _answer_has_complexity,
    _parse_markdown_sections,
    _remove_all_code_aggressive,
    _extract_explanation_only,
    _extract_approach_summary,
)


# --- API key extraction helper ---------------------------------------------------

def _clean_header_value(v: Optional[str]) -> Optional[str]:
    """Sanitize a header value: strip whitespace and reject null-ish strings."""
    t = (v or "").strip()
    if not t or t.lower() in {"null", "undefined", "none"}:
        return None
    return t


def _resolve_api_key(
    request: Request,
    x_api_key: Optional[str],
    x_gemini_key: Optional[str],
    authorization: Optional[str],
) -> tuple:
    """Return (api_key, is_demo) after applying BYOK / demo / server-key logic.

    Centralises the API-key resolution duplicated across every endpoint.
    """
    x_api_key = _clean_header_value(x_api_key)
    x_gemini_key = _clean_header_value(x_gemini_key)
    provided_key = x_gemini_key or x_api_key

    auth_key = extract_user_provided_api_key(
        request,
        x_api_key=None,
        x_gemini_key=None,
        authorization=authorization,
    )
    user_key = provided_key or auth_key
    is_demo = infer_user_type(request, user_provided_key=user_key) == "demo"

    # Enforce BYOK when required
    if settings.require_user_api_key and not user_key:
        raise HTTPException(
            status_code=401,
            detail=(
                "API key required for Interview Intelligence. "
                "Set your key in Bridge Settings (frontend) or send it via "
                "X-API-Key / X-Gemini-Key header, or Authorization: Bearer <key>."
            ),
        )

    api_key = user_key

    if not api_key:
        if is_demo:
            if settings.is_demo_key_pool_enabled():
                try:
                    from app.services.chat.demo_key_pool import demo_key_pool
                    demo_key = demo_key_pool.get_key()
                    if demo_key and demo_key.startswith("gsk_"):
                        api_key = demo_key
                except Exception:
                    pass
            if not api_key and settings.groq_api_key:
                api_key = settings.groq_api_key
            if not api_key:
                raise HTTPException(
                    status_code=503,
                    detail="Demo is temporarily unavailable (Groq is not configured).",
                )
        else:
            effective = settings.get_effective_provider(feature="default")
            if effective == "gemini" and settings.gemini_api_key:
                api_key = settings.gemini_api_key
            elif effective == "groq" and settings.groq_api_key:
                api_key = settings.groq_api_key
            else:
                api_key = settings.gemini_api_key or settings.groq_api_key

    return api_key, is_demo


def _resolve_api_key_simple(
    x_api_key: Optional[str],
    x_gemini_key: Optional[str],
    authorization: Optional[str],
) -> Optional[str]:
    """Simpler key resolution for enhanced/ultra endpoints (no demo logic)."""
    groq_key = _clean_header_value(x_api_key)
    gemini_key = _clean_header_value(x_gemini_key)
    if not groq_key and authorization and authorization.startswith("Bearer "):
        groq_key = authorization.split(" ", 1)[1].strip()
    api_key = gemini_key or groq_key
    if not api_key:
        api_key = settings.gemini_api_key or settings.groq_api_key
    return api_key


# --- Pydantic models for enhanced endpoints ---


class EnhancedSearchRequest(BaseModel):
	"""Enhanced search request with verification options"""
	query: str = Field(..., description="Search query")
	limit: int = Field(default=20, ge=1, le=50)
	verified_only: bool = Field(
		default=False, 
		description="Only return questions verified from real interviews"
	)
	min_credibility: float = Field(
		default=0.0,
		ge=0.0,
		le=1.0,
		description="Minimum credibility score (0.0-1.0)"
	)
	company: Optional[str] = Field(
		default=None,
		description="Filter by specific company (e.g., 'google', 'amazon')"
	)
	refresh: bool = Field(default=False)


class SearchMetadata(BaseModel):
	"""Metadata about search results"""
	total: int
	verified: int
	generated: int
	avg_credibility: float
	trust_level: str
	source_breakdown: dict
	warning: Optional[str] = None


class EnhancedSearchResponse(BaseModel):
	"""Search response with transparency metadata"""
	query: str
	questions: List[dict]
	count: int
	metadata: SearchMetadata
	# Helpful information
	tips: List[str] = Field(default_factory=list)


async def _search_and_build_response(
    query: str,
    limit: int,
    refresh: bool = False,
    api_key: Optional[str] = None,
    save_to_history: bool = True,
    request: Optional[Any] = None  # FastAPI Request object for user context
) -> SearchQuestionsResponse:
    """Helper to search and format response"""
    _ensure_intelligence_services_loaded()
    results = await interview_intelligence_service.search_questions(
        query,
        limit=limit,
        use_cache=not refresh,
        force_refresh=refresh,
        api_key=api_key
    )

    question_objects = [
        InterviewQuestion(
            question=r.get("question", ""),
            # APPLY FORMATTING HERE:
            answer=format_coding_answer_for_interview_tab(
                r.get("answer", ""),
                r.get("code_solution"),
                r.get("is_coding_question", False),
                r.get("language"),
                r.get("time_complexity"),
                r.get("space_complexity"),
            ),
            source=r.get("source", "llm_generated"),
            updated_at=r.get("updated_at", ""),
            topic=r.get("topic"),
            # Avoid returning the raw code_solution separately to prevent duplicate rendering
            code_solution=None,
            language=r.get("language"),
            is_coding_question=r.get("is_coding_question", False),
        )
        for r in results
    ]

    await auditor.log({
        "type": "interview_intelligence_search",
        "query": query,
        "count": len(question_objects),
        "refresh": refresh,
    })

    # HISTORY SAVE: Disabled here to prevent duplicates.
    # The frontend now controls history save via POST /api/history/ after receiving results.
    # This gives the UI full control and avoids double-saving.
    if save_to_history and len(question_objects) > 0:
        logger.debug(
            f"⏭️ Skipping backend auto-save for Interview Intelligence (frontend will save via /api/history/): query='{query}'"
        )


    return SearchQuestionsResponse(
        query=query,
        questions=question_objects,
        count=len(question_objects),
    )

@router.get("/topics", response_model=TopicListResponse)
async def get_topics():
    """
    Get list of all available interview question topics.
    
    Returns topics that have been generated or curated in the system.
    """
    try:
        _ensure_intelligence_services_loaded()
        topics = await interview_intelligence_service.get_all_topics()
        return TopicListResponse(topics=topics)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve topics: {str(e)}"
        )


@router.get("/questions/{topic}", response_model=InterviewQuestionsResponse)
async def get_questions_by_topic(
    request: Request,
    topic: str,
    limit: int = Query(default=50, ge=1, le=100),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    x_gemini_key: Optional[str] = Header(None, alias="X-Gemini-Key"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
):
    """Get interview questions for a specific topic."""
    api_key, _is_demo = _resolve_api_key(request, x_api_key, x_gemini_key, authorization)

    try:
        _ensure_intelligence_services_loaded()
        questions = await interview_intelligence_service.get_questions_by_topic(
            topic,
            limit=limit,
            api_key=api_key
        )
        
        if not questions:
            return InterviewQuestionsResponse(
                topic=topic,
                questions=[],
                count=0,
                message=f"No questions found for topic '{topic}'."
            )
        
        # APPLY FORMATTING:
        question_objects = [
            InterviewQuestion(
                question=q.get("question", ""),
                answer=format_coding_answer_for_interview_tab(
                    q.get("answer", ""),
                    q.get("code_solution"),
                    q.get("is_coding_question", False),
                    q.get("language"),
                    q.get("time_complexity"),
                    q.get("space_complexity"),
                ),
                source=q.get("source", "llm_generated"),
                updated_at=q.get("updated_at", ""),
                topic=q.get("topic"),
                # Do not expose raw code_solution separately to avoid duplication in the UI
                code_solution=None,
                language=q.get("language"),
                is_coding_question=q.get("is_coding_question", False),
            )
            for q in questions
        ]
        
        await auditor.log({
            "type": "interview_intelligence_topic_query",
            "topic": topic,
            "count": len(question_objects),
        })
        
        return InterviewQuestionsResponse(
            topic=topic,
            questions=question_objects,
            count=len(question_objects),
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed: {str(e)}")


@router.get("/search", response_model=SearchQuestionsResponse)
async def search_questions(
    request: Request,  # For user context
    q: str = Query(
        ...,
        description="Search query (e.g., 'python coding questions', 'system design for netflix')"
    ),
    limit: int = Query(
        default=20,
        ge=1,
        le=50,
        description="Maximum number of results"
    ),
    refresh: bool = Query(
        default=False,
        description="Generate fresh questions instead of using cache"
    ),
    save_to_history: bool = Query(
        default=True,
        description="Save this search to history (set to false to prevent duplicates)"
    ),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    x_gemini_key: Optional[str] = Header(None, alias="X-Gemini-Key"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
):
    """
    Search for interview questions using natural language.
    
    **Modern AI-Powered Search:**
    - Uses LLMs to generate highly relevant questions based on your query
    - Performs semantic search across existing questions
    - Grounds results with real-world interview examples
    - Automatically detects intent (coding, behavioral, system design)
    - Generates code solutions for programming questions
    
    **Examples:**
    - "Python coding questions for FAANG interviews"
    - "System design questions for senior engineer role"
    - "Easy SQL questions for beginners"
    - "Behavioral questions about leadership"
    - "Latest JavaScript interview questions 2025"
    
    **What makes this modern:**
    1. LLM-first generation (not web scraping)
    2. Semantic understanding of your query
    3. Fresh, high-quality content every time
    4. Comprehensive answers with examples
    5. Coding solutions with complexity analysis
    """
    if not q.strip():
        raise HTTPException(
            status_code=400,
            detail="Search query cannot be empty"
        )
    
    # API Key selection (Bridge Settings / BYOK)
    api_key, is_demo = _resolve_api_key(request, x_api_key, x_gemini_key, authorization)

    # Demo-mode clamp: keep demo responses small/cost-capped.
    if is_demo:
        limit = min(int(limit or 20), 2)

    try:
        return await _search_and_build_response(q, limit, refresh, api_key=api_key, save_to_history=save_to_history, request=request)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )


@router.post("/search", response_model=SearchQuestionsResponse)
async def search_questions_post(
    request: Request,
    payload: InterviewSearchRequest,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    x_gemini_key: Optional[str] = Header(None, alias="X-Gemini-Key"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
):
    """
    Search endpoint accepting JSON payload.
    """
    query = (payload.query or "").strip()
    if not query:
        raise HTTPException(
            status_code=400,
            detail="Search query cannot be empty"
        )

    limit = payload.limit or 20
    refresh = bool(payload.refresh)
    save_to_history = payload.save_to_history if payload.save_to_history is not None else True

    # API Key selection (Bridge Settings / BYOK)
    api_key, is_demo = _resolve_api_key(request, x_api_key, x_gemini_key, authorization)

    # Demo-mode clamp: keep demo responses small/cost-capped.
    if is_demo:
        limit = min(int(limit or 20), 2)

    try:
        return await _search_and_build_response(query, limit, refresh, api_key=api_key, save_to_history=save_to_history, request=request)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )


@router.post("/curate")
async def add_curated_question(question: dict, _user=Depends(get_current_user)):
    """
    Add a manually curated question to the database.
    
    Requires authentication — only logged-in users may add questions.
    
    **Request body:**
    ```json
    {
        "question": "Explain the differences between processes and threads",
        "answer": "Comprehensive answer here...",
        "topic": "operating-systems",
        "difficulty": "medium",
        "question_type": "technical",
        "key_concepts": ["processes", "threads", "concurrency"],
        "common_mistakes": ["Confusing threads with processes"],
        "companies": ["Google", "Amazon"]
    }
    ```
    """
    try:
        _ensure_intelligence_services_loaded()
        from app.services.interview.interview_intelligence_service import InterviewQuestion
        
        # Validate and create question object
        q = InterviewQuestion(**question)
        
        # Add to database
        await interview_intelligence_service.add_curated_question(q)
        
        return {
            "status": "ok",
            "message": "Question added successfully",
            "question": q.question[:100]
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to add question: {str(e)}"
        )


@router.get("/search/enhanced", response_model=EnhancedSearchResponse)
async def search_with_verification(
    q: str = Query(...),
    limit: int = Query(default=20, ge=1, le=50),
    verified_only: bool = Query(default=False),
    min_credibility: float = Query(default=0.0),
    company: Optional[str] = Query(default=None),
    refresh: bool = Query(default=False),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    x_gemini_key: Optional[str] = Header(None, alias="X-Gemini-Key"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
):
    """Enhanced search with verification."""
    api_key = _resolve_api_key_simple(x_api_key, x_gemini_key, authorization)

    try:
        _ensure_intelligence_services_loaded()
        logger.info(f"Enhanced search: q={q}, limit={limit}")
        
        questions = await enhanced_interview_service.search_questions(
            query=q,
            limit=limit,
            verified_only=verified_only,
            min_credibility=min_credibility,
            company=company,
            force_refresh=refresh,
            api_key=api_key
        )
        
        logger.info(f"Got {len(questions)} questions")
        
        # APPLY FORMATTING using helper:
        formatted_questions = apply_formatting_to_questions(questions)
        
        metadata_dict = await enhanced_interview_service.get_search_metadata(
            formatted_questions,
            verified_only=verified_only,
            min_credibility=min_credibility
        )
        metadata = SearchMetadata(**metadata_dict)
        
        tips = []
        if metadata.verified < metadata.total * 0.5:
            tips.append("💡 Add company name to get more verified questions")
        
        await auditor.log({
            "type": "enhanced_search",
            "query": q,
            "results": {"total": metadata.total, "verified": metadata.verified}
        })
        
        # Save to history
        if formatted_questions:
            await default_history_manager.initialize()
            tab_id = await default_history_manager.save_search(
                query=q,
                questions=formatted_questions,
                metadata=clean_history_metadata({
                    'limit': limit,
                    'verified_only': verified_only,
                    'min_credibility': min_credibility,
                    'company': company,
                    'refresh': refresh,
                    'enhanced': True
                })
            )
            logger.info(f"💾 Enhanced search saved to history: tab_id={tab_id}")
        else:
            logger.info(f"⏭️ Skipping history save (0 results) for enhanced search: '{q}'")
        
        return EnhancedSearchResponse(
            query=q,
            questions=formatted_questions,
            count=len(formatted_questions),
            metadata=metadata,
            tips=tips
        )
        
    except Exception as e:
        logger.error(f"Enhanced search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/search/enhanced", response_model=EnhancedSearchResponse)
async def search_with_verification_post(
    request: EnhancedSearchRequest,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    x_gemini_key: Optional[str] = Header(None, alias="X-Gemini-Key"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
):
    """POST version of enhanced search."""
    api_key = _resolve_api_key_simple(x_api_key, x_gemini_key, authorization)

    try:
        _ensure_intelligence_services_loaded()
        questions = await enhanced_interview_service.search_questions(
            query=request.query,
            limit=request.limit,
            verified_only=request.verified_only,
            min_credibility=request.min_credibility,
            company=request.company,
            force_refresh=request.refresh,
            api_key=api_key
        )
        
        # APPLY FORMATTING:
        formatted_questions = apply_formatting_to_questions(questions)
        
        metadata_dict = await enhanced_interview_service.get_search_metadata(
            formatted_questions,
            verified_only=request.verified_only,
            min_credibility=request.min_credibility
        )
        metadata = SearchMetadata(**metadata_dict)
        
        # Save to history
        if formatted_questions:
            await default_history_manager.initialize()
            tab_id = await default_history_manager.save_search(
                query=request.query,
                questions=formatted_questions,
                metadata=clean_history_metadata({
                    'limit': request.limit,
                    'verified_only': request.verified_only,
                    'min_credibility': request.min_credibility,
                    'company': request.company,
                    'refresh': request.refresh,
                    'enhanced': True
                })
            )
            logger.info(f"💾 Enhanced search (POST) saved to history: tab_id={tab_id}")
        else:
            logger.info(f"⏭️ Skipping history save (0 results) for enhanced search (POST): '{request.query}'")
        
        return EnhancedSearchResponse(
            query=request.query,
            questions=formatted_questions,
            count=len(formatted_questions),
            metadata=metadata,
            tips=[]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/sources/stats")
async def get_source_statistics():
    """Get statistics about available question sources"""
    try:
        _ensure_intelligence_services_loaded()
        stats = await enhanced_interview_service.source_manager.get_question_stats()
        return {
            "status": "ok",
            "statistics": stats,
            "source_info": {
                "leetcode": {
                    "name": "LeetCode",
                    "credibility": 0.95,
                    "description": "Company-tagged coding questions",
                    "verified": True,
                },
                "glassdoor": {
                    "name": "Glassdoor",
                    "credibility": 0.85,
                    "description": "Interview experiences and questions",
                    "verified": True,
                },
                "community": {
                    "name": "Community Submitted",
                    "credibility": 0.60,
                    "description": "User-submitted questions with votes",
                    "verified": "partial",
                },
                "llm_generated": {
                    "name": "AI-Generated",
                    "credibility": 0.30,
                    "description": "Practice questions generated by AI",
                    "verified": False,
                },
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/companies")
async def get_supported_companies():
	"""Get list of supported company filters for question search.

	These are illustrative companies commonly searched against.
	Actual question availability depends on real-time search results.
	"""
	companies = [
		{"name": "Amazon", "slug": "amazon"},
		{"name": "Google", "slug": "google"},
		{"name": "Meta/Facebook", "slug": "facebook"},
		{"name": "Microsoft", "slug": "microsoft"},
		{"name": "Apple", "slug": "apple"},
		{"name": "Netflix", "slug": "netflix"},
		{"name": "Tesla", "slug": "tesla"},
		{"name": "Bloomberg", "slug": "bloomberg"},
		{"name": "Adobe", "slug": "adobe"},
		{"name": "Uber", "slug": "uber"},
		{"name": "Airbnb", "slug": "airbnb"},
		{"name": "LinkedIn", "slug": "linkedin"},
		{"name": "Twitter", "slug": "twitter"},
		{"name": "Salesforce", "slug": "salesforce"},
		{"name": "Oracle", "slug": "oracle"},
	]
	return {
		"companies": companies,
		"total": len(companies),
		"note": "Use the 'slug' value as the company parameter when searching"
	}


@router.post("/community/submit")
async def submit_community_question(
    question: dict,
    submitted_by: str = Query(..., description="Anonymous user ID")
):
    """Submit a real interview question from the community"""
    try:
        _ensure_intelligence_services_loaded()
        from app.services.chat.dynamic_interview_sources import (
            VerifiedQuestion as VQ,
            SourceType as ST,
            VerificationStatus as VS,
            QuestionDomain,
        )

        # Minimal domain mapping; defaults to general.
        domain = QuestionDomain.GENERAL_TECHNICAL
        question_type = (question.get("question_type") or "technical").lower()
        topic = (question.get("topic") or "general").lower()
        if "system" in question_type or "design" in question_type or "system" in topic:
            domain = QuestionDomain.SYSTEM_DESIGN
        elif "behavior" in question_type:
            domain = QuestionDomain.BEHAVIORAL
        elif "devops" in question_type or "docker" in topic or "kubernetes" in topic:
            domain = QuestionDomain.DEVOPS
        elif "ml" in question_type or "ai" in question_type or "ml" in topic or "ai" in topic:
            domain = QuestionDomain.ML_AI

        vq = VQ(
            question=question.get("question"),
            answer=question.get("answer", ""),
            topic=question.get("topic", "general"),
            difficulty=question.get("difficulty", "medium"),
            question_type=question.get("question_type", "technical"),
            domain=domain,
            source_type=ST.COMMUNITY_VERIFIED,
            verification_status=VS.LIKELY_REAL,
            company=question.get("company"),
            position=question.get("position"),
            level=question.get("level"),
            interview_round=question.get("interview_round"),
            reported_count=1,
            credibility_score=0.5,
            submitted_by=submitted_by,
        )
        success = await enhanced_interview_service.source_manager.community.submit_question(vq, submitted_by)
        if success:
            return {
                "status": "success",
                "message": "Thank you for contributing! Your question will be reviewed.",
                "question": (question.get("question") or "")[:100],
                "credibility": 0.5,
                "note": "Credibility will increase as others verify this question",
            }
        raise HTTPException(status_code=500, detail="Failed to submit question")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/transparency")
async def get_transparency_info():
	"""Transparency details about data sources and credibility scoring"""
	return {
		"data_sources": {
			"verified_sources": {
				"leetcode": {
					"description": "Questions tagged by companies on LeetCode",
					"credibility": 0.95,
					"verification": "Confirmed by company tags on LeetCode"
				},
				"glassdoor": {
					"description": "Interview experience reports",
					"credibility": 0.85,
					"verification": "Multiple reports corroborate questions"
				},
				"community": {
					"description": "User-submitted questions",
					"credibility": "0.50-0.80 (based on votes)",
					"verification": "Verified by users who were also asked"
				}
			},
			"generated_sources": {
				"llm_generated": {
					"description": "Practice questions generated by AI",
					"credibility": 0.30,
					"verification": "Not verified"
				},
				"llm_grounded": {
					"description": "AI-generated with web validation",
					"credibility": 0.40,
					"verification": "Cross-referenced with interview prep sites"
				}
			}
		},
		"credibility_scoring": {
			"factors": [
				"Source type (LeetCode > Glassdoor > Community > AI)",
				"Number of reports/verifications",
				"User votes (upvotes/downvotes)",
				"Recency",
				"Frequency"
			],
			"ranges": {
				"0.90-1.00": "Confirmed real",
				"0.70-0.89": "Verified real",
				"0.50-0.69": "Likely real",
				"0.30-0.49": "AI-generated simulation",
				"0.00-0.29": "Unverified/low quality"
			}
		}
	}


@router.get("/health/enhanced")
async def health_check_enhanced():
    """Health check for enhanced service with source integration"""
    try:
        _ensure_intelligence_services_loaded()
        service = enhanced_interview_service
        checks = {
            "vector_db": service.vector_client is not None,
            "embedding_model": service.embed_model is not None,
            "source_manager": service.source_manager is not None,
            "leetcode_integration": True,
            "community_db": True,
        }
        all_healthy = all(checks.values())
        return {
            "status": "healthy" if all_healthy else "degraded",
            "components": checks,
            "features": {
                "verified_sources": True,
                "llm_generation": True,
                "community_submissions": True,
                "source_transparency": True,
            },
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@router.post("/update")
async def trigger_update():
    """
    Trigger system update (kept for API compatibility).
    
    Note: In the modern architecture, questions are generated on-demand,
    so there's no traditional "update" process. This endpoint exists
    for backward compatibility but is essentially a no-op.
    """
    _ensure_intelligence_services_loaded()
    await interview_intelligence_service.force_update()
    return {
        "status": "ok",
        "message": "Modern system generates fresh content on each search. No update needed.",
    }


@router.get("/stats")
async def get_statistics():
    """
    Get statistics about the question database.
    
    Returns information about:
    - Total questions in vector DB
    - Available topics
    - Question type distribution
    - Recent activity
    """
    try:
        _ensure_intelligence_services_loaded()
        topics = await interview_intelligence_service.get_all_topics()
        
        # Get collection stats from vector DB
        stats = {
            "status": "ok",
            "total_topics": len(topics),
            "available_topics": topics[:20],  # Show first 20
            "architecture": "modern_llm_first",
            "features": [
                "LLM-powered question generation",
                "Semantic vector search",
                "RAG-based grounding",
                "Real-time content generation",
                "Code solution generation",
                "Multi-language support"
            ],
            "note": "Questions are generated on-demand using advanced LLMs"
        }
        
        return stats
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get stats: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """Check if the service is healthy and ready"""
    try:
        _ensure_intelligence_services_loaded()
        # Verify components are initialized
        service = interview_intelligence_service
        
        checks = {
            "vector_db": service.vector_client is not None,
            "embedding_model": service.embed_model is not None,
            "llm_service": True,  # Assume healthy if import worked
        }
        
        all_healthy = all(checks.values())
        
        return {
            "status": "healthy" if all_healthy else "degraded",
            "components": checks,
            "architecture": "modern_llm_first_with_rag"
        }
    
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

class UltraSearchRequest(BaseModel):
    query: str
    limit: int = 20
    verified_only: bool = False
    min_credibility: float = 0.0
    company: Optional[str] = None
    enable_reranking: bool = Field(default=settings.enable_reranking)
    enable_query_expansion: bool = Field(default=settings.enable_query_expansion)
    user_id: Optional[str] = None
    refresh: bool = False

class CodeExecutionRequest(BaseModel):
    code: str
    language: str
    question: Optional[str] = None


@router.get("/search/ultra-production")
async def ultra_production_search(
    q: str = Query(...),
    limit: int = Query(default=20, ge=1, le=50),
    verified_only: bool = Query(default=False),
    min_credibility: float = Query(default=0.0),
    company: Optional[str] = Query(default=None),
    enable_reranking: bool = Query(default=True),
    enable_query_expansion: bool = Query(default=True),
    user_id: Optional[str] = Query(default=None),
    refresh: bool = Query(default=False),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    x_gemini_key: Optional[str] = Header(None, alias="X-Gemini-Key"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
):
    """🚀 Ultra Production Search."""
    api_key = _resolve_api_key_simple(x_api_key, x_gemini_key, authorization)

    try:
        _ensure_intelligence_services_loaded()
        start_time = time.time()
        
        questions = await ultra_production_service.search_questions(
            query=q,
            limit=limit,
            verified_only=verified_only,
            min_credibility=min_credibility,
            company=company,
            enable_reranking=enable_reranking,
            enable_query_expansion=enable_query_expansion,
            user_id=user_id,
            force_refresh=refresh,
            api_key=api_key
        )
        
        # APPLY FORMATTING:
        formatted_questions = apply_formatting_to_questions(questions)
        
        metadata = await ultra_production_service.get_search_metadata(
            formatted_questions,
            verified_only,
            min_credibility
        )
        
        elapsed = time.time() - start_time
        
        # Save to history
        if formatted_questions:
            await default_history_manager.initialize()
            tab_id = await default_history_manager.save_search(
                query=q,
                questions=formatted_questions,
                metadata=clean_history_metadata({
                    'verified_only': verified_only,
                    'min_credibility': min_credibility,
                    'company': company,
                    'total_results': len(formatted_questions),
                    **metadata
                })
            )
            logger.info(f"💾 Ultra-production search saved to history: tab_id={tab_id}")
        else:
            logger.info(f"⏭️ Skipping history save (0 results) for ultra-production search: '{q}'")
        
        return {
            "query": q,
            "questions": formatted_questions,
            "count": len(formatted_questions),
            "metadata": metadata,
            "performance": {
                "total_time_seconds": round(elapsed, 2),
                "questions_per_second": round(len(formatted_questions) / elapsed, 2) if elapsed > 0 else 0
            }
        }
    except Exception as e:
        logger.error(f"Ultra search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/code/execute")
async def execute_code(request: CodeExecutionRequest):
    """🔥 Execute and validate code"""
    try:
        _ensure_intelligence_services_loaded()
        result = await ultra_production_service.execute_and_validate_code(
            code=request.code,
            language=request.language,
            question=request.question or ""
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/questions/{question_id}/vote")
async def vote_question(
    question_id: str,
    user_id: str = Query(...),
    vote: int = Query(..., ge=-1, le=1),
    feedback: Optional[str] = Query(None)
):
    """👍👎 Vote on question quality"""
    try:
        _ensure_intelligence_services_loaded()
        await ultra_production_service.record_user_feedback(
            question_id, user_id, vote, feedback
        )
        return {"status": "ok", "message": "Vote recorded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/features")
async def get_features():
    """🎯 List available features"""
    _ensure_intelligence_services_loaded()
    return {
        "features": {
            "hybrid_search": {"available": True, "description": "Combines semantic + keyword search"},
            "reranking": {"available": ultra_production_service.enable_reranking, "description": "Re-ranks results by relevance"},
            "code_execution": {"available": True, "languages": ["python", "javascript", "java", "cpp"]},
            "query_expansion": {"available": True, "description": "Expands queries for broader coverage"},
            "real_time_streaming": {"available": True, "description": "Streams results via WebSocket"}
        }
    }


@router.websocket("/ws/search")
async def websocket_search(websocket: WebSocket):
    """
    🔴 REAL-TIME STREAMING SEARCH
    
    Stream search results as they're found from different sources.
    Much better UX - users see results immediately instead of waiting.
    
    Message Format:
    {
        "type": "search_started|source_searching|result|source_complete|search_complete|error",
        "source": "leetcode|stackoverflow|github|...",
        "data": {...},
        "timestamp": "ISO timestamp"
    }
    
    Usage:
    ```javascript
    const ws = new WebSocket('ws://localhost:8000/api/interview-intelligence/ws/search');
    ws.send(JSON.stringify({
        query: "python coding questions",
        limit: 20,
        enable_reranking: true,
        enable_query_expansion: true
    }));
    
    ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        if (msg.type === 'result') {
            displayQuestion(msg.data);  // Show result immediately
        }
    };
    ```
    """
    await websocket.accept()
    logger.info("WebSocket connection accepted for real-time search")
    
    try:
        _ensure_intelligence_services_loaded()
        # Wait for search request
        request_data = await websocket.receive_json()

        # Require an API key or auth token in the initial message
        ws_api_key = (request_data.get("api_key") or request_data.get("token") or "").strip()
        if not ws_api_key:
            # Fall back to server keys if available (same logic as REST endpoints)
            ws_api_key = settings.gemini_api_key or settings.groq_api_key
        if not ws_api_key:
            await websocket.send_json({
                "type": "error",
                "error": "Authentication required — send api_key or token in the first message",
                "timestamp": time.time(),
            })
            await websocket.close(code=4001)
            return
        
        query = request_data.get('query', '')
        limit = request_data.get('limit', 20)
        enable_reranking = request_data.get('enable_reranking', True)
        enable_query_expansion = request_data.get('enable_query_expansion', True)
        verified_only = request_data.get('verified_only', False)
        min_credibility = request_data.get('min_credibility', 0.0)
        company = request_data.get('company', None)
        
        if not query:
            await websocket.send_json({
                'type': 'error',
                'error': 'Query is required',
                'timestamp': time.time()
            })
            await websocket.close()
            return
        
        logger.info(f"Streaming search: query='{query}', limit={limit}")
        
        # Stream results
        from app.services.chat.ai_native_enhancements import RealTimeSearchStream
        
        # Create search sources (these will run in parallel)
        sources = []
        
        # We'll stream from the ultra production service
        # Send initial message
        await websocket.send_json({
            'type': 'search_started',
            'query': query,
            'timestamp': time.time()
        })
        
        # Start search (this runs all sources in parallel)
        questions = []
        seen_ids = set()
        
        # For streaming, we'll do a modified search that yields results as they come
        # This is a simplified version - ideally we'd modify the service to stream
        try:
            # Get expected sources for this query from the service
            expected_sources = ultra_production_service._get_expected_sources(query)
            
            # Send source_update for each expected source (searching status)
            for source in expected_sources[:3]:  # Show top 3 sources
                await websocket.send_json({
                    'type': 'source_update',
                    'source': source,
                    'status': 'searching',
                    'count': 0,
                    'timestamp': time.time()
                })
                await asyncio.sleep(0.1)  # Small delay for visual effect
            
            # Send searching status
            await websocket.send_json({
                'type': 'status',
                'message': f'Searching {", ".join(expected_sources[:3])}...',
                'timestamp': time.time()
            })
            
            # Execute search (we'll get all results but could be modified to stream)
            results = await ultra_production_service.search_questions(
                query=query,
                limit=limit,
                verified_only=verified_only,
                min_credibility=min_credibility,
                company=company,
                enable_reranking=enable_reranking,
                enable_query_expansion=enable_query_expansion,
                force_refresh=False
            )
            
            # Format questions
            formatted_questions = apply_formatting_to_questions(results)
            
            # Send source completion updates
            # Count results by source
            source_counts = {}
            for q in formatted_questions:
                source = q.get('source', 'Unknown')
                source_counts[source] = source_counts.get(source, 0) + 1
            
            # Send completion status for each source
            for source, count in source_counts.items():
                await websocket.send_json({
                    'type': 'source_update',
                    'source': source,
                    'status': 'complete',
                    'count': count,
                    'timestamp': time.time()
                })
                await asyncio.sleep(0.05)
            
            # Stream each result with a small delay for visual effect
            for idx, question in enumerate(formatted_questions):
                # DEBUG: Log source field and answer length
                answer_preview = question.get('answer', '')[:100] if question.get('answer') else 'MISSING'
                logger.info(f"📤 Sending question #{idx+1}: source='{question.get('source', 'MISSING')}', question='{question.get('question', '')[:50]}...', answer preview: {answer_preview}...")
                
                await websocket.send_json({
                    'type': 'result',
                    'data': question,
                    'progress': {
                        'current': idx + 1,
                        'total': len(formatted_questions)
                    },
                    'timestamp': time.time()
                })
                
                # Small delay to show streaming effect (optional)
                if idx < len(formatted_questions) - 1:
                    await asyncio.sleep(0.05)  # 50ms delay between results
            
            # Send completion
            metadata = await ultra_production_service.get_search_metadata(
                formatted_questions,
                verified_only,
                min_credibility
            )
            
            # Save to history using global singleton
            tab_id = None
            if formatted_questions:
                await default_history_manager.initialize()

                tab_id = await default_history_manager.save_search(
                    query=query,
                    questions=formatted_questions,
                    metadata=clean_history_metadata({
                        'verified_only': verified_only,
                        'min_credibility': min_credibility,
                        'company': company,
                        'total_results': len(formatted_questions),
                        'enhanced': True,  # Mark WebSocket searches as enhanced
                        **metadata
                    })
                )
                logger.info(f"💾 Saved to history: tab_id={tab_id}")
            else:
                logger.info(f"⏭️ Skipping history save (0 results) for websocket search: '{query}'")
            
            await websocket.send_json({
                'type': 'search_complete',
                'total_results': len(formatted_questions),
                'metadata': metadata,
                'tab_id': tab_id,  # null when not saved (e.g., 0 results)
                'timestamp': time.time()
            })
            
            logger.info(f"Streaming search complete: {len(formatted_questions)} results")
        
        except Exception as e:
            logger.error(f"Streaming search error: {e}", exc_info=True)
            await websocket.send_json({
                'type': 'error',
                'error': str(e),
                'timestamp': time.time()
            })
    
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        try:
            await websocket.send_json({
                'type': 'error',
                'error': str(e),
                'timestamp': time.time()
            })
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass