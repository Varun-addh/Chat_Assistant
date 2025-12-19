from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from app.config import settings
from app.utils.logging import configure_logging
from app.routers.questions import router as questions_router
from app.routers.ws import router as ws_router
from app.routers.diagrams import router as diagrams_router
from app.routers.evaluate import router as evaluate_router
from app.routers.interview_intelligence import router as interview_intelligence
from app.routers.history import router as history_router
from app.utils.audit import auditor
from app.services.llm_service import llm_service
from app.services.interview_intelligence_service import (
    interview_intelligence_service,
    base_interview_service,
    enhanced_interview_service,
    ultra_production_service
)

from app.routers.mock_interview import router as mock_interview_router
from app.services.mock_interview_service import initialize_mock_interview_service
from app.services.history_manager import default_history_manager

# Practice Mode imports
from app.routers.practice_mode import router as practice_router, init_practice_mode
from app.config_practice_mode import get_practice_config


configure_logging()
auditor.configure(settings.analytics_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan: startup and shutdown."""
    # Startup - initialize enhanced service first, then share its resources with base
    await enhanced_interview_service.initialize()
    
    # Share vector client and embed model to avoid Qdrant lock conflicts
    base_interview_service.vector_client = enhanced_interview_service.vector_client
    base_interview_service.embed_model = enhanced_interview_service.embed_model
    base_interview_service.collection_name = enhanced_interview_service.collection_name
    
    # Share resources with ultra service BEFORE initializing (to avoid Qdrant lock conflict)
    ultra_production_service.vector_client = enhanced_interview_service.vector_client
    ultra_production_service.embed_model = enhanced_interview_service.embed_model
    ultra_production_service.collection_name = enhanced_interview_service.collection_name
    
    # Initialize ultra production service (will skip creating new vector client since we shared it)
    await ultra_production_service.initialize()
    
    # Initialize mock interview service HERE (inside lifespan, after other services are ready)
    from app.services.llm_service import get_llm_service
    initialize_mock_interview_service(
        llm_service=get_llm_service("groq"),
        interview_intelligence_service=interview_intelligence_service
    )
    
    # Initialize history manager
    await default_history_manager.initialize()
    
    # Initialize Practice Mode
    if settings.gemini_api_key and settings.practice_mode_enabled:
        init_practice_mode(
            gemini_api_key=settings.gemini_api_key,
            gemini_model=settings.gemini_model,
            config=get_practice_config()
        )
        print("✅ Practice Mode initialized successfully!")
    else:
        if not settings.gemini_api_key:
            print("⚠️  Practice Mode not initialized (missing GEMINI_API_KEY)")
        else:
            print("⚠️  Practice Mode disabled in settings")
    
    yield
    
    # Shutdown - graceful cleanup
    # Clean up practice mode TTS resources
    from app.routers.practice_mode import cleanup_practice_mode
    cleanup_practice_mode()
    
    # Close interview intelligence services (only close enhanced, base and ultra share resources)
    await enhanced_interview_service.close()


app = FastAPI(
	title="Interview Assistant Backend",
	version="0.1.0",
	lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)


@app.get("/health")
async def health() -> JSONResponse:
	return JSONResponse({
		"status": "ok",
		"version": app.version,
		"llm": {"provider": settings.llm_provider, "enabled": llm_service.enabled}
	})

# Routers
app.include_router(history_router, prefix="/api/history", tags=["history"])
app.include_router(questions_router, prefix="/api", tags=["questions"])
app.include_router(diagrams_router, prefix="/api", tags=["diagrams"])
app.include_router(evaluate_router, prefix="/api", tags=["evaluation"]) 
app.include_router(interview_intelligence, prefix="/api/intelligence", tags=["interview-intelligence"])
app.include_router(ws_router, tags=["realtime"]) 
app.include_router(
    mock_interview_router,
    prefix="/api/mock-interview",
    tags=["mock-interview"]
)
app.include_router(practice_router)  # Practice Mode router