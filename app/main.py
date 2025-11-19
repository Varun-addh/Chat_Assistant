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
    enhanced_interview_service
)

from app.services.interview_intelligence_service import ultra_production_service


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
	
	yield
	# Shutdown - graceful cleanup (only close enhanced, base and ultra share resources)
	await enhanced_interview_service.close()


app = FastAPI(
	title="Interview Assistant Backend",
	version="0.1.0",
	lifespan=lifespan,
)

# CORS
app.add_middleware(
	CORSMiddleware,
	allow_origins=settings.cors_allow_origins,
	# Wildcard origins require credentials to be False per CORS spec
	allow_credentials=False if settings.cors_allow_origins == ["*"] else True,
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
app.include_router(history_router, prefix="/api/history_tabs", tags=["history-tabs"])
app.include_router(questions_router, prefix="/api", tags=["questions"]) 
app.include_router(history_router, prefix="/api/intel-history", tags=["intel-history"])
app.include_router(diagrams_router, prefix="/api", tags=["diagrams"])
app.include_router(evaluate_router, prefix="/api", tags=["evaluation"]) 
app.include_router(interview_intelligence, prefix="/api/intelligence", tags=["interview-intelligence"])
app.include_router(ws_router, tags=["realtime"]) 