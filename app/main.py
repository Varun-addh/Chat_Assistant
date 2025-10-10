from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from app.config import settings
from app.utils.logging import configure_logging
from app.routers.questions import router as questions_router
from app.routers.ws import router as ws_router
from app.routers.diagrams import router as diagrams_router
from app.utils.audit import auditor
from app.services.llm_service import llm_service


configure_logging()
auditor.configure(settings.analytics_path)
app = FastAPI(title="Interview Assistant Backend", version="0.1.0")

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
app.include_router(questions_router, prefix="/api", tags=["questions"]) 
app.include_router(diagrams_router, prefix="/api", tags=["diagrams"])
app.include_router(ws_router, tags=["realtime"]) 
