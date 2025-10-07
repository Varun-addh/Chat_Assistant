from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from app.config import settings
from app.utils.logging import configure_logging
from app.routers.questions import router as questions_router
from app.routers.ws import router as ws_router
from app.utils.audit import auditor
from app.services.llm_service import llm_service


configure_logging()
auditor.configure(settings.analytics_path)
app = FastAPI(title="Interview Assistant Backend", version="0.1.0")

# CORS
app.add_middleware(
	CORSMiddleware,
	allow_origins=settings.cors_allow_origins,
	allow_credentials=True,
	allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
	allow_headers=[
		"Accept",
		"Accept-Language",
		"Content-Language",
		"Content-Type",
		"Authorization",
		"X-Requested-With",
		"Origin",
		"Access-Control-Request-Method",
		"Access-Control-Request-Headers",
	],
	expose_headers=["*"],
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
app.include_router(ws_router, tags=["realtime"]) 
