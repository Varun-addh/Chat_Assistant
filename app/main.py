import asyncio
import warnings
from contextlib import asynccontextmanager
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from app.config import settings
from app.utils.logging import configure_logging
from app.routers.questions import router as questions_router
from app.routers.ws import router as ws_router
from app.routers.diagrams import router as diagrams_router
from app.routers.evaluate import router as evaluate_router
from app.routers.history import router as history_router
from app.routers.auth_routes import router as auth_router
from app.routers.code_execution import router as code_execution_router
from app.routers.health import router as health_router
from app.utils.audit import auditor
from app.services.chat.llm_service import llm_service
import logging

# Keep startup logs clean from known third-party deprecation noise.
warnings.filterwarnings(
    "ignore",
    message=r"torch\.utils\._pytree\._register_pytree_node is deprecated\..*",
)
warnings.filterwarnings(
    "ignore",
    message=r"Deprecated call to `pkg_resources\.declare_namespace\('google'\)`\..*",
)

# LangChain deprecation noise (we intentionally use the community embedding class
# to avoid dependency conflicts in constrained build environments).
warnings.filterwarnings(
    "ignore",
    message=r"(?s).*The class `HuggingFaceEmbeddings` was deprecated.*langchain-huggingface.*",
)


logger = logging.getLogger(__name__)

# Interview Intelligence depends on optional vector-db packages (qdrant_client)
# and heavy ML libraries (torch, sentence_transformers).
# When fast_startup is enabled, skip these imports entirely for quick dev startup.
INTERVIEW_INTELLIGENCE_AVAILABLE = True

if settings.fast_startup or settings.disable_interview_intelligence:
    # Skip heavy imports entirely — keeps startup fast for local dev
    INTERVIEW_INTELLIGENCE_AVAILABLE = False
    interview_intelligence = None  # type: ignore[assignment]
    interview_intelligence_service = None  # type: ignore[assignment]
    base_interview_service = None  # type: ignore[assignment]
    enhanced_interview_service = None  # type: ignore[assignment]
    ultra_production_service = None  # type: ignore[assignment]
    logger.info("Interview Intelligence imports skipped (fast_startup or disable_interview_intelligence)")
else:
    try:
        from app.routers.interview_intelligence import router as interview_intelligence
        from app.services.interview.interview_intelligence_service import (
            interview_intelligence_service,
            base_interview_service,
            enhanced_interview_service,
            ultra_production_service,
        )
    except ModuleNotFoundError as e:
        # Typical in local/dev setups that haven't installed all optional deps.
        if "qdrant_client" in str(e):
            INTERVIEW_INTELLIGENCE_AVAILABLE = False
            interview_intelligence = None  # type: ignore[assignment]
            interview_intelligence_service = None  # type: ignore[assignment]
            base_interview_service = None  # type: ignore[assignment]
            enhanced_interview_service = None  # type: ignore[assignment]
            ultra_production_service = None  # type: ignore[assignment]
            logging.getLogger(__name__).warning(
                "Interview Intelligence disabled: optional dependency missing (%s).",
                e,
            )
        else:
            raise

from app.routers.mock_interview import router as mock_interview_router
from app.services.interview.mock_interview_service import initialize_mock_interview_service
from app.services.core.history_manager import default_history_manager

# Practice Mode imports
from app.routers.practice_mode import router as practice_router, init_practice_mode
from app.config_practice_mode import get_practice_config


configure_logging()
auditor.configure(settings.analytics_path)

# Initialize Sentry for production error tracking
try:
    from app.utils.sentry import init_sentry
    init_sentry()
except Exception as e:
    logger.warning(f"Sentry initialization failed: {e}")

# Startup banner
logger = logging.getLogger(__name__)
logger.info("="*60)
logger.info("🚀 Stratax AI Backend Starting")
logger.info(f"   App Name: {settings.app_name}")
logger.info(f"   Developer: {settings.app_developer_name}")
logger.info(f"   Attribution Guard: ACTIVE")
logger.info("="*60)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan: startup and shutdown."""
    global INTERVIEW_INTELLIGENCE_AVAILABLE
    # Initialize database
    from app.database import init_db
    init_db()
    logger.info("✅ Database initialized")

    # Start in-memory rate limiter cleanup loop
    from app.middleware.rate_limit import rate_limiter
    rate_limiter.start()
    
    if INTERVIEW_INTELLIGENCE_AVAILABLE:
        try:
            # Startup - initialize enhanced service first, then share its resources with base
            # Timeout prevents hanging when Qdrant lock is held by another process (e.g. --reload)
            await asyncio.wait_for(enhanced_interview_service.initialize(), timeout=30)

            # Share vector client and embed model to avoid Qdrant lock conflicts
            base_interview_service.vector_client = enhanced_interview_service.vector_client
            base_interview_service.embed_model = enhanced_interview_service.embed_model
            base_interview_service.collection_name = enhanced_interview_service.collection_name

            # Share resources with ultra service BEFORE initializing (to avoid Qdrant lock conflict)
            ultra_production_service.vector_client = enhanced_interview_service.vector_client
            ultra_production_service.embed_model = enhanced_interview_service.embed_model
            ultra_production_service.collection_name = enhanced_interview_service.collection_name

            # Initialize ultra production service (will skip creating new vector client since we shared it)
            await asyncio.wait_for(ultra_production_service.initialize(), timeout=30)
        except asyncio.TimeoutError:
            logger.warning(
                "Interview Intelligence disabled: initialization timed out (Qdrant lock likely held by another process). "
                "Try running without --reload or set FAST_STARTUP=true."
            )
            INTERVIEW_INTELLIGENCE_AVAILABLE = False
        except RuntimeError as e:
            msg = str(e).lower()
            if "qdrant" in msg and ("locked" in msg or "already accessed" in msg or "storage folder" in msg):
                logger.warning(
                    "Interview Intelligence disabled due to local Qdrant lock. "
                    "Stop other uvicorn/python processes or run without --reload. Error: %s",
                    e,
                )
                # Degrade gracefully: keep the API up, but skip intelligence-backed features.
                # (This is common during dev when --reload spawns extra processes.)
                INTERVIEW_INTELLIGENCE_AVAILABLE = False
            else:
                raise

    # Initialize mock interview service (uses intelligence if available, else degraded mode).
    from app.services.chat.llm_service import get_llm_service
    initialize_mock_interview_service(
        llm_service=get_llm_service(feature="default"),
        interview_intelligence_service=(
            interview_intelligence_service if INTERVIEW_INTELLIGENCE_AVAILABLE else None
        ),
    )
    
    # Initialize history manager
    await default_history_manager.initialize()
    
    # Initialize Practice Mode
    # Always initialize if enabled, even without server key (user can provide their own in Bridge Settings)
    # NOTE: In fast_startup we skip Practice Mode initialization to keep startup light.
    if (not settings.fast_startup) and settings.practice_mode_enabled:
        # Use server key as default if available, otherwise just use a placeholder
        # The service will prioritize keys from request headers (Bridge Settings) anyway
        init_practice_mode(
            gemini_api_key=settings.gemini_api_key or "no-server-key-provided",
            gemini_model=settings.gemini_model,
            config=get_practice_config()
        )
        if settings.gemini_api_key:
            logger.info("Practice Mode initialized successfully with server key")
        else:
            logger.info("Practice Mode initialized (awaiting Bridge Settings key)")
    else:
        if settings.fast_startup:
            logger.info("Fast startup enabled: Practice Mode init skipped")
        else:
            logger.info("Practice Mode disabled in settings")
    
    yield
    
    # Shutdown - graceful cleanup
    # Stop in-memory rate limiter cleanup loop
    from app.middleware.rate_limit import rate_limiter
    await rate_limiter.shutdown()

    # Close shared Redis client (used by shared caches)
    try:
        from app.services.core.redis_client import close_redis

        await close_redis()
    except Exception:
        pass

    # Clean up practice mode TTS resources (only if it was initialized)
    if (not settings.fast_startup) and settings.practice_mode_enabled:
        from app.routers.practice_mode import cleanup_practice_mode
        cleanup_practice_mode()
    
    # Close interview intelligence services (only close enhanced, base and ultra share resources)
    if INTERVIEW_INTELLIGENCE_AVAILABLE:
        await enhanced_interview_service.close()


app = FastAPI(
	title="Stratax AI Backend",
	version="0.1.0",
	lifespan=lifespan,
)


@app.get("/")
async def root(logs: str | None = Query(default=None)) -> JSONResponse:
    """Root endpoint.

    Some platforms probe `/?logs=container` (or just `/`) during startup.
    Returning 200 here avoids noisy 404s in container logs.
    """
    return JSONResponse({"status": "ok", "logs": logs})

# CORS Middleware — uses settings.cors_allow_origins (env: CORS_ALLOW_ORIGINS)
_cors_credentials = "*" not in settings.cors_allow_origins  # credentials incompatible with wildcard
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=_cors_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

# User Authentication Middleware (for personalized history)
from app.middleware.auth import user_auth_middleware
app.middleware("http")(user_auth_middleware)

# Rate Limiting Middleware (protects API costs)
from app.middleware.rate_limit import rate_limit_middleware
app.middleware("http")(rate_limit_middleware)


@app.get("/health/simple")
async def health_simple() -> JSONResponse:
    """Lightweight health endpoint.

    The comprehensive health checks live in the health router at `GET /health`.
    """
    return JSONResponse({
        "status": "ok",
        "version": app.version,
        "llm": {"provider": settings.llm_provider, "enabled": llm_service.enabled},
    })

@app.get("/api/identity-check")
async def identity_check() -> JSONResponse:
	"""Diagnostic endpoint: confirms backend identity guard is active."""
	try:
		test_response = llm_service._identity_response_text("who developed you")
	except Exception as e:
		test_response = f"Error generating test response: {e}"
	
	return JSONResponse({
		"app_name": settings.app_name,
		"developer": settings.app_developer_name,
		"attribution": settings.app_developer_attribution,
		"identity_guard": "ACTIVE",
		"test_question": "who developed you",
		"test_response": test_response,
	})

# Routers
app.include_router(health_router)  # Health checks (no prefix - accessible at /health/*)
app.include_router(auth_router)  # Authentication routes (register, login, etc.)
app.include_router(history_router, prefix="/api/history", tags=["history"])
app.include_router(questions_router, prefix="/api", tags=["questions"])
app.include_router(diagrams_router, prefix="/api", tags=["diagrams"])
app.include_router(evaluate_router, prefix="/api", tags=["evaluation"]) 
app.include_router(code_execution_router)
if INTERVIEW_INTELLIGENCE_AVAILABLE and interview_intelligence is not None:
    app.include_router(interview_intelligence, prefix="/api/intelligence", tags=["interview-intelligence"])
app.include_router(ws_router, tags=["realtime"]) 
app.include_router(
    mock_interview_router,
    prefix="/api/mock-interview",
    tags=["mock-interview"]
)
app.include_router(practice_router)  # Practice Mode router