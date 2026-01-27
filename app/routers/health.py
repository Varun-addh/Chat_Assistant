"""
Health check endpoint with dependency status checks.
Provides operational visibility for load balancers, monitoring systems, and DevOps.
"""
from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from sqlalchemy import text
from datetime import datetime
import logging
import asyncio
from typing import Dict, Any

from app.database import SessionLocal
from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Health"])


async def check_database() -> Dict[str, Any]:
    """Check database connectivity and response time."""
    try:
        start = datetime.now()
        db = SessionLocal()
        try:
            db.execute(text("SELECT 1"))
            db.commit()
            latency_ms = (datetime.now() - start).total_seconds() * 1000
            return {
                "status": "healthy",
                "latency_ms": round(latency_ms, 2)
            }
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }


async def check_llm_service() -> Dict[str, Any]:
    """Check if LLM service configuration is valid."""
    try:
        from app.services.chat.llm_service import llm_service
        
        # Check if API keys are configured
        provider = settings.llm_provider.lower()
        if provider == "groq" and settings.groq_api_key:
            return {"status": "configured", "provider": provider}
        elif provider == "gemini" and settings.gemini_api_key:
            return {"status": "configured", "provider": provider}
        elif provider == "cohere" and settings.cohere_api_key:
            return {"status": "configured", "provider": provider}
        else:
            return {
                "status": "warning",
                "message": f"Provider {provider} configured but API key may be missing"
            }
    except Exception as e:
        logger.error(f"LLM service health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}


async def check_vector_db() -> Dict[str, Any]:
    """Check vector database availability (if Interview Intelligence is enabled)."""
    try:
        if settings.disable_interview_intelligence or settings.fast_startup:
            return {"status": "disabled", "message": "Interview Intelligence disabled"}
        
        try:
            from app.services.interview.interview_intelligence_service import enhanced_interview_service
            if enhanced_interview_service.vector_client:
                # Simple check - just verify client exists
                return {"status": "available"}
            else:
                return {"status": "not_initialized"}
        except ImportError:
            return {"status": "unavailable", "message": "qdrant_client not installed"}
    except Exception as e:
        logger.error(f"Vector DB health check failed: {e}")
        return {"status": "error", "error": str(e)}


@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """
    Comprehensive health check endpoint.
    
    Returns 200 if all critical services are healthy.
    Returns 503 if any critical service is down.
    
    Use for:
    - Load balancer health checks
    - Kubernetes readiness/liveness probes
    - Monitoring system integration
    - Deployment validation
    """
    # Run checks in parallel for speed
    db_check, llm_check, vector_check = await asyncio.gather(
        check_database(),
        check_llm_service(),
        check_vector_db(),
        return_exceptions=True
    )
    
    # Handle any exceptions from gather
    if isinstance(db_check, Exception):
        db_check = {"status": "error", "error": str(db_check)}
    if isinstance(llm_check, Exception):
        llm_check = {"status": "error", "error": str(llm_check)}
    if isinstance(vector_check, Exception):
        vector_check = {"status": "error", "error": str(vector_check)}
    
    # Determine overall health
    is_healthy = (
        db_check.get("status") == "healthy" and
        llm_check.get("status") in ("configured", "warning")
    )
    
    response_data = {
        "status": "healthy" if is_healthy else "unhealthy",
        "timestamp": datetime.utcnow().isoformat(),
        "app_name": settings.app_name,
        "environment": settings.app_env,
        "version": "1.0.0",  # TODO: Read from version file or git tag
        "checks": {
            "database": db_check,
            "llm_service": llm_check,
            "vector_db": vector_check
        }
    }
    
    status_code = status.HTTP_200_OK if is_healthy else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(content=response_data, status_code=status_code)


@router.get("/health/live", status_code=status.HTTP_200_OK)
async def liveness_check():
    """
    Kubernetes liveness probe - checks if app is alive.
    Returns 200 if the application is running (even if dependencies are down).
    """
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/health/ready", status_code=status.HTTP_200_OK)
async def readiness_check():
    """
    Kubernetes readiness probe - checks if app can serve traffic.
    Returns 200 only if critical dependencies (DB) are available.
    """
    db_check = await check_database()
    
    is_ready = db_check.get("status") == "healthy"
    
    response_data = {
        "status": "ready" if is_ready else "not_ready",
        "timestamp": datetime.utcnow().isoformat(),
        "database": db_check
    }
    
    status_code = status.HTTP_200_OK if is_ready else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(content=response_data, status_code=status_code)
