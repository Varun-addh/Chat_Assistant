"""
User authentication middleware for Stratax AI

Extracts user from JWT token and attaches to request state
for rate limiting, quotas, and personalized features.
"""
from fastapi import Request, Header
from typing import Optional
import logging
import jwt

logger = logging.getLogger(__name__)


def _extract_user_from_token(token: str):
    """
    Extract user from JWT token
    
    Returns User object or None
    """
    try:
        from app.auth import SECRET_KEY, ALGORITHM
        from app.database import get_db_context
        from app.models import User
        
        # Decode JWT
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        
        if not user_id:
            return None
        
        # Get user from database
        with get_db_context() as db:
            user = db.query(User).filter(User.id == user_id).first()
            return user
    
    except Exception as e:
        logger.debug(f"Token extraction failed: {e}")
        return None


async def user_auth_middleware(request: Request, call_next):
    """
    Middleware to extract and attach user to request state
    
    Checks for Authorization header with Bearer token (JWT)
    Attaches user object to request.state.user for downstream use
    
    If no valid token, request proceeds as guest (rate limited)
    """
    user = None
    
    # Check Authorization header (JWT token)
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
        user = _extract_user_from_token(token)
    
    # Attach user to request state (or None for guests)
    request.state.user = user
    
    # For backwards compatibility, also set user_id
    if user:
        request.state.user_id = user.id
        logger.debug(f"Authenticated request: {user.email} ({user.tier})")
    else:
        request.state.user_id = None
        logger.debug("Guest request (unauthenticated)")
    
    response = await call_next(request)
    return response


def get_user_id_from_request(request: Request) -> Optional[str]:
    """
    Helper function to get user_id from request state
    
    Args:
        request: FastAPI request object
        
    Returns:
        user_id if authenticated, None otherwise
    """
    return getattr(request.state, "user_id", None)

