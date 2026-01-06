"""
User authentication middleware for Stratax AI

Extracts user_id from various sources and attaches to request state
for personalized history and session management.
"""
from fastapi import Request, Header
from typing import Optional
import logging

logger = logging.getLogger(__name__)


async def user_auth_middleware(request: Request, call_next):
    """
    Middleware to extract and attach user_id to request state
    
    Checks for user_id in the following order:
    1. X-User-ID header
    2. Authorization header (Bearer token - extract user from JWT)
    3. Query parameter user_id
    4. Cookie user_id
    
    If no user_id found, request proceeds without authentication (guest mode)
    """
    user_id = None
    
    # 1. Check X-User-ID header (simplest method for frontend)
    user_id = request.headers.get("X-User-ID") or request.headers.get("x-user-id")
    
    # 2. Check Authorization header (for JWT tokens)
    if not user_id:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            # For now, use token as user_id (in production, decode JWT)
            # TODO: Implement JWT decoding to extract user_id
            user_id = _extract_user_from_token(token)
    
    # 3. Check query parameter
    if not user_id:
        user_id = request.query_params.get("user_id")
    
    # 4. Check cookie
    if not user_id:
        user_id = request.cookies.get("user_id")
    
    # Attach user_id to request state
    if user_id:
        request.state.user_id = user_id
        logger.debug(f"Authenticated request for user: {user_id}")
    else:
        # Guest mode - no user_id attached
        logger.debug("Guest request (no user_id)")
    
    response = await call_next(request)
    return response


def _extract_user_from_token(token: str) -> Optional[str]:
    """
    Extract user_id from JWT token
    
    For now, this is a placeholder. In production:
    1. Decode JWT using secret key
    2. Verify signature
    3. Extract user_id from payload
    4. Check expiration
    
    Args:
        token: JWT token string
        
    Returns:
        user_id if valid, None otherwise
    """
    try:
        # Placeholder: In production, use PyJWT library
        # import jwt
        # payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        # return payload.get("user_id")
        
        # For now, accept token as-is if it looks like a user_id
        # This allows testing without full JWT implementation
        if token and len(token) > 0:
            # Simple validation: alphanumeric + hyphens (UUID-like)
            if all(c.isalnum() or c in ['-', '_'] for c in token):
                return token
        
        return None
    except Exception as e:
        logger.warning(f"Failed to extract user from token: {e}")
        return None


def get_user_id_from_request(request: Request) -> Optional[str]:
    """
    Helper function to get user_id from request state
    
    Args:
        request: FastAPI request object
        
    Returns:
        user_id if authenticated, None otherwise
    """
    return getattr(request.state, "user_id", None)
