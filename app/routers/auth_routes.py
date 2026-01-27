"""Authentication routes - register, login, user management.

Includes Google OAuth popup flow endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
from typing import Optional
from urllib.parse import urlencode
import secrets
import httpx

from app.database import get_db
from app.auth import (
    UserRegister, UserLogin, Token, TokenData,
    create_user, authenticate_user, create_access_token,
    get_current_user, hash_password
)
from app.models import User, TIER_QUOTAS
from app.config import settings
from app.utils.secret_crypto import encrypt_secret, encryption_configured
from pydantic import BaseModel, ConfigDict
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["Authentication"])


def _frontend_callback_url(params: dict) -> str:
    """Build frontend callback URL with query params."""
    base = settings.frontend_url.rstrip("/")
    return f"{base}/auth/google/callback?{urlencode(params)}"


def _google_redirect_uri() -> str:
    """Build backend redirect URI registered in Google OAuth client."""
    return f"{settings.backend_base_url.rstrip('/')}/auth/google/callback"


def _ensure_unique_username(db: Session, base: str) -> str:
    """Generate a unique username derived from base."""
    candidate = base
    suffix = 0
    while True:
        exists = db.query(User).filter(User.username == candidate).first()
        if not exists:
            return candidate
        suffix += 1
        candidate = f"{base}{suffix}"


class UserResponse(BaseModel):
    """Public user information"""
    id: str
    email: str
    username: Optional[str]
    full_name: Optional[str]
    tier: str
    is_verified: bool
    created_at: str
    
    model_config = ConfigDict(from_attributes=True)


class UserUpdate(BaseModel):
    """User profile update"""
    full_name: Optional[str] = None
    username: Optional[str] = None
    user_groq_api_key: Optional[str] = None
    user_gemini_api_key: Optional[str] = None


class PasswordChange(BaseModel):
    """Change password"""
    current_password: str
    new_password: str


@router.post("/register", response_model=Token, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserRegister, db: Session = Depends(get_db)):
    """
    Register a new user account
    
    Returns JWT token for immediate login
    """
    try:
        user = create_user(
            db=db,
            email=user_data.email,
            password=user_data.password,
            full_name=user_data.full_name,
            username=user_data.username,
        )
        
        # Create access token
        access_token = create_access_token(
            data={"sub": user.id, "email": user.email, "tier": user.tier}
        )
        
        logger.info(f"✅ New user registered: {user.email} (ID: {user.id})")
        
        return Token(
            access_token=access_token,
            user_id=user.id,
            tier=user.tier,
            email=user.email,
            full_name=user.full_name,
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Registration failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed",
        )


@router.post("/login", response_model=Token)
async def login(credentials: UserLogin, db: Session = Depends(get_db)):
    """
    Login with email and password
    
    Returns JWT token for authenticated requests
    """
    user = authenticate_user(db, credentials.email, credentials.password)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is inactive",
        )
    
    # Create access token
    access_token = create_access_token(
        data={"sub": user.id, "email": user.email, "tier": user.tier}
    )
    
    logger.info(f"✅ User logged in: {user.email}")
    
    return Token(
        access_token=access_token,
        user_id=user.id,
        tier=user.tier,
        email=user.email,
        full_name=user.full_name,
    )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(user: User = Depends(get_current_user)):
    """
    Get current user information
    
    Requires authentication
    """
    return UserResponse(
        id=user.id,
        email=user.email,
        username=user.username,
        full_name=user.full_name,
        tier=user.tier,
        is_verified=user.is_verified,
        created_at=user.created_at.isoformat(),
    )


@router.put("/me", response_model=UserResponse)
async def update_profile(
    update_data: UserUpdate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Update user profile
    
    Requires authentication
    """

    def _maybe_encrypt_provider_key(value: Optional[str]) -> Optional[str]:
        v = (value or "").strip()
        if not v:
            return None
        if encryption_configured():
            return encrypt_secret(v)
        if (settings.app_env or "").strip().lower() == "production":
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Provider key storage requires STRATAX_SECRETS_ENCRYPTION_KEY in production",
            )
        # Backward-compatible dev/test behavior: allow plaintext storage when encryption is not configured.
        return v

    # Update fields
    if update_data.full_name is not None:
        user.full_name = update_data.full_name
    
    if update_data.username is not None:
        # Check if username is taken
        existing = db.query(User).filter(
            User.username == update_data.username,
            User.id != user.id
        ).first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already taken",
            )
        user.username = update_data.username
    
    if update_data.user_groq_api_key is not None:
        user.user_groq_api_key = _maybe_encrypt_provider_key(update_data.user_groq_api_key)
    
    if update_data.user_gemini_api_key is not None:
        user.user_gemini_api_key = _maybe_encrypt_provider_key(update_data.user_gemini_api_key)
    
    db.commit()
    db.refresh(user)
    
    logger.info(f"✅ User profile updated: {user.email}")
    
    return UserResponse(
        id=user.id,
        email=user.email,
        username=user.username,
        full_name=user.full_name,
        tier=user.tier,
        is_verified=user.is_verified,
        created_at=user.created_at.isoformat(),
    )


@router.post("/change-password")
async def change_password(
    password_data: PasswordChange,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Change user password
    
    Requires authentication and current password
    """
    from app.auth import verify_password
    
    # Verify current password
    if not verify_password(password_data.current_password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect current password",
        )
    
    # Update password
    user.hashed_password = hash_password(password_data.new_password)
    db.commit()
    
    logger.info(f"✅ Password changed for user: {user.email}")
    
    return {"message": "Password changed successfully"}


@router.get("/quota")
async def get_quota_info(user: User = Depends(get_current_user)):
    """
    Get current user's quota and tier limits
    
    Requires authentication
    """
    tier_info = TIER_QUOTAS.get(user.tier, TIER_QUOTAS["free"])
    
    return {
        "tier": user.tier,
        "limits": tier_info,
        "message": f"You are on the {user.tier.upper()} tier",
    }


@router.get("/google")
async def google_login():
    """Initiate Google OAuth flow (popup redirects here)."""
    if not settings.google_client_id:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Google OAuth not configured (missing GOOGLE_CLIENT_ID)",
        )

    state = secrets.token_urlsafe(32)
    redirect_uri = _google_redirect_uri()

    query = {
        "client_id": settings.google_client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "offline",
        "prompt": "consent",
        "state": state,
    }

    url = "https://accounts.google.com/o/oauth2/v2/auth?" + urlencode(query)
    response = RedirectResponse(url=url, status_code=302)
    # Store state in an httpOnly cookie to protect against CSRF.
    response.set_cookie(
        key="oauth_state",
        value=state,
        httponly=True,
        samesite="lax",
    )
    return response


@router.get("/google/callback")
async def google_callback(
    request: Request,
    code: Optional[str] = None,
    error: Optional[str] = None,
    state: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """Handle Google OAuth callback, create/link user, and redirect to frontend with JWT."""
    if error:
        return RedirectResponse(_frontend_callback_url({"error": error}))

    if not code:
        return RedirectResponse(_frontend_callback_url({"error": "no_code"}))

    expected_state = request.cookies.get("oauth_state")
    if not expected_state or not state or state != expected_state:
        return RedirectResponse(_frontend_callback_url({"error": "invalid_state"}))

    if not settings.google_client_id or not settings.google_client_secret:
        return RedirectResponse(_frontend_callback_url({"error": "oauth_not_configured"}))

    redirect_uri = _google_redirect_uri()
    token_url = "https://oauth2.googleapis.com/token"
    userinfo_url = "https://www.googleapis.com/oauth2/v2/userinfo"

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            token_resp = await client.post(
                token_url,
                data={
                    "code": code,
                    "client_id": settings.google_client_id,
                    "client_secret": settings.google_client_secret,
                    "redirect_uri": redirect_uri,
                    "grant_type": "authorization_code",
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            if token_resp.status_code != 200:
                logger.error("Google token exchange failed: %s %s", token_resp.status_code, token_resp.text)
                return RedirectResponse(_frontend_callback_url({"error": "token_exchange_failed"}))

            token_data = token_resp.json()
            access_token = token_data.get("access_token")
            if not access_token:
                return RedirectResponse(_frontend_callback_url({"error": "no_access_token"}))

            userinfo_resp = await client.get(
                userinfo_url,
                headers={"Authorization": f"Bearer {access_token}"},
            )
            if userinfo_resp.status_code != 200:
                logger.error("Google userinfo failed: %s %s", userinfo_resp.status_code, userinfo_resp.text)
                return RedirectResponse(_frontend_callback_url({"error": "userinfo_failed"}))

            userinfo = userinfo_resp.json()

        google_id = userinfo.get("id")
        email = userinfo.get("email")
        full_name = userinfo.get("name")
        verified_email = bool(userinfo.get("verified_email"))

        if not google_id or not email:
            return RedirectResponse(_frontend_callback_url({"error": "missing_profile"}))

        # Find by google_id first, then email (so returning users work even if email changes).
        user = db.query(User).filter(User.google_id == google_id).first()
        if not user:
            user = db.query(User).filter(User.email == email).first()

        if not user:
            # Create new user with an unusable random password hash (JWT login only).
            base_username = (email.split("@", 1)[0] or "user").replace(".", "")
            username = _ensure_unique_username(db, base_username)
            random_pw = secrets.token_urlsafe(32)
            user = User(
                email=email,
                username=username,
                full_name=full_name,
                hashed_password=hash_password(random_pw),
                tier="free",
                is_active=True,
                is_verified=verified_email,
                google_id=google_id,
            )
            db.add(user)
            db.commit()
            db.refresh(user)
            logger.info("✅ New Google user created: %s (ID: %s)", user.email, user.id)
        else:
            # Link google_id if missing
            if not getattr(user, "google_id", None):
                user.google_id = google_id
            # Mark verified if Google says email is verified
            if verified_email and not user.is_verified:
                user.is_verified = True
            db.commit()

        jwt_token = create_access_token(data={"sub": user.id, "email": user.email, "tier": user.tier})

        params = {
            "token": jwt_token,
            "user_id": user.id,
            "email": user.email,
            "full_name": user.full_name or "",
            "tier": user.tier,
        }
        response = RedirectResponse(_frontend_callback_url(params))
        # Clear state cookie
        response.delete_cookie("oauth_state")
        return response

    except Exception as e:
        logger.error("Google OAuth callback failed: %s", e, exc_info=True)
        return RedirectResponse(_frontend_callback_url({"error": "oauth_failed"}))
