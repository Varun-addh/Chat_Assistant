"""Authentication routes - register, login, user management.

Includes Google OAuth popup flow endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Request, BackgroundTasks
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from typing import Optional
from urllib.parse import urlencode
import secrets
import httpx
import time
from datetime import datetime, timedelta, timezone
import hashlib
import hmac

from app.database import get_db
from app.auth import (
    UserRegister, UserLogin, Token, TokenData,
    create_user, authenticate_user, create_access_token,
    get_current_user, hash_password
)
from app.models import User, TIER_QUOTAS
from app.config import settings
from app.utils.secret_crypto import encrypt_secret, encryption_configured
from app.utils.email_sender import send_email_best_effort, email_enabled
from pydantic import BaseModel, ConfigDict
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["Authentication"])

# ---------------------------------------------------------------------------
# Server-side OAuth state store (replaces cookies which fail behind HF proxy)
# ---------------------------------------------------------------------------
_OAUTH_STATE_TTL = 600  # 10 minutes
_oauth_states: dict[str, float] = {}


def _store_oauth_state(state: str) -> None:
    now = time.time()
    # prune expired entries
    for k in [k for k, v in _oauth_states.items() if v < now]:
        _oauth_states.pop(k, None)
    _oauth_states[state] = now + _OAUTH_STATE_TTL


def _consume_oauth_state(state: str) -> bool:
    expiry = _oauth_states.pop(state, None)
    return expiry is not None and time.time() <= expiry


def _frontend_callback_url(params: dict) -> str:
    """Build frontend callback URL with query params."""
    base = settings.frontend_url.rstrip("/")
    return f"{base}/auth/google/callback?{urlencode(params)}"


def _google_redirect_uri() -> str:
    """Build backend redirect URI registered in Google OAuth client."""
    return f"{settings.backend_base_url.rstrip('/')}/auth/google/callback"


def _ensure_unique_username(db: Session, base: str, max_attempts: int = 1000) -> str:
    """Generate a unique username derived from base."""
    candidate = base
    suffix = 0
    while suffix < max_attempts:
        exists = db.query(User).filter(User.username == candidate).first()
        if not exists:
            return candidate
        suffix += 1
        candidate = f"{base}{suffix}"
    # Fallback: use a UUID-based username
    return f"{base}_{secrets.token_hex(4)}"


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


class EmailOnly(BaseModel):
    """Email payload (avoid leaking user existence)."""
    email: str


class TokenOnly(BaseModel):
    token: str


class ResetPasswordBody(BaseModel):
    token: str
    new_password: str


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc_aware(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    # SQLite frequently round-trips timezone-aware datetimes as naive.
    # Treat naive values as UTC to avoid TypeError on comparisons.
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _hash_action_token(raw_token: str) -> str:
    """Hash tokens for DB storage.

    We use HMAC-SHA256 with COOKIE_SECRET (or JWT secret as fallback) so that even
    if DB is leaked, tokens can't be trivially reversed.
    """
    secret = (settings.cookie_secret or settings.jwt_secret_key or "").encode("utf-8")
    if not secret:
        # Should never happen in real deployments because settings enforces secrets,
        # but keep a safe fallback for unusual tests.
        secret = b"stratax-dev-secret"
    msg = (raw_token or "").encode("utf-8")
    return hmac.new(secret, msg, hashlib.sha256).hexdigest()


def _verification_link(token: str) -> str:
    base = settings.frontend_url.rstrip("/")
    return f"{base}/auth/verify-email?token={token}"


def _password_reset_link(token: str) -> str:
    base = settings.frontend_url.rstrip("/")
    return f"{base}/auth/reset-password?token={token}"


def _should_return_links_in_response() -> bool:
    # For local dev/testing UX we return the links.
    return (settings.app_env or "development").strip().lower() != "production"


@router.post("/request-email-verification")
async def request_email_verification(
    background_tasks: BackgroundTasks,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Request an email verification link for the current user."""
    if user.is_verified:
        return {"detail": "Email already verified"}

    token = secrets.token_urlsafe(32)
    token_hash = _hash_action_token(token)
    expires_at = _now_utc() + timedelta(minutes=int(getattr(settings, "email_verification_token_ttl_minutes", 1440) or 1440))

    user.email_verification_token_hash = token_hash
    user.email_verification_expires_at = expires_at
    # Do not toggle is_verified here.
    db.commit()

    link = _verification_link(token)
    body = (
        "Verify your email for Stratax AI\n\n"
        f"Click this link to verify: {link}\n\n"
        "If you did not request this, you can ignore this email."
    )
    if email_enabled():
        background_tasks.add_task(
            send_email_best_effort,
            to_email=user.email,
            subject="Verify your Stratax AI email",
            body_text=body,
        )

    resp = {"detail": "Verification link generated"}
    if _should_return_links_in_response():
        resp["verification_url"] = link
        resp["email_enabled"] = bool(email_enabled())
    return resp


@router.get("/verify-email")
async def verify_email(token: str, db: Session = Depends(get_db)):
    """Verify email by token (token is sent to the user's email)."""
    if not token or not token.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing token")

    token_hash = _hash_action_token(token)
    user = db.query(User).filter(User.email_verification_token_hash == token_hash).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid or expired token")

    now = _now_utc()
    exp = _as_utc_aware(user.email_verification_expires_at)
    if not exp or exp < now:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid or expired token")

    user.is_verified = True
    user.email_verified_at = now
    user.email_verification_token_hash = None
    user.email_verification_expires_at = None
    db.commit()

    return {"detail": "Email verified"}


@router.post("/forgot-password")
async def forgot_password(
    body: EmailOnly,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Request a password reset link.

    Always returns a generic response to avoid leaking whether a user exists.
    """
    email = (body.email or "").strip().lower()
    if not email:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing email")

    user = db.query(User).filter(User.email == email).first()
    reset_url = None
    if user and user.is_active:
        token = secrets.token_urlsafe(32)
        user.password_reset_token_hash = _hash_action_token(token)
        user.password_reset_expires_at = _now_utc() + timedelta(
            minutes=int(getattr(settings, "password_reset_token_ttl_minutes", 30) or 30)
        )
        user.password_reset_used_at = None
        db.commit()

        reset_url = _password_reset_link(token)
        body_text = (
            "Reset your Stratax AI password\n\n"
            f"Click this link to reset: {reset_url}\n\n"
            "If you did not request this, you can ignore this email."
        )
        if email_enabled():
            background_tasks.add_task(
                send_email_best_effort,
                to_email=user.email,
                subject="Reset your Stratax AI password",
                body_text=body_text,
            )

    resp = {"detail": "If an account exists for that email, a reset link was sent"}
    if _should_return_links_in_response() and reset_url:
        resp["reset_url"] = reset_url
        resp["email_enabled"] = bool(email_enabled())
    return resp


@router.post("/reset-password")
async def reset_password(body: ResetPasswordBody, db: Session = Depends(get_db)):
    """Reset password using a single-use token."""
    token = (body.token or "").strip()
    if not token:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing token")
    if not (body.new_password or "").strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing new_password")

    token_hash = _hash_action_token(token)
    user = db.query(User).filter(User.password_reset_token_hash == token_hash).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid or expired token")

    now = _now_utc()
    if user.password_reset_used_at is not None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid or expired token")
    exp = _as_utc_aware(user.password_reset_expires_at)
    if not exp or exp < now:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid or expired token")

    user.hashed_password = hash_password(body.new_password)
    user.password_reset_used_at = now
    user.password_reset_token_hash = None
    user.password_reset_expires_at = None
    user.last_login = None
    db.commit()

    return {"detail": "Password reset successful"}


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

    _store_oauth_state(state)
    logger.info("[OAUTH] Stored state=%s, total_states=%d", state, len(_oauth_states))

    url = "https://accounts.google.com/o/oauth2/v2/auth?" + urlencode(query)
    return RedirectResponse(url=url, status_code=302)


@router.get("/google/callback")
async def google_callback(
    request: Request,
    code: Optional[str] = None,
    error: Optional[str] = None,
    state: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """Handle Google OAuth callback, create/link user, and redirect to frontend with JWT."""
    logger.info("[OAUTH-CB] Entering callback. state=%s, code=%s, error=%s, frontend_url=%s",
                state, bool(code), error, settings.frontend_url)
    logger.info("[OAUTH-CB] Known states: %s", list(_oauth_states.keys()))

    if error:
        url = _frontend_callback_url({"error": error})
        logger.warning("[OAUTH-CB] Google returned error=%s, redirecting to %s", error, url)
        return RedirectResponse(url)

    if not code:
        url = _frontend_callback_url({"error": "no_code"})
        logger.warning("[OAUTH-CB] No code, redirecting to %s", url)
        return RedirectResponse(url)

    if not state or not _consume_oauth_state(state):
        url = _frontend_callback_url({"error": "invalid_state"})
        logger.warning("[OAUTH-CB] State mismatch! state=%s, remaining_states=%s, redirecting to %s",
                       state, list(_oauth_states.keys()), url)
        return RedirectResponse(url)

    logger.info("[OAUTH-CB] State validated OK, proceeding with token exchange")

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
            try:
                db.add(user)
                db.commit()
                db.refresh(user)
                logger.info("✅ New Google user created: %s (ID: %s)", user.email, user.id)
            except IntegrityError:
                db.rollback()
                # Race: another request created the user between our SELECT and INSERT.
                user = db.query(User).filter(User.email == email).first()
                if not user:
                    user = db.query(User).filter(User.google_id == google_id).first()
                if not user:
                    return RedirectResponse(_frontend_callback_url({"error": "account_conflict"}))
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
        url = _frontend_callback_url(params)
        logger.info("[OAUTH-CB] SUCCESS! Redirecting user %s to %s", user.email, url)
        return RedirectResponse(url)

    except Exception as e:
        logger.error("[OAUTH-CB] Exception: %s", e, exc_info=True)
        url = _frontend_callback_url({"error": "oauth_failed"})
        logger.error("[OAUTH-CB] Redirecting to %s", url)
        return RedirectResponse(url)
