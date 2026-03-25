"""
Usage tracking utilities for billing and analytics
"""
from datetime import datetime, timezone
from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from sqlalchemy import func
from app.models import UsageRecord, SessionRecord, User, UserTier, TIER_QUOTAS
import logging

logger = logging.getLogger(__name__)


def check_copilot_quota(db: Session, user: Optional[User]) -> tuple[bool, str]:
    """Check if the user has remaining copilot quota for today.

    Returns (allowed, reason).
    - Guests and users without a tier => allowed (no enforcement for unauthenticated).
    - Authenticated users => enforced against TIER_QUOTAS.
    """
    if user is None:
        return True, "guest"

    tier_str = getattr(user, "tier", UserTier.FREE)
    try:
        tier = UserTier(tier_str)
    except (ValueError, KeyError):
        tier = UserTier.FREE

    quota = TIER_QUOTAS.get(tier, TIER_QUOTAS[UserTier.FREE])
    daily_limit = quota.get("daily_copilot_questions", 10)
    if daily_limit < 0:  # unlimited
        return True, "unlimited"

    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    count = (
        db.query(func.count(UsageRecord.id))
        .filter(
            UsageRecord.user_id == user.id,
            UsageRecord.feature == "copilot",
            UsageRecord.timestamp >= today_start,
        )
        .scalar()
    ) or 0

    if count >= daily_limit:
        return False, f"Daily copilot quota reached ({daily_limit}/{daily_limit}). Upgrade your plan for more."
    return True, f"{count}/{daily_limit}"


def track_api_usage(
    db: Session,
    user: Optional[User],
    feature: str,
    endpoint: str,
    tokens_used: int = 0,
    cost_usd: float = 0.0,
    metadata: dict = None,
    guest_user_id: Optional[str] = None,
):
    """
    Track API usage for a user
    
    Args:
        db: Database session
        user: User object (None for guest)
        feature: Feature name (copilot, mock_interview, practice_mode, etc.)
        endpoint: API endpoint path
        tokens_used: Number of tokens consumed
        cost_usd: Estimated cost in USD
        metadata: Additional context
    """
    try:
        if user is None:
            # Track guest usage with special user_id
            # Prefer a stable per-client guest id (e.g. "guest_<hash>") when available.
            user_id = guest_user_id or "guest"

            # Ensure the guest user exists in `users` table to satisfy FK constraints.
            # Create a minimal stub user record for analytics if missing.
            if user_id and user_id.startswith("guest"):
                try:
                    existing_user = db.get(User, user_id)
                    if existing_user is None:
                        # Create a minimal placeholder user. Fill required fields with safe placeholders.
                        placeholder_email = f"{user_id}@noreply.stratax.internal"
                        # Use an impossible-to-match hash so verify_password() always
                        # returns False rather than crashing on an empty/invalid hash.
                        placeholder_password = "!GUEST_STUB_NO_LOGIN!"
                        stub = User(
                            id=user_id,
                            email=placeholder_email,
                            username=user_id,
                            hashed_password=placeholder_password,
                            is_active=False,
                        )
                        db.add(stub)
                        db.commit()
                        logger.debug(f"Created stub guest user for analytics: {user_id}")
                except Exception as e:
                    # If creating the stub user fails for any reason, log and continue; we will still
                    # attempt to insert the usage record and handle FK failures gracefully below.
                    logger.debug(f"Could not create stub guest user {user_id}: {e}")
        else:
            user_id = user.id
        
        record = UsageRecord(
            user_id=user_id,
            feature=feature,
            endpoint=endpoint,
            tokens_used=tokens_used,
            api_calls=1,
            cost_usd=cost_usd,
            extra_data=metadata or {},
            timestamp=datetime.now(timezone.utc)
        )
        
        db.add(record)
        db.commit()
        
        logger.debug(f"📊 Usage tracked: user={user_id}, feature={feature}, tokens={tokens_used}")
    
    except Exception as e:
        logger.error(f"❌ Failed to track usage: {e}")
        db.rollback()


def track_session(
    db: Session,
    user: User,
    session_id: str,
    session_type: str,
):
    """
    Track a new session
    
    Args:
        db: Database session
        user: User object
        session_id: Session identifier
        session_type: Type of session (qa, mock_interview, practice_mode)
    """
    try:
        now = datetime.now(timezone.utc)

        # Idempotency: if the session already exists (e.g. retry, refresh, race),
        # update activity instead of failing on UNIQUE constraint.
        existing = db.get(SessionRecord, session_id)
        if existing is not None:
            existing.last_activity = now
            existing.is_active = True
            # Keep metadata in sync (best-effort; avoid surprising flips)
            if existing.session_type != session_type:
                existing.session_type = session_type
            if existing.user_id != user.id:
                existing.user_id = user.id
            db.commit()
            logger.debug(
                "📊 Session updated: user=%s, session=%s, type=%s",
                user.id,
                session_id,
                session_type,
            )
            return

        record = SessionRecord(
            id=session_id,
            user_id=user.id,
            session_type=session_type,
            question_count=0,
            is_active=True,
            created_at=now,
            last_activity=now,
        )
        
        db.add(record)
        db.commit()
        
        logger.debug(f"📊 Session tracked: user={user.id}, session={session_id}, type={session_type}")

    except IntegrityError:
        # Handle race: another request inserted the same session_id between our
        # existence check and insert.
        db.rollback()
        try:
            now = datetime.now(timezone.utc)
            existing = db.get(SessionRecord, session_id)
            if existing is not None:
                existing.last_activity = now
                existing.is_active = True
                if existing.session_type != session_type:
                    existing.session_type = session_type
                if existing.user_id != user.id:
                    existing.user_id = user.id
                db.commit()
                logger.debug(
                    "📊 Session updated after race: user=%s, session=%s, type=%s",
                    user.id,
                    session_id,
                    session_type,
                )
                return
        except Exception:
            db.rollback()
        logger.warning(
            "Session tracking ignored duplicate session_id=%s (user=%s)",
            session_id,
            user.id,
        )

    except Exception as e:
        logger.error(f"❌ Failed to track session: {e}")
        db.rollback()


def get_user_usage_stats(db: Session, user_id: str, days: int = 1):
    """
    Get usage statistics for a user
    
    Args:
        db: Database session
        user_id: User ID
        days: Number of days to look back
    
    Returns:
        Dict with usage stats
    """
    from datetime import timedelta
    
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    
    # Query usage records
    records = db.query(UsageRecord).filter(
        UsageRecord.user_id == user_id,
        UsageRecord.timestamp >= cutoff
    ).all()
    
    # Aggregate stats
    stats = {
        "total_api_calls": len(records),
        "total_tokens": sum(r.tokens_used for r in records),
        "total_cost_usd": sum(r.cost_usd for r in records),
        "features_used": {},
    }
    
    # Group by feature
    for record in records:
        feature = record.feature
        if feature not in stats["features_used"]:
            stats["features_used"][feature] = {
                "calls": 0,
                "tokens": 0,
                "cost": 0.0
            }
        
        stats["features_used"][feature]["calls"] += record.api_calls
        stats["features_used"][feature]["tokens"] += record.tokens_used
        stats["features_used"][feature]["cost"] += record.cost_usd
    
    return stats
