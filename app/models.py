"""
Database models for Stratax AI
"""
from datetime import datetime, timezone
from typing import Optional, List
from enum import Enum
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import uuid

Base = declarative_base()


class UserTier(str, Enum):
    """User subscription tiers"""
    FREE = "free"
    BASIC = "basic"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class User(Base):
    """User account model"""
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, nullable=False, index=True)
    username = Column(String, unique=True, nullable=True, index=True)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String, nullable=True)

    # OAuth identities
    google_id = Column(String, unique=True, nullable=True, index=True)
    
    # Subscription
    tier = Column(String, default=UserTier.FREE, nullable=False)
    stripe_customer_id = Column(String, nullable=True)
    subscription_start = Column(DateTime(timezone=True), nullable=True)
    subscription_end = Column(DateTime(timezone=True), nullable=True)
    
    # API Keys (if user provides their own)
    user_groq_api_key = Column(String, nullable=True)
    user_gemini_api_key = Column(String, nullable=True)
    
    # Status
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    last_login = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    usage_records = relationship("UsageRecord", back_populates="user")
    sessions = relationship("SessionRecord", back_populates="user")
    
    def __repr__(self):
        return f"<User {self.email} ({self.tier})>"


class UsageRecord(Base):
    """Track API usage per user for billing and analytics"""
    __tablename__ = "usage_records"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    
    # What was used
    feature = Column(String, nullable=False, index=True)  # "copilot", "mock_interview", "practice_mode", etc.
    endpoint = Column(String, nullable=True)
    
    # Resource consumption
    tokens_used = Column(Integer, default=0)
    api_calls = Column(Integer, default=1)
    cost_usd = Column(Float, default=0.0)  # Estimated cost
    
    # Additional context (renamed from 'metadata' which is reserved by SQLAlchemy)
    extra_data = Column(JSON, nullable=True)  # Store additional context
    
    # Timestamp
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False, index=True)
    
    # Relationships
    user = relationship("User", back_populates="usage_records")
    
    def __repr__(self):
        return f"<UsageRecord user={self.user_id} feature={self.feature}>"


class SessionRecord(Base):
    """Track user sessions for Q&A, interviews, etc."""
    __tablename__ = "session_records"
    
    id = Column(String, primary_key=True)  # Use existing session IDs
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    
    # Session type
    session_type = Column(String, nullable=False)  # "qa", "mock_interview", "practice_mode"
    
    # Session data (simplified)
    question_count = Column(Integer, default=0)
    duration_seconds = Column(Integer, nullable=True)
    
    # Status
    is_active = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    last_activity = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    # Relationships
    user = relationship("User", back_populates="sessions")
    
    def __repr__(self):
        return f"<SessionRecord {self.id} type={self.session_type}>"


class RateLimitRecord(Base):
    """In-database rate limiting (use Redis in production for better performance)"""
    __tablename__ = "rate_limit_records"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, nullable=False, index=True)
    endpoint = Column(String, nullable=False, index=True)
    
    # Rate limit window
    window_start = Column(DateTime(timezone=True), nullable=False, index=True)
    request_count = Column(Integer, default=1)
    
    def __repr__(self):
        return f"<RateLimitRecord user={self.user_id} endpoint={self.endpoint} count={self.request_count}>"


# Tier quotas (requests per day)
TIER_QUOTAS = {
    UserTier.FREE: {
        "daily_api_calls": 50,
        "daily_copilot_questions": 10,
        "daily_mock_interviews": 1,
        "daily_practice_sessions": 1,
        "max_tokens_per_request": 2000,
        "features": ["copilot", "basic_search"],
    },
    UserTier.BASIC: {
        "daily_api_calls": 500,
        "daily_copilot_questions": 100,
        "daily_mock_interviews": 10,
        "daily_practice_sessions": 5,
        "max_tokens_per_request": 4000,
        "features": ["copilot", "mock_interview", "basic_search", "architecture"],
    },
    UserTier.PRO: {
        "daily_api_calls": 5000,
        "daily_copilot_questions": 1000,
        "daily_mock_interviews": 100,
        "daily_practice_sessions": 50,
        "max_tokens_per_request": 8000,
        "features": ["copilot", "mock_interview", "practice_mode", "advanced_search", "architecture", "priority_support"],
    },
    UserTier.ENTERPRISE: {
        "daily_api_calls": 50000,
        "daily_copilot_questions": -1,  # unlimited
        "daily_mock_interviews": -1,
        "daily_practice_sessions": -1,
        "max_tokens_per_request": 16000,
        "features": ["all"],
    },
}
