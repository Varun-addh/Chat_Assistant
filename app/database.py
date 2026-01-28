"""
Database connection and session management
"""
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from contextlib import contextmanager
from typing import Generator
from app.models import Base
from app.config import settings
import logging
import os

logger = logging.getLogger(__name__)

# Database URL
# NOTE: Prefer settings so env-file layering and aliases are respected.
DATABASE_URL = (settings.database_url or "sqlite:///./data/stratax.db").strip()

# Normalize Postgres URL to psycopg3 driver if user provided a bare scheme.
# Many setups use psycopg3 (psycopg) but SQLAlchemy defaults to psycopg2 for
# "postgresql://" unless a driver is specified.
if DATABASE_URL.startswith("postgresql://") and "+" not in DATABASE_URL.split("://", 1)[0]:
    logger.warning(
        "DATABASE_URL uses 'postgresql://' without an explicit driver; "
        "using psycopg3 by rewriting to 'postgresql+psycopg://'."
    )
    DATABASE_URL = "postgresql+psycopg://" + DATABASE_URL.removeprefix("postgresql://")

# Create engine
if DATABASE_URL.startswith("sqlite"):
    # SQLite-specific settings for better concurrency
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False,
    )
else:
    # PostgreSQL/MySQL settings
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=20,
        echo=False,
    )

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Initialize database tables"""
    try:
        # Ensure data directory exists
        os.makedirs("data", exist_ok=True)
        
        # Create all tables
        Base.metadata.create_all(bind=engine)

        # Lightweight SQLite "migration" for new columns we add over time.
        # (For production, prefer Alembic migrations.)
        if DATABASE_URL.startswith("sqlite"):
            with engine.connect() as conn:
                cols = [row[1] for row in conn.execute(text("PRAGMA table_info(users)"))]
                if "google_id" not in cols:
                    conn.execute(text("ALTER TABLE users ADD COLUMN google_id VARCHAR"))
                if "email_verified_at" not in cols:
                    conn.execute(text("ALTER TABLE users ADD COLUMN email_verified_at DATETIME"))
                if "email_verification_token_hash" not in cols:
                    conn.execute(text("ALTER TABLE users ADD COLUMN email_verification_token_hash VARCHAR"))
                if "email_verification_expires_at" not in cols:
                    conn.execute(text("ALTER TABLE users ADD COLUMN email_verification_expires_at DATETIME"))
                if "password_reset_token_hash" not in cols:
                    conn.execute(text("ALTER TABLE users ADD COLUMN password_reset_token_hash VARCHAR"))
                if "password_reset_expires_at" not in cols:
                    conn.execute(text("ALTER TABLE users ADD COLUMN password_reset_expires_at DATETIME"))
                if "password_reset_used_at" not in cols:
                    conn.execute(text("ALTER TABLE users ADD COLUMN password_reset_used_at DATETIME"))
                # Create a unique index to enforce uniqueness when google_id is present.
                conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS ix_users_google_id ON users (google_id)"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS ix_users_email_verification_token_hash ON users (email_verification_token_hash)"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS ix_users_password_reset_token_hash ON users (password_reset_token_hash)"))
                conn.commit()
        else:
            # Lightweight additive migrations for Postgres/MySQL-like DBs.
            # This keeps production deployments resilient even if Alembic migrations
            # weren't executed for a fresh database.
            with engine.connect() as conn:
                # Postgres supports IF NOT EXISTS; other dialects may ignore it.
                # We keep these statements simple and additive.
                conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS email_verified_at TIMESTAMPTZ"))
                conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS email_verification_token_hash VARCHAR"))
                conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS email_verification_expires_at TIMESTAMPTZ"))
                conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS password_reset_token_hash VARCHAR"))
                conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS password_reset_expires_at TIMESTAMPTZ"))
                conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS password_reset_used_at TIMESTAMPTZ"))
                # Indexes (safe no-ops if already present)
                conn.execute(text("CREATE INDEX IF NOT EXISTS ix_users_email_verification_token_hash ON users (email_verification_token_hash)"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS ix_users_password_reset_token_hash ON users (password_reset_token_hash)"))
                conn.commit()
        logger.info("✅ Database initialized: %s", DATABASE_URL)
        env = (getattr(settings, "app_env", "development") or "development").strip().lower()
        if env in {"production", "prod"} and DATABASE_URL.startswith("sqlite"):
            logger.warning("⚠️  Running SQLite in production environment (%s). Use Postgres for concurrency.", env)
    except Exception as e:
        logger.error(f"❌ Failed to initialize database: {e}")
        raise


def get_db() -> Generator[Session, None, None]:
    """
    Dependency for FastAPI routes to get database session
    
    Usage:
        @app.get("/users")
        def get_users(db: Session = Depends(get_db)):
            return db.query(User).all()
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_context():
    """
    Context manager for database session (for non-FastAPI code)
    
    Usage:
        with get_db_context() as db:
            user = db.query(User).filter_by(email=email).first()
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
