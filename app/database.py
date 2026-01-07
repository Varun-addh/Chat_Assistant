"""
Database connection and session management
"""
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from contextlib import contextmanager
from typing import Generator
from app.models import Base
import logging
import os

logger = logging.getLogger(__name__)

# Database URL (SQLite for now, easy to upgrade to PostgreSQL)
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/stratax.db")

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
                # Create a unique index to enforce uniqueness when google_id is present.
                conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS ix_users_google_id ON users (google_id)"))
                conn.commit()
        logger.info(f"✅ Database initialized: {DATABASE_URL}")
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
