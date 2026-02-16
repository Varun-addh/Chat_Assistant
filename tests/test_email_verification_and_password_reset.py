from __future__ import annotations

import os
from urllib.parse import urlparse, parse_qs

# Ensure importing app.database does not require Postgres drivers during unit tests.
os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")
os.environ.setdefault("APP_ENV", "development")

from fastapi import FastAPI, Depends
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from app.models import Base, User
from app.database import get_db
from app.auth import hash_password, verify_password, get_current_user
from app.routers.auth_routes import router as auth_router
from app.config import settings


def _token_from_url(url: str) -> str:
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    token_list = qs.get("token") or []
    assert token_list and token_list[0]
    return token_list[0]


def test_email_verification_roundtrip(monkeypatch):
    monkeypatch.setattr(settings, "app_env", "development")
    monkeypatch.setattr(settings, "email_enabled", False)

    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)

    with TestingSessionLocal() as db:
        user = User(
            email="u@example.com",
            username="u",
            hashed_password=hash_password("pw"),
            full_name="User",
            tier="free",
            is_active=True,
            is_verified=False,
        )
        db.add(user)
        db.commit()

    def override_get_db():
        db: Session = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    async def override_get_current_user(db: Session = Depends(get_db)) -> User:
        u = db.query(User).filter(User.email == "u@example.com").first()
        assert u is not None
        return u

    app = FastAPI()
    app.include_router(auth_router)
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_user] = override_get_current_user

    client = TestClient(app)

    resp = client.post("/auth/request-email-verification")
    assert resp.status_code == 200
    data = resp.json()
    assert "verification_url" in data

    token = _token_from_url(data["verification_url"])

    resp2 = client.get(f"/auth/verify-email?token={token}")
    assert resp2.status_code == 200

    with TestingSessionLocal() as db:
        updated = db.query(User).filter(User.email == "u@example.com").first()
        assert updated is not None
        assert updated.is_verified is True
        assert updated.email_verified_at is not None
        assert updated.email_verification_token_hash is None
        assert updated.email_verification_expires_at is None


def test_password_reset_roundtrip(monkeypatch):
    monkeypatch.setattr(settings, "app_env", "development")
    monkeypatch.setattr(settings, "email_enabled", False)

    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)

    with TestingSessionLocal() as db:
        user = User(
            email="u@example.com",
            username="u",
            hashed_password=hash_password("old_pw"),
            full_name="User",
            tier="free",
            is_active=True,
            is_verified=False,
        )
        db.add(user)
        db.commit()

    def override_get_db():
        db: Session = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app = FastAPI()
    app.include_router(auth_router)
    app.dependency_overrides[get_db] = override_get_db

    client = TestClient(app)

    resp = client.post("/auth/forgot-password", json={"email": "u@example.com"})
    assert resp.status_code == 200
    data = resp.json()
    assert "reset_url" in data
    token = _token_from_url(data["reset_url"])

    resp2 = client.post(
        "/auth/reset-password",
        json={"token": token, "new_password": "new_pw"},
    )
    assert resp2.status_code == 200

    with TestingSessionLocal() as db:
        updated = db.query(User).filter(User.email == "u@example.com").first()
        assert updated is not None
        assert verify_password("new_pw", updated.hashed_password) is True
        assert updated.password_reset_token_hash is None
        assert updated.password_reset_expires_at is None
        assert updated.password_reset_used_at is not None


def test_forgot_password_unknown_email_does_not_leak(monkeypatch):
    monkeypatch.setattr(settings, "app_env", "development")
    monkeypatch.setattr(settings, "email_enabled", False)

    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)

    def override_get_db():
        db: Session = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app = FastAPI()
    app.include_router(auth_router)
    app.dependency_overrides[get_db] = override_get_db

    client = TestClient(app)

    resp = client.post("/auth/forgot-password", json={"email": "missing@example.com"})
    assert resp.status_code == 200
    data = resp.json()
    assert "reset_url" not in data
