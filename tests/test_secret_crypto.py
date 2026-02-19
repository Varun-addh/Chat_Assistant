from __future__ import annotations

from fastapi import FastAPI, Depends
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from app.models import Base, User
from app.database import get_db
from app.auth import hash_password, get_current_user
from app.routers.auth_routes import router as auth_router
from app.utils.secret_crypto import encrypt_secret, decrypt_secret


def test_secret_crypto_roundtrip(monkeypatch):
    from cryptography.fernet import Fernet

    key = Fernet.generate_key().decode("utf-8")
    monkeypatch.setenv("STRATAX_SECRETS_ENCRYPTION_KEY", key)

    enc = encrypt_secret("gsk_test_123")
    assert enc is not None
    assert enc.startswith("enc:")
    assert "gsk_test_123" not in enc

    dec = decrypt_secret(enc)
    assert dec == "gsk_test_123"


def test_secret_crypto_plaintext_passthrough(monkeypatch):
    # Plaintext remains supported (backward compatibility)
    monkeypatch.delenv("STRATAX_SECRETS_ENCRYPTION_KEY", raising=False)
    assert decrypt_secret("gsk_plain_abc") == "gsk_plain_abc"


def test_secret_crypto_encrypt_requires_key(monkeypatch):
    monkeypatch.delenv("STRATAX_SECRETS_ENCRYPTION_KEY", raising=False)

    try:
        encrypt_secret("gsk_plain_abc")
        assert False, "Expected encrypt_secret to raise when encryption key is missing"
    except RuntimeError as e:
        assert "STRATAX_SECRETS_ENCRYPTION_KEY" in str(e)


def test_update_profile_encrypts_provider_key_when_configured(monkeypatch):
    from cryptography.fernet import Fernet

    key = Fernet.generate_key().decode("utf-8")
    monkeypatch.setenv("STRATAX_SECRETS_ENCRYPTION_KEY", key)

    # In-memory DB for the route
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)

    # Seed a user
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

    resp = client.put(
        "/auth/me",
        json={"user_groq_api_key": "gsk_test_123"},
    )
    assert resp.status_code == 200

    with TestingSessionLocal() as db:
        updated = db.query(User).filter(User.email == "u@example.com").first()
        assert updated is not None
        assert updated.user_groq_api_key is not None
        assert updated.user_groq_api_key.startswith("enc:")
        assert decrypt_secret(updated.user_groq_api_key) == "gsk_test_123"
