from __future__ import annotations

import os
from typing import Optional


_ENC_PREFIX = "enc:"


def _get_encryption_key() -> Optional[str]:
    # Prefer Settings env var; we read directly from env here to avoid import cycles.
    return (
        os.getenv("STRATAX_SECRETS_ENCRYPTION_KEY")
        or os.getenv("SECRETS_ENCRYPTION_KEY")
        or None
    )


def encryption_configured() -> bool:
    key = _get_encryption_key()
    return bool(key and key.strip())


def encrypt_secret(plain: Optional[str]) -> Optional[str]:
    """Encrypt a secret for DB storage.

    Returns values prefixed with 'enc:' so we can remain backward compatible
    with older plaintext rows.
    """

    if plain is None:
        return None
    p = (plain or "").strip()
    if not p:
        return None

    key = _get_encryption_key()
    if not key or not key.strip():
        raise RuntimeError(
            "Secrets encryption is not configured. Set STRATAX_SECRETS_ENCRYPTION_KEY to store provider keys."
        )

    try:
        from cryptography.fernet import Fernet

        f = Fernet(key.encode("utf-8"))
        token = f.encrypt(p.encode("utf-8")).decode("utf-8")
        return _ENC_PREFIX + token
    except Exception as e:
        raise RuntimeError(f"Failed to encrypt secret: {e}")


def decrypt_secret(value: Optional[str]) -> Optional[str]:
    """Decrypt a DB secret value.

    - If the value is not encrypted (no 'enc:' prefix), returns it as-is.
    - If encrypted but the key is missing/invalid, raises.
    """

    if value is None:
        return None
    v = (value or "").strip()
    if not v:
        return None

    if not v.startswith(_ENC_PREFIX):
        # Backward-compatible: old plaintext rows.
        return v

    key = _get_encryption_key()
    if not key or not key.strip():
        raise RuntimeError(
            "Secrets encryption key missing. Set STRATAX_SECRETS_ENCRYPTION_KEY to decrypt stored provider keys."
        )

    token = v[len(_ENC_PREFIX) :]
    try:
        from cryptography.fernet import Fernet

        f = Fernet(key.encode("utf-8"))
        return f.decrypt(token.encode("utf-8")).decode("utf-8")
    except Exception as e:
        raise RuntimeError(f"Failed to decrypt secret: {e}")
