"""In-process smoke test for auth endpoints.

This avoids needing a separate uvicorn process. It also enables FAST_STARTUP
so heavy model initialization is skipped.

Run:
  python scripts/smoke_auth_inprocess.py
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Ensure repo root is on sys.path so `import app` works even if the script is
# executed from a different working directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Must be set before importing app.main (settings are evaluated at import time)
os.environ.setdefault("FAST_STARTUP", "true")
os.environ.setdefault("DISABLE_INTERVIEW_INTELLIGENCE", "true")
os.environ.setdefault("PRACTICE_MODE_ENABLED", "false")

from fastapi.testclient import TestClient

from app.main import app


def main() -> None:
    client = TestClient(app)

    r = client.get("/health")
    print("health", r.status_code, r.json())

    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    payload = {
        "email": f"test_inprocess_{ts}@stratax.ai",
        "password": "SecurePass123!",
        "full_name": "Test User",
        "username": f"test_inprocess_user_{ts}",
    }

    r = client.post("/auth/register", json=payload)
    print("register", r.status_code, r.text)

    if r.status_code != 201:
        raise SystemExit(1)

    token = r.json()["access_token"]
    r2 = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    print("me", r2.status_code, r2.text)


if __name__ == "__main__":
    main()
