from __future__ import annotations

import os
from pathlib import Path


# Ensure tests never run against a developer's Postgres instance.
# This must execute before application modules (and Settings) are imported.
os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("DATABASE_URL", "sqlite:///./data/pytest.db")
os.environ.setdefault("STRATAX_SESSION_STORE", "file")

# Best-effort: start from a clean sqlite file each test run.
# (Most tests also call init_db(), which resets schema under pytest.)
_pytest_db = Path("data/pytest.db")
try:
    _pytest_db.parent.mkdir(parents=True, exist_ok=True)
    if _pytest_db.exists():
        _pytest_db.unlink()
except Exception:
    # If the file is locked on Windows, init_db() will still drop/recreate tables.
    pass
