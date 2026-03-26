"""One-time migration: file-based sessions/history -> database.

This imports:
- Sessions: `data/sessions/<user_id>/<session_id>.json`
- History tabs: `data/history/<user_id>_history.jsonl`

Design goals:
- Idempotent: safe to re-run.
- No deletion by default (no data loss).

Usage (example):
    set DATABASE_URL=postgresql+psycopg://user:pass@localhost:5432/stratax
    python -m scripts.migrate_file_storage_to_db

Optional env:
- STRATAX_MIGRATE_DELETE_FILES=1  (DANGEROUS: will delete source files after successful insert)
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from app.database import init_db, get_db_context
from app.models import ChatSession, HistoryTabRecord


SESSIONS_ROOT = Path("data/sessions")
HISTORY_ROOT = Path("data/history")


def _parse_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value:
        try:
            # Accept both naive and tz-aware ISO strings.
            dt = datetime.fromisoformat(value)
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            pass
    return datetime.now(timezone.utc)


def _migrate_session_file(db, *, user_id: str, session_path: Path) -> bool:
    try:
        raw = json.loads(session_path.read_text(encoding="utf-8"))
    except Exception:
        return False

    session_id = raw.get("session_id") or session_path.stem
    qna = list(raw.get("qna") or [])

    state = {
        "id": session_id,
        "user_id": user_id,
        "qna": qna,
        "partial_transcript": raw.get("partial_transcript") or "",
        "profile_text": raw.get("profile_text") or "",
        "custom_title": raw.get("custom_title"),
        "last_update": _parse_dt(raw.get("last_update")),
    }

    existing = (
        db.query(ChatSession)
        .filter(ChatSession.user_id == user_id)
        .filter(ChatSession.id == session_id)
        .first()
    )
    if existing is None:
        existing = ChatSession(id=session_id, user_id=user_id)
        db.add(existing)

    existing.qna = state["qna"]
    existing.partial_transcript = state["partial_transcript"]
    existing.profile_text = state["profile_text"]
    existing.custom_title = state["custom_title"]
    existing.last_update = state["last_update"]
    if existing.created_at is None:
        existing.created_at = state["last_update"]

    return True


def _migrate_history_file(db, *, user_id: str, history_path: Path) -> int:
    imported = 0
    try:
        lines = history_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return 0

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            raw: Dict[str, Any] = json.loads(line)
        except Exception:
            continue

        tab_id = raw.get("tab_id")
        if not tab_id:
            continue

        existing = (
            db.query(HistoryTabRecord)
            .filter(HistoryTabRecord.user_id == user_id)
            .filter(HistoryTabRecord.tab_id == tab_id)
            .first()
        )
        if existing is None:
            existing = HistoryTabRecord(tab_id=tab_id, user_id=user_id)
            db.add(existing)

        existing.query = raw.get("query") or ""
        existing.questions = list(raw.get("questions") or [])
        existing.extra_data = dict(raw.get("metadata") or {})
        existing.created_at = _parse_dt(raw.get("created_at"))
        imported += 1

    return imported


def main() -> None:
    init_db()

    delete_sources = (os.getenv("STRATAX_MIGRATE_DELETE_FILES", "") or "").strip() == "1"

    session_files = list(SESSIONS_ROOT.glob("*/**/*.json")) if SESSIONS_ROOT.exists() else []
    history_files = list(HISTORY_ROOT.glob("*_history.jsonl")) if HISTORY_ROOT.exists() else []

    migrated_sessions = 0
    migrated_tabs = 0
    files_to_delete: list[Path] = []

    with get_db_context() as db:
        for p in session_files:
            # data/sessions/<user_id>/<session_id>.json
            try:
                user_id = p.parent.name
            except Exception:
                continue
            ok = _migrate_session_file(db, user_id=user_id, session_path=p)
            if ok:
                migrated_sessions += 1
                if delete_sources:
                    files_to_delete.append(p)

        for hp in history_files:
            # data/history/<user_id>_history.jsonl
            name = hp.name
            if not name.endswith("_history.jsonl"):
                continue
            user_id = name[: -len("_history.jsonl")]
            migrated_tabs += _migrate_history_file(db, user_id=user_id, history_path=hp)
            if delete_sources:
                files_to_delete.append(hp)

        db.commit()

    # Delete source files only AFTER successful commit
    if delete_sources:
        for p in files_to_delete:
            try:
                p.unlink()
            except Exception:
                pass

    print(f"✅ Migration complete: sessions={migrated_sessions}, history_tabs={migrated_tabs}")
    if delete_sources:
        print("⚠️ Source files deletion was enabled (STRATAX_MIGRATE_DELETE_FILES=1)")


if __name__ == "__main__":
    main()
