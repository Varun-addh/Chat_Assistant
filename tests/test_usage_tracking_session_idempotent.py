from __future__ import annotations

import uuid

from app.database import get_db_context, init_db
from app.models import SessionRecord, User
from app.utils.usage_tracking import track_session


def test_track_session_idempotent_on_duplicate_session_id():
    init_db()

    user_email = f"test_session_{uuid.uuid4()}@example.com"
    session_id = str(uuid.uuid4())

    with get_db_context() as db:
        user = User(email=user_email, hashed_password="x")
        db.add(user)
        db.commit()
        db.refresh(user)

        # First insert
        track_session(db, user, session_id, session_type="qa")
        # Duplicate insert should not raise; it should update last_activity
        track_session(db, user, session_id, session_type="qa")

        rec = db.get(SessionRecord, session_id)
        assert rec is not None
        assert rec.user_id == user.id
        assert rec.session_type == "qa"
        assert rec.is_active is True
