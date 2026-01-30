"""Add practice media + proctoring tables.

Revision ID: 3f8a2c7b6c10
Revises: 9d1c1a2e3b4c
Create Date: 2026-01-29
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "3f8a2c7b6c10"
down_revision = "9d1c1a2e3b4c"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "practice_session_media",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("session_id", sa.String(), nullable=False),
        sa.Column("media_type", sa.String(), nullable=False),
        sa.Column("storage_url", sa.Text(), nullable=False),
        sa.Column("duration_seconds", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_practice_session_media_session_id", "practice_session_media", ["session_id"])
    op.create_index("ix_practice_session_media_media_type", "practice_session_media", ["media_type"])
    op.create_index("ix_practice_session_media_created_at", "practice_session_media", ["created_at"])

    op.create_table(
        "practice_proctoring_events",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("session_id", sa.String(), nullable=False),
        sa.Column("event_type", sa.String(), nullable=False),
        sa.Column("event_ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=True),
    )
    op.create_index("ix_practice_proctoring_events_session_id", "practice_proctoring_events", ["session_id"])
    op.create_index("ix_practice_proctoring_events_event_type", "practice_proctoring_events", ["event_type"])
    op.create_index("ix_practice_proctoring_events_event_ts", "practice_proctoring_events", ["event_ts"])


def downgrade() -> None:
    op.drop_index("ix_practice_proctoring_events_event_ts", table_name="practice_proctoring_events")
    op.drop_index("ix_practice_proctoring_events_event_type", table_name="practice_proctoring_events")
    op.drop_index("ix_practice_proctoring_events_session_id", table_name="practice_proctoring_events")
    op.drop_table("practice_proctoring_events")

    op.drop_index("ix_practice_session_media_created_at", table_name="practice_session_media")
    op.drop_index("ix_practice_session_media_media_type", table_name="practice_session_media")
    op.drop_index("ix_practice_session_media_session_id", table_name="practice_session_media")
    op.drop_table("practice_session_media")
