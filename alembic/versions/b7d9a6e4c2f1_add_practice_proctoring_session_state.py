"""Add practice proctoring session state table.

Revision ID: b7d9a6e4c2f1
Revises: a1f4c32c9d90
Create Date: 2026-03-17
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "b7d9a6e4c2f1"
down_revision = "a1f4c32c9d90"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "practice_proctoring_sessions",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("session_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=True),
        sa.Column("status", sa.String(), nullable=False, server_default="ACTIVE"),
        sa.Column("risk_level", sa.String(), nullable=False, server_default="LOW"),
        sa.Column("total_violations", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("serious_violations", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_event_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_heartbeat_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("terminated_reason", sa.String(), nullable=True),
        sa.Column("monitoring_metadata", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("session_id", name="uq_practice_proctoring_sessions_session_id"),
    )
    op.create_index("ix_practice_proctoring_sessions_session_id", "practice_proctoring_sessions", ["session_id"])
    op.create_index("ix_practice_proctoring_sessions_user_id", "practice_proctoring_sessions", ["user_id"])
    op.create_index("ix_practice_proctoring_sessions_status", "practice_proctoring_sessions", ["status"])
    op.create_index("ix_practice_proctoring_sessions_risk_level", "practice_proctoring_sessions", ["risk_level"])
    op.create_index("ix_practice_proctoring_sessions_last_event_at", "practice_proctoring_sessions", ["last_event_at"])
    op.create_index("ix_practice_proctoring_sessions_last_heartbeat_at", "practice_proctoring_sessions", ["last_heartbeat_at"])
    op.create_index("ix_practice_proctoring_sessions_created_at", "practice_proctoring_sessions", ["created_at"])
    op.create_index("ix_practice_proctoring_sessions_updated_at", "practice_proctoring_sessions", ["updated_at"])


def downgrade() -> None:
    op.drop_index("ix_practice_proctoring_sessions_updated_at", table_name="practice_proctoring_sessions")
    op.drop_index("ix_practice_proctoring_sessions_created_at", table_name="practice_proctoring_sessions")
    op.drop_index("ix_practice_proctoring_sessions_last_heartbeat_at", table_name="practice_proctoring_sessions")
    op.drop_index("ix_practice_proctoring_sessions_last_event_at", table_name="practice_proctoring_sessions")
    op.drop_index("ix_practice_proctoring_sessions_risk_level", table_name="practice_proctoring_sessions")
    op.drop_index("ix_practice_proctoring_sessions_status", table_name="practice_proctoring_sessions")
    op.drop_index("ix_practice_proctoring_sessions_user_id", table_name="practice_proctoring_sessions")
    op.drop_index("ix_practice_proctoring_sessions_session_id", table_name="practice_proctoring_sessions")
    op.drop_table("practice_proctoring_sessions")