"""Add mock interview session table.

Revision ID: a1f4c32c9d90
Revises: 3f8a2c7b6c10
Create Date: 2026-03-17
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "a1f4c32c9d90"
down_revision = "3f8a2c7b6c10"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "mock_interview_sessions",
        sa.Column("session_id", sa.String(), primary_key=True),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("session_payload", sa.JSON(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_mock_interview_sessions_user_id", "mock_interview_sessions", ["user_id"])
    op.create_index("ix_mock_interview_sessions_status", "mock_interview_sessions", ["status"])
    op.create_index("ix_mock_interview_sessions_started_at", "mock_interview_sessions", ["started_at"])
    op.create_index("ix_mock_interview_sessions_completed_at", "mock_interview_sessions", ["completed_at"])
    op.create_index("ix_mock_interview_sessions_updated_at", "mock_interview_sessions", ["updated_at"])


def downgrade() -> None:
    op.drop_index("ix_mock_interview_sessions_updated_at", table_name="mock_interview_sessions")
    op.drop_index("ix_mock_interview_sessions_completed_at", table_name="mock_interview_sessions")
    op.drop_index("ix_mock_interview_sessions_started_at", table_name="mock_interview_sessions")
    op.drop_index("ix_mock_interview_sessions_status", table_name="mock_interview_sessions")
    op.drop_index("ix_mock_interview_sessions_user_id", table_name="mock_interview_sessions")
    op.drop_table("mock_interview_sessions")