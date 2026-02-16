"""Add email verification and password reset columns

Revision ID: 9d1c1a2e3b4c
Revises: 62aab2f04f23
Create Date: 2026-01-28

"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "9d1c1a2e3b4c"
down_revision: Union[str, Sequence[str], None] = "62aab2f04f23"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("users") as batch:
        batch.add_column(sa.Column("email_verified_at", sa.DateTime(timezone=True), nullable=True))
        batch.add_column(sa.Column("email_verification_token_hash", sa.String(), nullable=True))
        batch.add_column(sa.Column("email_verification_expires_at", sa.DateTime(timezone=True), nullable=True))
        batch.add_column(sa.Column("password_reset_token_hash", sa.String(), nullable=True))
        batch.add_column(sa.Column("password_reset_expires_at", sa.DateTime(timezone=True), nullable=True))
        batch.add_column(sa.Column("password_reset_used_at", sa.DateTime(timezone=True), nullable=True))

    op.create_index(
        "ix_users_email_verification_token_hash",
        "users",
        ["email_verification_token_hash"],
        unique=False,
    )
    op.create_index(
        "ix_users_password_reset_token_hash",
        "users",
        ["password_reset_token_hash"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_users_password_reset_token_hash", table_name="users")
    op.drop_index("ix_users_email_verification_token_hash", table_name="users")

    with op.batch_alter_table("users") as batch:
        batch.drop_column("password_reset_used_at")
        batch.drop_column("password_reset_expires_at")
        batch.drop_column("password_reset_token_hash")
        batch.drop_column("email_verification_expires_at")
        batch.drop_column("email_verification_token_hash")
        batch.drop_column("email_verified_at")
