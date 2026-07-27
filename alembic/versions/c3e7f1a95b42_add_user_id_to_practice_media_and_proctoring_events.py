"""add user_id to practice_session_media and practice_proctoring_events

Revision ID: c3e7f1a95b42
Revises: b7d9a6e4c2f1
Create Date: 2026-07-27

Both tables were created by migration 3f8a2c7b6c10 without a ``user_id``
column. The column was later added to the models in app/models.py, but no
migration was written for it, and ``Base.metadata.create_all()`` (called from
init_db) only creates missing tables — it never ALTERs an existing one. Any
database created before that model change therefore lacks the column
permanently, which broke two endpoints at runtime:

    POST /api/practice/session/{id}/proctoring/event   -> 500
    POST /api/practice/session/{id}/media              -> 500
    GET  /api/practice/session/{id}/score              -> 500

    psycopg.errors.UndefinedColumn:
        column "user_id" of relation "practice_proctoring_events" does not exist

This migration is written to be idempotent. A database created by create_all()
from the *current* models already has the column, so applying the migration
there must be a no-op rather than an error.
"""

from alembic import op
import sqlalchemy as sa


revision = "c3e7f1a95b42"
down_revision = "b7d9a6e4c2f1"
branch_labels = None
depends_on = None


_TABLES = ("practice_session_media", "practice_proctoring_events")


def _existing_columns(table: str) -> set:
    inspector = sa.inspect(op.get_bind())
    if table not in inspector.get_table_names():
        return set()
    return {c["name"] for c in inspector.get_columns(table)}


def _existing_indexes(table: str) -> set:
    inspector = sa.inspect(op.get_bind())
    if table not in inspector.get_table_names():
        return set()
    return {i["name"] for i in inspector.get_indexes(table)}


def upgrade() -> None:
    for table in _TABLES:
        columns = _existing_columns(table)
        if not columns:
            # Table absent entirely — nothing to alter. create_all will build
            # it from the current models, user_id included.
            continue

        if "user_id" not in columns:
            # Nullable: existing rows have no user to attribute, and the model
            # declares it nullable too.
            op.add_column(table, sa.Column("user_id", sa.String(), nullable=True))

        index_name = f"ix_{table}_user_id"
        if index_name not in _existing_indexes(table):
            op.create_index(index_name, table, ["user_id"])


def downgrade() -> None:
    for table in _TABLES:
        columns = _existing_columns(table)
        if not columns:
            continue

        index_name = f"ix_{table}_user_id"
        if index_name in _existing_indexes(table):
            op.drop_index(index_name, table_name=table)

        if "user_id" in columns:
            op.drop_column(table, "user_id")
