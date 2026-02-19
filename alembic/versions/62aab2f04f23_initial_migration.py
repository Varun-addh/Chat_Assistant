"""Initial migration

Revision ID: 62aab2f04f23
Revises: 
Create Date: 2026-01-27 13:34:46.618494

"""
from typing import Sequence, Union

from alembic import op

# For an initial baseline migration, we create/drop the current SQLAlchemy
# metadata. This avoids baking in a huge auto-generated script and ensures a
# fresh Postgres database gets the full schema.
from app.models import Base

# revision identifiers, used by Alembic.
revision: str = '62aab2f04f23'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    bind = op.get_bind()
    Base.metadata.create_all(bind=bind)


def downgrade() -> None:
    """Downgrade schema."""
    bind = op.get_bind()
    Base.metadata.drop_all(bind=bind)
