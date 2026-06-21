"""bao_jobs

Revision ID: f5c2a3b4d6e7
Revises: e4b1f2c3d5a6
Create Date: 2026-06-21 03:55:00.000000

Adds the ``bao_jobs`` table for the async-mode BAO Service (Service 5).
Mirrors :mod:`e4b1f2c3d5a6_optimization_jobs`; ``bao_id`` is populated on
success.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "f5c2a3b4d6e7"
down_revision: Union[str, None] = "e4b1f2c3d5a6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "bao_jobs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("cache_key", sa.String(length=64), nullable=False),
        sa.Column("state", sa.String(length=20), nullable=False),
        sa.Column("progress", sa.Float(), nullable=True),
        sa.Column("stage", sa.String(length=32), nullable=True),
        sa.Column("message", sa.Text(), nullable=True),
        sa.Column("bao_id", sa.String(length=36), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_bao_jobs_cache_key", "bao_jobs", ["cache_key"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_bao_jobs_cache_key", table_name="bao_jobs")
    op.drop_table("bao_jobs")
