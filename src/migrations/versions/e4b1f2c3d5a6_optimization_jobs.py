"""optimization_jobs

Revision ID: e4b1f2c3d5a6
Revises: d3a9e3b5c6f4
Create Date: 2026-06-21 03:30:00.000000

Adds the ``optimization_jobs`` table used by the async-mode Optimization
Service (Service 4). Mirrors :mod:`d3a9e3b5c6f4_dose_jobs` minus the
dose/influence ``kind`` discriminator — optimization has a single output
kind, so the success-id column is just ``optimization_id``.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "e4b1f2c3d5a6"
down_revision: Union[str, None] = "d3a9e3b5c6f4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "optimization_jobs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("cache_key", sa.String(length=64), nullable=False),
        sa.Column("state", sa.String(length=20), nullable=False),
        sa.Column("progress", sa.Float(), nullable=True),
        sa.Column("stage", sa.String(length=32), nullable=True),
        sa.Column("message", sa.Text(), nullable=True),
        sa.Column("optimization_id", sa.String(length=36), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_optimization_jobs_cache_key",
        "optimization_jobs",
        ["cache_key"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_optimization_jobs_cache_key", table_name="optimization_jobs")
    op.drop_table("optimization_jobs")
