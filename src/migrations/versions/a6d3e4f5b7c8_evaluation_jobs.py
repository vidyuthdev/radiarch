"""evaluation_jobs

Revision ID: a6d3e4f5b7c8
Revises: f5c2a3b4d6e7
Create Date: 2026-06-21 04:10:00.000000

Adds the ``evaluation_jobs`` table for the async-mode Evaluation Service
(Service 6). Mirrors :mod:`f5c2a3b4d6e7_bao_jobs`; ``evaluation_id`` is
populated on success.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "a6d3e4f5b7c8"
down_revision: Union[str, None] = "f5c2a3b4d6e7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "evaluation_jobs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("cache_key", sa.String(length=64), nullable=False),
        sa.Column("state", sa.String(length=20), nullable=False),
        sa.Column("progress", sa.Float(), nullable=True),
        sa.Column("stage", sa.String(length=32), nullable=True),
        sa.Column("message", sa.Text(), nullable=True),
        sa.Column("evaluation_id", sa.String(length=36), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_evaluation_jobs_cache_key", "evaluation_jobs",
                    ["cache_key"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_evaluation_jobs_cache_key", table_name="evaluation_jobs")
    op.drop_table("evaluation_jobs")
