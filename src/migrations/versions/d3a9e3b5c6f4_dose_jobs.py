"""dose_jobs

Revision ID: d3a9e3b5c6f4
Revises: c2f8d2a4b5e3
Create Date: 2026-05-22 12:00:00.000000

Adds the ``dose_jobs`` table used by the async-mode Dose Service.
A single table backs both ``POST /api/v1/dose/compute`` and
``POST /api/v1/dose/influence`` — the ``kind`` column ('dose' /
'influence') discriminates which type of build a row tracks, and the
matching id column (``dose_id`` / ``influence_id``) is populated on
success.

Mirrors :mod:`b1e7c1f3a4f2_geometry_jobs` and
:mod:`c2f8d2a4b5e3_beam_model_jobs`.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "d3a9e3b5c6f4"
down_revision: Union[str, None] = "c2f8d2a4b5e3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "dose_jobs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("cache_key", sa.String(length=64), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("state", sa.String(length=20), nullable=False),
        sa.Column("progress", sa.Float(), nullable=True),
        sa.Column("stage", sa.String(length=32), nullable=True),
        sa.Column("message", sa.Text(), nullable=True),
        sa.Column("dose_id", sa.String(length=36), nullable=True),
        sa.Column("influence_id", sa.String(length=36), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_dose_jobs_cache_key",
        "dose_jobs",
        ["cache_key"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_dose_jobs_cache_key", table_name="dose_jobs")
    op.drop_table("dose_jobs")
