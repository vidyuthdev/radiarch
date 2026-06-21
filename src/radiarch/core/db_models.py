"""SQLAlchemy ORM models for plans, jobs, and artifacts."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import relationship

from .database import Base


def _utcnow():
    return datetime.now(timezone.utc)


class PlanRow(Base):
    __tablename__ = "plans"

    id = Column(String(36), primary_key=True)
    workflow_id = Column(String(64), nullable=False)
    status = Column(String(20), nullable=False, default="queued")
    study_instance_uid = Column(String(128), nullable=False)
    segmentation_uid = Column(String(128), nullable=True)
    prescription_gy = Column(Float, nullable=False)
    fraction_count = Column(Integer, nullable=False, default=1)
    beam_count = Column(Integer, nullable=False, default=1)
    notes = Column(Text, nullable=True)
    job_id = Column(String(36), nullable=True)
    qa_summary = Column(JSON, nullable=True)
    objectives = Column(JSON, nullable=True)     # Phase 8A: List[DoseObjective] as JSON
    robustness = Column(JSON, nullable=True)     # Phase 8C: RobustnessConfig as JSON
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False)

    # Relationships
    artifacts = relationship("ArtifactRow", back_populates="plan", cascade="all, delete-orphan")
    job = relationship("JobRow", back_populates="plan", uselist=False, cascade="all, delete-orphan")


class JobRow(Base):
    __tablename__ = "jobs"

    id = Column(String(36), primary_key=True)
    plan_id = Column(String(36), ForeignKey("plans.id"), nullable=False)
    state = Column(String(20), nullable=False, default="queued")
    progress = Column(Float, default=0.0)
    message = Column(Text, nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=True)
    finished_at = Column(DateTime(timezone=True), nullable=True)

    plan = relationship("PlanRow", back_populates="job")


class ArtifactRow(Base):
    __tablename__ = "artifacts"

    id = Column(String(36), primary_key=True)
    plan_id = Column(String(36), ForeignKey("plans.id"), nullable=False)
    file_path = Column(Text, nullable=False)
    content_type = Column(String(64), default="application/dicom")
    file_name = Column(String(256), default="")
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)

    plan = relationship("PlanRow", back_populates="artifacts")


class GeometryJobRow(Base):
    """Async tracking row for one ``POST /geometry/build`` invocation.

    Unlike ``JobRow`` this isn't tied to a plan. ``cache_key`` is indexed
    so we can look up in-flight builds for the same inputs (future dedup
    work) and ``geometry_id`` is populated when the build succeeds so
    clients polling the job endpoint can follow the link to the result.
    """

    __tablename__ = "geometry_jobs"

    id = Column(String(36), primary_key=True)
    cache_key = Column(String(64), nullable=False, index=True)
    state = Column(String(20), nullable=False, default="queued")
    progress = Column(Float, default=0.0)
    # Persist stage across restarts — clients polling want to see the
    # current pipeline phase (loading_dicom / rasterizing_contours / …).
    stage = Column(String(32), nullable=True, default="queued")
    message = Column(Text, nullable=True)
    geometry_id = Column(String(36), nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=True)
    finished_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)


class BeamModelJobRow(Base):
    """Async tracking row for one ``POST /beam-model/build`` invocation.

    Mirrors :class:`GeometryJobRow` exactly — same fields, different
    table. ``beam_model_id`` is populated when the build succeeds so
    clients polling the jobs endpoint can deep-link to the result.
    """

    __tablename__ = "beam_model_jobs"

    id = Column(String(36), primary_key=True)
    cache_key = Column(String(64), nullable=False, index=True)
    state = Column(String(20), nullable=False, default="queued")
    progress = Column(Float, default=0.0)
    stage = Column(String(32), nullable=True, default="queued")
    message = Column(Text, nullable=True)
    beam_model_id = Column(String(36), nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=True)
    finished_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)


class DoseJobRow(Base):
    """Async tracking row for one ``POST /dose/compute`` or ``/dose/influence``.

    A single table backs both kinds — ``kind`` discriminates and the
    matching id column (``dose_id`` / ``influence_id``) is populated on
    success.
    """

    __tablename__ = "dose_jobs"

    id = Column(String(36), primary_key=True)
    cache_key = Column(String(64), nullable=False, index=True)
    kind = Column(String(16), nullable=False, default="dose")
    state = Column(String(20), nullable=False, default="queued")
    progress = Column(Float, default=0.0)
    stage = Column(String(32), nullable=True, default="queued")
    message = Column(Text, nullable=True)
    dose_id = Column(String(36), nullable=True)
    influence_id = Column(String(36), nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=True)
    finished_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)


class OptimizationJobRow(Base):
    """Async tracking row for one ``POST /optimize/run`` invocation.

    Mirrors :class:`DoseJobRow` minus the dose/influence discriminator —
    optimization has a single output kind. ``optimization_id`` is populated
    when the run succeeds so clients polling the jobs endpoint can deep-link
    to the result.
    """

    __tablename__ = "optimization_jobs"

    id = Column(String(36), primary_key=True)
    cache_key = Column(String(64), nullable=False, index=True)
    state = Column(String(20), nullable=False, default="queued")
    progress = Column(Float, default=0.0)
    stage = Column(String(32), nullable=True, default="queued")
    message = Column(Text, nullable=True)
    optimization_id = Column(String(36), nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=True)
    finished_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)


class BAOJobRow(Base):
    """Async tracking row for one ``POST /bao/run`` invocation (Service 5).

    Mirrors :class:`OptimizationJobRow`; ``bao_id`` is populated on success so
    clients polling the jobs endpoint can deep-link to the result.
    """

    __tablename__ = "bao_jobs"

    id = Column(String(36), primary_key=True)
    cache_key = Column(String(64), nullable=False, index=True)
    state = Column(String(20), nullable=False, default="queued")
    progress = Column(Float, default=0.0)
    stage = Column(String(32), nullable=True, default="queued")
    message = Column(Text, nullable=True)
    bao_id = Column(String(36), nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=True)
    finished_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)


class EvaluationJobRow(Base):
    """Async tracking row for one ``POST /evaluate/run`` invocation (Service 6).

    Mirrors :class:`OptimizationJobRow`; ``evaluation_id`` is populated on
    success.
    """

    __tablename__ = "evaluation_jobs"

    id = Column(String(36), primary_key=True)
    cache_key = Column(String(64), nullable=False, index=True)
    state = Column(String(20), nullable=False, default="queued")
    progress = Column(Float, default=0.0)
    stage = Column(String(32), nullable=True, default="queued")
    message = Column(Text, nullable=True)
    evaluation_id = Column(String(36), nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=True)
    finished_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
