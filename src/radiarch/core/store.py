"""Store abstraction: StoreBase ABC, InMemoryStore, and SQLStore.

Tests use InMemoryStore.  Production uses SQLStore when RADIARCH_DATABASE_URL is set.
"""

from __future__ import annotations

import abc
import os
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

from ..models.plan import PlanDetail, PlanRequest, PlanSummary
from ..models.job import JobStatus, JobState
from ..models.artifact import ArtifactRecord
from ..models.geometry import GeometryJobStatus, GeometryStage
from ..models.beam_model import BeamModelJobStatus, BeamModelStage
from ..models.dose import DoseJobStatus, DoseStage
from ..models.optimization import OptimizationJobStatus, OptimizationStage
from ..models.bao import BAOJobStatus, BAOStage
from ..models.evaluation import EvaluationJobStatus, EvaluationStage


def _utcnow():
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class StoreBase(abc.ABC):
    """Interface that InMemoryStore and SQLStore both implement."""

    @abc.abstractmethod
    def create_plan(self, payload: PlanRequest) -> tuple[PlanDetail, JobStatus]: ...

    @abc.abstractmethod
    def list_plans(self) -> List[PlanSummary]: ...

    @abc.abstractmethod
    def get_plan(self, plan_id: str) -> Optional[PlanDetail]: ...

    @abc.abstractmethod
    def get_job(self, job_id: str) -> Optional[JobStatus]: ...

    @abc.abstractmethod
    def update_job(
        self,
        job_id: str,
        *,
        state: Optional[JobState] = None,
        progress: Optional[float] = None,
        message: Optional[str] = None,
        stage: Optional[str] = None,
        eta_seconds: Optional[float] = None,
    ) -> Optional[JobStatus]: ...

    @abc.abstractmethod
    def register_artifact(
        self,
        plan_id: str,
        file_path: str,
        content_type: str = "application/dicom",
        file_name: str = "",
    ) -> Optional[str]: ...

    @abc.abstractmethod
    def get_artifact(self, artifact_id: str) -> Optional[ArtifactRecord]: ...

    @abc.abstractmethod
    def attach_artifact(self, plan_id: str, artifact_id: str) -> None: ...

    @abc.abstractmethod
    def set_plan_summary(self, plan_id: str, summary: dict) -> None: ...

    @abc.abstractmethod
    def delete_plan(self, plan_id: str) -> bool:
        """Delete a plan and all associated jobs/artifacts. Returns True if deleted."""
        ...

    # ---- Geometry async jobs -----------------------------------------

    @abc.abstractmethod
    def create_geometry_job(self, cache_key: str) -> GeometryJobStatus:
        """Record a new queued geometry build. Returns the new status row."""
        ...

    @abc.abstractmethod
    def get_geometry_job(self, job_id: str) -> Optional[GeometryJobStatus]: ...

    @abc.abstractmethod
    def update_geometry_job(
        self,
        job_id: str,
        *,
        state: Optional[JobState] = None,
        progress: Optional[float] = None,
        message: Optional[str] = None,
        stage: Optional[GeometryStage] = None,
        geometry_id: Optional[str] = None,
    ) -> Optional[GeometryJobStatus]: ...

    # ---- Beam model async jobs ---------------------------------------

    @abc.abstractmethod
    def create_beam_model_job(self, cache_key: str) -> BeamModelJobStatus:
        """Record a new queued beam-model build."""
        ...

    @abc.abstractmethod
    def get_beam_model_job(self, job_id: str) -> Optional[BeamModelJobStatus]: ...

    @abc.abstractmethod
    def update_beam_model_job(
        self,
        job_id: str,
        *,
        state: Optional[JobState] = None,
        progress: Optional[float] = None,
        message: Optional[str] = None,
        stage: Optional[BeamModelStage] = None,
        beam_model_id: Optional[str] = None,
    ) -> Optional[BeamModelJobStatus]: ...

    # ---- Dose async jobs ---------------------------------------------

    @abc.abstractmethod
    def create_dose_job(self, cache_key: str, kind: str = "dose") -> DoseJobStatus:
        """Record a new queued dose (or influence) build."""
        ...

    @abc.abstractmethod
    def get_dose_job(self, job_id: str) -> Optional[DoseJobStatus]: ...

    @abc.abstractmethod
    def update_dose_job(
        self,
        job_id: str,
        *,
        state: Optional[JobState] = None,
        progress: Optional[float] = None,
        message: Optional[str] = None,
        stage: Optional[DoseStage] = None,
        dose_id: Optional[str] = None,
        influence_id: Optional[str] = None,
    ) -> Optional[DoseJobStatus]: ...

    # ---- Optimization async jobs -------------------------------------

    @abc.abstractmethod
    def create_optimization_job(self, cache_key: str) -> OptimizationJobStatus:
        """Record a new queued optimization run."""
        ...

    @abc.abstractmethod
    def get_optimization_job(self, job_id: str) -> Optional[OptimizationJobStatus]: ...

    @abc.abstractmethod
    def update_optimization_job(
        self,
        job_id: str,
        *,
        state: Optional[JobState] = None,
        progress: Optional[float] = None,
        message: Optional[str] = None,
        stage: Optional[OptimizationStage] = None,
        optimization_id: Optional[str] = None,
    ) -> Optional[OptimizationJobStatus]: ...

    @abc.abstractmethod
    def list_optimization_jobs(self) -> List[OptimizationJobStatus]: ...

    # ---- BAO async jobs ----------------------------------------------

    @abc.abstractmethod
    def create_bao_job(self, cache_key: str) -> BAOJobStatus:
        """Record a new queued beam-angle-optimization run."""
        ...

    @abc.abstractmethod
    def get_bao_job(self, job_id: str) -> Optional[BAOJobStatus]: ...

    @abc.abstractmethod
    def update_bao_job(
        self,
        job_id: str,
        *,
        state: Optional[JobState] = None,
        progress: Optional[float] = None,
        message: Optional[str] = None,
        stage: Optional[BAOStage] = None,
        bao_id: Optional[str] = None,
    ) -> Optional[BAOJobStatus]: ...

    @abc.abstractmethod
    def list_bao_jobs(self) -> List[BAOJobStatus]: ...

    # ---- Evaluation async jobs ---------------------------------------

    @abc.abstractmethod
    def create_evaluation_job(self, cache_key: str) -> EvaluationJobStatus:
        """Record a new queued evaluation run."""
        ...

    @abc.abstractmethod
    def get_evaluation_job(self, job_id: str) -> Optional[EvaluationJobStatus]: ...

    @abc.abstractmethod
    def update_evaluation_job(
        self,
        job_id: str,
        *,
        state: Optional[JobState] = None,
        progress: Optional[float] = None,
        message: Optional[str] = None,
        stage: Optional[EvaluationStage] = None,
        evaluation_id: Optional[str] = None,
    ) -> Optional[EvaluationJobStatus]: ...

    @abc.abstractmethod
    def list_evaluation_jobs(self) -> List[EvaluationJobStatus]: ...


# ---------------------------------------------------------------------------
# In-memory implementation (used in tests & dev)
# ---------------------------------------------------------------------------

class InMemoryStore(StoreBase):
    def __init__(self):
        self._plans: Dict[str, PlanDetail] = {}
        self._jobs: Dict[str, JobStatus] = {}
        self._artifacts: Dict[str, ArtifactRecord] = {}
        self._geometry_jobs: Dict[str, GeometryJobStatus] = {}
        self._beam_model_jobs: Dict[str, BeamModelJobStatus] = {}
        self._dose_jobs: Dict[str, DoseJobStatus] = {}
        self._optimization_jobs: Dict[str, OptimizationJobStatus] = {}
        self._bao_jobs: Dict[str, BAOJobStatus] = {}
        self._evaluation_jobs: Dict[str, EvaluationJobStatus] = {}

    def create_plan(self, payload: PlanRequest) -> tuple[PlanDetail, JobStatus]:
        plan_id = str(uuid.uuid4())
        job_id = str(uuid.uuid4())
        now = _utcnow()
        detail = PlanDetail(
            id=plan_id,
            workflow_id=payload.workflow_id,
            status=JobState.queued.value,
            created_at=now,
            updated_at=now,
            prescription_gy=payload.prescription_gy,
            artifact_ids=[],
            study_instance_uid=payload.study_instance_uid,
            segmentation_uid=payload.segmentation_uid,
            fraction_count=payload.fraction_count,
            beam_count=payload.beam_count,
            notes=payload.notes,
            job_id=job_id,
            objectives=payload.objectives,
            robustness=payload.robustness,
        )
        job = JobStatus(id=job_id, plan_id=plan_id, state=JobState.queued, progress=0.0)
        self._plans[plan_id] = detail
        self._jobs[job_id] = job
        return detail, job

    def list_plans(self):
        return list(self._plans.values())

    def get_plan(self, plan_id: str) -> PlanDetail | None:
        return self._plans.get(plan_id)

    def get_job(self, job_id: str) -> JobStatus | None:
        return self._jobs.get(job_id)

    def update_job(
        self,
        job_id: str,
        *,
        state: JobState | None = None,
        progress: float | None = None,
        message: str | None = None,
        stage: str | None = None,
        eta_seconds: float | None = None,
    ) -> JobStatus | None:
        job = self._jobs.get(job_id)
        if not job:
            return None
        update_data = job.model_dump()
        now = _utcnow()
        if state:
            update_data["state"] = state
            if state == JobState.running and not update_data.get("started_at"):
                update_data["started_at"] = now
            if state in {JobState.succeeded, JobState.failed, JobState.cancelled}:
                update_data["finished_at"] = now
        if progress is not None:
            update_data["progress"] = progress
        if message is not None:
            update_data["message"] = message
        if stage is not None:
            update_data["stage"] = stage
        if eta_seconds is not None:
            update_data["eta_seconds"] = eta_seconds
        updated = JobStatus(**update_data)
        self._jobs[job_id] = updated
        plan = self._plans.get(job.plan_id)
        if plan:
            plan.status = updated.state.value
            plan.updated_at = _utcnow()
        return updated

    def register_artifact(
        self,
        plan_id: str,
        file_path: str,
        content_type: str = "application/dicom",
        file_name: str = "",
    ) -> Optional[str]:
        artifact_id = str(uuid.uuid4())
        record = ArtifactRecord(
            id=artifact_id,
            plan_id=plan_id,
            file_path=file_path,
            content_type=content_type,
            file_name=file_name or file_path.rsplit("/", 1)[-1],
            created_at=_utcnow(),
        )
        self._artifacts[artifact_id] = record
        self.attach_artifact(plan_id, artifact_id)
        return artifact_id

    def get_artifact(self, artifact_id: str) -> Optional[ArtifactRecord]:
        return self._artifacts.get(artifact_id)

    def attach_artifact(self, plan_id: str, artifact_id: str):
        plan = self._plans.get(plan_id)
        if not plan:
            return
        artifacts = list(plan.artifact_ids)
        artifacts.append(artifact_id)
        plan.artifact_ids = artifacts
        plan.updated_at = _utcnow()

    def set_plan_summary(self, plan_id: str, summary: dict):
        plan = self._plans.get(plan_id)
        if not plan:
            return
        plan.qa_summary = summary
        plan.updated_at = _utcnow()

    def delete_plan(self, plan_id: str) -> bool:
        plan = self._plans.pop(plan_id, None)
        if not plan:
            return False
        # Remove associated job
        if plan.job_id:
            self._jobs.pop(plan.job_id, None)
        # Remove associated artifacts
        artifact_ids_to_remove = [aid for aid, a in self._artifacts.items() if a.plan_id == plan_id]
        for aid in artifact_ids_to_remove:
            self._artifacts.pop(aid, None)
        return True

    # ---- Geometry async jobs -----------------------------------------

    def create_geometry_job(self, cache_key: str) -> GeometryJobStatus:
        job_id = str(uuid.uuid4())
        now = _utcnow()
        job = GeometryJobStatus(
            id=job_id,
            cache_key=cache_key,
            state=JobState.queued,
            progress=0.0,
            stage=GeometryStage.queued,
            created_at=now,
        )
        self._geometry_jobs[job_id] = job
        return job

    def get_geometry_job(self, job_id: str) -> Optional[GeometryJobStatus]:
        return self._geometry_jobs.get(job_id)

    def update_geometry_job(
        self,
        job_id: str,
        *,
        state: Optional[JobState] = None,
        progress: Optional[float] = None,
        message: Optional[str] = None,
        stage: Optional[GeometryStage] = None,
        geometry_id: Optional[str] = None,
    ) -> Optional[GeometryJobStatus]:
        job = self._geometry_jobs.get(job_id)
        if not job:
            return None
        data = job.model_dump()
        now = _utcnow()
        if state:
            data["state"] = state
            if state == JobState.running and not data.get("started_at"):
                data["started_at"] = now
            if state in {JobState.succeeded, JobState.failed, JobState.cancelled}:
                data["finished_at"] = now
        if progress is not None:
            data["progress"] = progress
        if message is not None:
            data["message"] = message
        if stage is not None:
            data["stage"] = stage
        if geometry_id is not None:
            data["geometry_id"] = geometry_id
        updated = GeometryJobStatus(**data)
        self._geometry_jobs[job_id] = updated
        return updated

    # ---- Beam model async jobs ---------------------------------------

    def create_beam_model_job(self, cache_key: str) -> BeamModelJobStatus:
        job_id = str(uuid.uuid4())
        now = _utcnow()
        job = BeamModelJobStatus(
            id=job_id,
            cache_key=cache_key,
            state=JobState.queued,
            progress=0.0,
            stage=BeamModelStage.queued,
            created_at=now,
        )
        self._beam_model_jobs[job_id] = job
        return job

    def get_beam_model_job(self, job_id: str) -> Optional[BeamModelJobStatus]:
        return self._beam_model_jobs.get(job_id)

    def update_beam_model_job(
        self,
        job_id: str,
        *,
        state: Optional[JobState] = None,
        progress: Optional[float] = None,
        message: Optional[str] = None,
        stage: Optional[BeamModelStage] = None,
        beam_model_id: Optional[str] = None,
    ) -> Optional[BeamModelJobStatus]:
        job = self._beam_model_jobs.get(job_id)
        if not job:
            return None
        data = job.model_dump()
        now = _utcnow()
        if state:
            data["state"] = state
            if state == JobState.running and not data.get("started_at"):
                data["started_at"] = now
            if state in {JobState.succeeded, JobState.failed, JobState.cancelled}:
                data["finished_at"] = now
        if progress is not None:
            data["progress"] = progress
        if message is not None:
            data["message"] = message
        if stage is not None:
            data["stage"] = stage
        if beam_model_id is not None:
            data["beam_model_id"] = beam_model_id
        updated = BeamModelJobStatus(**data)
        self._beam_model_jobs[job_id] = updated
        return updated

    # ---- Dose async jobs ---------------------------------------------

    def create_dose_job(self, cache_key: str, kind: str = "dose") -> DoseJobStatus:
        job_id = str(uuid.uuid4())
        now = _utcnow()
        job = DoseJobStatus(
            id=job_id,
            cache_key=cache_key,
            kind=kind,
            state=JobState.queued,
            progress=0.0,
            stage=DoseStage.queued,
            created_at=now,
        )
        self._dose_jobs[job_id] = job
        return job

    def get_dose_job(self, job_id: str) -> Optional[DoseJobStatus]:
        return self._dose_jobs.get(job_id)

    def update_dose_job(
        self,
        job_id: str,
        *,
        state: Optional[JobState] = None,
        progress: Optional[float] = None,
        message: Optional[str] = None,
        stage: Optional[DoseStage] = None,
        dose_id: Optional[str] = None,
        influence_id: Optional[str] = None,
    ) -> Optional[DoseJobStatus]:
        job = self._dose_jobs.get(job_id)
        if not job:
            return None
        data = job.model_dump()
        now = _utcnow()
        if state:
            data["state"] = state
            if state == JobState.running and not data.get("started_at"):
                data["started_at"] = now
            if state in {JobState.succeeded, JobState.failed, JobState.cancelled}:
                data["finished_at"] = now
        if progress is not None:
            data["progress"] = progress
        if message is not None:
            data["message"] = message
        if stage is not None:
            data["stage"] = stage
        if dose_id is not None:
            data["dose_id"] = dose_id
        if influence_id is not None:
            data["influence_id"] = influence_id
        updated = DoseJobStatus(**data)
        self._dose_jobs[job_id] = updated
        return updated

    # ---- Optimization async jobs -------------------------------------

    def create_optimization_job(self, cache_key: str) -> OptimizationJobStatus:
        job_id = str(uuid.uuid4())
        now = _utcnow()
        job = OptimizationJobStatus(
            id=job_id,
            cache_key=cache_key,
            state=JobState.queued,
            progress=0.0,
            stage=OptimizationStage.queued,
            created_at=now,
        )
        self._optimization_jobs[job_id] = job
        return job

    def get_optimization_job(self, job_id: str) -> Optional[OptimizationJobStatus]:
        return self._optimization_jobs.get(job_id)

    def update_optimization_job(
        self,
        job_id: str,
        *,
        state: Optional[JobState] = None,
        progress: Optional[float] = None,
        message: Optional[str] = None,
        stage: Optional[OptimizationStage] = None,
        optimization_id: Optional[str] = None,
    ) -> Optional[OptimizationJobStatus]:
        job = self._optimization_jobs.get(job_id)
        if not job:
            return None
        data = job.model_dump()
        now = _utcnow()
        if state:
            data["state"] = state
            if state == JobState.running and not data.get("started_at"):
                data["started_at"] = now
            if state in {JobState.succeeded, JobState.failed, JobState.cancelled}:
                data["finished_at"] = now
        if progress is not None:
            data["progress"] = progress
        if message is not None:
            data["message"] = message
        if stage is not None:
            data["stage"] = stage
        if optimization_id is not None:
            data["optimization_id"] = optimization_id
        updated = OptimizationJobStatus(**data)
        self._optimization_jobs[job_id] = updated
        return updated

    def list_optimization_jobs(self) -> List[OptimizationJobStatus]:
        return list(self._optimization_jobs.values())

    # ---- BAO async jobs ----------------------------------------------

    def create_bao_job(self, cache_key: str) -> BAOJobStatus:
        job_id = str(uuid.uuid4())
        job = BAOJobStatus(
            id=job_id, cache_key=cache_key, state=JobState.queued,
            progress=0.0, stage=BAOStage.queued, created_at=_utcnow(),
        )
        self._bao_jobs[job_id] = job
        return job

    def get_bao_job(self, job_id: str) -> Optional[BAOJobStatus]:
        return self._bao_jobs.get(job_id)

    def update_bao_job(
        self,
        job_id: str,
        *,
        state: Optional[JobState] = None,
        progress: Optional[float] = None,
        message: Optional[str] = None,
        stage: Optional[BAOStage] = None,
        bao_id: Optional[str] = None,
    ) -> Optional[BAOJobStatus]:
        job = self._bao_jobs.get(job_id)
        if not job:
            return None
        data = job.model_dump()
        now = _utcnow()
        if state:
            data["state"] = state
            if state == JobState.running and not data.get("started_at"):
                data["started_at"] = now
            if state in {JobState.succeeded, JobState.failed, JobState.cancelled}:
                data["finished_at"] = now
        if progress is not None:
            data["progress"] = progress
        if message is not None:
            data["message"] = message
        if stage is not None:
            data["stage"] = stage
        if bao_id is not None:
            data["bao_id"] = bao_id
        updated = BAOJobStatus(**data)
        self._bao_jobs[job_id] = updated
        return updated

    def list_bao_jobs(self) -> List[BAOJobStatus]:
        return list(self._bao_jobs.values())

    # ---- Evaluation async jobs ---------------------------------------

    def create_evaluation_job(self, cache_key: str) -> EvaluationJobStatus:
        job_id = str(uuid.uuid4())
        job = EvaluationJobStatus(
            id=job_id, cache_key=cache_key, state=JobState.queued,
            progress=0.0, stage=EvaluationStage.queued, created_at=_utcnow(),
        )
        self._evaluation_jobs[job_id] = job
        return job

    def get_evaluation_job(self, job_id: str) -> Optional[EvaluationJobStatus]:
        return self._evaluation_jobs.get(job_id)

    def update_evaluation_job(
        self,
        job_id: str,
        *,
        state: Optional[JobState] = None,
        progress: Optional[float] = None,
        message: Optional[str] = None,
        stage: Optional[EvaluationStage] = None,
        evaluation_id: Optional[str] = None,
    ) -> Optional[EvaluationJobStatus]:
        job = self._evaluation_jobs.get(job_id)
        if not job:
            return None
        data = job.model_dump()
        now = _utcnow()
        if state:
            data["state"] = state
            if state == JobState.running and not data.get("started_at"):
                data["started_at"] = now
            if state in {JobState.succeeded, JobState.failed, JobState.cancelled}:
                data["finished_at"] = now
        if progress is not None:
            data["progress"] = progress
        if message is not None:
            data["message"] = message
        if stage is not None:
            data["stage"] = stage
        if evaluation_id is not None:
            data["evaluation_id"] = evaluation_id
        updated = EvaluationJobStatus(**data)
        self._evaluation_jobs[job_id] = updated
        return updated

    def list_evaluation_jobs(self) -> List[EvaluationJobStatus]:
        return list(self._evaluation_jobs.values())


# ---------------------------------------------------------------------------
# SQL implementation (production)
# ---------------------------------------------------------------------------

class SQLStore(StoreBase):
    """Persistent store backed by SQLAlchemy."""

    def __init__(self, session_factory):
        self._session_factory = session_factory

    def _session(self):
        return self._session_factory()

    def create_plan(self, payload: PlanRequest) -> tuple[PlanDetail, JobStatus]:
        from .db_models import PlanRow, JobRow

        plan_id = str(uuid.uuid4())
        job_id = str(uuid.uuid4())
        now = _utcnow()

        session = self._session()
        try:
            plan_row = PlanRow(
                id=plan_id,
                workflow_id=payload.workflow_id.value if hasattr(payload.workflow_id, 'value') else payload.workflow_id,
                status=JobState.queued.value,
                study_instance_uid=payload.study_instance_uid,
                segmentation_uid=payload.segmentation_uid,
                prescription_gy=payload.prescription_gy,
                fraction_count=payload.fraction_count,
                beam_count=payload.beam_count,
                notes=payload.notes,
                job_id=job_id,
                objectives=[obj.model_dump() for obj in payload.objectives] if payload.objectives else None,
                robustness=payload.robustness.model_dump() if payload.robustness else None,
                created_at=now,
                updated_at=now,
            )
            job_row = JobRow(
                id=job_id,
                plan_id=plan_id,
                state=JobState.queued.value,
                progress=0.0,
                started_at=None,
                finished_at=None,
            )
            session.add(plan_row)
            session.add(job_row)
            session.commit()

            detail = self._plan_row_to_detail(plan_row, session)
            job = self._job_row_to_status(job_row)
            return detail, job
        finally:
            session.close()

    def list_plans(self) -> list:
        from .db_models import PlanRow

        session = self._session()
        try:
            rows = session.query(PlanRow).order_by(PlanRow.created_at.desc()).all()
            return [self._plan_row_to_detail(r, session) for r in rows]
        finally:
            session.close()

    def get_plan(self, plan_id: str) -> Optional[PlanDetail]:
        from .db_models import PlanRow

        session = self._session()
        try:
            row = session.query(PlanRow).filter_by(id=plan_id).first()
            if not row:
                return None
            return self._plan_row_to_detail(row, session)
        finally:
            session.close()

    def get_job(self, job_id: str) -> Optional[JobStatus]:
        from .db_models import JobRow

        session = self._session()
        try:
            row = session.query(JobRow).filter_by(id=job_id).first()
            if not row:
                return None
            return self._job_row_to_status(row)
        finally:
            session.close()

    def update_job(
        self,
        job_id: str,
        *,
        state: Optional[JobState] = None,
        progress: Optional[float] = None,
        message: Optional[str] = None,
        stage: Optional[str] = None,
        eta_seconds: Optional[float] = None,
    ) -> Optional[JobStatus]:
        from .db_models import JobRow, PlanRow

        session = self._session()
        try:
            row = session.query(JobRow).filter_by(id=job_id).first()
            if not row:
                return None
            now = _utcnow()
            if state:
                row.state = state.value if hasattr(state, 'value') else state
                if state == JobState.running and not row.started_at:
                    row.started_at = now
                if state in {JobState.succeeded, JobState.failed, JobState.cancelled}:
                    row.finished_at = now
            if progress is not None:
                row.progress = progress
            if message is not None:
                row.message = message
            # Note: stage/eta_seconds are transient (not stored in SQL).
            # They are only meaningful for InMemoryStore or real-time queries.
            # Update plan status
            plan_row = session.query(PlanRow).filter_by(id=row.plan_id).first()
            if plan_row:
                plan_row.status = row.state
                plan_row.updated_at = now
            session.commit()
            return self._job_row_to_status(row)
        finally:
            session.close()

    def register_artifact(
        self,
        plan_id: str,
        file_path: str,
        content_type: str = "application/dicom",
        file_name: str = "",
    ) -> Optional[str]:
        from .db_models import ArtifactRow

        artifact_id = str(uuid.uuid4())
        session = self._session()
        try:
            row = ArtifactRow(
                id=artifact_id,
                plan_id=plan_id,
                file_path=file_path,
                content_type=content_type,
                file_name=file_name or file_path.rsplit("/", 1)[-1],
                created_at=_utcnow(),
            )
            session.add(row)
            session.commit()
            return artifact_id
        finally:
            session.close()

    def get_artifact(self, artifact_id: str) -> Optional[ArtifactRecord]:
        from .db_models import ArtifactRow

        session = self._session()
        try:
            row = session.query(ArtifactRow).filter_by(id=artifact_id).first()
            if not row:
                return None
            return ArtifactRecord(
                id=row.id,
                plan_id=row.plan_id,
                file_path=row.file_path,
                content_type=row.content_type,
                file_name=row.file_name,
                created_at=row.created_at,
            )
        finally:
            session.close()

    def attach_artifact(self, plan_id: str, artifact_id: str) -> None:
        # In SQL store, artifacts are linked via plan_id FK — no separate attach needed.
        pass

    def set_plan_summary(self, plan_id: str, summary: dict) -> None:
        from .db_models import PlanRow

        session = self._session()
        try:
            row = session.query(PlanRow).filter_by(id=plan_id).first()
            if row:
                row.qa_summary = summary
                row.updated_at = _utcnow()
                session.commit()
        finally:
            session.close()

    def delete_plan(self, plan_id: str) -> bool:
        from .db_models import PlanRow

        session = self._session()
        try:
            row = session.query(PlanRow).filter_by(id=plan_id).first()
            if not row:
                return False
            session.delete(row)  # cascade deletes jobs + artifacts
            session.commit()
            return True
        finally:
            session.close()

    # ---- Geometry async jobs -----------------------------------------

    def create_geometry_job(self, cache_key: str) -> GeometryJobStatus:
        from .db_models import GeometryJobRow

        job_id = str(uuid.uuid4())
        now = _utcnow()
        session = self._session()
        try:
            row = GeometryJobRow(
                id=job_id,
                cache_key=cache_key,
                state=JobState.queued.value,
                progress=0.0,
                stage=GeometryStage.queued.value,
                created_at=now,
            )
            session.add(row)
            session.commit()
            return self._geometry_job_row_to_status(row)
        finally:
            session.close()

    def get_geometry_job(self, job_id: str) -> Optional[GeometryJobStatus]:
        from .db_models import GeometryJobRow

        session = self._session()
        try:
            row = session.query(GeometryJobRow).filter_by(id=job_id).first()
            if not row:
                return None
            return self._geometry_job_row_to_status(row)
        finally:
            session.close()

    def update_geometry_job(
        self,
        job_id: str,
        *,
        state: Optional[JobState] = None,
        progress: Optional[float] = None,
        message: Optional[str] = None,
        stage: Optional[GeometryStage] = None,
        geometry_id: Optional[str] = None,
    ) -> Optional[GeometryJobStatus]:
        from .db_models import GeometryJobRow

        session = self._session()
        try:
            row = session.query(GeometryJobRow).filter_by(id=job_id).first()
            if not row:
                return None
            now = _utcnow()
            if state:
                row.state = state.value if hasattr(state, "value") else state
                if state == JobState.running and not row.started_at:
                    row.started_at = now
                if state in {JobState.succeeded, JobState.failed, JobState.cancelled}:
                    row.finished_at = now
            if progress is not None:
                row.progress = progress
            if message is not None:
                row.message = message
            if stage is not None:
                row.stage = stage.value if hasattr(stage, "value") else stage
            if geometry_id is not None:
                row.geometry_id = geometry_id
            session.commit()
            return self._geometry_job_row_to_status(row)
        finally:
            session.close()

    @staticmethod
    def _geometry_job_row_to_status(row) -> GeometryJobStatus:
        return GeometryJobStatus(
            id=row.id,
            cache_key=row.cache_key,
            state=row.state,
            progress=row.progress or 0.0,
            stage=row.stage,
            message=row.message,
            geometry_id=row.geometry_id,
            started_at=row.started_at,
            finished_at=row.finished_at,
            created_at=row.created_at,
        )

    # ---- Beam model async jobs ---------------------------------------

    def create_beam_model_job(self, cache_key: str) -> BeamModelJobStatus:
        from .db_models import BeamModelJobRow

        job_id = str(uuid.uuid4())
        now = _utcnow()
        session = self._session()
        try:
            row = BeamModelJobRow(
                id=job_id,
                cache_key=cache_key,
                state=JobState.queued.value,
                progress=0.0,
                stage=BeamModelStage.queued.value,
                created_at=now,
            )
            session.add(row)
            session.commit()
            return self._beam_model_job_row_to_status(row)
        finally:
            session.close()

    def get_beam_model_job(self, job_id: str) -> Optional[BeamModelJobStatus]:
        from .db_models import BeamModelJobRow

        session = self._session()
        try:
            row = session.query(BeamModelJobRow).filter_by(id=job_id).first()
            if not row:
                return None
            return self._beam_model_job_row_to_status(row)
        finally:
            session.close()

    def update_beam_model_job(
        self,
        job_id: str,
        *,
        state: Optional[JobState] = None,
        progress: Optional[float] = None,
        message: Optional[str] = None,
        stage: Optional[BeamModelStage] = None,
        beam_model_id: Optional[str] = None,
    ) -> Optional[BeamModelJobStatus]:
        from .db_models import BeamModelJobRow

        session = self._session()
        try:
            row = session.query(BeamModelJobRow).filter_by(id=job_id).first()
            if not row:
                return None
            now = _utcnow()
            if state:
                row.state = state.value if hasattr(state, "value") else state
                if state == JobState.running and not row.started_at:
                    row.started_at = now
                if state in {JobState.succeeded, JobState.failed, JobState.cancelled}:
                    row.finished_at = now
            if progress is not None:
                row.progress = progress
            if message is not None:
                row.message = message
            if stage is not None:
                row.stage = stage.value if hasattr(stage, "value") else stage
            if beam_model_id is not None:
                row.beam_model_id = beam_model_id
            session.commit()
            return self._beam_model_job_row_to_status(row)
        finally:
            session.close()

    @staticmethod
    def _beam_model_job_row_to_status(row) -> BeamModelJobStatus:
        return BeamModelJobStatus(
            id=row.id,
            cache_key=row.cache_key,
            state=row.state,
            progress=row.progress or 0.0,
            stage=row.stage,
            message=row.message,
            beam_model_id=row.beam_model_id,
            started_at=row.started_at,
            finished_at=row.finished_at,
            created_at=row.created_at,
        )

    # ---- Dose async jobs ---------------------------------------------

    def create_dose_job(self, cache_key: str, kind: str = "dose") -> DoseJobStatus:
        from .db_models import DoseJobRow

        job_id = str(uuid.uuid4())
        now = _utcnow()
        session = self._session()
        try:
            row = DoseJobRow(
                id=job_id,
                cache_key=cache_key,
                kind=kind,
                state=JobState.queued.value,
                progress=0.0,
                stage=DoseStage.queued.value,
                created_at=now,
            )
            session.add(row)
            session.commit()
            return self._dose_job_row_to_status(row)
        finally:
            session.close()

    def get_dose_job(self, job_id: str) -> Optional[DoseJobStatus]:
        from .db_models import DoseJobRow

        session = self._session()
        try:
            row = session.query(DoseJobRow).filter_by(id=job_id).first()
            if not row:
                return None
            return self._dose_job_row_to_status(row)
        finally:
            session.close()

    def update_dose_job(
        self,
        job_id: str,
        *,
        state: Optional[JobState] = None,
        progress: Optional[float] = None,
        message: Optional[str] = None,
        stage: Optional[DoseStage] = None,
        dose_id: Optional[str] = None,
        influence_id: Optional[str] = None,
    ) -> Optional[DoseJobStatus]:
        from .db_models import DoseJobRow

        session = self._session()
        try:
            row = session.query(DoseJobRow).filter_by(id=job_id).first()
            if not row:
                return None
            now = _utcnow()
            if state:
                row.state = state.value if hasattr(state, "value") else state
                if state == JobState.running and not row.started_at:
                    row.started_at = now
                if state in {JobState.succeeded, JobState.failed, JobState.cancelled}:
                    row.finished_at = now
            if progress is not None:
                row.progress = progress
            if message is not None:
                row.message = message
            if stage is not None:
                row.stage = stage.value if hasattr(stage, "value") else stage
            if dose_id is not None:
                row.dose_id = dose_id
            if influence_id is not None:
                row.influence_id = influence_id
            session.commit()
            return self._dose_job_row_to_status(row)
        finally:
            session.close()

    @staticmethod
    def _dose_job_row_to_status(row) -> DoseJobStatus:
        return DoseJobStatus(
            id=row.id,
            cache_key=row.cache_key,
            kind=row.kind,
            state=row.state,
            progress=row.progress or 0.0,
            stage=row.stage,
            message=row.message,
            dose_id=row.dose_id,
            influence_id=row.influence_id,
            started_at=row.started_at,
            finished_at=row.finished_at,
            created_at=row.created_at,
        )

    # ---- Optimization async jobs -------------------------------------

    def create_optimization_job(self, cache_key: str) -> OptimizationJobStatus:
        from .db_models import OptimizationJobRow

        job_id = str(uuid.uuid4())
        now = _utcnow()
        session = self._session()
        try:
            row = OptimizationJobRow(
                id=job_id,
                cache_key=cache_key,
                state=JobState.queued.value,
                progress=0.0,
                stage=OptimizationStage.queued.value,
                created_at=now,
            )
            session.add(row)
            session.commit()
            return self._optimization_job_row_to_status(row)
        finally:
            session.close()

    def get_optimization_job(self, job_id: str) -> Optional[OptimizationJobStatus]:
        from .db_models import OptimizationJobRow

        session = self._session()
        try:
            row = session.query(OptimizationJobRow).filter_by(id=job_id).first()
            if not row:
                return None
            return self._optimization_job_row_to_status(row)
        finally:
            session.close()

    def update_optimization_job(
        self,
        job_id: str,
        *,
        state: Optional[JobState] = None,
        progress: Optional[float] = None,
        message: Optional[str] = None,
        stage: Optional[OptimizationStage] = None,
        optimization_id: Optional[str] = None,
    ) -> Optional[OptimizationJobStatus]:
        from .db_models import OptimizationJobRow

        session = self._session()
        try:
            row = session.query(OptimizationJobRow).filter_by(id=job_id).first()
            if not row:
                return None
            now = _utcnow()
            if state:
                row.state = state.value if hasattr(state, "value") else state
                if state == JobState.running and not row.started_at:
                    row.started_at = now
                if state in {JobState.succeeded, JobState.failed, JobState.cancelled}:
                    row.finished_at = now
            if progress is not None:
                row.progress = progress
            if message is not None:
                row.message = message
            if stage is not None:
                row.stage = stage.value if hasattr(stage, "value") else stage
            if optimization_id is not None:
                row.optimization_id = optimization_id
            session.commit()
            return self._optimization_job_row_to_status(row)
        finally:
            session.close()

    def list_optimization_jobs(self) -> List[OptimizationJobStatus]:
        from .db_models import OptimizationJobRow

        session = self._session()
        try:
            rows = (session.query(OptimizationJobRow)
                    .order_by(OptimizationJobRow.created_at.desc()).all())
            return [self._optimization_job_row_to_status(r) for r in rows]
        finally:
            session.close()

    @staticmethod
    def _optimization_job_row_to_status(row) -> OptimizationJobStatus:
        return OptimizationJobStatus(
            id=row.id,
            cache_key=row.cache_key,
            state=row.state,
            progress=row.progress or 0.0,
            stage=row.stage,
            message=row.message,
            optimization_id=row.optimization_id,
            started_at=row.started_at,
            finished_at=row.finished_at,
            created_at=row.created_at,
        )

    # ---- BAO async jobs ----------------------------------------------

    def create_bao_job(self, cache_key: str) -> BAOJobStatus:
        from .db_models import BAOJobRow

        job_id = str(uuid.uuid4())
        now = _utcnow()
        session = self._session()
        try:
            row = BAOJobRow(
                id=job_id, cache_key=cache_key, state=JobState.queued.value,
                progress=0.0, stage=BAOStage.queued.value, created_at=now,
            )
            session.add(row)
            session.commit()
            return self._bao_job_row_to_status(row)
        finally:
            session.close()

    def get_bao_job(self, job_id: str) -> Optional[BAOJobStatus]:
        from .db_models import BAOJobRow

        session = self._session()
        try:
            row = session.query(BAOJobRow).filter_by(id=job_id).first()
            if not row:
                return None
            return self._bao_job_row_to_status(row)
        finally:
            session.close()

    def update_bao_job(
        self,
        job_id: str,
        *,
        state: Optional[JobState] = None,
        progress: Optional[float] = None,
        message: Optional[str] = None,
        stage: Optional[BAOStage] = None,
        bao_id: Optional[str] = None,
    ) -> Optional[BAOJobStatus]:
        from .db_models import BAOJobRow

        session = self._session()
        try:
            row = session.query(BAOJobRow).filter_by(id=job_id).first()
            if not row:
                return None
            now = _utcnow()
            if state:
                row.state = state.value if hasattr(state, "value") else state
                if state == JobState.running and not row.started_at:
                    row.started_at = now
                if state in {JobState.succeeded, JobState.failed, JobState.cancelled}:
                    row.finished_at = now
            if progress is not None:
                row.progress = progress
            if message is not None:
                row.message = message
            if stage is not None:
                row.stage = stage.value if hasattr(stage, "value") else stage
            if bao_id is not None:
                row.bao_id = bao_id
            session.commit()
            return self._bao_job_row_to_status(row)
        finally:
            session.close()

    def list_bao_jobs(self) -> List[BAOJobStatus]:
        from .db_models import BAOJobRow

        session = self._session()
        try:
            rows = session.query(BAOJobRow).order_by(BAOJobRow.created_at.desc()).all()
            return [self._bao_job_row_to_status(r) for r in rows]
        finally:
            session.close()

    @staticmethod
    def _bao_job_row_to_status(row) -> BAOJobStatus:
        return BAOJobStatus(
            id=row.id,
            cache_key=row.cache_key,
            state=row.state,
            progress=row.progress or 0.0,
            stage=row.stage,
            message=row.message,
            bao_id=row.bao_id,
            started_at=row.started_at,
            finished_at=row.finished_at,
            created_at=row.created_at,
        )

    # ---- Evaluation async jobs ---------------------------------------

    def create_evaluation_job(self, cache_key: str) -> EvaluationJobStatus:
        from .db_models import EvaluationJobRow

        job_id = str(uuid.uuid4())
        now = _utcnow()
        session = self._session()
        try:
            row = EvaluationJobRow(
                id=job_id, cache_key=cache_key, state=JobState.queued.value,
                progress=0.0, stage=EvaluationStage.queued.value, created_at=now,
            )
            session.add(row)
            session.commit()
            return self._evaluation_job_row_to_status(row)
        finally:
            session.close()

    def get_evaluation_job(self, job_id: str) -> Optional[EvaluationJobStatus]:
        from .db_models import EvaluationJobRow

        session = self._session()
        try:
            row = session.query(EvaluationJobRow).filter_by(id=job_id).first()
            if not row:
                return None
            return self._evaluation_job_row_to_status(row)
        finally:
            session.close()

    def update_evaluation_job(
        self,
        job_id: str,
        *,
        state: Optional[JobState] = None,
        progress: Optional[float] = None,
        message: Optional[str] = None,
        stage: Optional[EvaluationStage] = None,
        evaluation_id: Optional[str] = None,
    ) -> Optional[EvaluationJobStatus]:
        from .db_models import EvaluationJobRow

        session = self._session()
        try:
            row = session.query(EvaluationJobRow).filter_by(id=job_id).first()
            if not row:
                return None
            now = _utcnow()
            if state:
                row.state = state.value if hasattr(state, "value") else state
                if state == JobState.running and not row.started_at:
                    row.started_at = now
                if state in {JobState.succeeded, JobState.failed, JobState.cancelled}:
                    row.finished_at = now
            if progress is not None:
                row.progress = progress
            if message is not None:
                row.message = message
            if stage is not None:
                row.stage = stage.value if hasattr(stage, "value") else stage
            if evaluation_id is not None:
                row.evaluation_id = evaluation_id
            session.commit()
            return self._evaluation_job_row_to_status(row)
        finally:
            session.close()

    def list_evaluation_jobs(self) -> List[EvaluationJobStatus]:
        from .db_models import EvaluationJobRow

        session = self._session()
        try:
            rows = (session.query(EvaluationJobRow)
                    .order_by(EvaluationJobRow.created_at.desc()).all())
            return [self._evaluation_job_row_to_status(r) for r in rows]
        finally:
            session.close()

    @staticmethod
    def _evaluation_job_row_to_status(row) -> EvaluationJobStatus:
        return EvaluationJobStatus(
            id=row.id,
            cache_key=row.cache_key,
            state=row.state,
            progress=row.progress or 0.0,
            stage=row.stage,
            message=row.message,
            evaluation_id=row.evaluation_id,
            started_at=row.started_at,
            finished_at=row.finished_at,
            created_at=row.created_at,
        )

    # -- helpers --

    @staticmethod
    def _plan_row_to_detail(row, session) -> PlanDetail:
        from .db_models import ArtifactRow
        from ..models.plan import DoseObjective, RobustnessConfig

        artifact_ids = [
            a.id for a in session.query(ArtifactRow).filter_by(plan_id=row.id).all()
        ]

        # Deserialize objectives from JSON
        objectives = None
        if row.objectives:
            objectives = [DoseObjective(**obj) for obj in row.objectives]

        # Deserialize robustness from JSON
        robustness = None
        if row.robustness:
            robustness = RobustnessConfig(**row.robustness)

        return PlanDetail(
            id=row.id,
            workflow_id=row.workflow_id,
            status=row.status,
            created_at=row.created_at,
            updated_at=row.updated_at,
            prescription_gy=row.prescription_gy,
            artifact_ids=artifact_ids,
            study_instance_uid=row.study_instance_uid,
            segmentation_uid=row.segmentation_uid,
            fraction_count=row.fraction_count,
            beam_count=getattr(row, 'beam_count', 1) or 1,
            notes=row.notes,
            job_id=row.job_id,
            qa_summary=row.qa_summary,
            objectives=objectives,
            robustness=robustness,
        )

    @staticmethod
    def _job_row_to_status(row) -> JobStatus:
        return JobStatus(
            id=row.id,
            plan_id=row.plan_id,
            state=row.state,
            progress=row.progress,
            message=row.message,
            started_at=row.started_at,
            finished_at=row.finished_at,
        )


# ---------------------------------------------------------------------------
# Factory + module-level convenience
# ---------------------------------------------------------------------------

_store: Optional[StoreBase] = None


def get_store() -> StoreBase:
    """Return the configured store instance.

    - If RADIARCH_DATABASE_URL is set and non-empty → SQLStore
    - Otherwise (tests, dev) → InMemoryStore
    """
    global _store
    if _store is not None:
        return _store

    db_url = os.environ.get("RADIARCH_DATABASE_URL", "")
    if db_url:
        from . import database
        database.get_engine()  # ensure engine is built
        if database.SessionLocal is not None:
            _store = SQLStore(database.SessionLocal)
        else:
            _store = InMemoryStore()
    else:
        _store = InMemoryStore()
    return _store


def reset_store():
    """Reset the global store (for testing)."""
    global _store
    _store = None


# Backwards-compatible module-level singleton (used by existing code).
# Uses a lazy proxy so that get_store() is called on first attribute access,
# allowing config (RADIARCH_DATABASE_URL) to determine the store type.
class _StoreProxy:
    """Lazy proxy that forwards all attribute access to get_store()."""
    def __getattr__(self, name):
        return getattr(get_store(), name)

store = _StoreProxy()
