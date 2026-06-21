"""Celery tasks for async dose + influence builds.

Two tasks — one per kind. Both follow the same pattern as
``build_geometry_job`` and ``build_beam_model_job``:

* Pull the matching job row from :mod:`radiarch.core.store`.
* Drive the corresponding :class:`DoseService` method, mirroring its
  ``stage`` + ``fraction`` + ``message`` progress to the DB row.
* On success, stash the resulting id (``dose_id`` or ``influence_id``)
  so polling clients can deep-link.
* On failure, mark the job ``failed`` and write the exception type +
  message into ``message``.

In ``environment=dev`` Celery is in eager mode, so these run in-process
in the API worker — the same code path tests exercise.
"""

from __future__ import annotations

import time

from celery.exceptions import SoftTimeLimitExceeded
from loguru import logger

from .celery_app import celery_app
from ..core.store import store
from ..models.dose import (
    DoseComputeRequest,
    DoseStage,
    InfluenceBuildRequest,
)
from ..models.job import JobState
from ..services.audit import emit, make_event


@celery_app.task(
    name="radiarch.dose.compute",
    autoretry_for=(ConnectionError, OSError),
    retry_backoff=True,
    retry_backoff_max=120,
    max_retries=3,
)
def build_dose_job(job_id: str, request_payload: dict):
    """Execute :meth:`DoseService.compute_dose` and mirror progress."""
    from ..services.dose import DoseService

    job = store.get_dose_job(job_id)
    if not job:
        logger.error("build_dose_job called with unknown job_id=%s", job_id)
        return

    t0 = time.monotonic()

    def _on_progress(stage: DoseStage, fraction: float, message: str) -> None:
        store.update_dose_job(
            job_id,
            state=JobState.running if fraction < 1.0 else job.state,
            progress=round(fraction, 3),
            stage=stage,
            message=message,
        )

    request: DoseComputeRequest | None = None
    try:
        store.update_dose_job(
            job_id,
            state=JobState.running,
            progress=0.0,
            stage=DoseStage.queued,
            message="Queued → running",
        )
        request = DoseComputeRequest.model_validate(request_payload)
        service = DoseService()
        result = service.compute_dose(request, progress_callback=_on_progress)

    except SoftTimeLimitExceeded:
        logger.error(f"Dose compute timed out for job {job_id}")
        store.update_dose_job(
            job_id,
            state=JobState.failed,
            progress=1.0,
            stage=DoseStage.done,
            message="Timed out",
        )
        emit(make_event(
            "dose.compute",
            state="failed",
            error_type="SoftTimeLimitExceeded",
            error_message="dose compute task hit soft time limit",
            duration_s=round(time.monotonic() - t0, 3),
            geometry_id=getattr(request, "geometry_id", None),
            beam_model_id=getattr(request, "beam_model_id", None),
            engine_name=getattr(getattr(request, "engine", None), "name", None),
            extra={"job_id": job_id},
        ))
        return

    except Exception as exc:  # pragma: no cover — exercised in tests
        logger.exception(f"Dose compute failed for job {job_id}")
        store.update_dose_job(
            job_id,
            state=JobState.failed,
            progress=1.0,
            stage=DoseStage.done,
            message=f"{type(exc).__name__}: {exc}",
        )
        emit(make_event(
            "dose.compute",
            state="failed",
            error_type=type(exc).__name__,
            error_message=str(exc)[:500],
            duration_s=round(time.monotonic() - t0, 3),
            geometry_id=getattr(request, "geometry_id", None),
            beam_model_id=getattr(request, "beam_model_id", None),
            engine_name=getattr(getattr(request, "engine", None), "name", None),
            extra={"job_id": job_id},
        ))
        return

    store.update_dose_job(
        job_id,
        state=JobState.succeeded,
        progress=1.0,
        stage=DoseStage.done,
        message=f"Built in {time.monotonic() - t0:.1f}s",
        dose_id=result.dose_id,
    )
    emit(make_event(
        "dose.compute",
        state="succeeded",
        cache_key=result.cache_key,
        dose_id=result.dose_id,
        geometry_id=request.geometry_id,
        beam_model_id=request.beam_model_id,
        engine_name=request.engine.name,
        engine_version=request.engine.version,
        modality=result.modality.value if hasattr(result.modality, "value") else str(result.modality),
        scenario_count=len(result.scenario_doses or []),
        duration_s=round(time.monotonic() - t0, 3),
        extra={"job_id": job_id},
    ))


@celery_app.task(
    name="radiarch.dose.influence",
    autoretry_for=(ConnectionError, OSError),
    retry_backoff=True,
    retry_backoff_max=120,
    max_retries=3,
)
def build_influence_job(job_id: str, request_payload: dict):
    """Execute :meth:`DoseService.build_influence` and mirror progress."""
    from ..services.dose import DoseService

    job = store.get_dose_job(job_id)
    if not job:
        logger.error("build_influence_job called with unknown job_id=%s", job_id)
        return

    t0 = time.monotonic()

    def _on_progress(stage: DoseStage, fraction: float, message: str) -> None:
        store.update_dose_job(
            job_id,
            state=JobState.running if fraction < 1.0 else job.state,
            progress=round(fraction, 3),
            stage=stage,
            message=message,
        )

    request: InfluenceBuildRequest | None = None
    try:
        store.update_dose_job(
            job_id,
            state=JobState.running,
            progress=0.0,
            stage=DoseStage.queued,
            message="Queued → running",
        )
        request = InfluenceBuildRequest.model_validate(request_payload)
        service = DoseService()
        result = service.build_influence(request, progress_callback=_on_progress)

    except SoftTimeLimitExceeded:
        logger.error(f"Influence build timed out for job {job_id}")
        store.update_dose_job(
            job_id,
            state=JobState.failed,
            progress=1.0,
            stage=DoseStage.done,
            message="Timed out",
        )
        emit(make_event(
            "dose.influence",
            state="failed",
            error_type="SoftTimeLimitExceeded",
            error_message="influence build task hit soft time limit",
            duration_s=round(time.monotonic() - t0, 3),
            geometry_id=getattr(request, "geometry_id", None),
            beam_model_id=getattr(request, "beam_model_id", None),
            engine_name=getattr(getattr(request, "engine", None), "name", None),
            extra={"job_id": job_id},
        ))
        return

    except Exception as exc:  # pragma: no cover
        logger.exception(f"Influence build failed for job {job_id}")
        store.update_dose_job(
            job_id,
            state=JobState.failed,
            progress=1.0,
            stage=DoseStage.done,
            message=f"{type(exc).__name__}: {exc}",
        )
        emit(make_event(
            "dose.influence",
            state="failed",
            error_type=type(exc).__name__,
            error_message=str(exc)[:500],
            duration_s=round(time.monotonic() - t0, 3),
            geometry_id=getattr(request, "geometry_id", None),
            beam_model_id=getattr(request, "beam_model_id", None),
            engine_name=getattr(getattr(request, "engine", None), "name", None),
            extra={"job_id": job_id},
        ))
        return

    store.update_dose_job(
        job_id,
        state=JobState.succeeded,
        progress=1.0,
        stage=DoseStage.done,
        message=f"Built in {time.monotonic() - t0:.1f}s",
        influence_id=result.influence_id,
    )
    emit(make_event(
        "dose.influence",
        state="succeeded",
        cache_key=result.cache_key,
        influence_id=result.influence_id,
        geometry_id=request.geometry_id,
        beam_model_id=request.beam_model_id,
        engine_name=request.engine.name,
        duration_s=round(time.monotonic() - t0, 3),
        extra={
            "job_id": job_id,
            "n_voxels": getattr(result, "n_voxels", None),
            "n_elements": getattr(result, "n_elements", None),
            "nnz": getattr(result, "nnz", None),
        },
    ))
