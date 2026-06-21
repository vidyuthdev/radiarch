"""Celery task for async inverse-plan optimization (Service 4).

Mirrors ``dose_tasks.build_dose_job``:

* Pull the matching job row from :mod:`radiarch.core.store`.
* Drive :meth:`OptimizationService.run`, mirroring its ``stage`` + ``fraction``
  + ``message`` progress to the DB row.
* On success, stash the resulting ``optimization_id`` so polling clients can
  deep-link.
* On failure (including ``SoftTimeLimitExceeded``), mark the job ``failed`` and
  emit an audit event.

In ``environment=dev`` Celery is eager, so this runs in-process — the same code
path the tests exercise.
"""

from __future__ import annotations

import time

from celery.exceptions import SoftTimeLimitExceeded
from loguru import logger

from .celery_app import celery_app
from ..core.store import store
from ..models.job import JobState
from ..models.optimization import OptimizationRunRequest, OptimizationStage
from ..services.audit import emit, make_event


@celery_app.task(
    name="radiarch.optimize.run",
    autoretry_for=(ConnectionError, OSError),
    retry_backoff=True,
    retry_backoff_max=120,
    max_retries=3,
)
def run_optimization_job(job_id: str, request_payload: dict):
    """Execute :meth:`OptimizationService.run` and mirror progress."""
    from ..services.optimization import OptimizationService

    job = store.get_optimization_job(job_id)
    if not job:
        logger.error("run_optimization_job called with unknown job_id=%s", job_id)
        return

    t0 = time.monotonic()

    def _on_progress(stage: OptimizationStage, fraction: float, message: str) -> None:
        store.update_optimization_job(
            job_id,
            state=JobState.running if fraction < 1.0 else job.state,
            progress=round(fraction, 3),
            stage=stage,
            message=message,
        )

    request: OptimizationRunRequest | None = None
    try:
        store.update_optimization_job(
            job_id,
            state=JobState.running,
            progress=0.0,
            stage=OptimizationStage.queued,
            message="Queued → running",
        )
        request = OptimizationRunRequest.model_validate(request_payload)
        service = OptimizationService()
        result = service.run(request, progress_callback=_on_progress)

    except SoftTimeLimitExceeded:
        logger.error(f"Optimization timed out for job {job_id}")
        store.update_optimization_job(
            job_id,
            state=JobState.failed,
            progress=1.0,
            stage=OptimizationStage.done,
            message="Timed out",
        )
        emit(make_event(
            "optimize.run",
            state="failed",
            error_type="SoftTimeLimitExceeded",
            error_message="optimization task hit soft time limit",
            duration_s=round(time.monotonic() - t0, 3),
            geometry_id=getattr(request, "geometry_id", None),
            beam_model_id=getattr(request, "beam_model_id", None),
            engine_name=getattr(getattr(request, "dose_engine", None), "name", None),
            extra={"job_id": job_id},
        ))
        return

    except Exception as exc:  # pragma: no cover — exercised in tests
        logger.exception(f"Optimization failed for job {job_id}")
        store.update_optimization_job(
            job_id,
            state=JobState.failed,
            progress=1.0,
            stage=OptimizationStage.done,
            message=f"{type(exc).__name__}: {exc}",
        )
        emit(make_event(
            "optimize.run",
            state="failed",
            error_type=type(exc).__name__,
            error_message=str(exc)[:500],
            duration_s=round(time.monotonic() - t0, 3),
            geometry_id=getattr(request, "geometry_id", None),
            beam_model_id=getattr(request, "beam_model_id", None),
            engine_name=getattr(getattr(request, "dose_engine", None), "name", None),
            extra={"job_id": job_id},
        ))
        return

    store.update_optimization_job(
        job_id,
        state=JobState.succeeded,
        progress=1.0,
        stage=OptimizationStage.done,
        message=f"Optimized in {time.monotonic() - t0:.1f}s",
        optimization_id=result.optimization_id,
    )
    emit(make_event(
        "optimize.run",
        state="succeeded",
        cache_key=result.cache_key,
        geometry_id=request.geometry_id,
        beam_model_id=request.beam_model_id,
        engine_name=request.dose_engine.name,
        engine_version=request.dose_engine.version,
        duration_s=round(time.monotonic() - t0, 3),
        extra={
            "job_id": job_id,
            "optimization_id": result.optimization_id,
            "iterations": result.convergence.iterations,
            "final_cost": result.convergence.final_cost,
            "solver_method": request.solver.method,
        },
    ))
