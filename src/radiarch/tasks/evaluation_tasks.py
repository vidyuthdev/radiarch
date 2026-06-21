"""Celery task for async plan evaluation (Service 6).

Mirrors ``optimization_tasks.run_optimization_job``: pull the job row, drive
:meth:`EvaluationService.run` while mirroring stage/progress to the DB row, stash
``evaluation_id`` on success, and emit audit events.
"""

from __future__ import annotations

import time

from celery.exceptions import SoftTimeLimitExceeded
from loguru import logger

from .celery_app import celery_app
from ..core.store import store
from ..models.evaluation import EvaluationRequest, EvaluationStage
from ..models.job import JobState
from ..services.audit import emit, make_event


@celery_app.task(
    name="radiarch.evaluate.run",
    autoretry_for=(ConnectionError, OSError),
    retry_backoff=True,
    retry_backoff_max=120,
    max_retries=3,
)
def run_evaluation_job(job_id: str, request_payload: dict):
    """Execute :meth:`EvaluationService.run` and mirror progress."""
    from ..services.evaluation import EvaluationService

    job = store.get_evaluation_job(job_id)
    if not job:
        logger.error("run_evaluation_job called with unknown job_id=%s", job_id)
        return

    t0 = time.monotonic()

    def _on_progress(stage: EvaluationStage, fraction: float, message: str) -> None:
        store.update_evaluation_job(
            job_id,
            state=JobState.running if fraction < 1.0 else job.state,
            progress=round(fraction, 3), stage=stage, message=message,
        )

    request: EvaluationRequest | None = None
    try:
        store.update_evaluation_job(
            job_id, state=JobState.running, progress=0.0,
            stage=EvaluationStage.queued, message="Queued → running",
        )
        request = EvaluationRequest.model_validate(request_payload)
        result = EvaluationService().run(request, progress_callback=_on_progress)

    except SoftTimeLimitExceeded:
        logger.error(f"Evaluation timed out for job {job_id}")
        store.update_evaluation_job(job_id, state=JobState.failed, progress=1.0,
                                    stage=EvaluationStage.done, message="Timed out")
        emit(make_event(
            "evaluate.run", state="failed", error_type="SoftTimeLimitExceeded",
            error_message="evaluation task hit soft time limit",
            duration_s=round(time.monotonic() - t0, 3),
            geometry_id=getattr(request, "geometry_id", None),
            extra={"job_id": job_id},
        ))
        return

    except Exception as exc:  # pragma: no cover — exercised in tests
        logger.exception(f"Evaluation failed for job {job_id}")
        store.update_evaluation_job(job_id, state=JobState.failed, progress=1.0,
                                    stage=EvaluationStage.done,
                                    message=f"{type(exc).__name__}: {exc}")
        emit(make_event(
            "evaluate.run", state="failed", error_type=type(exc).__name__,
            error_message=str(exc)[:500],
            duration_s=round(time.monotonic() - t0, 3),
            geometry_id=getattr(request, "geometry_id", None),
            extra={"job_id": job_id},
        ))
        return

    store.update_evaluation_job(
        job_id, state=JobState.succeeded, progress=1.0, stage=EvaluationStage.done,
        message=f"Evaluated in {time.monotonic() - t0:.1f}s",
        evaluation_id=result.evaluation_id,
    )
    emit(make_event(
        "evaluate.run", state="succeeded", cache_key=result.cache_key,
        geometry_id=request.geometry_id, dose_id=request.dose_id,
        duration_s=round(time.monotonic() - t0, 3),
        extra={
            "job_id": job_id, "evaluation_id": result.evaluation_id,
            "dvh_curves": len(result.dvh_curves),
            "has_indices": result.indices is not None,
            "has_gamma": result.gamma is not None,
        },
    ))
