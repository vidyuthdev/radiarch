"""Celery task for async beam-angle optimization (Service 5).

Mirrors ``optimization_tasks.run_optimization_job``: pull the job row, drive
:meth:`BAOService.run` while mirroring stage/progress to the DB row, stash
``bao_id`` on success, and emit audit events.
"""

from __future__ import annotations

import time

from celery.exceptions import SoftTimeLimitExceeded
from loguru import logger

from .celery_app import celery_app
from ..core.store import store
from ..models.bao import BAORunRequest, BAOStage
from ..models.job import JobState
from ..services.audit import emit, make_event


@celery_app.task(
    name="radiarch.bao.run",
    autoretry_for=(ConnectionError, OSError),
    retry_backoff=True,
    retry_backoff_max=120,
    max_retries=3,
)
def run_bao_job(job_id: str, request_payload: dict):
    """Execute :meth:`BAOService.run` and mirror progress."""
    from ..services.bao import BAOService

    job = store.get_bao_job(job_id)
    if not job:
        logger.error("run_bao_job called with unknown job_id=%s", job_id)
        return

    t0 = time.monotonic()

    def _on_progress(stage: BAOStage, fraction: float, message: str) -> None:
        store.update_bao_job(
            job_id,
            state=JobState.running if fraction < 1.0 else job.state,
            progress=round(fraction, 3),
            stage=stage,
            message=message,
        )

    request: BAORunRequest | None = None
    try:
        store.update_bao_job(
            job_id, state=JobState.running, progress=0.0,
            stage=BAOStage.queued, message="Queued → running",
        )
        request = BAORunRequest.model_validate(request_payload)
        result = BAOService().run(request, progress_callback=_on_progress)

    except SoftTimeLimitExceeded:
        logger.error(f"BAO timed out for job {job_id}")
        store.update_bao_job(job_id, state=JobState.failed, progress=1.0,
                             stage=BAOStage.done, message="Timed out")
        emit(make_event(
            "bao.run", state="failed", error_type="SoftTimeLimitExceeded",
            error_message="bao task hit soft time limit",
            duration_s=round(time.monotonic() - t0, 3),
            geometry_id=getattr(request, "geometry_id", None),
            engine_name=getattr(getattr(request, "dose_engine", None), "name", None),
            extra={"job_id": job_id},
        ))
        return

    except Exception as exc:  # pragma: no cover — exercised in tests
        logger.exception(f"BAO failed for job {job_id}")
        store.update_bao_job(job_id, state=JobState.failed, progress=1.0,
                             stage=BAOStage.done,
                             message=f"{type(exc).__name__}: {exc}")
        emit(make_event(
            "bao.run", state="failed", error_type=type(exc).__name__,
            error_message=str(exc)[:500],
            duration_s=round(time.monotonic() - t0, 3),
            geometry_id=getattr(request, "geometry_id", None),
            engine_name=getattr(getattr(request, "dose_engine", None), "name", None),
            extra={"job_id": job_id},
        ))
        return

    store.update_bao_job(
        job_id, state=JobState.succeeded, progress=1.0, stage=BAOStage.done,
        message=f"Selected {len(result.selected_angles)} beams in "
                f"{time.monotonic() - t0:.1f}s",
        bao_id=result.bao_id,
    )
    emit(make_event(
        "bao.run", state="succeeded", cache_key=result.cache_key,
        geometry_id=request.geometry_id, engine_name=request.dose_engine.name,
        engine_version=request.dose_engine.version,
        duration_s=round(time.monotonic() - t0, 3),
        extra={
            "job_id": job_id, "bao_id": result.bao_id,
            "selected": [c.key() for c in result.selected_angles],
            "final_score": result.final_score, "search": request.search,
        },
    ))
