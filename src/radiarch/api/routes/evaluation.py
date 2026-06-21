"""FastAPI routes for the Evaluation Service (Service 6).

Mirrors ``api/routes/optimization.py``: router-level auth, the
200/202/422/401 contract, audit emission.

Endpoints
---------
``POST   /api/v1/evaluate/run``                — run / reuse cached evaluation.
``GET    /api/v1/evaluate/jobs/{job_id}``      — poll an async run.
``GET    /api/v1/evaluate/{evaluation_id}``    — retrieve the report.
``DELETE /api/v1/evaluate/{evaluation_id}``    — drop from cache.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel

from ...core.store import store
from ...models.evaluation import (
    EvaluationJobStatus,
    EvaluationRequest,
    EvaluationResult,
)
from ...services.audit import emit, make_event
from ...services.evaluation import EvaluationService
from ..security import api_key_auth

router = APIRouter(
    prefix="/evaluate",
    tags=["evaluation"],
    dependencies=[Depends(api_key_auth)],
)


@lru_cache(maxsize=1)
def _service() -> EvaluationService:
    return EvaluationService()


class EvaluationRunResponse(BaseModel):
    job_id: str
    cache_key: str
    state: str = "queued"
    message: str = "Run dispatched; poll /evaluate/jobs/{job_id} for progress."


@router.post(
    "/run",
    summary="Run (or reuse cached) plan evaluation (DVH + indices + gamma).",
    responses={
        200: {"description": "Cache hit — returned the existing report inline."},
        202: {"description": "Cache miss — Celery job dispatched."},
        401: {"description": "Missing / invalid API key."},
        422: {"description": "Request validation error."},
    },
)
async def run_evaluation(
    request: EvaluationRequest,
    response: Response,
    api_key_prefix: Optional[str] = Depends(api_key_auth),
):
    try:
        cache_key = request.compute_cache_key()
        service = _service()
        cached = service.store.lookup_by_cache_key(cache_key)
        if cached is not None:
            response.status_code = status.HTTP_200_OK
            emit(make_event(
                "evaluate.run", state="cache_hit", cache_key=cache_key,
                geometry_id=request.geometry_id, dose_id=request.dose_id,
                api_key_prefix=api_key_prefix,
                extra={"evaluation_id": cached.evaluation_id},
            ))
            return cached

        job = store.create_evaluation_job(cache_key)
        from ...tasks.evaluation_tasks import run_evaluation_job
        run_evaluation_job.delay(job.id, request.model_dump(mode="json"))
        response.status_code = status.HTTP_202_ACCEPTED
        emit(make_event(
            "evaluate.run", state="dispatched", cache_key=cache_key,
            geometry_id=request.geometry_id, dose_id=request.dose_id,
            api_key_prefix=api_key_prefix, extra={"job_id": job.id},
        ))
        return EvaluationRunResponse(job_id=job.id, cache_key=cache_key)

    except ValueError as exc:
        emit(make_event(
            "evaluate.run", state="failed",
            geometry_id=request.geometry_id, dose_id=request.dose_id,
            error_type=type(exc).__name__, error_message=str(exc)[:200],
            api_key_prefix=api_key_prefix,
        ))
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get(
    "/jobs/{job_id}",
    response_model=EvaluationJobStatus,
    summary="Poll an async evaluation run.",
)
async def get_evaluation_job(job_id: str) -> EvaluationJobStatus:
    job = store.get_evaluation_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Evaluation job not found: {job_id}")
    return job


@router.get(
    "/{evaluation_id}",
    response_model=EvaluationResult,
    summary="Retrieve a completed evaluation report.",
)
async def get_evaluation(evaluation_id: str) -> EvaluationResult:
    result = _service().store.get_by_id(evaluation_id)
    if result is None:
        raise HTTPException(status_code=404,
                            detail=f"Evaluation not found: {evaluation_id}")
    return result


@router.delete(
    "/{evaluation_id}",
    status_code=204,
    response_class=Response,
    summary="Delete a cached evaluation report.",
)
async def delete_evaluation(evaluation_id: str):
    svc = _service()
    if svc.store.get_by_id(evaluation_id) is None:
        raise HTTPException(status_code=404,
                            detail=f"Evaluation not found: {evaluation_id}")
    svc.store.delete_by_id(evaluation_id)
    return Response(status_code=204)
