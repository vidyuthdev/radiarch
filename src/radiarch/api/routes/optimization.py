"""FastAPI routes for the Optimization Service (Service 4).

Mirrors ``api/routes/dose.py`` — the same auth dependency at router level, the
same 200 (cache hit) / 202 (async dispatch) / 422 (validation) / 401 (auth)
status-code contract, and the same audit-emission pattern.

Endpoints
---------
``POST   /api/v1/optimize/run``                  — solve / reuse a cached plan.
``GET    /api/v1/optimize/jobs/{job_id}``        — poll an async run.
``GET    /api/v1/optimize/{opt_id}``             — retrieve the result.
``GET    /api/v1/optimize/{opt_id}/weights``     — stream the optimal weights (.npy).
``GET    /api/v1/optimize/{opt_id}/checkpoints`` — list checkpoint snapshots.
``DELETE /api/v1/optimize/{opt_id}``             — drop from cache.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ...core.store import store
from ...models.optimization import (
    CheckpointInfo,
    OptimizationJobStatus,
    OptimizationResult,
    OptimizationRunRequest,
)
from ...services.audit import emit, make_event
from ...services.dose_engines import EngineRegistryError
from ...services.optimization import OptimizationService
from ...services.optimization_persistence import WEIGHTS_FILENAME
from ..security import api_key_auth

router = APIRouter(
    prefix="/optimize",
    tags=["optimization"],
    dependencies=[Depends(api_key_auth)],
)


@lru_cache(maxsize=1)
def _service() -> OptimizationService:
    """Singleton service instance, reused across requests."""
    return OptimizationService()


class OptimizationRunResponse(BaseModel):
    """Returned by ``POST /run`` on a cache miss (async dispatch)."""

    job_id: str
    cache_key: str
    state: str = "queued"
    message: str = "Run dispatched; poll /optimize/jobs/{job_id} for progress."


# ---------------------------------------------------------------------------
# POST /run
# ---------------------------------------------------------------------------

@router.post(
    "/run",
    summary="Run (or reuse cached) inverse-plan optimization.",
    responses={
        200: {"description": "Cache hit — returned the existing result inline."},
        202: {"description": "Cache miss — Celery job dispatched."},
        401: {"description": "Missing / invalid API key."},
        422: {"description": "Request validation error."},
    },
)
async def run_optimization(
    request: OptimizationRunRequest,
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
                "optimize.run", state="cache_hit", cache_key=cache_key,
                geometry_id=request.geometry_id,
                beam_model_id=request.beam_model_id,
                engine_name=request.dose_engine.name,
                engine_version=request.dose_engine.version,
                api_key_prefix=api_key_prefix,
                extra={"optimization_id": cached.optimization_id},
            ))
            return cached

        job = store.create_optimization_job(cache_key)
        from ...tasks.optimization_tasks import run_optimization_job
        run_optimization_job.delay(job.id, request.model_dump(mode="json"))
        response.status_code = status.HTTP_202_ACCEPTED
        emit(make_event(
            "optimize.run", state="dispatched", cache_key=cache_key,
            geometry_id=request.geometry_id,
            beam_model_id=request.beam_model_id,
            engine_name=request.dose_engine.name,
            engine_version=request.dose_engine.version,
            api_key_prefix=api_key_prefix,
            extra={"job_id": job.id},
        ))
        return OptimizationRunResponse(job_id=job.id, cache_key=cache_key)

    except (ValueError, EngineRegistryError) as exc:
        emit(make_event(
            "optimize.run", state="failed",
            geometry_id=request.geometry_id,
            beam_model_id=request.beam_model_id,
            engine_name=request.dose_engine.name,
            error_type=type(exc).__name__,
            error_message=str(exc)[:200],
            api_key_prefix=api_key_prefix,
        ))
        raise HTTPException(status_code=422, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# GET /jobs/{job_id}
# ---------------------------------------------------------------------------

@router.get(
    "/jobs/{job_id}",
    response_model=OptimizationJobStatus,
    summary="Poll an async optimization run.",
)
async def get_optimization_job(job_id: str) -> OptimizationJobStatus:
    job = store.get_optimization_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Optimization job not found: {job_id}")
    return job


# ---------------------------------------------------------------------------
# Result + artifacts + delete
# ---------------------------------------------------------------------------

@router.get(
    "/{opt_id}",
    response_model=OptimizationResult,
    summary="Retrieve a completed optimization result.",
)
async def get_optimization(opt_id: str) -> OptimizationResult:
    result = _service().store.get_by_id(opt_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Optimization not found: {opt_id}")
    return result


@router.get(
    "/{opt_id}/weights",
    summary="Stream the optimal weight vector (.npy).",
    response_class=FileResponse,
)
async def get_optimization_weights(opt_id: str):
    svc = _service()
    if svc.store.get_by_id(opt_id) is None:
        raise HTTPException(status_code=404, detail=f"Optimization not found: {opt_id}")
    weights_path = svc.store.base_dir / opt_id / WEIGHTS_FILENAME
    if not os.path.isfile(weights_path):
        raise HTTPException(status_code=410, detail=f"{WEIGHTS_FILENAME} no longer on disk")
    return FileResponse(
        path=str(weights_path),
        media_type="application/octet-stream",
        filename=WEIGHTS_FILENAME,
    )


@router.get(
    "/{opt_id}/checkpoints",
    response_model=List[CheckpointInfo],
    summary="List checkpoint snapshots for a run.",
)
async def get_optimization_checkpoints(opt_id: str) -> List[CheckpointInfo]:
    result = _service().store.get_by_id(opt_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Optimization not found: {opt_id}")
    return result.checkpoints


@router.delete(
    "/{opt_id}",
    status_code=204,
    response_class=Response,
    summary="Delete a cached optimization.",
)
async def delete_optimization(opt_id: str):
    svc = _service()
    if svc.store.get_by_id(opt_id) is None:
        raise HTTPException(status_code=404, detail=f"Optimization not found: {opt_id}")
    svc.store.delete_by_id(opt_id)
    return Response(status_code=204)
