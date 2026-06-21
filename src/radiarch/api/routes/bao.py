"""FastAPI routes for the BAO Service (Service 5 — Beam Angle Optimization).

Mirrors ``api/routes/optimization.py``: router-level auth, the
200/202/422/401 contract, and audit emission.

Endpoints
---------
``POST   /api/v1/bao/run``             — run / reuse cached beam-angle selection.
``GET    /api/v1/bao/jobs/{job_id}``   — poll an async run.
``GET    /api/v1/bao/{bao_id}``        — retrieve the result.
``DELETE /api/v1/bao/{bao_id}``        — drop from cache.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel

from ...core.store import store
from ...models.bao import BAOJobStatus, BAORunRequest, BAOResult
from ...services.audit import emit, make_event
from ...services.bao import BAOService
from ...services.dose_engines import EngineRegistryError
from ..security import api_key_auth

router = APIRouter(
    prefix="/bao",
    tags=["bao"],
    dependencies=[Depends(api_key_auth)],
)


@lru_cache(maxsize=1)
def _service() -> BAOService:
    return BAOService()


class BAORunResponse(BaseModel):
    job_id: str
    cache_key: str
    state: str = "queued"
    message: str = "Run dispatched; poll /bao/jobs/{job_id} for progress."


@router.post(
    "/run",
    summary="Run (or reuse cached) beam-angle optimization.",
    responses={
        200: {"description": "Cache hit — returned the existing result inline."},
        202: {"description": "Cache miss — Celery job dispatched."},
        401: {"description": "Missing / invalid API key."},
        422: {"description": "Request validation error."},
    },
)
async def run_bao(
    request: BAORunRequest,
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
                "bao.run", state="cache_hit", cache_key=cache_key,
                geometry_id=request.geometry_id,
                engine_name=request.dose_engine.name,
                api_key_prefix=api_key_prefix,
                extra={"bao_id": cached.bao_id},
            ))
            return cached

        job = store.create_bao_job(cache_key)
        from ...tasks.bao_tasks import run_bao_job
        run_bao_job.delay(job.id, request.model_dump(mode="json"))
        response.status_code = status.HTTP_202_ACCEPTED
        emit(make_event(
            "bao.run", state="dispatched", cache_key=cache_key,
            geometry_id=request.geometry_id,
            engine_name=request.dose_engine.name,
            api_key_prefix=api_key_prefix,
            extra={"job_id": job.id},
        ))
        return BAORunResponse(job_id=job.id, cache_key=cache_key)

    except (ValueError, EngineRegistryError) as exc:
        emit(make_event(
            "bao.run", state="failed",
            geometry_id=request.geometry_id,
            engine_name=request.dose_engine.name,
            error_type=type(exc).__name__,
            error_message=str(exc)[:200],
            api_key_prefix=api_key_prefix,
        ))
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get(
    "/jobs/{job_id}",
    response_model=BAOJobStatus,
    summary="Poll an async beam-angle-optimization run.",
)
async def get_bao_job(job_id: str) -> BAOJobStatus:
    job = store.get_bao_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"BAO job not found: {job_id}")
    return job


@router.get(
    "/{bao_id}",
    response_model=BAOResult,
    summary="Retrieve a completed beam-angle-optimization result.",
)
async def get_bao(bao_id: str) -> BAOResult:
    result = _service().store.get_by_id(bao_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"BAO result not found: {bao_id}")
    return result


@router.delete(
    "/{bao_id}",
    status_code=204,
    response_class=Response,
    summary="Delete a cached BAO result.",
)
async def delete_bao(bao_id: str):
    svc = _service()
    if svc.store.get_by_id(bao_id) is None:
        raise HTTPException(status_code=404, detail=f"BAO result not found: {bao_id}")
    svc.store.delete_by_id(bao_id)
    return Response(status_code=204)
