"""End-to-end tests for the /bao/* routes (Service 5).

Same pattern as ``test_api_optimization.py``: inject fake beam-model +
optimization services into every ``BAOService`` (route singleton and the one the
Celery task builds), point the store at a temp dir, run Celery eagerly, and
bypass the Postgres lifespan init.
"""

from __future__ import annotations

import tempfile
import types
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from radiarch import app as radiarch_app
from radiarch.api import security as security_module
from radiarch.api.routes import bao as bao_route
from radiarch.app import create_app
from radiarch.core import store as store_module
from radiarch.services.bao import BAOService
from radiarch.tasks import bao_tasks as bao_tasks_module


class _FakeBeamModelService:
    def build(self, req):
        gantries = "_".join(f"{b.gantry_deg:g}" for b in req.beam_set.beams)
        return SimpleNamespace(beam_model_id=f"bm-{gantries}")


class _FakeOptimizationService:
    def run(self, opt_req):
        gs = [float(x) for x in opt_req.beam_model_id[3:].split("_") if x]
        cost = sum((g - 180.0) ** 2 for g in gs) / len(gs)
        return SimpleNamespace(convergence=SimpleNamespace(final_cost=cost))


@pytest.fixture
def client(monkeypatch):
    tmp = tempfile.TemporaryDirectory()
    original_init = BAOService.__init__

    def _init(self, base_dir=None, beam_model_service=None, optimization_service=None):
        original_init(self, base_dir=tmp.name,
                      beam_model_service=_FakeBeamModelService(),
                      optimization_service=_FakeOptimizationService())

    monkeypatch.setattr(BAOService, "__init__", _init)

    svc = BAOService()
    store_module.reset_store()
    bao_route._service.cache_clear()
    monkeypatch.setattr(bao_route, "_service", lambda: svc)
    monkeypatch.setattr(radiarch_app, "init_db", lambda: None)

    def _eager(job_id, payload):
        bao_tasks_module.run_bao_job.run(job_id, payload)
        return types.SimpleNamespace(id=job_id)

    monkeypatch.setattr(bao_tasks_module.run_bao_job, "delay", _eager)

    app = create_app()
    with TestClient(app) as c:
        yield c
    tmp.cleanup()
    store_module.reset_store()


def _payload(**over) -> dict:
    base = {
        "geometry_id": "g-1",
        "dose_engine": {"name": "analytic", "version": "0.1.0", "params": {}},
        "objectives": [
            {"type": "DUniform", "structure_name": "PTV", "dose_gy": 10.0, "weight": 1.0}
        ],
        "n_beams": 2,
        "angle_step_deg": 90.0,
        "search": "greedy",
    }
    base.update(over)
    return base


class TestRunDispatch:
    def test_first_run_returns_202(self, client):
        r = client.post("/api/v1/bao/run", json=_payload())
        assert r.status_code == 202, r.text
        assert r.json()["job_id"]

    def test_cache_hit_returns_200(self, client):
        client.post("/api/v1/bao/run", json=_payload())
        r = client.post("/api/v1/bao/run", json=_payload())
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["bao_id"]
        assert any(c["gantry_deg"] == 180.0 for c in body["selected_angles"])


class TestJobsAndResult:
    def test_job_links_bao_id(self, client):
        job_id = client.post("/api/v1/bao/run", json=_payload()).json()["job_id"]
        poll = client.get(f"/api/v1/bao/jobs/{job_id}").json()
        assert poll["state"] == "succeeded"
        assert poll["bao_id"]

    def test_unknown_job_404(self, client):
        assert client.get("/api/v1/bao/jobs/nope").status_code == 404

    def test_get_and_delete(self, client):
        client.post("/api/v1/bao/run", json=_payload())
        result = client.post("/api/v1/bao/run", json=_payload()).json()
        bid = result["bao_id"]
        assert client.get(f"/api/v1/bao/{bid}").status_code == 200
        assert client.delete(f"/api/v1/bao/{bid}").status_code == 204
        assert client.get(f"/api/v1/bao/{bid}").status_code == 404


class TestValidation:
    def test_no_candidates_422(self, client):
        bad = _payload()
        del bad["angle_step_deg"]
        assert client.post("/api/v1/bao/run", json=bad).status_code == 422

    def test_n_beams_exceeds_candidates_fails_job(self, client):
        r = client.post("/api/v1/bao/run", json=_payload(n_beams=10))
        assert r.status_code == 202
        job = client.get(f"/api/v1/bao/jobs/{r.json()['job_id']}").json()
        assert job["state"] == "failed"
        assert "candidate" in (job["message"] or "").lower()


class TestAuth:
    def test_missing_key_rejected(self, client, monkeypatch):
        monkeypatch.setattr(
            security_module, "get_settings",
            lambda: SimpleNamespace(api_key="secret", api_key_header="X-API-Key"),
        )
        assert client.post("/api/v1/bao/run", json=_payload()).status_code == 401
        ok = client.post("/api/v1/bao/run", json=_payload(),
                         headers={"X-API-Key": "secret"})
        assert ok.status_code in (200, 202)
