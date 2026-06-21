"""End-to-end tests for the /optimize/* routes (Service 4, task O16).

Same pattern as ``test_api_dose.py``: stub the geometry + beam-model loaders so
SimpleITK + the on-disk stores are never touched, force every ``OptimizationService``
(route singleton *and* the one the Celery task builds) to use isolated temp dirs,
patch ``.delay`` to ``.run`` so Celery runs synchronously without Redis, and bypass
the Postgres lifespan init.
"""

from __future__ import annotations

import tempfile
import types

import numpy as np
import pytest
from fastapi.testclient import TestClient

from radiarch import app as radiarch_app
from radiarch.api import security as security_module
from radiarch.api.routes import optimization as opt_route
from radiarch.app import create_app
from radiarch.core import store as store_module
from radiarch.models.beam_model import (
    BeamModelResult,
    FluenceElementSet,
    Modality,
    PerBeamElements,
)
from radiarch.models.geometry import CTMetadata, GeometryResult, GridSpec
from radiarch.services.dose import DoseService
from radiarch.services.dose_engines.protocol import BeamModelBundle, GeometryBundle
from radiarch.services.optimization import OptimizationService
from radiarch.tasks import optimization_tasks as opt_tasks_module


def _stub_geometry_bundle() -> GeometryBundle:
    nz, ny, nx = 4, 6, 6
    masks = np.zeros((nz, ny, nx), dtype=np.uint16)
    masks[1:3, 2:4, 2:4] = 1  # PTV
    return GeometryBundle(
        result=GeometryResult(
            geometry_id="g-1",
            density_grid_uri="/tmp/d.nii.gz",
            structure_masks_uri="/tmp/m.nii.gz",
            structure_index={"PTV": 1},
            grid_spec=GridSpec(spacing_mm=(2, 2, 3), origin_mm=(0, 0, 0),
                               size=(nx, ny, nz)),
            frame_of_reference_uid="1.2.3",
            ct_metadata=CTMetadata(num_slices=nz),
            cache_key="g",
        ),
        density=np.ones((nz, ny, nx), dtype=np.float32),
        masks=masks,
        spacing_mm=(2.0, 2.0, 3.0),
    )


def _stub_beam_model_bundle() -> BeamModelBundle:
    return BeamModelBundle(
        result=BeamModelResult(
            beam_model_id="bm-1", geometry_id="g-1", modality=Modality.proton_pbs,
            fluence_elements=FluenceElementSet(
                total_count=4,
                per_beam=[PerBeamElements(beam_id="B1", element_count=4,
                                          energy_layers=[100.0], spots_per_layer=[4])],
            ),
            beam_model_ref_uri="/tmp/plan.pkl",
            machine_model_id="default",
            cache_key="bm",
        ),
        plan=object(),
    )


@pytest.fixture
def client(monkeypatch):
    opt_tmp = tempfile.TemporaryDirectory()
    dose_tmp = tempfile.TemporaryDirectory()
    infl_tmp = tempfile.TemporaryDirectory()

    # Stub DoseService loaders at the class level so every instance — including
    # the one OptimizationService builds inside the Celery task — uses them.
    monkeypatch.setattr(DoseService, "_load_geometry",
                        lambda self, gid: _stub_geometry_bundle())
    monkeypatch.setattr(DoseService, "_load_beam_model",
                        lambda self, bid: _stub_beam_model_bundle())

    # Force every OptimizationService to use the isolated temp dirs.
    original_opt_init = OptimizationService.__init__

    def _opt_init(self, base_dir=None, dose_service=None):
        ds = DoseService(dose_dir=dose_tmp.name, influence_dir=infl_tmp.name)
        original_opt_init(self, base_dir=opt_tmp.name, dose_service=ds)

    monkeypatch.setattr(OptimizationService, "__init__", _opt_init)

    svc = OptimizationService()
    store_module.reset_store()
    opt_route._service.cache_clear()
    monkeypatch.setattr(opt_route, "_service", lambda: svc)
    monkeypatch.setattr(radiarch_app, "init_db", lambda: None)

    def _eager_delay(job_id, request_payload):
        opt_tasks_module.run_optimization_job.run(job_id, request_payload)
        return types.SimpleNamespace(id=job_id)

    monkeypatch.setattr(opt_tasks_module.run_optimization_job, "delay", _eager_delay)

    app = create_app()
    with TestClient(app) as c:
        yield c
    opt_tmp.cleanup()
    dose_tmp.cleanup()
    infl_tmp.cleanup()
    store_module.reset_store()


def _payload(**over) -> dict:
    base = {
        "geometry_id": "g-1",
        "beam_model_id": "bm-1",
        "dose_engine": {"name": "analytic", "version": "0.1.0", "params": {}},
        "objectives": [
            {"type": "DUniform", "structure_name": "PTV", "dose_gy": 10.0, "weight": 1.0}
        ],
        "solver": {"method": "L-BFGS-B", "max_iterations": 50},
    }
    base.update(over)
    return base


# ---------------------------------------------------------------------------
# Dispatch + cache
# ---------------------------------------------------------------------------

class TestRunDispatch:
    def test_first_run_returns_202(self, client):
        r = client.post("/api/v1/optimize/run", json=_payload())
        assert r.status_code == 202, r.text
        body = r.json()
        assert body["job_id"]
        assert "optimization_id" not in body

    def test_cache_hit_returns_200_full_result(self, client):
        first = client.post("/api/v1/optimize/run", json=_payload())
        assert first.status_code == 202
        second = client.post("/api/v1/optimize/run", json=_payload())
        assert second.status_code == 200, second.text
        body = second.json()
        assert body["optimization_id"]
        assert body["engine_name"] == "analytic"
        assert body["convergence"]["final_cost"] >= 0.0


class TestJobsEndpoint:
    def test_succeeded_job_links_optimization_id(self, client):
        first = client.post("/api/v1/optimize/run", json=_payload())
        job_id = first.json()["job_id"]
        poll = client.get(f"/api/v1/optimize/jobs/{job_id}").json()
        assert poll["state"] == "succeeded"
        assert poll["optimization_id"]

    def test_unknown_job_404(self, client):
        assert client.get("/api/v1/optimize/jobs/nope").status_code == 404


class TestResultAndArtifacts:
    def _run(self, client) -> dict:
        client.post("/api/v1/optimize/run", json=_payload())
        return client.post("/api/v1/optimize/run", json=_payload()).json()

    def test_get_result(self, client):
        result = self._run(client)
        r = client.get(f"/api/v1/optimize/{result['optimization_id']}")
        assert r.status_code == 200
        assert r.json()["optimization_id"] == result["optimization_id"]

    def test_get_unknown_404(self, client):
        assert client.get("/api/v1/optimize/nope").status_code == 404

    def test_weights_stream(self, client):
        result = self._run(client)
        r = client.get(f"/api/v1/optimize/{result['optimization_id']}/weights")
        assert r.status_code == 200
        assert r.headers["content-type"] == "application/octet-stream"
        assert len(r.content) > 0

    def test_checkpoints_list(self, client):
        client.post("/api/v1/optimize/run",
                    json=_payload(solver={"method": "ProjectedGradient",
                                          "max_iterations": 6},
                                  checkpoint_interval=2))
        result = client.post("/api/v1/optimize/run",
                             json=_payload(solver={"method": "ProjectedGradient",
                                                   "max_iterations": 6},
                                           checkpoint_interval=2)).json()
        r = client.get(f"/api/v1/optimize/{result['optimization_id']}/checkpoints")
        assert r.status_code == 200
        assert len(r.json()) >= 1

    def test_delete(self, client):
        result = self._run(client)
        oid = result["optimization_id"]
        assert client.delete(f"/api/v1/optimize/{oid}").status_code == 204
        assert client.get(f"/api/v1/optimize/{oid}").status_code == 404


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_dvhmin_without_volume_fraction_422(self, client):
        # Pydantic rejects the request at the POST boundary.
        bad = _payload(objectives=[
            {"type": "DVHMin", "structure_name": "PTV", "dose_gy": 10.0, "weight": 1.0}
        ])
        r = client.post("/api/v1/optimize/run", json=bad)
        assert r.status_code == 422, r.text

    def test_unknown_engine_fails_job(self, client):
        bad = _payload(dose_engine={"name": "no-such", "version": "1", "params": {}})
        r = client.post("/api/v1/optimize/run", json=bad)
        assert r.status_code == 202
        job = client.get(f"/api/v1/optimize/jobs/{r.json()['job_id']}").json()
        assert job["state"] == "failed"

    def test_unknown_structure_fails_job(self, client):
        bad = _payload(objectives=[
            {"type": "DUniform", "structure_name": "NOPE", "dose_gy": 10.0, "weight": 1.0}
        ])
        r = client.post("/api/v1/optimize/run", json=bad)
        assert r.status_code == 202
        job = client.get(f"/api/v1/optimize/jobs/{r.json()['job_id']}").json()
        assert job["state"] == "failed"
        assert "NOPE" in (job["message"] or "")


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

class TestAuth:
    def test_missing_key_rejected_when_auth_enabled(self, client, monkeypatch):
        monkeypatch.setattr(
            security_module, "get_settings",
            lambda: types.SimpleNamespace(api_key="secret", api_key_header="X-API-Key"),
        )
        # No header → 401.
        r = client.post("/api/v1/optimize/run", json=_payload())
        assert r.status_code == 401
        # Correct header → not 401 (202 dispatch or 200 cache hit).
        ok = client.post("/api/v1/optimize/run", json=_payload(),
                         headers={"X-API-Key": "secret"})
        assert ok.status_code in (200, 202), ok.text
