"""End-to-end tests for the /evaluate/* routes (Service 6).

Same pattern as ``test_api_optimization.py``: stub the geometry loader, supply
the dose as a ``file://`` NIfTI, force every ``EvaluationService`` onto a temp
store, run Celery eagerly, and bypass the Postgres lifespan init.
"""

from __future__ import annotations

import tempfile
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import SimpleITK as sitk
from fastapi.testclient import TestClient

from radiarch import app as radiarch_app
from radiarch.api import security as security_module
from radiarch.api.routes import evaluation as eval_route
from radiarch.app import create_app
from radiarch.core import store as store_module
from radiarch.models.geometry import CTMetadata, GeometryResult, GridSpec
from radiarch.services.dose import DoseService
from radiarch.services.dose_engines.protocol import GeometryBundle
from radiarch.services.evaluation import EvaluationService
from radiarch.tasks import evaluation_tasks as eval_tasks_module


def _geometry_bundle() -> GeometryBundle:
    nz, ny, nx = 6, 6, 6
    masks = np.zeros((nz, ny, nx), dtype=np.uint16)
    masks[2:4, 2:4, 2:4] = 1
    return GeometryBundle(
        result=GeometryResult(
            geometry_id="g-1", density_grid_uri="/tmp/d.nii.gz",
            structure_masks_uri="/tmp/m.nii.gz", structure_index={"PTV": 1},
            grid_spec=GridSpec(spacing_mm=(2, 2, 2), origin_mm=(0, 0, 0),
                               size=(nx, ny, nz)),
            frame_of_reference_uid="1.2.3",
            ct_metadata=CTMetadata(num_slices=nz), cache_key="g",
        ),
        density=np.ones((nz, ny, nx), dtype=np.float32),
        masks=masks, spacing_mm=(2.0, 2.0, 2.0),
    )


@pytest.fixture
def client(monkeypatch):
    tmp = tempfile.TemporaryDirectory()

    dose = np.zeros((6, 6, 6), dtype=np.float32)
    dose[2:4, 2:4, 2:4] = 60.0
    dose_path = Path(tmp.name) / "dose.nii.gz"
    img = sitk.GetImageFromArray(dose)
    img.SetSpacing([2.0, 2.0, 2.0])
    sitk.WriteImage(img, str(dose_path), useCompression=True)

    monkeypatch.setattr(DoseService, "_load_geometry",
                        lambda self, gid: _geometry_bundle())

    original_init = EvaluationService.__init__

    def _init(self, base_dir=None, dose_service=None):
        original_init(self, base_dir=tmp.name,
                      dose_service=DoseService(dose_dir=tmp.name, influence_dir=tmp.name))

    monkeypatch.setattr(EvaluationService, "__init__", _init)

    svc = EvaluationService()
    store_module.reset_store()
    eval_route._service.cache_clear()
    monkeypatch.setattr(eval_route, "_service", lambda: svc)
    monkeypatch.setattr(radiarch_app, "init_db", lambda: None)

    def _eager(job_id, payload):
        eval_tasks_module.run_evaluation_job.run(job_id, payload)
        return types.SimpleNamespace(id=job_id)

    monkeypatch.setattr(eval_tasks_module.run_evaluation_job, "delay", _eager)

    app = create_app()
    with TestClient(app) as c:
        c._dose_uri = f"file://{dose_path}"
        yield c
    tmp.cleanup()
    store_module.reset_store()


def _payload(client, **over) -> dict:
    base = {
        "dose_ref_uri": client._dose_uri,
        "geometry_id": "g-1",
        "prescription_gy": 60.0,
        "target_structure": "PTV",
    }
    base.update(over)
    return base


class TestRunDispatch:
    def test_first_run_returns_202(self, client):
        r = client.post("/api/v1/evaluate/run", json=_payload(client))
        assert r.status_code == 202, r.text
        assert r.json()["job_id"]

    def test_cache_hit_returns_200(self, client):
        client.post("/api/v1/evaluate/run", json=_payload(client))
        r = client.post("/api/v1/evaluate/run", json=_payload(client))
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["evaluation_id"]
        assert body["indices"]["coverage_pct"] == pytest.approx(100.0)
        assert len(body["dvh_curves"]) == 1


class TestJobsAndResult:
    def test_job_links_evaluation_id(self, client):
        job_id = client.post("/api/v1/evaluate/run",
                             json=_payload(client)).json()["job_id"]
        poll = client.get(f"/api/v1/evaluate/jobs/{job_id}").json()
        assert poll["state"] == "succeeded"
        assert poll["evaluation_id"]

    def test_unknown_job_404(self, client):
        assert client.get("/api/v1/evaluate/jobs/nope").status_code == 404

    def test_get_and_delete(self, client):
        client.post("/api/v1/evaluate/run", json=_payload(client))
        result = client.post("/api/v1/evaluate/run", json=_payload(client)).json()
        eid = result["evaluation_id"]
        assert client.get(f"/api/v1/evaluate/{eid}").status_code == 200
        assert client.delete(f"/api/v1/evaluate/{eid}").status_code == 204
        assert client.get(f"/api/v1/evaluate/{eid}").status_code == 404


class TestValidation:
    def test_no_dose_source_422(self, client):
        bad = _payload(client)
        del bad["dose_ref_uri"]
        assert client.post("/api/v1/evaluate/run", json=bad).status_code == 422

    def test_unknown_structure_fails_job(self, client):
        r = client.post("/api/v1/evaluate/run",
                        json=_payload(client, structures=["NOPE"]))
        assert r.status_code == 202
        job = client.get(f"/api/v1/evaluate/jobs/{r.json()['job_id']}").json()
        assert job["state"] == "failed"
        assert "NOPE" in (job["message"] or "")


class TestAuth:
    def test_missing_key_rejected(self, client, monkeypatch):
        monkeypatch.setattr(
            security_module, "get_settings",
            lambda: SimpleNamespace(api_key="secret", api_key_header="X-API-Key"),
        )
        assert client.post("/api/v1/evaluate/run",
                           json=_payload(client)).status_code == 401
        ok = client.post("/api/v1/evaluate/run", json=_payload(client),
                         headers={"X-API-Key": "secret"})
        assert ok.status_code in (200, 202)
