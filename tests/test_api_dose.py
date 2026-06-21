"""End-to-end tests for the /dose/* routes.

Same pattern as ``test_api_beam_model.py``: stub the service loaders so
SimpleITK + the geometry store never get touched, patch ``.delay`` to
``.run`` so Celery runs synchronously without Redis, and bypass the
Postgres lifespan init.
"""

from __future__ import annotations

import tempfile
import types
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from radiarch import app as radiarch_app
from radiarch.api.routes import dose as dose_route
from radiarch.app import create_app
from radiarch.core import store as store_module
from radiarch.models.beam_model import (
    BeamModelResult,
    FluenceElementSet,
    Modality,
    PerBeamElements,
)
from radiarch.models.geometry import (
    CTMetadata,
    GeometryResult,
    GridSpec,
)
from radiarch.services.dose import DoseService
from radiarch.services.dose_engines.protocol import (
    BeamModelBundle,
    GeometryBundle,
)
from radiarch.tasks import dose_tasks as dose_tasks_module


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

def _stub_geometry_bundle() -> GeometryBundle:
    nz, ny, nx = 4, 6, 6
    return GeometryBundle(
        result=GeometryResult(
            geometry_id="g-1",
            density_grid_uri="/tmp/d.nii.gz",
            structure_masks_uri="/tmp/m.nii.gz",
            structure_index={"PTV": 1},
            grid_spec=GridSpec(spacing_mm=(2, 2, 3),
                               origin_mm=(0, 0, 0), size=(nx, ny, nz)),
            frame_of_reference_uid="1.2.3",
            ct_metadata=CTMetadata(num_slices=nz),
            cache_key="g",
        ),
        density=np.ones((nz, ny, nx), dtype=np.float32),
        masks=np.zeros((nz, ny, nx), dtype=np.uint16),
        spacing_mm=(2.0, 2.0, 3.0),
    )


def _stub_beam_model_bundle(modality: Modality = Modality.proton_pbs) -> BeamModelBundle:
    return BeamModelBundle(
        result=BeamModelResult(
            beam_model_id="bm-1", geometry_id="g-1", modality=modality,
            fluence_elements=FluenceElementSet(
                total_count=4,
                per_beam=[PerBeamElements(beam_id="B1", element_count=4,
                                          energy_layers=[100.0],
                                          spots_per_layer=[4])],
            ),
            beam_model_ref_uri="/tmp/plan.pkl",
            machine_model_id="default",
            cache_key="bm",
        ),
        plan=object(),
    )


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def client(monkeypatch):
    dose_tmp = tempfile.TemporaryDirectory()
    infl_tmp = tempfile.TemporaryDirectory()
    svc = DoseService(dose_dir=dose_tmp.name, influence_dir=infl_tmp.name)

    # Stub loaders on both the instance the route uses AND any new
    # service instance built inside the Celery task body.
    monkeypatch.setattr(svc, "_load_geometry",
                        lambda gid: _stub_geometry_bundle())
    monkeypatch.setattr(svc, "_load_beam_model",
                        lambda bid: _stub_beam_model_bundle())
    monkeypatch.setattr(DoseService, "_load_geometry",
                        lambda self, gid: _stub_geometry_bundle())
    monkeypatch.setattr(DoseService, "_load_beam_model",
                        lambda self, bid: _stub_beam_model_bundle())

    # Force any new service instantiated inside the Celery task to use
    # the same tempdirs.
    original_init = DoseService.__init__

    def _init_to_tmp(self, dose_dir=None, influence_dir=None):
        original_init(self,
                      dose_dir=dose_tmp.name,
                      influence_dir=infl_tmp.name)

    monkeypatch.setattr(DoseService, "__init__", _init_to_tmp)

    # Reset and re-point singletons.
    store_module.reset_store()
    dose_route._service.cache_clear()
    monkeypatch.setattr(dose_route, "_service", lambda: svc)

    # Bypass the Postgres lifespan init.
    monkeypatch.setattr(radiarch_app, "init_db", lambda: None)

    # Bypass Celery for both task types.
    def _eager_dose_delay(job_id, request_payload):
        dose_tasks_module.build_dose_job.run(job_id, request_payload)
        return types.SimpleNamespace(id=job_id)

    def _eager_inf_delay(job_id, request_payload):
        dose_tasks_module.build_influence_job.run(job_id, request_payload)
        return types.SimpleNamespace(id=job_id)

    monkeypatch.setattr(
        dose_tasks_module.build_dose_job, "delay", _eager_dose_delay
    )
    monkeypatch.setattr(
        dose_tasks_module.build_influence_job, "delay", _eager_inf_delay
    )

    app = create_app()
    with TestClient(app) as c:
        yield c
    dose_tmp.cleanup()
    infl_tmp.cleanup()
    store_module.reset_store()


def _dose_payload(**over) -> dict:
    base = {
        "geometry_id": "g-1",
        "beam_model_id": "bm-1",
        "engine": {"name": "analytic", "version": "0.1.0", "params": {}},
        "weights": {"length": 4, "values": [1.0, 1.0, 1.0, 1.0]},
    }
    base.update(over)
    return base


def _influence_payload(**over) -> dict:
    base = {
        "geometry_id": "g-1",
        "beam_model_id": "bm-1",
        "engine": {"name": "analytic", "version": "0.1.0", "params": {}},
    }
    base.update(over)
    return base


# ---------------------------------------------------------------------------
# Async dispatch
# ---------------------------------------------------------------------------

class TestDoseDispatch:
    def test_first_compute_returns_202(self, client):
        r = client.post("/api/v1/dose/compute", json=_dose_payload())
        assert r.status_code == 202, r.text
        body = r.json()
        assert body["job_id"]
        assert body["kind"] == "dose"
        assert "dose_id" not in body

    def test_first_influence_returns_202(self, client):
        r = client.post("/api/v1/dose/influence", json=_influence_payload())
        assert r.status_code == 202, r.text
        assert r.json()["kind"] == "influence"

    def test_cache_hit_returns_200_with_full_result(self, client):
        first = client.post("/api/v1/dose/compute", json=_dose_payload())
        assert first.status_code == 202
        second = client.post("/api/v1/dose/compute", json=_dose_payload())
        assert second.status_code == 200, second.text
        body = second.json()
        assert body["dose_id"]
        assert body["engine_name"] == "analytic"
        assert body["statistics"]["max_gy"] > 0

    def test_influence_cache_hit(self, client):
        client.post("/api/v1/dose/influence", json=_influence_payload())
        r = client.post("/api/v1/dose/influence", json=_influence_payload())
        assert r.status_code == 200
        body = r.json()
        assert body["influence_id"]
        assert body["n_elements"] == 4


# ---------------------------------------------------------------------------
# Polling job endpoint
# ---------------------------------------------------------------------------

class TestJobsEndpoint:
    def test_succeeded_job_links_dose_id(self, client):
        first = client.post("/api/v1/dose/compute", json=_dose_payload())
        job_id = first.json()["job_id"]
        poll = client.get(f"/api/v1/dose/jobs/{job_id}")
        assert poll.status_code == 200
        body = poll.json()
        assert body["state"] == "succeeded"
        assert body["dose_id"]
        assert body["kind"] == "dose"

    def test_influence_job_links_influence_id(self, client):
        first = client.post("/api/v1/dose/influence", json=_influence_payload())
        job_id = first.json()["job_id"]
        poll = client.get(f"/api/v1/dose/jobs/{job_id}").json()
        assert poll["state"] == "succeeded"
        assert poll["influence_id"]
        assert poll["kind"] == "influence"

    def test_unknown_job_id_404(self, client):
        r = client.get("/api/v1/dose/jobs/does-not-exist")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# GET / DELETE
# ---------------------------------------------------------------------------

class TestGetAndDelete:
    def test_get_dose(self, client):
        client.post("/api/v1/dose/compute", json=_dose_payload())
        result = client.post("/api/v1/dose/compute", json=_dose_payload()).json()
        r = client.get(f"/api/v1/dose/{result['dose_id']}")
        assert r.status_code == 200
        assert r.json()["dose_id"] == result["dose_id"]

    def test_get_unknown_dose_404(self, client):
        r = client.get("/api/v1/dose/no-such-id")
        assert r.status_code == 404

    def test_delete_dose(self, client):
        client.post("/api/v1/dose/compute", json=_dose_payload())
        result = client.post("/api/v1/dose/compute", json=_dose_payload()).json()
        d = client.delete(f"/api/v1/dose/{result['dose_id']}")
        assert d.status_code == 204
        assert client.get(f"/api/v1/dose/{result['dose_id']}").status_code == 404

    def test_get_influence(self, client):
        client.post("/api/v1/dose/influence", json=_influence_payload())
        r = client.post("/api/v1/dose/influence", json=_influence_payload()).json()
        get = client.get(f"/api/v1/dose/influence/{r['influence_id']}")
        assert get.status_code == 200
        assert get.json()["influence_id"] == r["influence_id"]

    def test_delete_influence(self, client):
        client.post("/api/v1/dose/influence", json=_influence_payload())
        r = client.post("/api/v1/dose/influence", json=_influence_payload()).json()
        d = client.delete(f"/api/v1/dose/influence/{r['influence_id']}")
        assert d.status_code == 204


# ---------------------------------------------------------------------------
# Artifact streaming
# ---------------------------------------------------------------------------

class TestArtifactStreaming:
    def test_dose_artifact_returned(self, client):
        client.post("/api/v1/dose/compute", json=_dose_payload())
        result = client.post("/api/v1/dose/compute", json=_dose_payload()).json()
        r = client.get(f"/api/v1/dose/{result['dose_id']}/artifact")
        assert r.status_code == 200
        assert r.headers["content-type"] == "application/octet-stream"
        assert len(r.content) > 0

    def test_influence_artifact_returned(self, client):
        client.post("/api/v1/dose/influence", json=_influence_payload())
        r = client.post("/api/v1/dose/influence", json=_influence_payload()).json()
        out = client.get(f"/api/v1/dose/influence/{r['influence_id']}/artifact")
        assert out.status_code == 200
        assert len(out.content) > 0


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

class TestValidationErrors:
    def test_unknown_engine_422(self, client):
        payload = _dose_payload(engine={"name": "no-such", "version": "1", "params": {}})
        r = client.post("/api/v1/dose/compute", json=payload)
        # Engine validation surfaces during the Celery task body — so the
        # POST returns 202 and the job ends up failed. We poll to confirm.
        assert r.status_code == 202
        job_id = r.json()["job_id"]
        poll = client.get(f"/api/v1/dose/jobs/{job_id}").json()
        assert poll["state"] == "failed"
        assert "not registered" in (poll["message"] or "").lower() or \
               "no-such" in (poll["message"] or "")

    def test_weight_length_mismatch_fails_job(self, client):
        payload = _dose_payload(weights={"length": 99, "values": [1.0] * 99})
        r = client.post("/api/v1/dose/compute", json=payload)
        assert r.status_code == 202
        job = client.get(f"/api/v1/dose/jobs/{r.json()['job_id']}").json()
        assert job["state"] == "failed"
        assert "length" in (job["message"] or "").lower()


# ---------------------------------------------------------------------------
# Modality enforcement at the API level
# ---------------------------------------------------------------------------

class TestModalityEnforcementAtAPI:
    def test_mcsquare_on_photon_fails_job(self, client, monkeypatch):
        # Override the loader to return a photon beam model.
        monkeypatch.setattr(
            DoseService, "_load_beam_model",
            lambda self, bid: _stub_beam_model_bundle(modality=Modality.photon_imrt),
        )
        payload = _dose_payload(engine={"name": "mcsquare", "version": "0.1.0", "params": {}})
        r = client.post("/api/v1/dose/compute", json=payload)
        assert r.status_code == 202
        job = client.get(f"/api/v1/dose/jobs/{r.json()['job_id']}").json()
        assert job["state"] == "failed"
        assert "modality" in (job["message"] or "").lower() or \
               "PHOTON_IMRT" in (job["message"] or "")
