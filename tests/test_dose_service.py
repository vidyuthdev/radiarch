"""Tests for the :class:`DoseService` orchestrator.

Stubs out the geometry + beam-model loaders so the tests never touch
SimpleITK ReadImage or the on-disk beam-model store. The analytic engine
is registered by default and used end-to-end for nominal + scenario +
influence flows. Each test gets a fresh temp dir for the dose / influence
stores so caches don't leak across tests.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from radiarch.models.beam_model import (
    BeamModelResult,
    FluenceElementSet,
    Modality,
    PerBeamElements,
)
from radiarch.models.dose import (
    DoseComputeRequest,
    EngineSpec,
    InfluenceBuildRequest,
    ScenarioSetSpec,
    ScenarioSpec,
    WeightVector,
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


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

def _fake_geometry_bundle() -> GeometryBundle:
    nz, ny, nx = 4, 8, 8
    density = np.ones((nz, ny, nx), dtype=np.float32)
    masks = np.zeros((nz, ny, nx), dtype=np.uint16)
    masks[1:3, 2:6, 2:6] = 1
    return GeometryBundle(
        result=GeometryResult(
            geometry_id="g-1",
            density_grid_uri="/tmp/dx.nii.gz",
            structure_masks_uri="/tmp/mx.nii.gz",
            structure_index={"PTV": 1},
            grid_spec=GridSpec(spacing_mm=(2, 2, 3),
                               origin_mm=(0, 0, 0), size=(nx, ny, nz)),
            frame_of_reference_uid="1.2.3",
            ct_metadata=CTMetadata(num_slices=nz),
            cache_key="g-cache",
        ),
        density=density,
        masks=masks,
        spacing_mm=(2.0, 2.0, 3.0),
    )


def _fake_beam_model_bundle(total: int = 4,
                            modality: Modality = Modality.proton_pbs,
                            beam_model_id: str = "bm-1") -> BeamModelBundle:
    return BeamModelBundle(
        result=BeamModelResult(
            beam_model_id=beam_model_id, geometry_id="g-1", modality=modality,
            fluence_elements=FluenceElementSet(
                total_count=total,
                per_beam=[PerBeamElements(beam_id="B1", element_count=total,
                                          energy_layers=[100.0],
                                          spots_per_layer=[total])],
            ),
            beam_model_ref_uri="/tmp/plan.pkl",
            machine_model_id="default",
            cache_key="bm-cache",
        ),
        plan=object(),
    )


@pytest.fixture
def svc(monkeypatch):
    """A DoseService with stubbed loaders and isolated on-disk stores."""
    dose_tmp = tempfile.TemporaryDirectory()
    infl_tmp = tempfile.TemporaryDirectory()
    s = DoseService(dose_dir=dose_tmp.name, influence_dir=infl_tmp.name)

    # Stub the two loading seams.
    monkeypatch.setattr(s, "_load_geometry",
                        lambda gid: _fake_geometry_bundle())
    monkeypatch.setattr(s, "_load_beam_model",
                        lambda bid: _fake_beam_model_bundle())

    yield s
    dose_tmp.cleanup()
    infl_tmp.cleanup()


# ---------------------------------------------------------------------------
# compute_dose — nominal
# ---------------------------------------------------------------------------

class TestComputeDoseNominal:
    def _req(self, **over):
        kwargs = dict(
            geometry_id="g-1",
            beam_model_id="bm-1",
            engine=EngineSpec(name="analytic"),
            weights=WeightVector(length=4, values=[1, 1, 1, 1]),
        )
        kwargs.update(over)
        return DoseComputeRequest(**kwargs)

    def test_nominal_only_returns_result(self, svc):
        r = svc.compute_dose(self._req())
        assert r.dose_id
        assert r.modality == Modality.proton_pbs
        assert r.engine_name == "analytic"
        assert r.scenario_doses is None
        assert r.statistics.max_gy > 0
        # NIfTI on disk.
        assert Path(r.dose_grid_uri).is_file()

    def test_cache_hit_returns_same_id(self, svc):
        first = svc.compute_dose(self._req())
        second = svc.compute_dose(self._req())
        assert first.dose_id == second.dose_id
        assert first.cache_key == second.cache_key

    def test_different_weights_different_id(self, svc):
        a = svc.compute_dose(self._req())
        b = svc.compute_dose(
            self._req(weights=WeightVector(length=4, values=[2, 2, 2, 2]))
        )
        assert a.dose_id != b.dose_id
        assert b.statistics.max_gy > a.statistics.max_gy

    def test_weight_length_mismatch_rejected(self, svc):
        with pytest.raises(ValueError, match="length"):
            svc.compute_dose(
                self._req(weights=WeightVector(length=99, values=[1] * 99))
            )


# ---------------------------------------------------------------------------
# compute_dose — scenarios
# ---------------------------------------------------------------------------

class TestComputeDoseWithScenarios:
    def _req(self):
        return DoseComputeRequest(
            geometry_id="g-1", beam_model_id="bm-1",
            engine=EngineSpec(name="analytic"),
            weights=WeightVector(length=4, values=[1, 1, 1, 1]),
            scenarios=ScenarioSetSpec(
                scenarios=[
                    ScenarioSpec(name="nominal"),
                    ScenarioSpec(name="range_up", range_scale=1.05),
                    ScenarioSpec(name="density_down", density_scale=0.95),
                ]
            ),
        )

    def test_scenario_doses_populated(self, svc):
        r = svc.compute_dose(self._req())
        assert r.scenario_doses is not None
        # nominal is the first scenario in our list — the orchestrator
        # treats it as nominal, the rest become entries.
        assert len(r.scenario_doses) == 2
        names = {e.scenario_name for e in r.scenario_doses}
        assert names == {"range_up", "density_down"}

    def test_scenario_files_written(self, svc):
        r = svc.compute_dose(self._req())
        for entry in r.scenario_doses or []:
            assert Path(entry.dose_grid_uri).is_file()


# ---------------------------------------------------------------------------
# Modality gating
# ---------------------------------------------------------------------------

class TestModalityGating:
    def test_mcsquare_rejects_photon(self, svc, monkeypatch):
        monkeypatch.setattr(svc, "_load_beam_model",
                            lambda bid: _fake_beam_model_bundle(modality=Modality.photon_imrt))
        req = DoseComputeRequest(
            geometry_id="g-1", beam_model_id="bm-1",
            engine=EngineSpec(name="mcsquare"),
            weights=WeightVector(length=4, values=[1, 1, 1, 1]),
        )
        with pytest.raises(ValueError, match="does not support"):
            svc.compute_dose(req)

    def test_unknown_engine_rejected(self, svc):
        req = DoseComputeRequest(
            geometry_id="g-1", beam_model_id="bm-1",
            engine=EngineSpec(name="no-such-engine"),
            weights=WeightVector(length=4, values=[1, 1, 1, 1]),
        )
        with pytest.raises(ValueError):
            svc.compute_dose(req)


# ---------------------------------------------------------------------------
# build_influence
# ---------------------------------------------------------------------------

class TestBuildInfluence:
    def _req(self, **over):
        kwargs = dict(
            geometry_id="g-1", beam_model_id="bm-1",
            engine=EngineSpec(name="analytic"),
        )
        kwargs.update(over)
        return InfluenceBuildRequest(**kwargs)

    def test_influence_result_metadata(self, svc):
        r = svc.build_influence(self._req())
        assert r.influence_id
        assert r.n_elements == 4
        assert r.n_voxels == 4 * 8 * 8
        assert r.nnz > 0

    def test_influence_cache_hit(self, svc):
        a = svc.build_influence(self._req())
        b = svc.build_influence(self._req())
        assert a.influence_id == b.influence_id

    def test_influence_file_round_trips(self, svc):
        r = svc.build_influence(self._req())
        inf = svc.influence_store.load_influence(r.influence_id)
        assert inf.n_voxels == r.n_voxels
        assert inf.n_elements == r.n_elements
        assert inf.nnz == r.nnz

    def test_apply_influence_matches_compute_dose(self, svc):
        """End-to-end: apply(Dij, w) ≈ compute_dose(w).

        This is the key invariant the optimizer relies on.
        """
        compute_req = DoseComputeRequest(
            geometry_id="g-1", beam_model_id="bm-1",
            engine=EngineSpec(name="analytic"),
            weights=WeightVector(length=4, values=[0.2, 0.4, 0.6, 0.8]),
        )
        compute = svc.compute_dose(compute_req)
        infl = svc.build_influence(self._req())

        # Apply the saved Dij to the same weights via the engine.
        from radiarch.services.dose_engines import get_engine
        from radiarch.services.dose_persistence import read_dose_volume
        engine = get_engine("analytic")
        dij = svc.influence_store.load_influence(infl.influence_id)
        w = np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float32)
        # Bundle shape matches what _load_geometry returned.
        applied = engine.apply_influence(dij, w, (4, 8, 8)).dose
        from_disk = read_dose_volume(Path(compute.dose_grid_uri))
        np.testing.assert_allclose(applied, from_disk, rtol=1e-4, atol=1e-6)
