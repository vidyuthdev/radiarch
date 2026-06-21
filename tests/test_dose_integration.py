"""End-to-end integration test for Service 3.

This test does NOT stub the geometry or beam-model loaders — instead it
seeds the on-disk stores with real, persisted geometry + beam-model
artifacts and then drives :meth:`DoseService.compute_dose` and
:meth:`DoseService.build_influence` through the actual ``_load_*``
seams.

It uses the *analytic* engine end-to-end so the test runs without
OpenTPS or MCsquare.

The point of this test is to catch wiring bugs that purely-stubbed
tests can't: NIfTI round-tripping through SimpleITK, beam-model store
lookup paths, cache-key plumbing across services, etc.
"""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

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
from radiarch.services.beam_persistence import BeamModelStore
from radiarch.services.dose import DoseService
from radiarch.services.dose_persistence import read_dose_volume
from radiarch.services.persistence import GeometryStore


@pytest.fixture
def seeded_stores(tmp_path):
    """Create real on-disk geometry and beam-model entries.

    Returns (geometry_id, beam_model_id, geometry_dir, beam_model_dir,
    dose_dir, influence_dir).
    """
    geom_dir = tmp_path / "geometries"
    bm_dir = tmp_path / "beam_models"
    dose_dir = tmp_path / "doses"
    infl_dir = tmp_path / "influence"
    for d in [geom_dir, bm_dir, dose_dir, infl_dir]:
        d.mkdir()

    # ---- Geometry ----------------------------------------------------
    nz, ny, nx = 4, 8, 8
    density = np.ones((nz, ny, nx), dtype=np.float32)
    masks = np.zeros((nz, ny, nx), dtype=np.uint16)
    masks[1:3, 2:6, 2:6] = 1

    geometry_id = "g-int-001"
    geom_root = geom_dir / geometry_id
    geom_root.mkdir()
    density_path = geom_root / "density.nii.gz"
    masks_path = geom_root / "masks.nii.gz"

    img = sitk.GetImageFromArray(density)
    img.SetSpacing([2.0, 2.0, 3.0])
    sitk.WriteImage(img, str(density_path), useCompression=True)

    m_img = sitk.GetImageFromArray(masks)
    m_img.SetSpacing([2.0, 2.0, 3.0])
    sitk.WriteImage(m_img, str(masks_path), useCompression=True)

    geom_result = GeometryResult(
        geometry_id=geometry_id,
        density_grid_uri=str(density_path),
        structure_masks_uri=str(masks_path),
        structure_index={"PTV": 1},
        grid_spec=GridSpec(spacing_mm=(2.0, 2.0, 3.0),
                           origin_mm=(0.0, 0.0, 0.0),
                           size=(nx, ny, nz)),
        frame_of_reference_uid="1.2.840.10008.test",
        ct_metadata=CTMetadata(num_slices=nz),
        cache_key="cache-g",
    )
    meta_path = geom_root / "meta.json"
    meta_path.write_text(geom_result.model_dump_json(indent=2))
    # Index entry.
    (geom_dir / "_index.json").write_text(
        f'{{"cache-g": "{geometry_id}"}}'
    )

    # ---- Beam model --------------------------------------------------
    beam_model_id = "bm-int-001"
    bm_result = BeamModelResult(
        beam_model_id=beam_model_id,
        geometry_id=geometry_id,
        modality=Modality.proton_pbs,
        fluence_elements=FluenceElementSet(
            total_count=4,
            per_beam=[PerBeamElements(beam_id="B1", element_count=4,
                                      energy_layers=[100.0, 110.0],
                                      spots_per_layer=[2, 2])],
        ),
        beam_model_ref_uri=str(bm_dir / beam_model_id / "plan.pkl"),
        machine_model_id="default",
        cache_key="cache-bm",
    )
    bm_store = BeamModelStore(bm_dir)
    bm_store.save(
        beam_model_id=beam_model_id,
        cache_key="cache-bm",
        plan={"mock": "plan"},  # picklable
        result=bm_result,
    )

    return {
        "geometry_id": geometry_id,
        "beam_model_id": beam_model_id,
        "geom_dir": geom_dir,
        "bm_dir": bm_dir,
        "dose_dir": dose_dir,
        "infl_dir": infl_dir,
    }


@pytest.fixture
def patched_service(monkeypatch, seeded_stores):
    """A DoseService whose loaders use the seeded stores above.

    We monkeypatch GeometryService().store to use the test geom dir, and
    BeamModelStore's base_dir likewise — these are what the real
    ``_load_geometry`` and ``_load_beam_model`` reach for.
    """
    seeded = seeded_stores
    svc = DoseService(
        dose_dir=seeded["dose_dir"],
        influence_dir=seeded["infl_dir"],
    )

    # Replace the lookups inside the loaders to hit our test dirs.
    from radiarch.services import dose as dose_mod
    from radiarch.services.persistence import GeometryStore

    real_geom_store = GeometryStore(seeded["geom_dir"])

    class _StubGeoSvc:
        store = real_geom_store

    monkeypatch.setattr(dose_mod, "GeometryService", lambda: _StubGeoSvc())

    # For beam-model loader: rebind BeamModelStore inside _load_beam_model
    # by patching its base-dir resolution.
    from radiarch.config import get_settings as real_get_settings
    real_settings = real_get_settings()

    class _StubSettings:
        artifact_dir = str(seeded["bm_dir"].parent)

    monkeypatch.setattr(dose_mod, "get_settings",
                        lambda: _StubSettings())
    # _load_beam_model resolves bm_dir as Path(settings.artifact_dir) / "beam_models".
    # Our seeded bm_dir IS exactly that path, since we put it at tmp_path/"beam_models".

    return svc, seeded


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestIntegrationCompute:
    def test_end_to_end_nominal(self, patched_service):
        svc, seeded = patched_service
        req = DoseComputeRequest(
            geometry_id=seeded["geometry_id"],
            beam_model_id=seeded["beam_model_id"],
            engine=EngineSpec(name="analytic"),
            weights=WeightVector(length=4, values=[0.5, 0.5, 0.5, 0.5]),
        )
        result = svc.compute_dose(req)
        assert result.dose_id
        assert result.modality == Modality.proton_pbs
        assert result.engine_name == "analytic"
        # NIfTI round-trip
        arr = read_dose_volume(Path(result.dose_grid_uri))
        assert arr.shape == (4, 8, 8)
        assert arr.max() > 0

    def test_cache_hit_skips_rebuild(self, patched_service):
        svc, seeded = patched_service
        req = DoseComputeRequest(
            geometry_id=seeded["geometry_id"],
            beam_model_id=seeded["beam_model_id"],
            engine=EngineSpec(name="analytic"),
            weights=WeightVector(length=4, values=[1, 1, 1, 1]),
        )
        a = svc.compute_dose(req)
        b = svc.compute_dose(req)
        assert a.dose_id == b.dose_id

    def test_with_scenarios(self, patched_service):
        svc, seeded = patched_service
        req = DoseComputeRequest(
            geometry_id=seeded["geometry_id"],
            beam_model_id=seeded["beam_model_id"],
            engine=EngineSpec(name="analytic"),
            weights=WeightVector(length=4, values=[1, 1, 1, 1]),
            scenarios=ScenarioSetSpec(
                scenarios=[
                    ScenarioSpec(name="nominal"),
                    ScenarioSpec(name="up", range_scale=1.05),
                ]
            ),
        )
        result = svc.compute_dose(req)
        assert result.scenario_doses is not None
        # exactly the non-nominal entries persisted
        for entry in result.scenario_doses:
            assert Path(entry.dose_grid_uri).is_file()


class TestIntegrationInfluence:
    def test_build_and_round_trip(self, patched_service):
        svc, seeded = patched_service
        req = InfluenceBuildRequest(
            geometry_id=seeded["geometry_id"],
            beam_model_id=seeded["beam_model_id"],
            engine=EngineSpec(name="analytic"),
        )
        result = svc.build_influence(req)
        assert result.n_elements == 4
        assert result.n_voxels == 4 * 8 * 8
        # Sanity: load back, multiply, expect a sensible dose.
        inf = svc.influence_store.load_influence(result.influence_id)
        from radiarch.services.dose_engines import get_engine
        eng = get_engine("analytic")
        w = np.array([1, 1, 1, 1], dtype=np.float32)
        applied = eng.apply_influence(inf, w, (4, 8, 8)).dose
        assert applied.max() > 0
