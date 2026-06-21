"""Tests for the engine plugins beyond the analytic engine.

The analytic engine is covered in ``test_dose_d1.py``. This file focuses
on the modality gating in MCsquareEngine and CCCEngine, plus the
``EngineUnavailableError`` contract for unimplemented operations.

These tests do NOT require OpenTPS or MCsquare to be importable — both
plugins are designed to register cleanly and surface a clean error when
the backing tooling isn't available.
"""

from __future__ import annotations

import numpy as np
import pytest

from radiarch.models.beam_model import (
    BeamModelResult,
    FluenceElementSet,
    Modality,
    PerBeamElements,
)
from radiarch.models.dose import ScenarioSpec
from radiarch.models.geometry import CTMetadata, GeometryResult, GridSpec
from radiarch.services.dose_engines import (
    EngineUnavailableError,
    get_engine,
    list_engines,
)
from radiarch.services.dose_engines.ccc import CCCEngine
from radiarch.services.dose_engines.mcsquare import MCsquareEngine
from radiarch.services.dose_engines.protocol import (
    BeamModelBundle,
    GeometryBundle,
)


# ---------------------------------------------------------------------------
# Bundles
# ---------------------------------------------------------------------------

def _geom() -> GeometryBundle:
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
        spacing_mm=(2, 2, 3),
    )


def _bm(modality: Modality, total: int = 4) -> BeamModelBundle:
    return BeamModelBundle(
        result=BeamModelResult(
            beam_model_id="bm-1", geometry_id="g-1", modality=modality,
            fluence_elements=FluenceElementSet(
                total_count=total,
                per_beam=[PerBeamElements(beam_id="B1", element_count=total,
                                          energy_layers=[100.0],
                                          spots_per_layer=[total])],
            ),
            beam_model_ref_uri="/tmp/plan.pkl",
            machine_model_id="default",
            cache_key="bm",
        ),
        plan=object(),
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

class TestEnginesRegistered:
    def test_all_three_registered(self):
        names = list_engines()
        assert "analytic" in names
        assert "mcsquare" in names
        assert "ccc" in names

    def test_each_has_protocol_methods(self):
        for name in ["analytic", "mcsquare", "ccc"]:
            e = get_engine(name)
            for method in ("validate", "compute_dose", "build_influence",
                           "apply_influence", "compute_grad"):
                assert callable(getattr(e, method))

    def test_modalities_are_correct(self):
        assert "PROTON_PBS" in get_engine("mcsquare").modalities
        assert "PHOTON_IMRT" in get_engine("ccc").modalities
        # analytic claims both for testability
        assert "PROTON_PBS" in get_engine("analytic").modalities
        assert "PHOTON_IMRT" in get_engine("analytic").modalities


# ---------------------------------------------------------------------------
# MCsquare
# ---------------------------------------------------------------------------

class TestMCsquareGating:
    def test_rejects_photon_modality(self):
        eng = MCsquareEngine()
        issues = eng.validate(_geom(), _bm(Modality.photon_imrt), {})
        assert any("PROTON_PBS" in i for i in issues)

    def test_accepts_proton_modality(self):
        eng = MCsquareEngine()
        issues = eng.validate(_geom(), _bm(Modality.proton_pbs), {})
        # validation may still flag things related to OpenTPS, but not modality
        assert not any("PROTON_PBS" in i and "got" in i for i in issues)

    def test_compute_dose_without_backend_raises_unavailable(self, monkeypatch):
        # Force the "no opentps" branch even if it happens to be importable.
        import radiarch.services.dose_engines.mcsquare as mc
        monkeypatch.setattr(mc, "_opentps_available", lambda: False)

        eng = MCsquareEngine()
        with pytest.raises(EngineUnavailableError):
            eng.compute_dose(_geom(), _bm(Modality.proton_pbs),
                             np.ones(4, dtype=np.float32))

    def test_influence_unavailable_without_opentps(self, monkeypatch):
        """D8.1 — build_influence works in principle, but still needs OpenTPS.

        On a machine without OpenTPS importable, MCsquare's
        beamlet path raises EngineUnavailableError just like
        compute_dose does — same fail-fast behavior.
        """
        from radiarch.services.dose_engines import mcsquare as mc
        monkeypatch.setattr(mc, "_opentps_available", lambda: False)
        eng = MCsquareEngine()
        with pytest.raises(EngineUnavailableError):
            eng.build_influence(_geom(), _bm(Modality.proton_pbs))

    def test_apply_influence_is_csr_matvec(self):
        """D8.1 — apply_influence is the standard sparse matvec.

        Pre-D8.1 this raised EngineUnavailableError; now it does the
        same Dij @ w that the analytic engine does (the math is
        engine-agnostic — only build_influence is engine-specific).
        Verify it produces the right values from a hand-computable input.
        """
        from radiarch.services.dose_engines.protocol import InfluenceData
        from scipy.sparse import csr_matrix
        import numpy as np

        # 3 voxels × 4 fluence elements. Hand-computable so the
        # expected output is obvious from the matrix below.
        dense = np.array([
            [1.0, 0.0, 0.5, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float32)
        csr = csr_matrix(dense)
        influence = InfluenceData(
            indptr=csr.indptr.astype(np.int64),
            indices=csr.indices.astype(np.int32),
            data=csr.data.astype(np.float32),
            n_voxels=dense.shape[0],
            n_elements=dense.shape[1],
        )

        eng = MCsquareEngine()
        weights = np.array([1.0, 2.0, 4.0, 8.0], dtype=np.float32)
        result = eng.apply_influence(influence, weights, (3, 1, 1))
        # Expected: [1*1 + 0.5*4, 2*2, 1*8] = [3.0, 4.0, 8.0]
        np.testing.assert_allclose(result.dose.flatten(), [3.0, 4.0, 8.0])

    def test_grad_not_implemented(self):
        eng = MCsquareEngine()
        with pytest.raises(EngineUnavailableError):
            eng.compute_grad(_geom(), _bm(Modality.proton_pbs),
                             np.ones(4), np.zeros((4, 6, 6)))


# ---------------------------------------------------------------------------
# CCC
# ---------------------------------------------------------------------------

class TestCCCGating:
    def test_rejects_proton_modality(self):
        eng = CCCEngine()
        issues = eng.validate(_geom(), _bm(Modality.proton_pbs), {})
        assert any("PHOTON_IMRT" in i for i in issues)

    def test_compute_dose_always_unavailable_in_v1(self):
        eng = CCCEngine()
        with pytest.raises(EngineUnavailableError):
            eng.compute_dose(_geom(), _bm(Modality.photon_imrt),
                             np.ones(4, dtype=np.float32))

    def test_influence_not_implemented(self):
        eng = CCCEngine()
        with pytest.raises(EngineUnavailableError):
            eng.build_influence(_geom(), _bm(Modality.photon_imrt))

    def test_grad_not_implemented(self):
        eng = CCCEngine()
        with pytest.raises(EngineUnavailableError):
            eng.compute_grad(_geom(), _bm(Modality.photon_imrt),
                             np.ones(4), np.zeros((4, 6, 6)))
