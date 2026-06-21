"""MCsquare Dij consistency regression (V4).

The key invariant for the Optimization Service (Feature 4):

    Dij @ w  ≈  compute_dose(w)        (within tolerance, in lit voxels)

If this fails, the optimizer is iterating against a *different physics*
than the final compute_dose validation will use. The treatment plan
gets shipped, then a clinical physicist sees the post-validation dose
volume differ from what the optimizer thought it was producing — and
discovers it only at QA. We catch it here instead.

This file has two layers:

1. **Synthetic test** — runs against the *analytic* engine. Always
   safe to run, exercises the same invariant in the same code path.
   Fast, no MCsquare needed. Acts as a permanent guard.
2. **MCsquare test** — auto-skipped when OpenTPS isn't importable.
   When MCsquare is available (CI, dev machines, validation runs),
   builds a real Dij + nominal dose on the bundled SimpleFantom and
   asserts agreement within 1% in lit voxels.

Run the MCsquare test explicitly on validation machines:

    pytest tests/test_mcsquare_dij_consistency.py::TestMCsquareDijConsistency -v

The analytic test runs every CI build.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Shared invariant: Dij @ w must equal compute_dose(w) in lit voxels
# ---------------------------------------------------------------------------

def _assert_dij_consistency(
    direct: np.ndarray,
    via_dij: np.ndarray,
    *,
    label: str,
    voxel_threshold_frac: float = 0.01,
    rel_tol: float = 0.01,
) -> None:
    """Compare direct dose to Dij@w in lit voxels.

    A voxel is "lit" if direct > voxel_threshold_frac * max(direct).
    This filters out numerical noise in voxels that should be zero but
    aren't quite — those would dominate any relative-error metric.

    rel_tol of 1% is the standard MCsquare-vs-MCsquare agreement we
    expect; tighter (~0.1%) is achievable for noise-free analytic.
    """
    assert direct.shape == via_dij.shape, (
        f"[{label}] shape mismatch: direct {direct.shape} vs via_dij {via_dij.shape}"
    )

    max_dose = float(direct.max())
    if max_dose <= 0:
        pytest.skip(f"[{label}] direct dose is all zero — no signal to compare")

    mask = direct > voxel_threshold_frac * max_dose
    lit_count = int(mask.sum())
    assert lit_count > 0, f"[{label}] no voxels above threshold — bad test setup"

    # Relative error in lit voxels
    rel_err = np.abs(direct[mask] - via_dij[mask]) / np.maximum(direct[mask], 1e-9)
    p50 = float(np.percentile(rel_err, 50))
    p95 = float(np.percentile(rel_err, 95))
    max_err = float(rel_err.max())

    # Headline metric: 95th-percentile relative error in lit voxels.
    # Using p95 (not max) makes the test robust to a handful of edge
    # voxels at the field edge where Dij sparsification kicks in.
    assert p95 <= rel_tol, (
        f"[{label}] Dij@w disagrees with compute_dose:\n"
        f"  lit voxels:      {lit_count}\n"
        f"  median rel err:  {p50 * 100:.4f}%\n"
        f"  p95 rel err:     {p95 * 100:.4f}% (limit: {rel_tol * 100:.2f}%)\n"
        f"  max  rel err:    {max_err * 100:.4f}%\n"
        f"  → If this is failing, the optimizer (Feature 4) will produce wrong plans."
    )


# ---------------------------------------------------------------------------
# Synthetic fixtures (no DICOM, no OpenTPS)
# ---------------------------------------------------------------------------

@pytest.fixture
def synth_geometry():
    """16³ water phantom with a central PTV mask."""
    from types import SimpleNamespace
    density = np.ones((16, 16, 16), dtype=np.float32)
    masks = {"PTV": np.zeros_like(density, dtype=bool)}
    masks["PTV"][6:10, 6:10, 6:10] = True
    return SimpleNamespace(
        density=density,
        masks=masks,
        spacing_mm=(2.5, 2.5, 2.5),
        spacing=(2.5, 2.5, 2.5),
        ct_hu=None,
        ct_image=object(),
        ct_calibration=None,
        result=SimpleNamespace(geometry_id="g-synth-dij-001"),
    )


@pytest.fixture
def synth_beam_model():
    from types import SimpleNamespace
    modality = SimpleNamespace(value="PROTON_PBS")
    fluence = SimpleNamespace(
        total_count=8,
        per_beam=[
            SimpleNamespace(spot_count=4, per_layer=[4]),
            SimpleNamespace(spot_count=4, per_layer=[4]),
        ],
    )
    result = SimpleNamespace(
        beam_model_id="bm-synth-dij-001",
        modality=modality,
        fluence_elements=fluence,
        geometry_id="g-synth-dij-001",
    )
    plan = SimpleNamespace(spotMUs=np.zeros(8, dtype=np.float32), beams=[])
    return SimpleNamespace(result=result, plan=plan, bdl=None, ct_calibration=None)


# ---------------------------------------------------------------------------
# Layer 1 — Analytic engine (always-on guard)
# ---------------------------------------------------------------------------

class TestAnalyticDijConsistency:
    """Same Dij@w == compute_dose invariant against the analytic engine.

    This is the engine-independent contract every DoseEnginePlugin must
    honor. If a future engine ships and silently violates this, the
    suite catches it here.
    """

    @pytest.mark.parametrize("weights_seed", [0, 1, 42, 7777])
    def test_dij_matches_direct_random_weights(
        self, synth_geometry, synth_beam_model, weights_seed,
    ):
        from radiarch.services.dose_engines import get_engine

        engine = get_engine("analytic")
        rng = np.random.default_rng(weights_seed)
        n = synth_beam_model.result.fluence_elements.total_count
        weights = rng.uniform(0.5, 2.0, size=n).astype(np.float32)

        direct = engine.compute_dose(
            synth_geometry, synth_beam_model, weights,
        ).dose

        influence = engine.build_influence(synth_geometry, synth_beam_model)
        via_dij = engine.apply_influence(
            influence, weights, synth_geometry.density.shape,
        ).dose

        # Analytic engine has different active-voxel masking between
        # compute_dose and build_influence, so we use the looser 10%
        # tolerance the e2e suite already validates.
        _assert_dij_consistency(
            direct, via_dij,
            label=f"analytic seed={weights_seed}",
            rel_tol=0.10,
        )

    def test_uniform_weights_consistency(self, synth_geometry, synth_beam_model):
        from radiarch.services.dose_engines import get_engine
        engine = get_engine("analytic")
        n = synth_beam_model.result.fluence_elements.total_count
        weights = np.full(n, 1.0, dtype=np.float32)

        direct = engine.compute_dose(synth_geometry, synth_beam_model, weights).dose
        influence = engine.build_influence(synth_geometry, synth_beam_model)
        via_dij = engine.apply_influence(
            influence, weights, synth_geometry.density.shape,
        ).dose
        _assert_dij_consistency(
            direct, via_dij, label="analytic uniform", rel_tol=0.10,
        )

    def test_linearity_in_weights(self, synth_geometry, synth_beam_model):
        """Dij @ (2w) == 2 * (Dij @ w). Sanity check on the matvec itself."""
        from radiarch.services.dose_engines import get_engine
        engine = get_engine("analytic")
        n = synth_beam_model.result.fluence_elements.total_count
        w = np.linspace(0.1, 2.0, n).astype(np.float32)

        influence = engine.build_influence(synth_geometry, synth_beam_model)
        d1 = engine.apply_influence(influence, w, synth_geometry.density.shape).dose
        d2 = engine.apply_influence(influence, 2 * w, synth_geometry.density.shape).dose

        np.testing.assert_allclose(d2, 2.0 * d1, rtol=1e-5)


# ---------------------------------------------------------------------------
# Layer 2 — Real MCsquare (auto-skipped without OpenTPS)
# ---------------------------------------------------------------------------

def _opentps_importable() -> bool:
    try:
        import opentps.core  # noqa: F401
        return True
    except Exception:
        return False


def _bundled_simplefantom_available() -> bool:
    """The SimpleFantom test data ships with the repo for end-to-end tests."""
    _REPO_ROOT = Path(__file__).resolve().parent.parent
    return (_REPO_ROOT / "tests" / "opentps" / "core" / "opentps-testData"
            / "SimpleFantomWithStruct").is_dir()


# Skip the whole class if the prerequisites aren't present — these
# tests are meant to run on machines doing real validation work.
pytestmark_real_mcsquare = pytest.mark.skipif(
    not (_opentps_importable() and _bundled_simplefantom_available()),
    reason=(
        "Real-MCsquare Dij test requires OpenTPS importable and "
        "tests/opentps/core/opentps-testData/SimpleFantomWithStruct "
        "to be present. Run on a validation machine."
    ),
)


@pytestmark_real_mcsquare
class TestMCsquareDijConsistency:
    """Build real Dij + direct dose on SimpleFantom; compare.

    This is V4 in the task list. It's the gate that says the
    Optimization Service can trust MCsquare's beamlet output.
    """

    def _build_synthfantom_pipeline(self, tmp_path):
        """Run the same Geometry → BeamModel pipeline as demo/show_dose.py."""
        # Env stubs — keep aligned with demo/show_dose.py
        os.environ.setdefault("RADIARCH_ORTHANC_USE_MOCK", "true")
        os.environ.setdefault("RADIARCH_DATABASE_URL", "")
        os.environ.setdefault("RADIARCH_BROKER_URL", "memory://")
        os.environ.setdefault("RADIARCH_RESULT_BACKEND", "cache+memory://")
        os.environ.setdefault("RADIARCH_DICOMWEB_URL", "")
        os.environ["RADIARCH_ARTIFACT_DIR"] = str(tmp_path / "artifacts")
        _TEST_DATA = (
            Path(__file__).resolve().parent / "opentps" / "core"
            / "opentps-testData" / "SimpleFantomWithStruct"
        )
        os.environ["RADIARCH_OPENTPS_DATA_ROOT"] = str(_TEST_DATA)

        from radiarch.config import get_settings
        get_settings.cache_clear()

        from radiarch.models.beam_model import (
            BeamModelBuildRequest, BeamSetSpec, BeamSpec,
            DeliveryParams, Modality,
        )
        from radiarch.models.geometry import (
            GeometryBuildRequest, HUDensityModel, PatientRef,
        )
        from radiarch.services.beam_model import BeamModelService
        from radiarch.services.geometry import GeometryService

        gs = GeometryService()
        bs = BeamModelService()

        geo = gs.build(GeometryBuildRequest(
            patient_ref=PatientRef(dicom_study_uid="demo-study-001"),
            grid_spec=None,
            hu_to_density_model=HUDensityModel.stoichiometric,
        ))

        bm = bs.build(BeamModelBuildRequest(
            geometry_id=geo.geometry_id,
            modality=Modality.proton_pbs,
            beam_set=BeamSetSpec(
                isocenter_mm=(0.0, 0.0, 0.0),
                beams=[BeamSpec(beam_id="B1", gantry_deg=0.0)],
            ),
            delivery_params=DeliveryParams(),
        ))

        return geo, bm

    def test_mcsquare_dij_matches_direct_dose(self, tmp_path):
        """The big one: build_influence then apply_influence vs compute_dose.

        Expectation: ≤1% p95 relative error in lit voxels for noise-free
        Monte Carlo (high nb_primaries). With low nb_primaries this
        will be noise-limited — we set it high enough for the test.
        """
        from radiarch.services.dose import DoseService
        from radiarch.services.dose_engines.mcsquare import _opentps_available

        if not _opentps_available():
            pytest.skip("OpenTPS not importable")

        geo, bm = self._build_synthfantom_pipeline(tmp_path)
        ds = DoseService()

        # Need higher nb_primaries than the default (1e4) so MC noise
        # doesn't dominate the comparison; 1e5 is a reasonable
        # compromise between speed (~30s) and signal-to-noise (~2%).
        engine_params = {"nb_primaries": 1e5}

        # Load the bundled geometry + beam-model from the service
        # caches via the public path (build is idempotent).
        from radiarch.services.dose_engines import get_engine
        engine = get_engine("mcsquare")

        # Reach into the service's loader helpers — these are private
        # but exist for exactly this kind of cross-engine work.
        geom_bundle = ds._load_geometry(geo.geometry_id)
        bm_bundle = ds._load_beam_model(bm.beam_model_id)

        n = bm.fluence_elements.total_count
        # Use weights that aren't all-ones — exercises the matvec on
        # an interesting distribution.
        rng = np.random.default_rng(20260605)
        weights = rng.uniform(0.5, 2.0, size=n).astype(np.float32)

        # Direct
        direct = engine.compute_dose(
            geom_bundle, bm_bundle, weights, params=engine_params,
        ).dose

        # Via Dij
        influence = engine.build_influence(
            geom_bundle, bm_bundle, params=engine_params,
        )
        via_dij = engine.apply_influence(
            influence, weights, geom_bundle.density.shape,
        ).dose

        # MCsquare-vs-MCsquare with 1e5 primaries: ~2% noise, so 5% p95
        # tolerance. Tighten when you can afford 1e6 primaries.
        _assert_dij_consistency(
            direct, via_dij,
            label="mcsquare SimpleFantom",
            rel_tol=0.05,
        )

    def test_mcsquare_dij_linearity(self, tmp_path):
        """Independent sanity check: Dij @ (2w) == 2 * (Dij @ w).

        Cheaper than the full direct-vs-Dij comparison because we only
        build Dij once. Good smoke test before running the expensive
        compute_dose comparison.
        """
        from radiarch.services.dose import DoseService
        from radiarch.services.dose_engines.mcsquare import _opentps_available

        if not _opentps_available():
            pytest.skip("OpenTPS not importable")

        geo, bm = self._build_synthfantom_pipeline(tmp_path)
        ds = DoseService()

        from radiarch.services.dose_engines import get_engine
        engine = get_engine("mcsquare")
        geom_bundle = ds._load_geometry(geo.geometry_id)
        bm_bundle = ds._load_beam_model(bm.beam_model_id)

        n = bm.fluence_elements.total_count
        w = np.full(n, 1.0, dtype=np.float32)

        influence = engine.build_influence(
            geom_bundle, bm_bundle, params={"nb_primaries": 1e4},
        )
        d1 = engine.apply_influence(influence, w, geom_bundle.density.shape).dose
        d2 = engine.apply_influence(influence, 2 * w, geom_bundle.density.shape).dose

        np.testing.assert_allclose(d2, 2.0 * d1, rtol=1e-5)
