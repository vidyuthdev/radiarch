"""Unit tests for Dose Service D1: models, engine protocol/registry,
persistence, and scenario expansion.

No FastAPI / Celery / Postgres in this file — pure in-process exercising
of the building blocks. Patterns mirror ``test_api_geometry.py`` and the
existing beam-model unit tests.
"""

from __future__ import annotations

import json
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
    DoseJobStatus,
    DoseResult,
    DoseStage,
    DoseStatistics,
    EngineSpec,
    InfluenceBuildRequest,
    InfluenceResult,
    ScenarioDoseEntry,
    ScenarioSetSpec,
    ScenarioSpec,
    WeightVector,
)
from radiarch.models.geometry import (
    CTMetadata,
    GeometryResult,
    GridSpec,
)
from radiarch.services.dose_engines import (
    AnalyticEngine,
    DoseEnginePlugin,
    EngineRegistryError,
    get_engine,
    list_engines,
    register_engine,
    reset_registry,
)
from radiarch.services.dose_engines.analytic import AnalyticEngine as _AnalyticCls
from radiarch.services.dose_engines.ccc import CCCEngine as _CCCCls
from radiarch.services.dose_engines.mcsquare import MCsquareEngine as _MCsquareCls
from radiarch.services.dose_engines.protocol import (
    BeamModelBundle,
    EngineParamError,
    GeometryBundle,
    InfluenceData,
)
from radiarch.services.dose_persistence import (
    DoseStore,
    InfluenceStore,
    read_dose_volume,
)
from radiarch.services.scenarios import expand_scenarios


# ---------------------------------------------------------------------------
# Helpers — small fakes shared by multiple tests
# ---------------------------------------------------------------------------

def _make_geometry_result(geometry_id: str = "g-1") -> GeometryResult:
    return GeometryResult(
        geometry_id=geometry_id,
        density_grid_uri="/tmp/density.nii.gz",
        structure_masks_uri="/tmp/masks.nii.gz",
        structure_index={"PTV": 1, "Cord": 2},
        grid_spec=GridSpec(
            spacing_mm=(2.0, 2.0, 3.0),
            origin_mm=(0.0, 0.0, 0.0),
            size=(8, 8, 4),
        ),
        frame_of_reference_uid="1.2.3.4",
        ct_metadata=CTMetadata(num_slices=4),
        cache_key="g-cache",
    )


def _make_beam_model_result(total_count: int = 5,
                            modality: Modality = Modality.proton_pbs,
                            beam_model_id: str = "bm-1") -> BeamModelResult:
    return BeamModelResult(
        beam_model_id=beam_model_id,
        geometry_id="g-1",
        modality=modality,
        fluence_elements=FluenceElementSet(
            total_count=total_count,
            per_beam=[PerBeamElements(beam_id="B1", element_count=total_count,
                                      energy_layers=[100.0], spots_per_layer=[total_count])],
        ),
        beam_model_ref_uri="/tmp/plan.pkl",
        machine_model_id="default",
        cache_key="bm-cache",
    )


def _make_geometry_bundle() -> GeometryBundle:
    """Small (4, 8, 8) bundle for engine tests — keeps tests fast."""
    nz, ny, nx = 4, 8, 8
    density = np.ones((nz, ny, nx), dtype=np.float32) * 1.0  # ~water
    masks = np.zeros((nz, ny, nx), dtype=np.uint16)
    masks[1:3, 2:6, 2:6] = 1  # PTV
    return GeometryBundle(
        result=_make_geometry_result(),
        density=density,
        masks=masks,
        spacing_mm=(2.0, 2.0, 3.0),
    )


def _make_beam_model_bundle(total_count: int = 5) -> BeamModelBundle:
    return BeamModelBundle(
        result=_make_beam_model_result(total_count=total_count),
        plan=object(),
    )


# ---------------------------------------------------------------------------
# 1. Pydantic models — shape, validation, cache keys
# ---------------------------------------------------------------------------

class TestEngineSpec:
    def test_minimal_valid(self):
        e = EngineSpec(name="mcsquare")
        assert e.name == "mcsquare"
        assert e.version == "default"
        assert e.params == {}

    def test_empty_name_rejected(self):
        with pytest.raises(Exception):
            EngineSpec(name="")


class TestScenarioSpec:
    def test_nominal_default(self):
        s = ScenarioSpec()
        assert s.is_nominal()
        assert s.name == "nominal"

    def test_with_perturbations_not_nominal(self):
        s = ScenarioSpec(setup_shift_mm=(1.0, 0.0, 0.0))
        assert not s.is_nominal()

    def test_range_scale_bounds(self):
        with pytest.raises(Exception):
            ScenarioSpec(range_scale=0.0)
        with pytest.raises(Exception):
            ScenarioSpec(range_scale=3.0)

    def test_hash_is_stable_and_short(self):
        h = ScenarioSpec(name="x", setup_shift_mm=(1, 2, 3)).hash()
        assert len(h) == 16
        # repeat → same hash
        assert h == ScenarioSpec(name="x", setup_shift_mm=(1, 2, 3)).hash()

    def test_hash_differs_when_content_differs(self):
        a = ScenarioSpec(name="a", range_scale=1.01).hash()
        b = ScenarioSpec(name="a", range_scale=1.05).hash()
        assert a != b


class TestScenarioSetSpec:
    def test_explicit_mode(self):
        s = ScenarioSetSpec(scenarios=[ScenarioSpec(), ScenarioSpec(range_scale=1.05)])
        assert s.scenarios is not None
        assert len(s.scenarios) == 2

    def test_generator_mode_needs_count(self):
        with pytest.raises(Exception):
            ScenarioSetSpec(setup_sigma_mm=3.0)  # missing count

    def test_empty_rejected(self):
        with pytest.raises(Exception):
            ScenarioSetSpec()

    def test_hash_is_stable(self):
        s = ScenarioSetSpec(setup_sigma_mm=2.0, count=5)
        assert s.hash() == ScenarioSetSpec(setup_sigma_mm=2.0, count=5).hash()


class TestWeightVector:
    def test_inline_values_round_trip(self):
        w = WeightVector(length=3, values=[0.5, 0.5, 0.5])
        assert w.values == [0.5, 0.5, 0.5]
        assert w.weights_uri is None

    def test_uri_only(self):
        w = WeightVector(length=100, weights_uri="file:///tmp/w.npy")
        assert w.weights_uri == "file:///tmp/w.npy"
        assert w.values is None

    def test_neither_rejected(self):
        with pytest.raises(Exception):
            WeightVector(length=3)

    def test_both_rejected(self):
        with pytest.raises(Exception):
            WeightVector(length=2, values=[1, 2], weights_uri="x")

    def test_length_mismatch_rejected(self):
        with pytest.raises(Exception):
            WeightVector(length=2, values=[1, 2, 3])

    def test_hash_invariant_to_print_noise(self):
        a = WeightVector(length=2, values=[1.0000000001, 2.0]).hash()
        b = WeightVector(length=2, values=[1.0, 2.0]).hash()
        # rounding to 1e-9 → identical
        assert a == b


class TestDoseComputeRequestCacheKey:
    def _base(self, **overrides):
        kwargs = dict(
            geometry_id="g-1",
            beam_model_id="bm-1",
            engine=EngineSpec(name="analytic"),
            weights=WeightVector(length=3, values=[1, 1, 1]),
        )
        kwargs.update(overrides)
        return DoseComputeRequest(**kwargs)

    def test_cache_key_is_64_char_hex(self):
        k = self._base().compute_cache_key()
        assert len(k) == 64
        int(k, 16)  # parses as hex

    def test_cache_key_stable(self):
        assert self._base().compute_cache_key() == self._base().compute_cache_key()

    def test_plan_id_does_not_affect_cache_key(self):
        a = self._base(plan_id=None).compute_cache_key()
        b = self._base(plan_id="some-plan").compute_cache_key()
        assert a == b

    def test_weights_affect_cache_key(self):
        a = self._base().compute_cache_key()
        b = self._base(
            weights=WeightVector(length=3, values=[2, 2, 2])
        ).compute_cache_key()
        assert a != b

    def test_engine_affects_cache_key(self):
        a = self._base().compute_cache_key()
        b = self._base(
            engine=EngineSpec(name="analytic", params={"mu_per_cm": 0.1})
        ).compute_cache_key()
        assert a != b

    def test_scenarios_affect_cache_key(self):
        a = self._base().compute_cache_key()
        b = self._base(
            scenarios=ScenarioSetSpec(setup_sigma_mm=3.0, count=4)
        ).compute_cache_key()
        assert a != b


class TestInfluenceBuildRequestCacheKey:
    def _base(self, **overrides):
        kwargs = dict(
            geometry_id="g-1",
            beam_model_id="bm-1",
            engine=EngineSpec(name="analytic"),
        )
        kwargs.update(overrides)
        return InfluenceBuildRequest(**kwargs)

    def test_cache_key_is_hex(self):
        int(self._base().compute_cache_key(), 16)

    def test_scenario_affects_cache_key(self):
        a = self._base().compute_cache_key()
        b = self._base(scenario=ScenarioSpec(range_scale=1.05)).compute_cache_key()
        assert a != b


class TestDoseResultValidation:
    def _stats(self):
        return DoseStatistics(max_gy=60.0, mean_gy=20.0, p95_gy=55.0, nonzero_voxel_count=100)

    def test_minimal_valid(self):
        r = DoseResult(
            dose_id="d-1", geometry_id="g-1", beam_model_id="bm-1",
            modality=Modality.proton_pbs, engine_name="analytic",
            engine_version="0.1.0", dose_grid_uri="/tmp/d.nii.gz",
            statistics=self._stats(), cache_key="x",
        )
        assert r.dose_id == "d-1"
        assert r.scenario_doses is None

    def test_scenario_doses_optional_list(self):
        r = DoseResult(
            dose_id="d-1", geometry_id="g-1", beam_model_id="bm-1",
            modality=Modality.proton_pbs, engine_name="analytic",
            engine_version="0.1.0", dose_grid_uri="/tmp/d.nii.gz",
            statistics=self._stats(), cache_key="x",
            scenario_doses=[ScenarioDoseEntry(
                scenario_name="nominal", scenario_hash="abc",
                dose_grid_uri="/tmp/s.nii.gz", statistics=self._stats(),
            )],
        )
        assert len(r.scenario_doses) == 1


class TestDoseJobStatus:
    def test_valid_kinds(self):
        DoseJobStatus(id="j1", cache_key="k", kind="dose")
        DoseJobStatus(id="j1", cache_key="k", kind="influence")

    def test_invalid_kind_rejected(self):
        with pytest.raises(Exception):
            DoseJobStatus(id="j1", cache_key="k", kind="not-a-kind")


# ---------------------------------------------------------------------------
# 2. Engine protocol + registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def setup_method(self):
        # Reset and re-register all three engines so later tests in
        # other files still see the full registry.
        reset_registry()
        register_engine(_AnalyticCls())
        register_engine(_MCsquareCls())
        register_engine(_CCCCls())

    def teardown_method(self):
        reset_registry()
        register_engine(_AnalyticCls())
        register_engine(_MCsquareCls())
        register_engine(_CCCCls())

    def test_analytic_is_registered_by_default(self):
        assert "analytic" in list_engines()

    def test_get_engine_returns_protocol(self):
        e = get_engine("analytic")
        assert isinstance(e, DoseEnginePlugin)
        assert e.name == "analytic"

    def test_unknown_engine_raises(self):
        with pytest.raises(EngineRegistryError):
            get_engine("does-not-exist")

    def test_register_overwrites(self):
        # AnalyticEngine is a dataclass, so subclass-attribute overrides
        # don't work for fields with defaults — mutate the instance.
        other = _AnalyticCls()
        other.version = "9.9.9"
        register_engine(other)
        assert get_engine("analytic").version == "9.9.9"

    def test_empty_name_rejected(self):
        bad = _AnalyticCls()
        bad.name = ""
        with pytest.raises(ValueError):
            register_engine(bad)


class TestAnalyticEngineKernel:
    def test_validate_clean(self):
        eng = AnalyticEngine()
        issues = eng.validate(_make_geometry_bundle(),
                              _make_beam_model_bundle(), {})
        assert issues == []

    def test_validate_catches_mismatched_shapes(self):
        eng = AnalyticEngine()
        g = _make_geometry_bundle()
        g.masks = np.zeros((1, 1, 1), dtype=np.uint16)
        issues = eng.validate(g, _make_beam_model_bundle(), {})
        assert any("disagree" in i for i in issues)

    def test_compute_dose_shape(self):
        eng = AnalyticEngine()
        g = _make_geometry_bundle()
        bm = _make_beam_model_bundle(total_count=5)
        w = np.ones(5, dtype=np.float32)
        result = eng.compute_dose(g, bm, w)
        assert result.dose.shape == g.density.shape
        assert result.dose.dtype == np.float32
        assert result.dose.max() > 0  # something deposited

    def test_compute_dose_linear_in_weights(self):
        eng = AnalyticEngine()
        g = _make_geometry_bundle()
        bm = _make_beam_model_bundle(total_count=5)
        w1 = np.ones(5, dtype=np.float32)
        w2 = np.ones(5, dtype=np.float32) * 2.0
        d1 = eng.compute_dose(g, bm, w1).dose
        d2 = eng.compute_dose(g, bm, w2).dose
        # 2x weight → 2x dose
        np.testing.assert_allclose(d2, 2.0 * d1, rtol=1e-5)

    def test_compute_dose_wrong_weight_length_rejected(self):
        eng = AnalyticEngine()
        with pytest.raises(EngineParamError):
            eng.compute_dose(_make_geometry_bundle(),
                             _make_beam_model_bundle(total_count=5),
                             np.ones(7, dtype=np.float32))

    def test_scenario_range_scale_changes_dose(self):
        eng = AnalyticEngine()
        g = _make_geometry_bundle()
        bm = _make_beam_model_bundle()
        w = np.ones(5, dtype=np.float32)
        nominal = eng.compute_dose(g, bm, w).dose
        perturbed = eng.compute_dose(g, bm, w, scenario=ScenarioSpec(range_scale=1.20)).dose
        assert not np.allclose(nominal, perturbed)

    def test_scenario_density_scale_scales_dose_linearly(self):
        eng = AnalyticEngine()
        g = _make_geometry_bundle()
        bm = _make_beam_model_bundle()
        w = np.ones(5, dtype=np.float32)
        nominal = eng.compute_dose(g, bm, w).dose
        scaled = eng.compute_dose(
            g, bm, w, scenario=ScenarioSpec(density_scale=1.5),
        ).dose
        np.testing.assert_allclose(scaled, 1.5 * nominal, rtol=1e-5)


class TestAnalyticEngineInfluence:
    def test_build_influence_dimensions(self):
        eng = AnalyticEngine()
        g = _make_geometry_bundle()
        bm = _make_beam_model_bundle(total_count=4)
        inf = eng.build_influence(g, bm)
        assert inf.n_voxels == int(np.prod(g.density.shape))
        assert inf.n_elements == 4
        assert inf.nnz > 0

    def test_apply_influence_matches_compute_dose(self):
        """End-to-end consistency: ``Dij @ w`` ≈ ``compute_dose(w)``.

        This is the fundamental contract every engine must satisfy.
        """
        eng = AnalyticEngine()
        g = _make_geometry_bundle()
        bm = _make_beam_model_bundle(total_count=4)
        w = np.array([0.3, 0.7, 0.2, 0.5], dtype=np.float32)

        direct = eng.compute_dose(g, bm, w).dose
        inf = eng.build_influence(g, bm)
        applied = eng.apply_influence(inf, w, g.density.shape).dose

        np.testing.assert_allclose(applied, direct, rtol=1e-4, atol=1e-6)

    def test_apply_influence_wrong_weight_length_rejected(self):
        eng = AnalyticEngine()
        inf = eng.build_influence(_make_geometry_bundle(),
                                  _make_beam_model_bundle(total_count=3))
        with pytest.raises(EngineParamError):
            eng.apply_influence(inf, np.ones(99, dtype=np.float32),
                                (4, 8, 8))

    def test_compute_grad_shape(self):
        eng = AnalyticEngine()
        g = _make_geometry_bundle()
        bm = _make_beam_model_bundle(total_count=5)
        w = np.ones(5, dtype=np.float32)
        dL = np.ones_like(g.density, dtype=np.float32)
        grad = eng.compute_grad(g, bm, w, dL)
        assert grad.shape == (5,)
        assert grad.dtype == np.float32


# ---------------------------------------------------------------------------
# 3. Persistence — DoseStore + InfluenceStore round-trips
# ---------------------------------------------------------------------------

class TestDoseStore:
    def _result(self, dose_id: str = "d-1", cache_key: str = "k1") -> DoseResult:
        return DoseResult(
            dose_id=dose_id, geometry_id="g-1", beam_model_id="bm-1",
            modality=Modality.proton_pbs,
            engine_name="analytic", engine_version="0.1.0",
            dose_grid_uri=f"/tmp/{dose_id}/dose.nii.gz",
            statistics=DoseStatistics(
                max_gy=1.0, mean_gy=0.5, p95_gy=0.9, nonzero_voxel_count=10,
            ),
            cache_key=cache_key,
        )

    def test_round_trip_nominal_dose(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = DoseStore(tmp)
            dose = np.ones((4, 4, 4), dtype=np.float32) * 0.5
            store.save(
                dose_id="d-1", cache_key="k1",
                nominal_dose=dose, spacing_mm=(2.0, 2.0, 3.0),
                scenario_doses=None, result=self._result(),
            )
            got = store.get_by_id("d-1")
            assert got is not None
            assert got.dose_id == "d-1"
            # round-trip the NIfTI
            arr = read_dose_volume(Path(tmp) / "d-1" / "dose.nii.gz")
            np.testing.assert_allclose(arr, dose)

    def test_lookup_by_cache_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = DoseStore(tmp)
            dose = np.zeros((2, 2, 2), dtype=np.float32)
            store.save(dose_id="d-2", cache_key="lookup-key",
                       nominal_dose=dose, spacing_mm=(1, 1, 1),
                       scenario_doses=None, result=self._result("d-2", "lookup-key"))
            hit = store.lookup_by_cache_key("lookup-key")
            assert hit is not None and hit.dose_id == "d-2"
            assert store.lookup_by_cache_key("nope") is None

    def test_save_with_scenario_doses(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = DoseStore(tmp)
            nominal = np.zeros((2, 2, 2), dtype=np.float32)
            scenarios = {
                "h-aaaa": np.ones((2, 2, 2), dtype=np.float32) * 0.1,
                "h-bbbb": np.ones((2, 2, 2), dtype=np.float32) * 0.2,
            }
            store.save(
                dose_id="d-s", cache_key="ks",
                nominal_dose=nominal, spacing_mm=(1, 1, 1),
                scenario_doses=scenarios, result=self._result("d-s", "ks"),
            )
            root = Path(tmp) / "d-s"
            assert (root / "scenario_h-aaaa.nii.gz").exists()
            assert (root / "scenario_h-bbbb.nii.gz").exists()

    def test_delete_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = DoseStore(tmp)
            dose = np.zeros((2, 2, 2), dtype=np.float32)
            store.save(dose_id="d-del", cache_key="kd",
                       nominal_dose=dose, spacing_mm=(1, 1, 1),
                       scenario_doses=None, result=self._result("d-del", "kd"))
            assert store.delete_by_id("d-del") is True
            assert store.get_by_id("d-del") is None
            assert store.lookup_by_cache_key("kd") is None
            assert store.delete_by_id("d-del") is False  # idempotent

    def test_list_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = DoseStore(tmp)
            dose = np.zeros((1, 1, 1), dtype=np.float32)
            for did in ["a", "b", "c"]:
                store.save(dose_id=did, cache_key=f"k-{did}",
                           nominal_dose=dose, spacing_mm=(1, 1, 1),
                           scenario_doses=None, result=self._result(did, f"k-{did}"))
            assert store.list_ids() == ["a", "b", "c"]

    def test_corrupt_index_treated_as_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = DoseStore(tmp)
            (Path(tmp) / "_index.json").write_text("not json")
            assert store.lookup_by_cache_key("anything") is None


class TestInfluenceStore:
    def _infl(self) -> InfluenceData:
        # Tiny CSR: 4-voxel, 2-element matrix.
        return InfluenceData(
            indptr=np.array([0, 2, 2, 4, 6], dtype=np.int64),
            indices=np.array([0, 1, 0, 1, 0, 1], dtype=np.int32),
            data=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32),
            n_voxels=4, n_elements=2,
        )

    def _result(self, influence_id: str, cache_key: str) -> InfluenceResult:
        return InfluenceResult(
            influence_id=influence_id, geometry_id="g-1", beam_model_id="bm-1",
            modality=Modality.proton_pbs,
            engine_name="analytic", engine_version="0.1.0",
            influence_uri=f"/tmp/{influence_id}/dij.npz",
            n_voxels=4, n_elements=2, nnz=6,
            cache_key=cache_key,
        )

    def test_save_and_load_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = InfluenceStore(tmp)
            inf = self._infl()
            store.save(
                influence_id="i-1", cache_key="ki",
                influence=inf, result=self._result("i-1", "ki"),
            )
            got = store.get_by_id("i-1")
            assert got is not None and got.influence_id == "i-1"
            roundtrip = store.load_influence("i-1")
            np.testing.assert_array_equal(roundtrip.indptr, inf.indptr)
            np.testing.assert_array_equal(roundtrip.indices, inf.indices)
            np.testing.assert_allclose(roundtrip.data, inf.data)
            assert roundtrip.n_voxels == 4
            assert roundtrip.n_elements == 2

    def test_lookup_by_cache_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = InfluenceStore(tmp)
            store.save(influence_id="i-2", cache_key="kii",
                       influence=self._infl(),
                       result=self._result("i-2", "kii"))
            assert store.lookup_by_cache_key("kii").influence_id == "i-2"
            assert store.lookup_by_cache_key("missing") is None

    def test_delete_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = InfluenceStore(tmp)
            store.save(influence_id="i-d", cache_key="kdi",
                       influence=self._infl(),
                       result=self._result("i-d", "kdi"))
            assert store.delete_by_id("i-d") is True
            assert store.delete_by_id("i-d") is False


# ---------------------------------------------------------------------------
# 4. Scenario expansion
# ---------------------------------------------------------------------------

class TestExpandScenarios:
    def test_explicit_pass_through_keeps_order(self):
        spec = ScenarioSetSpec(scenarios=[
            ScenarioSpec(name="nominal"),
            ScenarioSpec(name="a", range_scale=1.05),
        ])
        out = expand_scenarios(spec)
        assert [s.name for s in out] == ["nominal", "a"]

    def test_explicit_inserts_nominal_if_missing(self):
        spec = ScenarioSetSpec(scenarios=[
            ScenarioSpec(name="a", range_scale=1.05),
        ])
        out = expand_scenarios(spec)
        assert out[0].is_nominal()

    def test_setup_generator_adds_8_corners_plus_nominal(self):
        spec = ScenarioSetSpec(setup_sigma_mm=3.0, count=9)
        out = expand_scenarios(spec)
        # nominal + 8 corners = 9
        assert len(out) == 9
        assert out[0].is_nominal()
        names = {s.name for s in out}
        assert all(any(n.startswith("setup_corner_") for n in names)
                   for _ in range(1))

    def test_range_generator_adds_pos_and_neg(self):
        spec = ScenarioSetSpec(range_sigma=0.035, count=3)
        out = expand_scenarios(spec)
        # nominal + range_pos + range_neg = 3
        assert len(out) == 3
        names = {s.name for s in out}
        assert "range_pos" in names and "range_neg" in names

    def test_generator_deterministic(self):
        spec_a = ScenarioSetSpec(setup_sigma_mm=2.0, count=20)
        spec_b = ScenarioSetSpec(setup_sigma_mm=2.0, count=20)
        out_a = expand_scenarios(spec_a)
        out_b = expand_scenarios(spec_b)
        # same input → same scenarios (same shifts)
        for a, b in zip(out_a, out_b):
            assert a.setup_shift_mm == b.setup_shift_mm

    def test_count_smaller_than_corner_count_truncates(self):
        spec = ScenarioSetSpec(setup_sigma_mm=3.0, count=3)
        out = expand_scenarios(spec)
        assert len(out) == 3
        assert out[0].is_nominal()
