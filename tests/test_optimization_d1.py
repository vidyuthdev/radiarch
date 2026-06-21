"""Unit tests for Optimization Service O1: Pydantic models.

Pure in-process exercising of the building blocks — no FastAPI / Celery /
Postgres. Patterns mirror ``tests/test_dose_d1.py``: model round-trip,
cache-key stability, cache-key sensitivity to each field, and validator
rejections.
"""

from __future__ import annotations

import pytest

from radiarch.models.dose import EngineSpec, ScenarioSpec
from radiarch.models.job import JobState
from radiarch.models.optimization import (
    CheckpointInfo,
    ConstraintSpec,
    ConvergenceInfo,
    ObjectiveSpec,
    OptimizationJobStatus,
    OptimizationResult,
    OptimizationRunRequest,
    OptimizationStage,
    RegularizationConfig,
    RobustnessSpec,
    RobustStats,
    SolverConfig,
)


# ---------------------------------------------------------------------------
# 1. ObjectiveSpec — shape + validators
# ---------------------------------------------------------------------------

class TestObjectiveSpec:
    def test_point_objective_minimal_valid(self):
        o = ObjectiveSpec(type="DMin", structure_name="PTV", dose_gy=60.0, weight=1.0)
        assert o.type == "DMin"
        assert o.volume_fraction is None

    def test_round_trip(self):
        o = ObjectiveSpec(type="DVHMax", structure_name="Cord", dose_gy=45.0,
                          weight=2.0, volume_fraction=0.1)
        again = ObjectiveSpec.model_validate(o.model_dump())
        assert again == o

    def test_unknown_type_rejected(self):
        with pytest.raises(Exception):
            ObjectiveSpec(type="DBogus", structure_name="PTV", dose_gy=1.0, weight=1.0)

    def test_dvhmin_requires_volume_fraction(self):
        with pytest.raises(Exception):
            ObjectiveSpec(type="DVHMin", structure_name="PTV", dose_gy=60.0, weight=1.0)

    def test_dvhmax_requires_volume_fraction(self):
        with pytest.raises(Exception):
            ObjectiveSpec(type="DVHMax", structure_name="Cord", dose_gy=45.0, weight=1.0)

    def test_dvh_with_volume_fraction_ok(self):
        o = ObjectiveSpec(type="DVHMin", structure_name="PTV", dose_gy=60.0,
                          weight=1.0, volume_fraction=0.95)
        assert o.volume_fraction == 0.95

    def test_point_type_does_not_need_volume_fraction(self):
        # DUniform / EUD / DMin / DMax must not require volume_fraction
        for t in ("DMin", "DMax", "DUniform", "EUD"):
            ObjectiveSpec(type=t, structure_name="X", dose_gy=10.0, weight=1.0)

    def test_negative_weight_rejected(self):
        with pytest.raises(Exception):
            ObjectiveSpec(type="DMin", structure_name="PTV", dose_gy=60.0, weight=0.0)

    def test_hash_stable_and_short(self):
        a = ObjectiveSpec(type="DMin", structure_name="PTV", dose_gy=60.0, weight=1.0).hash()
        b = ObjectiveSpec(type="DMin", structure_name="PTV", dose_gy=60.0, weight=1.0).hash()
        assert a == b and len(a) == 16

    def test_hash_differs_on_content(self):
        a = ObjectiveSpec(type="DMin", structure_name="PTV", dose_gy=60.0, weight=1.0).hash()
        b = ObjectiveSpec(type="DMin", structure_name="PTV", dose_gy=70.0, weight=1.0).hash()
        assert a != b


class TestConstraintSpec:
    def test_minimal_valid(self):
        c = ConstraintSpec(structure_name="Cord", type="max_dose", op="<=",
                           value_gy=45.0, weight=1.0)
        assert c.op == "<="

    def test_bad_op_rejected(self):
        with pytest.raises(Exception):
            ConstraintSpec(structure_name="Cord", type="max_dose", op="!=",
                           value_gy=45.0, weight=1.0)

    def test_round_trip(self):
        c = ConstraintSpec(structure_name="Cord", type="max_dose", op=">=",
                           value_gy=45.0, weight=3.0)
        assert ConstraintSpec.model_validate(c.model_dump()) == c


class TestRegularizationAndSolver:
    def test_regularization_defaults_none(self):
        r = RegularizationConfig()
        assert r.fluence_smoothness is None and r.total_variation is None

    def test_solver_minimal_valid(self):
        s = SolverConfig(method="L-BFGS-B", max_iterations=100)
        assert s.regularization.fluence_smoothness is None

    def test_solver_bad_method_rejected(self):
        with pytest.raises(Exception):
            SolverConfig(method="Newton", max_iterations=100)

    def test_solver_round_trip(self):
        s = SolverConfig(method="Adam", max_iterations=500, convergence_tol=1e-6,
                         regularization=RegularizationConfig(fluence_smoothness=0.1))
        assert SolverConfig.model_validate(s.model_dump()) == s


class TestRobustnessSpec:
    def test_default_disabled(self):
        r = RobustnessSpec()
        assert r.enabled is False and r.scenarios == []

    def test_enabled_requires_scenarios(self):
        with pytest.raises(Exception):
            RobustnessSpec(enabled=True, scenarios=[])

    def test_bad_aggregation_rejected(self):
        with pytest.raises(Exception):
            RobustnessSpec(aggregation="MEDIAN")

    def test_enabled_with_scenarios_ok(self):
        r = RobustnessSpec(enabled=True, aggregation="WORST_CASE",
                           scenarios=[ScenarioSpec(), ScenarioSpec(range_scale=1.05)])
        assert len(r.scenarios) == 2


# ---------------------------------------------------------------------------
# 2. OptimizationRunRequest — round-trip + cache key
# ---------------------------------------------------------------------------

class TestOptimizationRunRequest:
    def _base(self, **overrides) -> OptimizationRunRequest:
        kwargs = dict(
            geometry_id="g-1",
            beam_model_id="bm-1",
            dose_engine=EngineSpec(name="analytic"),
            objectives=[ObjectiveSpec(type="DMin", structure_name="PTV",
                                      dose_gy=60.0, weight=1.0)],
            constraints=[ConstraintSpec(structure_name="Cord", type="max_dose",
                                        op="<=", value_gy=45.0, weight=1.0)],
            solver=SolverConfig(method="L-BFGS-B", max_iterations=100),
        )
        kwargs.update(overrides)
        return OptimizationRunRequest(**kwargs)

    def test_round_trip(self):
        req = self._base()
        again = OptimizationRunRequest.model_validate(req.model_dump())
        assert again == req

    def test_requires_at_least_one_objective(self):
        with pytest.raises(Exception):
            self._base(objectives=[])

    def test_cache_key_is_64_char_hex(self):
        k = self._base().compute_cache_key()
        assert len(k) == 64
        int(k, 16)

    def test_cache_key_stable(self):
        assert self._base().compute_cache_key() == self._base().compute_cache_key()

    def test_plan_id_does_not_affect_cache_key(self):
        a = self._base(plan_id=None).compute_cache_key()
        b = self._base(plan_id="some-plan").compute_cache_key()
        assert a == b

    def test_checkpoint_interval_does_not_affect_cache_key(self):
        a = self._base(checkpoint_interval=None).compute_cache_key()
        b = self._base(checkpoint_interval=10).compute_cache_key()
        assert a == b

    def test_geometry_id_affects_cache_key(self):
        assert self._base().compute_cache_key() != self._base(geometry_id="g-2").compute_cache_key()

    def test_beam_model_id_affects_cache_key(self):
        assert self._base().compute_cache_key() != self._base(beam_model_id="bm-2").compute_cache_key()

    def test_engine_affects_cache_key(self):
        a = self._base().compute_cache_key()
        b = self._base(dose_engine=EngineSpec(name="analytic", params={"k": 1})).compute_cache_key()
        c = self._base(dose_engine=EngineSpec(name="mcsquare")).compute_cache_key()
        assert a != b and a != c

    def test_objectives_affect_cache_key(self):
        a = self._base().compute_cache_key()
        b = self._base(objectives=[ObjectiveSpec(type="DMax", structure_name="PTV",
                                                 dose_gy=60.0, weight=1.0)]).compute_cache_key()
        assert a != b

    def test_constraints_affect_cache_key(self):
        a = self._base().compute_cache_key()
        b = self._base(constraints=[]).compute_cache_key()
        assert a != b

    def test_solver_affects_cache_key(self):
        a = self._base().compute_cache_key()
        b = self._base(solver=SolverConfig(method="Adam", max_iterations=100)).compute_cache_key()
        assert a != b

    def test_solver_regularization_affects_cache_key(self):
        a = self._base().compute_cache_key()
        b = self._base(solver=SolverConfig(
            method="L-BFGS-B", max_iterations=100,
            regularization=RegularizationConfig(total_variation=0.5),
        )).compute_cache_key()
        assert a != b

    def test_init_weights_uri_affects_cache_key(self):
        a = self._base().compute_cache_key()
        b = self._base(init_weights_uri="file:///tmp/w.npy").compute_cache_key()
        assert a != b

    def test_robustness_affects_cache_key(self):
        a = self._base().compute_cache_key()
        b = self._base(robustness=RobustnessSpec(
            enabled=True, aggregation="WORST_CASE",
            scenarios=[ScenarioSpec(range_scale=1.05)],
        )).compute_cache_key()
        assert a != b


# ---------------------------------------------------------------------------
# 3. Result + status models
# ---------------------------------------------------------------------------

class TestResultModels:
    def _convergence(self) -> ConvergenceInfo:
        return ConvergenceInfo(success=True, iterations=42, final_cost=1.23,
                               cost_history=[5.0, 3.0, 1.23],
                               constraint_violations={"Cord<=45": 0.0})

    def test_convergence_round_trip(self):
        c = self._convergence()
        assert ConvergenceInfo.model_validate(c.model_dump()) == c

    def test_checkpoint_info(self):
        cp = CheckpointInfo(iteration=10, weights_uri="file:///tmp/iter_10.npy", cost=2.0)
        assert CheckpointInfo.model_validate(cp.model_dump()) == cp

    def test_result_minimal_valid(self):
        r = OptimizationResult(
            optimization_id="opt-1", cache_key="k",
            weights_ref_uri="file:///tmp/w.npy",
            dose_ref_uri="file:///tmp/dose.nii.gz",
            convergence=self._convergence(),
            compute_time_s=3.5,
            geometry_id="g-1", beam_model_id="bm-1",
            engine_name="analytic", engine_version="0.1.0",
        )
        assert r.robust_stats is None
        assert r.checkpoints == []

    def test_result_with_robust_stats_round_trip(self):
        r = OptimizationResult(
            optimization_id="opt-2", cache_key="k",
            weights_ref_uri="file:///tmp/w.npy",
            dose_ref_uri="file:///tmp/dose.nii.gz",
            convergence=self._convergence(),
            robust_stats=RobustStats(
                scenario_doses={"nominal": {"max_gy": 60.0}},
                worst_case_metrics={"ptv_d95": 58.0},
            ),
            compute_time_s=10.0,
            checkpoints=[CheckpointInfo(iteration=5, weights_uri="file:///tmp/c.npy", cost=1.0)],
            geometry_id="g-1", beam_model_id="bm-1",
            engine_name="analytic", engine_version="0.1.0",
        )
        again = OptimizationResult.model_validate(r.model_dump())
        assert again == r


class TestOptimizationJobStatus:
    def test_defaults(self):
        s = OptimizationJobStatus(id="j1", cache_key="k")
        assert s.state == JobState.queued
        assert s.stage == OptimizationStage.queued
        assert s.optimization_id is None

    def test_round_trip(self):
        s = OptimizationJobStatus(
            id="j1", cache_key="k", state=JobState.succeeded,
            progress=1.0, stage=OptimizationStage.done,
            message="ok", optimization_id="opt-1",
        )
        assert OptimizationJobStatus.model_validate(s.model_dump()) == s

    def test_all_stages_valid(self):
        for stage in OptimizationStage:
            OptimizationJobStatus(id="j", cache_key="k", stage=stage)
