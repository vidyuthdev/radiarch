"""Tests for :class:`OptimizationService` (Service 4, tasks O9–O13).

Mirrors ``tests/test_dose_service.py``: the geometry + beam-model loaders are
stubbed so no SimpleITK / on-disk beam model is touched, and the analytic engine
is used end-to-end. Each test gets isolated temp dirs for the optimization +
influence stores so caches don't leak.

The analytic engine makes every fluence element contribute the *same* kernel, so
the dose depends only on ``sum(w)`` and the per-weight gradient is identical
across elements. That degeneracy is fine for these tests: cost still decreases
monotonically, the analytic gradient still matches finite differences, and the
robustness aggregation still picks the worst scenario.
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
from radiarch.models.dose import EngineSpec, ScenarioSpec
from radiarch.models.geometry import CTMetadata, GeometryResult, GridSpec
from radiarch.models.optimization import (
    ConstraintSpec,
    ObjectiveSpec,
    OptimizationRunRequest,
    RobustnessSpec,
    SolverConfig,
)
from radiarch.services.dose import DoseService
from radiarch.services.dose_engines.protocol import BeamModelBundle, GeometryBundle
from radiarch.services.optimization import OptimizationService


# ---------------------------------------------------------------------------
# Stubs — geometry with PTV (label 1) + OAR (label 2)
# ---------------------------------------------------------------------------

def _fake_geometry_bundle() -> GeometryBundle:
    nz, ny, nx = 4, 8, 8
    density = np.ones((nz, ny, nx), dtype=np.float32)
    masks = np.zeros((nz, ny, nx), dtype=np.uint16)
    masks[1:3, 2:6, 2:6] = 1   # PTV
    masks[1:3, 6:8, 2:6] = 2   # OAR (adjacent slab)
    return GeometryBundle(
        result=GeometryResult(
            geometry_id="g-1",
            density_grid_uri="/tmp/dx.nii.gz",
            structure_masks_uri="/tmp/mx.nii.gz",
            structure_index={"PTV": 1, "OAR": 2},
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


def _fake_beam_model_bundle(total: int = 6) -> BeamModelBundle:
    return BeamModelBundle(
        result=BeamModelResult(
            beam_model_id="bm-1", geometry_id="g-1", modality=Modality.proton_pbs,
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
    """An OptimizationService with stubbed loaders + isolated stores."""
    opt_tmp = tempfile.TemporaryDirectory()
    dose_tmp = tempfile.TemporaryDirectory()
    infl_tmp = tempfile.TemporaryDirectory()
    dose_service = DoseService(dose_dir=dose_tmp.name, influence_dir=infl_tmp.name)
    monkeypatch.setattr(dose_service, "_load_geometry",
                        lambda gid: _fake_geometry_bundle())
    monkeypatch.setattr(dose_service, "_load_beam_model",
                        lambda bid: _fake_beam_model_bundle())
    s = OptimizationService(base_dir=opt_tmp.name, dose_service=dose_service)
    yield s
    opt_tmp.cleanup()
    dose_tmp.cleanup()
    infl_tmp.cleanup()


def _req(**over) -> OptimizationRunRequest:
    kwargs = dict(
        geometry_id="g-1",
        beam_model_id="bm-1",
        dose_engine=EngineSpec(name="analytic"),
        objectives=[ObjectiveSpec(type="DUniform", structure_name="PTV",
                                  dose_gy=10.0, weight=1.0)],
        solver=SolverConfig(method="L-BFGS-B", max_iterations=200),
    )
    kwargs.update(over)
    return OptimizationRunRequest(**kwargs)


# ---------------------------------------------------------------------------
# Convergence + persistence
# ---------------------------------------------------------------------------

class TestConvergence:
    def test_single_objective_converges(self, svc):
        result = svc.run(_req())
        ch = result.convergence.cost_history
        assert len(ch) >= 1
        assert result.convergence.final_cost <= ch[0] + 1e-9
        # Monotone (non-increasing) cost history for L-BFGS-B.
        assert all(b <= a + 1e-6 for a, b in zip(ch, ch[1:]))

    def test_persisted_artifacts_exist(self, svc):
        result = svc.run(_req())
        assert Path(result.weights_ref_uri).is_file()
        assert Path(result.dose_ref_uri).is_file()
        w = np.load(result.weights_ref_uri)
        assert w.shape == (6,)
        assert np.all(w >= 0.0)  # non-negativity enforced

    def test_cache_hit_same_id(self, svc):
        a = svc.run(_req())
        b = svc.run(_req())
        assert a.optimization_id == b.optimization_id
        assert a.cache_key == b.cache_key

    def test_composite_objective_balances(self, svc):
        # Push PTV up to 10 Gy, hold OAR under 2 Gy.
        result = svc.run(_req(
            objectives=[
                ObjectiveSpec(type="DMin", structure_name="PTV",
                              dose_gy=10.0, weight=1.0),
                ObjectiveSpec(type="DMax", structure_name="OAR",
                              dose_gy=2.0, weight=1.0),
            ],
        ))
        assert result.convergence.final_cost >= 0.0
        assert Path(result.dose_ref_uri).is_file()

    def test_constraint_residuals_reported(self, svc):
        result = svc.run(_req(
            constraints=[ConstraintSpec(structure_name="OAR", type="max_dose",
                                        op="<=", value_gy=1.0, weight=1.0)],
        ))
        assert result.convergence.constraint_violations  # non-empty dict
        assert all(v >= 0.0 for v in result.convergence.constraint_violations.values())


# ---------------------------------------------------------------------------
# Gradient correctness (O10)
# ---------------------------------------------------------------------------

class TestGradient:
    def test_analytic_grad_matches_finite_difference(self, svc):
        # Use smooth (quadratic) objectives so the central-difference check
        # isn't polluted by the kink in max(0,·)^2 — the kinked objectives have
        # their own per-objective FD checks in test_objectives.py. This test
        # specifically validates the service's Dijᵀ weight-gradient (O10).
        req = _req(objectives=[
            ObjectiveSpec(type="DUniform", structure_name="PTV", dose_gy=10.0, weight=1.0),
            ObjectiveSpec(type="DUniform", structure_name="OAR", dose_gy=2.0, weight=1.0),
        ])
        prob = svc._assemble_problem(req)
        rng = np.random.default_rng(0)
        w = rng.uniform(0.2, 1.5, size=prob.n_elements)
        _, grad = prob.cost_and_grad(w)

        eps = 1e-3
        num = np.zeros_like(grad)
        for i in range(w.size):
            wp = w.copy(); wp[i] += eps
            wm = w.copy(); wm[i] -= eps
            num[i] = (prob.cost_and_grad(wp)[0] - prob.cost_and_grad(wm)[0]) / (2 * eps)
        denom = np.linalg.norm(num) + 1e-12
        rel = np.linalg.norm(grad - num) / denom
        assert rel < 1e-3, f"gradient rel err {rel:.2e}"


# ---------------------------------------------------------------------------
# Checkpointing + warm start (O11)
# ---------------------------------------------------------------------------

class TestCheckpointAndWarmStart:
    def test_checkpoints_written_and_loadable(self, svc):
        result = svc.run(_req(
            solver=SolverConfig(method="ProjectedGradient", max_iterations=10),
            checkpoint_interval=2,
        ))
        assert result.checkpoints
        for cp in result.checkpoints:
            arr = np.load(cp.weights_uri)
            assert arr.shape == (6,)
            assert cp.iteration % 2 == 0

    def test_warm_start_from_converged_weights(self, svc):
        first = svc.run(_req())
        # Warm-start a different problem from the first run's weights.
        result = svc.run(_req(
            objectives=[ObjectiveSpec(type="DMin", structure_name="PTV",
                                      dose_gy=10.0, weight=1.0)],
            init_weights_uri="file://" + first.weights_ref_uri,
        ))
        assert Path(result.weights_ref_uri).is_file()

    def test_warm_start_wrong_length_rejected(self, svc, tmp_path):
        bad = tmp_path / "bad.npy"
        np.save(bad, np.ones(99, dtype=np.float32))
        with pytest.raises(ValueError, match="init_weights length"):
            svc.run(_req(init_weights_uri="file://" + str(bad)))


# ---------------------------------------------------------------------------
# Robust optimization (O12/O13)
# ---------------------------------------------------------------------------

class TestRobust:
    def _robust_req(self, aggregation: str) -> OptimizationRunRequest:
        return _req(
            objectives=[ObjectiveSpec(type="DUniform", structure_name="PTV",
                                      dose_gy=10.0, weight=1.0)],
            robustness=RobustnessSpec(
                enabled=True,
                aggregation=aggregation,
                scenarios=[
                    ScenarioSpec(name="range_up", range_scale=1.1),
                    ScenarioSpec(name="range_down", range_scale=0.9),
                    ScenarioSpec(name="dens_down", density_scale=0.9),
                ],
            ),
        )

    @pytest.mark.parametrize("agg", ["WORST_CASE", "EXPECTED", "CVAR"])
    def test_robust_stats_populated(self, svc, agg):
        result = svc.run(self._robust_req(agg))
        assert result.robust_stats is not None
        # nominal + 3 scenarios.
        assert len(result.robust_stats.scenario_doses) == 4
        assert "worst_cost" in result.robust_stats.worst_case_metrics

    def test_worst_case_picks_worst_scenario(self, svc):
        result = svc.run(self._robust_req("WORST_CASE"))
        costs = {k: v["cost"] for k, v in result.robust_stats.scenario_doses.items()}
        worst_name = max(costs, key=costs.get)
        names = ["nominal", "range_up", "range_down", "dens_down"]
        idx = result.robust_stats.worst_case_metrics["worst_scenario_index"]
        assert names[int(idx)] == worst_name
