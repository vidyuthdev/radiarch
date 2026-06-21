"""``OptimizationService`` — the public entry point for Service 4.

Inverse planning: given a geometry, a beam model, a dose engine, and a set of
clinical objectives + constraints, solve for the fluence-element weights ``w*``
that minimize a composite cost ``L(w) = Σ_i obj_i(D(w))`` where ``D(w) = Dij·w``
is the dose produced by the engine's influence matrix.

Design (mirrors :class:`radiarch.services.dose.DoseService`):

* Content-addressable: identical requests → same ``cache_key`` → the stored
  result is returned with no solver call.
* Engine-agnostic by construction. The *only* engine interaction during the
  solver loop is the forward matvec ``apply_influence`` (D·w). The gradient
  ``dL/dw = Dijᵀ · Σ_i ∂obj_i/∂D`` (O10) is assembled once in the service from
  the same sparse ``Dij`` — so it works for any engine that can produce a Dij.
* Robust optimization (O12/O13): the cost and gradient are evaluated over a set
  of perturbed scenarios (each with its own Dij) and aggregated WORST_CASE /
  EXPECTED / CVaR. The aggregation is applied identically to the cost (a scalar)
  and the gradient (a vector) so the search direction stays consistent.

Geometry/beam-model loading and the cached ``Dij`` build are delegated to
:class:`DoseService` so the two services agree on bundle shapes and share the
influence cache.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger
from scipy.sparse import csr_matrix

from ..config import get_settings
from ..models.dose import InfluenceBuildRequest, ScenarioSpec
from ..models.optimization import (
    CheckpointInfo,
    ConstraintSpec,
    ObjectiveSpec,
    OptimizationResult,
    OptimizationRunRequest,
    OptimizationStage,
    RobustStats,
)
from .dose import DoseService
from .dose_engines import get_engine
from .objectives import (
    ConstraintPenalty,
    DMax,
    DMin,
    DUniform,
    DVHMax,
    DVHMin,
    EUD,
    SmoothnessRegularizer,
    TotalVariationRegularizer,
    layer_neighbor_pairs,
)
from .optimization_persistence import OptimizationStore
from .optimization_solvers import get_solver

# Stage + fraction + message — same shape as the other services' callbacks.
OptimizationProgressCallback = Callable[[OptimizationStage, float, str], None]

# CVaR tail fraction: aggregate over the worst 10% of scenarios by default.
_CVAR_ALPHA = 0.1


@dataclass
class _Problem:
    """Everything needed to run + report one optimization, assembled once.

    Built by :meth:`OptimizationService._assemble_problem` and consumed by
    :meth:`OptimizationService.run`. Exposing it (and ``cost_and_grad``)
    separately makes the gradient (O10) and the composite objective directly
    unit-testable without driving a full solver loop.
    """

    geometry: object
    beam_model: object
    engine: object
    grid_shape: tuple
    n_elements: int
    scenarios: List[ScenarioSpec]
    scenario_dij: Dict[str, Tuple[object, csr_matrix]]
    obj_terms: list
    constraint_terms: list
    regularizers: list
    aggregation: str
    cost_and_grad: Callable[[np.ndarray], Tuple[float, np.ndarray]]
    eval_dose_terms: Callable[[np.ndarray], Tuple[float, np.ndarray]]
    w0: np.ndarray


class OptimizationService:
    """Stateless orchestrator. Reused across requests; persistence on disk."""

    def __init__(
        self,
        base_dir: Optional[str | Path] = None,
        dose_service: Optional[DoseService] = None,
    ) -> None:
        if base_dir is None:
            base_dir = Path(get_settings().artifact_dir) / "optimization"
        self.store = OptimizationStore(base_dir)
        # Reuse the dose service for geometry/beam-model loading + Dij cache.
        self.dose_service = dose_service or DoseService()

    # -----------------------------------------------------------------
    # run
    # -----------------------------------------------------------------

    def run(
        self,
        request: OptimizationRunRequest,
        progress_callback: Optional[OptimizationProgressCallback] = None,
    ) -> OptimizationResult:
        on_progress = progress_callback or _noop_progress
        t0 = time.monotonic()
        cache_key = request.compute_cache_key()

        cached = self.store.lookup_by_cache_key(cache_key)
        if cached is not None:
            logger.info("Optimization cache hit %s → %s",
                        cache_key[:10], cached.optimization_id)
            on_progress(OptimizationStage.done, 1.0, "cache hit")
            return cached

        prob = self._assemble_problem(request, on_progress)

        # --- solver loop with progress + checkpoints (O11) -----------
        opt_id = str(uuid.uuid4())
        self.store.prepare(opt_id)
        checkpoints: List[CheckpointInfo] = []
        max_iter = request.solver.max_iterations
        ckpt_interval = request.checkpoint_interval

        def _callback(it: int, cost: float, w: np.ndarray) -> None:
            frac = 0.35 + 0.5 * min(1.0, it / max(max_iter, 1))
            on_progress(OptimizationStage.optimizing, frac,
                        f"iter {it} cost={cost:.4g}")
            if ckpt_interval and it % ckpt_interval == 0:
                checkpoints.append(
                    self.store.write_checkpoint(opt_id, it, w, cost)
                )

        on_progress(OptimizationStage.optimizing, 0.35,
                    f"Solving ({request.solver.method})")
        solver = get_solver(request.solver.method)
        tol = request.solver.convergence_tol or 1e-9
        w_final, convergence = solver.run(
            prob.cost_and_grad, prob.w0, max_iter, tol, callback=_callback
        )
        w_final = np.clip(np.asarray(w_final, dtype=np.float32).ravel(), 0.0, None)

        # --- final dose on the nominal scenario ----------------------
        on_progress(OptimizationStage.computing_final_dose, 0.88,
                    "Computing final dose")
        nominal = prob.scenarios[0]
        inf_nom, _ = prob.scenario_dij[nominal.hash()]
        final_dose = prob.engine.apply_influence(
            inf_nom, w_final, prob.grid_shape
        ).dose

        convergence.constraint_violations = self._constraint_residuals(
            prob.constraint_terms, final_dose
        )

        robust_stats = None
        if request.robustness.enabled:
            robust_stats = self._robust_stats(prob, w_final)

        # --- persist -------------------------------------------------
        on_progress(OptimizationStage.persisting, 0.95, "Writing results")
        paths = self.store.prepare(opt_id)
        result = OptimizationResult(
            optimization_id=opt_id,
            cache_key=cache_key,
            weights_ref_uri=str(paths.weights),
            dose_ref_uri=str(paths.dose),
            convergence=convergence,
            robust_stats=robust_stats,
            compute_time_s=round(time.monotonic() - t0, 4),
            checkpoints=checkpoints,
            geometry_id=prob.geometry.result.geometry_id,
            beam_model_id=prob.beam_model.result.beam_model_id,
            engine_name=prob.engine.name,
            engine_version=prob.engine.version,
        )
        self.store.save(
            opt_id=opt_id, cache_key=cache_key,
            weights=w_final, dose=final_dose,
            spacing_mm=prob.geometry.spacing_mm, result=result,
        )

        on_progress(OptimizationStage.done, 1.0, f"optimization_id={opt_id}")
        logger.info(
            "Optimization %s done: solver=%s iters=%d final_cost=%.6g time=%.1fs",
            opt_id, request.solver.method, convergence.iterations,
            convergence.final_cost, result.compute_time_s,
        )
        return result

    # -----------------------------------------------------------------
    # Problem assembly (shared by run + tests)
    # -----------------------------------------------------------------

    def _assemble_problem(
        self,
        request: OptimizationRunRequest,
        on_progress: Optional[OptimizationProgressCallback] = None,
    ) -> _Problem:
        """Load bundles, build the composite objective + per-scenario Dij, and
        return a :class:`_Problem` whose ``cost_and_grad`` is ready to solve.
        """
        on_progress = on_progress or _noop_progress

        on_progress(OptimizationStage.loading, 0.05, "Loading geometry + beam model")
        geometry = self.dose_service._load_geometry(request.geometry_id)
        beam_model = self.dose_service._load_beam_model(request.beam_model_id)
        self.dose_service._check_modality(
            request.dose_engine.name, beam_model.result.modality
        )
        engine = get_engine(request.dose_engine.name)
        issues = engine.validate(geometry, beam_model, request.dose_engine.params)
        if issues:
            raise ValueError(
                f"Engine {request.dose_engine.name} rejected the request: "
                + "; ".join(issues)
            )

        grid_shape = tuple(geometry.density.shape)
        n_elements = int(beam_model.result.fluence_elements.total_count)

        on_progress(OptimizationStage.building_objective, 0.20, "Building objective")
        obj_terms = self._build_objective_terms(
            request.objectives, geometry, grid_shape
        )
        constraint_terms = self._build_constraint_terms(
            request.constraints, geometry, grid_shape
        )
        regularizers = self._build_regularizers(request, beam_model)

        scenarios = self._resolve_scenarios(request)
        on_progress(OptimizationStage.building_objective, 0.30,
                    f"Building influence ({len(scenarios)} scenario(s))")
        scenario_dij = self._build_scenario_dij(request, scenarios)
        aggregation = (request.robustness.aggregation
                       if request.robustness.enabled else "EXPECTED")

        def eval_dose_terms(dose: np.ndarray) -> Tuple[float, np.ndarray]:
            total = 0.0
            grad_dose = np.zeros(grid_shape, dtype=np.float64)
            for obj, mask in obj_terms:
                loss, g = obj(dose, mask)
                total += loss
                grad_dose += g
            for con, mask in constraint_terms:
                loss, g = con(dose, mask)
                total += loss
                grad_dose += g
            return total, grad_dose

        def cost_and_grad(w: np.ndarray) -> Tuple[float, np.ndarray]:
            w = np.asarray(w, dtype=np.float64).ravel()
            per_loss: List[float] = []
            per_grad: List[np.ndarray] = []
            for sc in scenarios:
                inf, mat = scenario_dij[sc.hash()]
                dose = engine.apply_influence(inf, w.astype(np.float32),
                                              grid_shape).dose
                loss, grad_dose = eval_dose_terms(dose)
                grad_w = mat.T @ grad_dose.ravel(order="C")  # O10: Dijᵀ·dL/dD
                per_loss.append(loss)
                per_grad.append(np.asarray(grad_w, dtype=np.float64))
            loss, grad = _aggregate(per_loss, per_grad, aggregation)
            for reg in regularizers:  # weight-space, added once
                rl, rg = reg(w)
                loss += rl
                grad = grad + rg
            return loss, grad

        w0 = self._initial_weights(request.init_weights_uri, n_elements)

        return _Problem(
            geometry=geometry, beam_model=beam_model, engine=engine,
            grid_shape=grid_shape, n_elements=n_elements,
            scenarios=scenarios, scenario_dij=scenario_dij,
            obj_terms=obj_terms, constraint_terms=constraint_terms,
            regularizers=regularizers, aggregation=aggregation,
            cost_and_grad=cost_and_grad, eval_dose_terms=eval_dose_terms,
            w0=w0,
        )

    # -----------------------------------------------------------------
    # Objective / constraint / regularizer construction
    # -----------------------------------------------------------------

    def _resolve_mask(self, geometry, structure_name: str,
                      grid_shape: tuple) -> np.ndarray:
        """Resolve a structure name to a float {0,1} mask on the dose grid."""
        index = geometry.result.structure_index
        if structure_name not in index:
            raise ValueError(
                f"structure {structure_name!r} not in geometry "
                f"(have: {sorted(index)})"
            )
        label = index[structure_name]
        mask = (np.asarray(geometry.masks) == label).astype(np.float64)
        if mask.shape != tuple(grid_shape):
            raise ValueError(
                f"mask shape {mask.shape} != dose grid {tuple(grid_shape)}"
            )
        return mask

    def _build_objective_terms(self, specs: List[ObjectiveSpec], geometry,
                               grid_shape: tuple):
        """Map each ObjectiveSpec to a ``(callable, mask)`` pair."""
        terms = []
        for s in specs:
            mask = self._resolve_mask(geometry, s.structure_name, grid_shape)
            obj = _objective_from_spec(s)
            terms.append((obj, mask))
        return terms

    def _build_constraint_terms(self, specs: List[ConstraintSpec], geometry,
                                grid_shape: tuple):
        terms = []
        for s in specs:
            mask = self._resolve_mask(geometry, s.structure_name, grid_shape)
            con = ConstraintPenalty(
                structure_name=s.structure_name, op=s.op,
                value_gy=s.value_gy, weight=s.weight, label=s.type,
            )
            terms.append((con, mask))
        return terms

    def _build_regularizers(self, request: OptimizationRunRequest, beam_model):
        reg = request.solver.regularization
        out = []
        if reg.fluence_smoothness:
            pairs = layer_neighbor_pairs(beam_model.result)
            out.append(SmoothnessRegularizer(reg.fluence_smoothness, pairs))
        if reg.total_variation:
            out.append(TotalVariationRegularizer(reg.total_variation))
        return out

    # -----------------------------------------------------------------
    # Scenarios + influence
    # -----------------------------------------------------------------

    def _resolve_scenarios(self, request: OptimizationRunRequest) -> List[ScenarioSpec]:
        """Nominal first, then the robustness scenarios (deduped on nominal)."""
        scenarios: List[ScenarioSpec] = [ScenarioSpec(name="nominal")]
        if request.robustness.enabled:
            for sc in request.robustness.scenarios:
                if not sc.is_nominal():
                    scenarios.append(sc)
        return scenarios

    def _build_scenario_dij(
        self, request: OptimizationRunRequest, scenarios: List[ScenarioSpec]
    ) -> Dict[str, Tuple[object, csr_matrix]]:
        """Build/load a cached Dij per scenario and wrap it in a CSR matrix.

        Delegates to :meth:`DoseService.build_influence` so the influence cache
        is shared with the Dose Service. The CSR matrix is reused for both the
        forward matvec (via ``engine.apply_influence``) and the transpose used
        in the gradient (O10).
        """
        out: Dict[str, Tuple[object, csr_matrix]] = {}
        for sc in scenarios:
            scenario_arg = None if sc.is_nominal() else sc
            inf_req = InfluenceBuildRequest(
                plan_id=request.plan_id,
                geometry_id=request.geometry_id,
                beam_model_id=request.beam_model_id,
                engine=request.dose_engine,
                scenario=scenario_arg,
            )
            inf_result = self.dose_service.build_influence(inf_req)
            inf = self.dose_service.influence_store.load_influence(
                inf_result.influence_id
            )
            mat = csr_matrix(
                (inf.data, inf.indices, inf.indptr),
                shape=(inf.n_voxels, inf.n_elements),
            )
            out[sc.hash()] = (inf, mat)
        return out

    # -----------------------------------------------------------------
    # Weights + residuals + robust stats
    # -----------------------------------------------------------------

    def _initial_weights(self, init_weights_uri: Optional[str],
                         n_elements: int) -> np.ndarray:
        """Warm-start from a stored vector, or uniform ones (O11)."""
        if not init_weights_uri:
            return np.ones(n_elements, dtype=np.float32)
        uri = init_weights_uri
        if uri.startswith("file://"):
            path = uri[len("file://"):]
        elif "://" in uri:
            raise ValueError(
                f"unsupported init_weights_uri scheme: {uri!r} (file:// only)"
            )
        else:
            path = uri
        arr = np.load(path).astype(np.float32).ravel()
        if arr.shape != (n_elements,):
            raise ValueError(
                f"init_weights length {arr.shape[0]} != beam-model element "
                f"count {n_elements}"
            )
        return arr

    @staticmethod
    def _constraint_residuals(constraint_terms, dose) -> Dict[str, float]:
        residuals: Dict[str, float] = {}
        for con, mask in constraint_terms:
            loss, _ = con(dose, mask)
            residuals[con.name] = float(loss)
        return residuals

    def _robust_stats(self, prob: _Problem, w_final) -> RobustStats:
        per_scenario: Dict[str, dict] = {}
        worst_cost = -np.inf
        worst_name = None
        for sc in prob.scenarios:
            inf, _ = prob.scenario_dij[sc.hash()]
            dose = prob.engine.apply_influence(inf, w_final, prob.grid_shape).dose
            cost, _ = prob.eval_dose_terms(dose)
            per_scenario[sc.name] = {
                "cost": float(cost),
                "max_gy": float(np.max(dose)) if dose.size else 0.0,
                "mean_gy": float(np.mean(dose)) if dose.size else 0.0,
            }
            if cost > worst_cost:
                worst_cost = cost
                worst_name = sc.name
        names = [s.name for s in prob.scenarios]
        return RobustStats(
            scenario_doses=per_scenario,
            worst_case_metrics={
                "worst_cost": float(worst_cost),
                "worst_scenario_index": float(
                    names.index(worst_name) if worst_name is not None else -1
                ),
            },
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _objective_from_spec(s: ObjectiveSpec):
    """Construct the objective callable for one validated :class:`ObjectiveSpec`.

    Note: ``EUD`` needs an exponent ``a`` that the v0.1 :class:`ObjectiveSpec`
    does not carry, so a serial-organ default (``a = 10``) is used. Pass a
    dedicated EUD spec field in a future revision to expose it.
    """
    t = s.type
    if t == "DMin":
        return DMin(s.structure_name, s.dose_gy, s.weight)
    if t == "DMax":
        return DMax(s.structure_name, s.dose_gy, s.weight)
    if t == "DUniform":
        return DUniform(s.structure_name, s.dose_gy, s.weight)
    if t == "DVHMin":
        return DVHMin(s.structure_name, s.dose_gy, s.volume_fraction, s.weight)
    if t == "DVHMax":
        return DVHMax(s.structure_name, s.dose_gy, s.volume_fraction, s.weight)
    if t == "EUD":
        return EUD(s.structure_name, s.dose_gy, 10.0, s.weight)
    raise ValueError(f"unknown objective type {t!r}")


def _aggregate(per_loss: List[float], per_grad: List[np.ndarray],
               aggregation: str) -> Tuple[float, np.ndarray]:
    """Aggregate per-scenario (loss, grad) into a robust (loss, grad) (O12/O13).

    The gradient aggregation matches the cost aggregation so the descent
    direction is the (sub)gradient of the aggregated cost:

    * WORST_CASE → the argmax-scenario's loss + its gradient (subgradient).
    * EXPECTED   → the mean loss + mean gradient.
    * CVaR       → mean over the worst ``ceil(alpha·S)`` scenarios (≥1).
    """
    losses = np.asarray(per_loss, dtype=np.float64)
    n = len(per_loss)
    if n == 1:
        return float(losses[0]), per_grad[0]

    if aggregation == "WORST_CASE":
        idx = int(np.argmax(losses))
        return float(losses[idx]), per_grad[idx]

    if aggregation == "CVAR":
        k = max(1, int(np.ceil(_CVAR_ALPHA * n)))
        order = np.argsort(losses)[::-1][:k]  # worst k
        loss = float(np.mean(losses[order]))
        grad = np.mean([per_grad[i] for i in order], axis=0)
        return loss, grad

    # EXPECTED (default)
    loss = float(np.mean(losses))
    grad = np.mean(per_grad, axis=0)
    return loss, grad


def _noop_progress(stage: OptimizationStage, fraction: float, message: str) -> None:
    del stage, fraction, message


__all__ = ["OptimizationService", "OptimizationProgressCallback"]
