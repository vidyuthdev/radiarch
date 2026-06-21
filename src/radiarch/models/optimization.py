"""Pydantic I/O models for the Optimization Service (Service 4).

The Optimization Service is the fourth stage of the TPS pipeline. Given a
built geometry (Service 1), a built beam model (Service 2), and a dose
engine (Service 3), it solves the inverse-planning problem: find the
fluence-element weights ``w*`` that minimize a composite objective built
from clinical dose goals (objectives), hard limits (constraints), and
regularizers.

The cost function is ``L(w) = sum_i obj_i(D(w))`` where ``D(w) = Dij @ w``
is the dose produced by the engine's influence matrix. The optimizer
iterates entirely in weight space; the only engine interaction during
iteration is the ``Dij @ w`` matvec, which keeps the optimizer
engine-agnostic.

Robust optimization is supported by aggregating the cost (and gradient)
over a set of perturbed scenarios — see :class:`RobustnessSpec`.

The :class:`EngineSpec` and :class:`ScenarioSpec` types are reused from
``models/dose.py`` so the optimizer and dose services agree on engine
pinning and scenario semantics. The cache key mirrors
``DoseComputeRequest.compute_cache_key`` — SHA256 of a normalized JSON
representation that folds in the geometry/beam-model ids, the engine
name+version+params, and per-section content hashes. Transient fields
(``plan_id``, timestamps, ``checkpoint_interval``) are deliberately
excluded so identical optimization problems share a cache entry.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator

# Reuse the exact engine + scenario contracts from the Dose Service so the
# two services agree on engine pinning and robustness semantics.
from .dose import EngineSpec, ScenarioSpec
from .job import JobState


# ---------------------------------------------------------------------------
# Objectives — clinical dose goals
# ---------------------------------------------------------------------------

class ObjectiveSpec(BaseModel):
    """A single dose objective evaluated on one structure.

    Objectives are *soft* goals folded into the composite cost with a
    per-objective ``weight``. Supported types:

    * ``DMin`` / ``DMax`` — penalize voxels below / above ``dose_gy``.
    * ``DUniform`` — penalize deviation from ``dose_gy`` in both directions.
    * ``DVHMin`` / ``DVHMax`` — dose-volume-histogram goals: a fraction
      (``volume_fraction``) of the structure must receive at least /
      at most ``dose_gy``. Differentiable via a sigmoid surrogate.
    * ``EUD`` — generalized equivalent uniform dose (Niemierko).

    ``volume_fraction`` is required for the two DVH types and ignored
    otherwise; the validator enforces this.
    """

    type: str = Field(
        ...,
        description='One of "DMin", "DMax", "DUniform", "DVHMin", "DVHMax", "EUD".',
    )
    structure_name: str = Field(..., min_length=1, max_length=128,
                                description="Name of the target structure mask.")
    dose_gy: float = Field(..., ge=0.0,
                           description="Target / limit dose in Gy.")
    weight: float = Field(..., gt=0.0,
                          description="Relative importance in the composite cost.")
    volume_fraction: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Required for DVH* types: fraction of structure volume.",
    )

    _ALLOWED_TYPES = {"DMin", "DMax", "DUniform", "DVHMin", "DVHMax", "EUD"}
    _DVH_TYPES = {"DVHMin", "DVHMax"}

    @model_validator(mode="after")
    def _check_type_and_volume_fraction(self) -> "ObjectiveSpec":
        if self.type not in self._ALLOWED_TYPES:
            raise ValueError(
                f"type must be one of {sorted(self._ALLOWED_TYPES)}, got {self.type!r}"
            )
        if self.type in self._DVH_TYPES and self.volume_fraction is None:
            raise ValueError(
                f"volume_fraction is required for DVH objective type {self.type!r}"
            )
        return self

    def hash(self) -> str:
        """Deterministic short hash folded into the run cache key."""
        payload = {
            "type": self.type,
            "structure_name": self.structure_name,
            "dose_gy": round(self.dose_gy, 9),
            "weight": round(self.weight, 9),
            "volume_fraction": (
                round(self.volume_fraction, 9)
                if self.volume_fraction is not None
                else None
            ),
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


class ConstraintSpec(BaseModel):
    """A hard dose limit, applied as a penalty in the composite cost.

    ``op`` chooses the direction: ``">="`` penalizes dose below ``value_gy``,
    ``"<="`` penalizes dose above it. The penalty is ``weight * (excess)^2``.
    """

    structure_name: str = Field(..., min_length=1, max_length=128)
    type: str = Field(..., min_length=1, max_length=64,
                      description="Constraint kind label (e.g. 'max_dose').")
    op: str = Field(..., description='Comparison operator: ">=" or "<=".')
    value_gy: float = Field(..., ge=0.0, description="Dose limit in Gy.")
    weight: float = Field(..., gt=0.0,
                          description="Penalty weight in the composite cost.")

    _ALLOWED_OPS = {">=", "<="}

    @model_validator(mode="after")
    def _check_op(self) -> "ConstraintSpec":
        if self.op not in self._ALLOWED_OPS:
            raise ValueError(
                f'op must be one of {sorted(self._ALLOWED_OPS)}, got {self.op!r}'
            )
        return self

    def hash(self) -> str:
        payload = {
            "structure_name": self.structure_name,
            "type": self.type,
            "op": self.op,
            "value_gy": round(self.value_gy, 9),
            "weight": round(self.weight, 9),
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Regularization + solver configuration
# ---------------------------------------------------------------------------

class RegularizationConfig(BaseModel):
    """Optional regularizers added to the composite cost.

    * ``fluence_smoothness`` — penalize differences between spatially
      neighboring fluence elements (smooth spot maps).
    * ``total_variation`` — 1D total-variation penalty in fluence-element
      order; a cheaper alternative to full smoothness.

    Both default to ``None`` (disabled).
    """

    fluence_smoothness: Optional[float] = Field(
        default=None, ge=0.0,
        description="Weight of the spatial-smoothness regularizer.",
    )
    total_variation: Optional[float] = Field(
        default=None, ge=0.0,
        description="Weight of the total-variation regularizer.",
    )

    def hash(self) -> str:
        payload = {
            "fluence_smoothness": (
                round(self.fluence_smoothness, 9)
                if self.fluence_smoothness is not None
                else None
            ),
            "total_variation": (
                round(self.total_variation, 9)
                if self.total_variation is not None
                else None
            ),
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


class SolverConfig(BaseModel):
    """Which solver to run and its stopping criteria.

    ``method`` selects a registered solver plugin. ``convergence_tol``
    is solver-specific (e.g. L-BFGS-B's ``ftol``) and may be ``None`` to
    fall back to the solver's default.
    """

    method: str = Field(
        ...,
        description='One of "L-BFGS-B", "Adam", "ProjectedGradient".',
    )
    max_iterations: int = Field(..., ge=1, le=1_000_000)
    convergence_tol: Optional[float] = Field(default=None, gt=0.0)
    regularization: RegularizationConfig = Field(
        default_factory=RegularizationConfig,
    )

    _ALLOWED_METHODS = {"L-BFGS-B", "Adam", "ProjectedGradient"}

    @model_validator(mode="after")
    def _check_method(self) -> "SolverConfig":
        if self.method not in self._ALLOWED_METHODS:
            raise ValueError(
                f"method must be one of {sorted(self._ALLOWED_METHODS)}, "
                f"got {self.method!r}"
            )
        return self

    def hash(self) -> str:
        payload = {
            "method": self.method,
            "max_iterations": self.max_iterations,
            "convergence_tol": (
                round(self.convergence_tol, 12)
                if self.convergence_tol is not None
                else None
            ),
            "regularization": self.regularization.hash(),
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------

class RobustnessSpec(BaseModel):
    """Robust-optimization configuration.

    When ``enabled``, the cost is evaluated over each scenario in
    ``scenarios`` and aggregated according to ``aggregation``:

    * ``WORST_CASE`` — minimize the maximum per-scenario cost.
    * ``EXPECTED`` — minimize the mean per-scenario cost.
    * ``CVAR`` — minimize the mean over the worst tail of scenarios.

    The scenarios reuse :class:`ScenarioSpec` from the Dose Service.
    """

    enabled: bool = Field(default=False)
    scenarios: List[ScenarioSpec] = Field(default_factory=list, max_length=64)
    aggregation: str = Field(
        default="EXPECTED",
        description='One of "WORST_CASE", "EXPECTED", "CVAR".',
    )

    _ALLOWED_AGG = {"WORST_CASE", "EXPECTED", "CVAR"}

    @model_validator(mode="after")
    def _check(self) -> "RobustnessSpec":
        if self.aggregation not in self._ALLOWED_AGG:
            raise ValueError(
                f"aggregation must be one of {sorted(self._ALLOWED_AGG)}, "
                f"got {self.aggregation!r}"
            )
        if self.enabled and not self.scenarios:
            raise ValueError(
                "robustness.enabled=True requires a non-empty scenarios list"
            )
        return self

    def hash(self) -> str:
        payload = {
            "enabled": self.enabled,
            "aggregation": self.aggregation,
            "scenarios": [s.hash() for s in self.scenarios],
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

class OptimizationStage(str, Enum):
    """Stages an :meth:`OptimizationService.run` call passes through."""

    queued = "queued"
    loading = "loading"
    building_objective = "building_objective"
    optimizing = "optimizing"
    computing_final_dose = "computing_final_dose"
    persisting = "persisting"
    done = "done"


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

class OptimizationRunRequest(BaseModel):
    """Input for ``POST /api/v1/optimize/run``.

    Caller supplies the geometry + beam-model ids, the dose engine to use,
    the clinical objectives + constraints, the solver configuration, and
    an optional robustness spec. ``init_weights_uri`` optionally warm-starts
    from a previously-stored weight vector. ``checkpoint_interval`` controls
    how often the solver snapshots weights to disk.
    """

    plan_id: Optional[str] = Field(
        default=None, max_length=36,
        description="Owning plan id (downstream ref only — not in cache key).",
    )
    geometry_id: str = Field(..., min_length=1, max_length=36)
    beam_model_id: str = Field(..., min_length=1, max_length=36)
    dose_engine: EngineSpec
    objectives: List[ObjectiveSpec] = Field(..., min_length=1, max_length=256)
    constraints: List[ConstraintSpec] = Field(default_factory=list, max_length=256)
    solver: SolverConfig
    init_weights_uri: Optional[str] = Field(
        default=None,
        description="Optional warm-start weight vector URI (file:// for v0.1).",
    )
    robustness: RobustnessSpec = Field(default_factory=RobustnessSpec)
    checkpoint_interval: Optional[int] = Field(
        default=None, ge=1,
        description="Snapshot weights every N iterations (None = no checkpoints).",
    )

    def compute_cache_key(self) -> str:
        """SHA256 of a normalized JSON view of the optimization problem.

        Mirrors ``DoseComputeRequest.compute_cache_key``: identical
        problems share a cache entry. ``plan_id``, ``checkpoint_interval``
        and any timestamps are excluded as transient. Each variable-length
        section folds in via its own content hash so order and content both
        matter.
        """
        payload = {
            "geometry_id": self.geometry_id,
            "beam_model_id": self.beam_model_id,
            "engine": {
                "name": self.dose_engine.name,
                "version": self.dose_engine.version,
                "params": self.dose_engine.params,
            },
            "objectives_hash": [o.hash() for o in self.objectives],
            "constraints_hash": [c.hash() for c in self.constraints],
            "solver_config_hash": self.solver.hash(),
            "init_weights_hash": self.init_weights_uri,
            "robustness_hash": self.robustness.hash(),
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

class ConvergenceInfo(BaseModel):
    """Solver convergence summary for a completed optimization."""

    success: bool = Field(..., description="Did the solver report convergence?")
    iterations: int = Field(..., ge=0)
    final_cost: float = Field(..., description="Composite cost at the final weights.")
    cost_history: List[float] = Field(
        default_factory=list,
        description="Composite cost per iteration (for convergence plots).",
    )
    constraint_violations: Dict[str, float] = Field(
        default_factory=dict,
        description="Residual penalty per constraint at the final weights.",
    )


class RobustStats(BaseModel):
    """Per-scenario robustness summary, populated only for robust runs."""

    scenario_doses: Dict[str, Any] = Field(
        default_factory=dict,
        description="Per-scenario dose summaries keyed by scenario name/hash.",
    )
    worst_case_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Worst-case dose metrics across scenarios.",
    )


class CheckpointInfo(BaseModel):
    """One persisted weight snapshot from the solver loop."""

    iteration: int = Field(..., ge=0)
    weights_uri: str = Field(..., description="URI/path to the snapshot .npy.")
    cost: float = Field(..., description="Composite cost at this iteration.")


class OptimizationResult(BaseModel):
    """Output of a completed optimization run."""

    optimization_id: str
    cache_key: str
    weights_ref_uri: str = Field(
        ..., description="URI/path to the optimal weight vector (.npy)."
    )
    dose_ref_uri: str = Field(
        ..., description="URI/path to the final dose volume (NIfTI, Gy)."
    )
    convergence: ConvergenceInfo
    robust_stats: Optional[RobustStats] = None
    compute_time_s: float = Field(..., ge=0.0)
    checkpoints: List[CheckpointInfo] = Field(default_factory=list)
    geometry_id: str
    beam_model_id: str
    engine_name: str
    engine_version: str
    created_at: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Async job tracking
# ---------------------------------------------------------------------------

class OptimizationJobStatus(BaseModel):
    """Status row for an async ``POST /optimize/run``.

    Mirrors :class:`radiarch.models.dose.DoseJobStatus` and reuses the
    shared :class:`JobState` enum so the API contract is consistent
    across services.
    """

    id: str
    cache_key: str
    state: JobState = JobState.queued
    progress: float = 0.0
    stage: Optional[OptimizationStage] = OptimizationStage.queued
    message: Optional[str] = None
    optimization_id: Optional[str] = None  # populated when state=='succeeded'
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    created_at: Optional[datetime] = None


__all__ = [
    "ObjectiveSpec",
    "ConstraintSpec",
    "RegularizationConfig",
    "SolverConfig",
    "RobustnessSpec",
    "OptimizationStage",
    "OptimizationRunRequest",
    "ConvergenceInfo",
    "RobustStats",
    "CheckpointInfo",
    "OptimizationResult",
    "OptimizationJobStatus",
]
