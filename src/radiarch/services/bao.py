"""``BAOService`` — the public entry point for Service 5 (Beam Angle Optimization).

BAO selects the beam directions to use *before* fluence optimization. It is built
directly on top of Service 4: an angle set is scored by how good a plan it
produces — build a beam model for those angles (Service 2), then run a short
fluence optimization (Service 4) and take the achieved composite cost as the
score (lower = better). A pluggable search strategy (greedy / top-k) selects the
best ``n_beams`` subset.

Design (mirrors the other service orchestrators):

* Content-addressable: identical requests → same ``cache_key`` → stored result.
* Engine-agnostic: scoring goes through OptimizationService, which only touches
  the engine via the influence matvec.
* The expensive inner loop (build beam model → optimize) is wrapped in a single
  ``_score_angle_set`` closure handed to the search strategy, so the search
  logic stays independent of how a score is produced.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Callable, List, Optional

from loguru import logger

from ..config import get_settings
from ..models.bao import (
    BAORunRequest,
    BAOResult,
    BAOStage,
    CandidateAngle,
)
from ..models.beam_model import (
    BeamModelBuildRequest,
    BeamSetSpec,
    BeamSpec,
    DeliveryParams,
    Modality,
)
from ..models.dose import EngineSpec
from ..models.optimization import OptimizationRunRequest, SolverConfig
from .bao_persistence import BAOStore
from .bao_search import get_search_strategy
from .beam_model import BeamModelService
from .optimization import OptimizationService

BAOProgressCallback = Callable[[BAOStage, float, str], None]


class BAOService:
    """Stateless orchestrator. Reused across requests; persistence on disk."""

    def __init__(
        self,
        base_dir: Optional[str | Path] = None,
        beam_model_service: Optional[BeamModelService] = None,
        optimization_service: Optional[OptimizationService] = None,
    ) -> None:
        if base_dir is None:
            base_dir = Path(get_settings().artifact_dir) / "bao"
        self.store = BAOStore(base_dir)
        self.beam_model_service = beam_model_service or BeamModelService()
        self.optimization_service = optimization_service or OptimizationService()

    def run(
        self,
        request: BAORunRequest,
        progress_callback: Optional[BAOProgressCallback] = None,
    ) -> BAOResult:
        on_progress = progress_callback or _noop_progress
        t0 = time.monotonic()
        cache_key = request.compute_cache_key()

        cached = self.store.lookup_by_cache_key(cache_key)
        if cached is not None:
            logger.info("BAO cache hit %s → %s", cache_key[:10], cached.bao_id)
            on_progress(BAOStage.done, 1.0, "cache hit")
            return cached

        on_progress(BAOStage.enumerating, 0.05, "Enumerating candidate angles")
        candidates = request.resolve_candidates()
        if request.n_beams > len(candidates):
            raise ValueError(
                f"n_beams={request.n_beams} exceeds candidate count "
                f"{len(candidates)}"
            )
        logger.info("BAO: %d candidates, selecting %d beams via %s",
                    len(candidates), request.n_beams, request.search)

        # --- scoring closure: angle set → composite plan cost --------
        scored = {"n": 0}
        total_evals = self._estimate_evals(request, len(candidates))

        def score_angle_set(angles: List[CandidateAngle]) -> float:
            cost, _ = self._score_angle_set(request, angles)
            scored["n"] += 1
            frac = 0.10 + 0.75 * min(1.0, scored["n"] / max(total_evals, 1))
            on_progress(BAOStage.scoring, frac,
                        f"scored {scored['n']}/{total_evals} angle sets")
            return cost

        on_progress(BAOStage.selecting, 0.10, f"Searching ({request.search})")
        strategy = get_search_strategy(request.search)
        selected, per_angle, history, final_score = strategy.select(
            candidates, request.n_beams, score_angle_set
        )

        # --- build the final selected beam model for downstream use --
        on_progress(BAOStage.persisting, 0.92, "Building selected beam model")
        beam_model_id = self._build_beam_model(request, selected)

        bao_id = str(uuid.uuid4())
        result = BAOResult(
            bao_id=bao_id,
            cache_key=cache_key,
            selected_angles=selected,
            per_angle_scores=per_angle,
            selection_history=history,
            final_score=final_score,
            beam_model_id=beam_model_id,
            compute_time_s=round(time.monotonic() - t0, 4),
            geometry_id=request.geometry_id,
            engine_name=request.dose_engine.name,
            engine_version=request.dose_engine.version,
            scoring=request.scoring,
            search=request.search,
        )
        self.store.save(bao_id=bao_id, cache_key=cache_key, result=result)
        on_progress(BAOStage.done, 1.0, f"bao_id={bao_id}")
        logger.info("BAO %s done: selected %s final_score=%.6g time=%.1fs",
                    bao_id, [c.key() for c in selected], final_score,
                    result.compute_time_s)
        return result

    # -----------------------------------------------------------------
    # Scoring + beam-model construction
    # -----------------------------------------------------------------

    @staticmethod
    def _estimate_evals(request: BAORunRequest, n_candidates: int) -> int:
        """Rough count of score_fn calls, for progress reporting."""
        if request.search == "top_k":
            return n_candidates + 1
        # greedy: sum over steps of remaining candidates.
        n = min(request.n_beams, n_candidates)
        return sum(n_candidates - s for s in range(n))

    def _build_beam_model(self, request: BAORunRequest,
                          angles: List[CandidateAngle]) -> str:
        """Build (and cache) a beam model for the given angle set; return its id."""
        beams = [
            BeamSpec(beam_id=f"B{i}", gantry_deg=a.gantry_deg, couch_deg=a.couch_deg)
            for i, a in enumerate(angles)
        ]
        bm_req = BeamModelBuildRequest(
            plan_id=request.plan_id,
            geometry_id=request.geometry_id,
            modality=Modality(request.modality),
            machine_model_id=request.machine_model_id,
            beam_set=BeamSetSpec(
                isocenter_mm=tuple(request.isocenter_mm), beams=beams
            ),
            delivery_params=DeliveryParams(),
        )
        return self.beam_model_service.build(bm_req).beam_model_id

    def _score_angle_set(self, request: BAORunRequest,
                         angles: List[CandidateAngle]) -> tuple:
        """Score an angle set = achieved composite cost of a short fluence opt.

        Builds a beam model for ``angles`` and runs OptimizationService with the
        request's objectives + engine and a small iteration budget. Returns
        ``(final_cost, beam_model_id)``.
        """
        beam_model_id = self._build_beam_model(request, angles)
        opt_req = OptimizationRunRequest(
            plan_id=request.plan_id,
            geometry_id=request.geometry_id,
            beam_model_id=beam_model_id,
            dose_engine=request.dose_engine,
            objectives=request.objectives,
            solver=SolverConfig(method="L-BFGS-B",
                                max_iterations=request.scoring_iterations),
        )
        result = self.optimization_service.run(opt_req)
        return float(result.convergence.final_cost), beam_model_id


def _noop_progress(stage: BAOStage, fraction: float, message: str) -> None:
    del stage, fraction, message


__all__ = ["BAOService", "BAOProgressCallback"]
