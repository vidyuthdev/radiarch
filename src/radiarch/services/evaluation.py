"""``EvaluationService`` — the public entry point for Service 6 (Evaluation).

The read-only end of the pipeline: given a computed dose volume (from Service 3
or the final dose of Service 4) and the geometry's structure masks, it produces a
clinician-readable report — per-structure DVHs, target plan-quality indices
(conformity / homogeneity / coverage), and an optional gamma comparison against a
reference dose.

It does no engine or solver work. Geometry loading is delegated to
:class:`DoseService` so the two services agree on bundle shapes; the dose itself
is resolved either from the dose cache (``dose_id``) or a direct ``file://`` URI.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
from loguru import logger

from ..config import get_settings
from ..models.evaluation import (
    DVHCurve,
    EvaluationRequest,
    EvaluationResult,
    EvaluationStage,
)
from .dose import DoseService
from .dose_persistence import read_dose_volume
from .dvh import cumulative_dvh, dvh_metrics
from .evaluation_persistence import EvaluationStore
from .gamma import gamma_index
from .indices import dose_indices

EvaluationProgressCallback = Callable[[EvaluationStage, float, str], None]


class EvaluationService:
    """Stateless orchestrator. Reused across requests; persistence on disk."""

    def __init__(
        self,
        base_dir: Optional[str | Path] = None,
        dose_service: Optional[DoseService] = None,
    ) -> None:
        if base_dir is None:
            base_dir = Path(get_settings().artifact_dir) / "evaluation"
        self.store = EvaluationStore(base_dir)
        self.dose_service = dose_service or DoseService()

    def run(
        self,
        request: EvaluationRequest,
        progress_callback: Optional[EvaluationProgressCallback] = None,
    ) -> EvaluationResult:
        on_progress = progress_callback or _noop_progress
        t0 = time.monotonic()
        cache_key = request.compute_cache_key()

        cached = self.store.lookup_by_cache_key(cache_key)
        if cached is not None:
            logger.info("Evaluation cache hit %s → %s", cache_key[:10],
                        cached.evaluation_id)
            on_progress(EvaluationStage.done, 1.0, "cache hit")
            return cached

        on_progress(EvaluationStage.loading, 0.05, "Loading dose + geometry")
        dose = self._load_dose(request)
        geometry = self.dose_service._load_geometry(request.geometry_id)
        masks = np.asarray(geometry.masks)
        if dose.shape != masks.shape:
            raise ValueError(
                f"dose shape {dose.shape} != geometry mask shape {masks.shape}"
            )
        index = geometry.result.structure_index
        voxel_volume_cc = float(np.prod(geometry.spacing_mm)) / 1000.0

        # --- DVH per structure ---------------------------------------
        on_progress(EvaluationStage.computing_dvh, 0.25, "Computing DVHs")
        names = request.structures or list(index.keys())
        curves: List[DVHCurve] = []
        for name in names:
            if name not in index:
                raise ValueError(
                    f"structure {name!r} not in geometry (have: {sorted(index)})"
                )
            mask = masks == index[name]
            dose_bins, vol = cumulative_dvh(dose, mask, request.dvh_bins)
            metrics = dvh_metrics(dose, mask, request.prescription_gy, voxel_volume_cc)
            curves.append(DVHCurve(
                structure_name=name,
                dose_bins_gy=[float(x) for x in dose_bins],
                volume_pct=[float(x) for x in vol],
                metrics=metrics,
            ))

        # --- indices (target) ----------------------------------------
        indices = None
        if request.target_structure:
            on_progress(EvaluationStage.computing_indices, 0.55, "Computing indices")
            if request.target_structure not in index:
                raise ValueError(
                    f"target_structure {request.target_structure!r} not in geometry"
                )
            tgt_mask = masks == index[request.target_structure]
            indices = dose_indices(dose, tgt_mask, request.target_structure,
                                   request.prescription_gy)

        # --- gamma (optional) ----------------------------------------
        gamma = None
        if request.gamma is not None:
            on_progress(EvaluationStage.computing_gamma, 0.70, "Computing gamma")
            reference = self._load_reference_dose(request.gamma)
            # spacing_mm is (sx, sy, sz); dose arrays are (nz, ny, nx).
            sx, sy, sz = geometry.spacing_mm
            gamma = gamma_index(
                dose, reference, spacing_mm=(sz, sy, sx),
                dose_percent=request.gamma.dose_percent,
                distance_mm=request.gamma.distance_mm,
                threshold_pct=request.gamma.threshold_pct,
                local=request.gamma.local,
            )

        # --- persist -------------------------------------------------
        on_progress(EvaluationStage.persisting, 0.92, "Writing report")
        evaluation_id = str(uuid.uuid4())
        result = EvaluationResult(
            evaluation_id=evaluation_id,
            cache_key=cache_key,
            dvh_curves=curves,
            indices=indices,
            gamma=gamma,
            dose_id=request.dose_id,
            geometry_id=request.geometry_id,
            compute_time_s=round(time.monotonic() - t0, 4),
        )
        self.store.save(evaluation_id=evaluation_id, cache_key=cache_key,
                        result=result)
        on_progress(EvaluationStage.done, 1.0, f"evaluation_id={evaluation_id}")
        logger.info("Evaluation %s done: %d DVH curves, indices=%s, gamma=%s",
                    evaluation_id, len(curves), indices is not None,
                    gamma is not None)
        return result

    # -----------------------------------------------------------------
    # Dose loading
    # -----------------------------------------------------------------

    def _load_dose(self, request: EvaluationRequest) -> np.ndarray:
        if request.dose_id:
            result = self.dose_service.dose_store.get_by_id(request.dose_id)
            if result is None:
                raise ValueError(f"dose_id {request.dose_id!r} not found")
            return read_dose_volume(Path(result.dose_grid_uri))
        return self._read_uri(request.dose_ref_uri)

    def _load_reference_dose(self, gamma_spec) -> np.ndarray:
        if gamma_spec.reference_dose_id:
            result = self.dose_service.dose_store.get_by_id(gamma_spec.reference_dose_id)
            if result is None:
                raise ValueError(
                    f"reference_dose_id {gamma_spec.reference_dose_id!r} not found"
                )
            return read_dose_volume(Path(result.dose_grid_uri))
        return self._read_uri(gamma_spec.reference_dose_uri)

    @staticmethod
    def _read_uri(uri: Optional[str]) -> np.ndarray:
        if not uri:
            raise ValueError("missing dose URI")
        path = uri[len("file://"):] if uri.startswith("file://") else uri
        if "://" in path:
            raise ValueError(f"unsupported dose URI scheme: {uri!r} (file:// only)")
        return read_dose_volume(Path(path))


def _noop_progress(stage: EvaluationStage, fraction: float, message: str) -> None:
    del stage, fraction, message


__all__ = ["EvaluationService", "EvaluationProgressCallback"]
