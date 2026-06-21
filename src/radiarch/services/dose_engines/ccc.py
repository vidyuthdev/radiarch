"""Collapsed-Cone-Convolution photon dose engine.

CCC is the photon counterpart to MCsquare: a depth-dose convolution of
TERMA (total energy released per unit mass) with a polyenergetic kernel
that's been "collapsed" along a small number of cone axes for speed.

Like :mod:`.mcsquare` this is a graceful-degradation engine — if the
backing CCC implementation isn't available (we don't ship one in v1),
the plugin still registers but raises :class:`EngineUnavailableError`
on every call. The Dose Service handles that cleanly (failed job with a
501-equivalent error message).

When a real CCC backend lands, this is where the dispatch lives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from loguru import logger

from ...models.dose import ScenarioSpec
from .protocol import (
    BeamModelBundle,
    DoseEnginePlugin,
    EngineParamError,
    EngineRuntimeError,
    EngineUnavailableError,
    GeometryBundle,
    InfluenceData,
    NominalDose,
)
from .registry import register_engine


def _ccc_backend_available() -> bool:
    """Always False in v1 — no CCC backend is vendored yet."""
    return False


@dataclass
class CCCEngine:
    """Photon dose engine — pluggable surface, no backend yet in v1."""

    name: str = "ccc"
    version: str = "0.1.0"
    modalities: List[str] = field(default_factory=lambda: ["PHOTON_IMRT"])

    def validate(self, geometry, beam_model, params: dict) -> List[str]:
        issues: List[str] = []
        if beam_model.result.modality.value != "PHOTON_IMRT":
            issues.append(
                f"ccc engine requires PHOTON_IMRT, got "
                f"{beam_model.result.modality.value}"
            )
        if geometry.density.ndim != 3:
            issues.append("geometry density must be 3D")
        if beam_model.result.fluence_elements.total_count < 1:
            issues.append("beam model has no fluence elements")
        if not _ccc_backend_available():
            issues.append(
                "CCC backend is not available in this build — "
                "this engine will fail at compute time."
            )
        return issues

    def compute_dose(
        self,
        geometry: GeometryBundle,
        beam_model: BeamModelBundle,
        weights: np.ndarray,
        scenario: Optional[ScenarioSpec] = None,
        params: Optional[dict] = None,
    ) -> NominalDose:
        if not _ccc_backend_available():
            raise EngineUnavailableError(
                "CCC backend is not implemented in this Radiarch build. "
                "Use the analytic engine for tests; a CCC backend will "
                "ship in a future iteration."
            )

        # Reserved for future implementation:
        expected = beam_model.result.fluence_elements.total_count
        if weights.shape != (expected,):
            raise EngineParamError(
                f"weights shape {weights.shape} != ({expected},)"
            )
        logger.warning("CCC compute_dose reached an unreachable branch")
        raise EngineRuntimeError("CCC compute_dose unreachable")  # pragma: no cover

    def build_influence(self, geometry, beam_model, scenario=None, params=None):
        raise EngineUnavailableError(
            "CCC influence is not implemented in v1."
        )

    def apply_influence(self, influence, weights, grid_shape):
        raise EngineUnavailableError(
            "CCC influence is not implemented in v1."
        )

    def compute_grad(self, geometry, beam_model, weights, dL_dDose,
                     scenario=None, params=None):
        raise EngineUnavailableError(
            "CCC gradient is not implemented in v1."
        )


register_engine(CCCEngine())


__all__ = ["CCCEngine"]
