"""Analytic test/demo engine — no OpenTPS, no MCsquare.

Implements the full :class:`DoseEnginePlugin` protocol with a
deterministic toy physics model so the Dose Service can run end-to-end
in tests, in CI, and in the demo without external dependencies.

Physics is intentionally trivial:

* The "beam" deposits dose along the central axial slab with an
  exponential depth-falloff parameterized by ``mu`` (1/cm).
* Per-element contribution scales linearly with that element's weight.
* Scenarios apply a rigid shift in z (proxy for setup uncertainty) and
  scale the depth attenuation by ``range_scale`` (proxy for range error).
* Density-scale scenarios scale the dose linearly (proxy for HU
  calibration drift).

It's the wrong physics, but the right *shape*: the dose array has the
expected dimensions, scales linearly with weights (so Dij @ w == dose),
and reacts to scenarios. Real engines plug in alongside this one and
replace the registry entry for their modality.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ...models.dose import ScenarioSpec
from .protocol import (
    BeamModelBundle,
    DoseEnginePlugin,
    EngineParamError,
    GeometryBundle,
    InfluenceData,
    NominalDose,
)
from .registry import register_engine


def _resolve_param(params: Optional[dict], key: str, default):
    if not params:
        return default
    return params.get(key, default)


@dataclass
class AnalyticEngine:
    """Trivial depth-falloff engine. Implements DoseEnginePlugin."""

    name: str = "analytic"
    version: str = "0.1.0"
    modalities: List[str] = field(default_factory=lambda: ["PROTON_PBS", "PHOTON_IMRT"])

    # -----------------------------------------------------------------
    # validation
    # -----------------------------------------------------------------

    def validate(self, geometry, beam_model, params: dict) -> List[str]:
        issues: List[str] = []
        if geometry.density.ndim != 3:
            issues.append(
                f"density must be 3D, got shape {geometry.density.shape!r}"
            )
        if geometry.density.shape != geometry.masks.shape:
            issues.append(
                f"density {geometry.density.shape} and masks "
                f"{geometry.masks.shape} shapes disagree"
            )
        if beam_model.result.fluence_elements.total_count < 1:
            issues.append("beam model has zero fluence elements")
        mu = _resolve_param(params, "mu_per_cm", 0.05)
        if mu <= 0:
            issues.append(f"mu_per_cm must be > 0, got {mu}")
        return issues

    # -----------------------------------------------------------------
    # core kernel
    # -----------------------------------------------------------------

    def _kernel(
        self,
        geometry: GeometryBundle,
        scenario: Optional[ScenarioSpec],
        mu_per_cm: float,
    ) -> np.ndarray:
        """Per-unit-weight dose deposition. Same shape as geometry.density.

        The deposition is constant in x,y across the central axial slab
        and exponentially attenuated in z (depth). Scenario modifies mu
        and shifts the z baseline.
        """
        nz, ny, nx = geometry.density.shape
        sx, sy, sz = geometry.spacing_mm
        if scenario is not None and scenario.range_scale is not None:
            mu_per_cm = mu_per_cm * scenario.range_scale
        z_shift_vox = 0
        if scenario is not None and scenario.setup_shift_mm is not None:
            _, _, dz_mm = scenario.setup_shift_mm
            if sz > 0:
                z_shift_vox = int(round(dz_mm / sz))

        # depth in cm relative to the (shifted) origin slice
        depth_cm = (np.arange(nz, dtype=np.float64) - z_shift_vox) * (sz / 10.0)
        # negative depth = upstream of skin → no dose
        attenuation = np.where(depth_cm >= 0,
                               np.exp(-mu_per_cm * depth_cm),
                               0.0)
        # Density-weighted (dose ~ ρ × fluence) — small effect with our toy
        # densities, but exercises the code path.
        per_slice = attenuation.astype(np.float32)
        # Tile across x,y.
        kernel = np.broadcast_to(
            per_slice[:, None, None], (nz, ny, nx)
        ).astype(np.float32)
        # Density modulation
        out = kernel * geometry.density.astype(np.float32)
        if scenario is not None and scenario.density_scale is not None:
            out = out * float(scenario.density_scale)
        return out

    # -----------------------------------------------------------------
    # compute_dose
    # -----------------------------------------------------------------

    def compute_dose(
        self,
        geometry: GeometryBundle,
        beam_model: BeamModelBundle,
        weights: np.ndarray,
        scenario: Optional[ScenarioSpec] = None,
        params: Optional[dict] = None,
    ) -> NominalDose:
        expected = beam_model.result.fluence_elements.total_count
        if weights.shape != (expected,):
            raise EngineParamError(
                f"weights shape {weights.shape} != ({expected},)"
            )
        mu = float(_resolve_param(params, "mu_per_cm", 0.05))
        kernel = self._kernel(geometry, scenario, mu)
        total = float(weights.sum())
        dose = (kernel * total).astype(np.float32)
        return NominalDose(dose=dose)

    # -----------------------------------------------------------------
    # build_influence
    # -----------------------------------------------------------------

    def build_influence(
        self,
        geometry: GeometryBundle,
        beam_model: BeamModelBundle,
        scenario: Optional[ScenarioSpec] = None,
        params: Optional[dict] = None,
    ) -> InfluenceData:
        """Sparse Dij where each column j contributes the same kernel.

        This makes ``dose == sum_j w_j * kernel``, which matches what
        ``compute_dose`` returns (since the dose there scales linearly
        with total weight). The matrix is therefore consistent.

        Memory note
        -----------
        Each active row contributes ``n_elements`` entries, so the matrix
        has ``len(active) * n_elements * 4`` bytes (float32 data) plus
        equal bytes of int32 indices. For full clinical CTs (8M voxels)
        with 800+ spots this would blow well past 10 GB.

        We cap the active row count via ``params["max_active_voxels"]``
        (default 250_000 voxels = ~800 MB for 1000 spots) — voxels are
        ranked by kernel intensity so the highest-dose voxels are kept.
        Set the param explicitly to override for small tests.
        """
        from loguru import logger as _logger

        nz, ny, nx = geometry.density.shape
        mu = float(_resolve_param(params, "mu_per_cm", 0.05))
        threshold = float(_resolve_param(params, "sparsity_threshold", 1e-6))
        max_active = int(_resolve_param(params, "max_active_voxels", 250_000))

        kernel = self._kernel(geometry, scenario, mu).ravel(order="C")
        active = np.where(np.abs(kernel) > threshold)[0]

        n_voxels = int(nz * ny * nx)
        n_elements = int(beam_model.result.fluence_elements.total_count)

        # Cap active voxels — keep the highest-intensity ones.
        if active.size > max_active:
            _logger.info(
                "Analytic Dij: capping active voxels %d → %d "
                "(set engine params.max_active_voxels to override)",
                active.size, max_active,
            )
            top = np.argpartition(-np.abs(kernel[active]), max_active)[:max_active]
            active = np.sort(active[top])

        active_vals = kernel[active].astype(np.float32)
        row_count = int(active.size)

        # Build CSR — each row that's active gets `n_elements` entries.
        indptr = np.zeros(n_voxels + 1, dtype=np.int64)
        indptr[active + 1] = n_elements
        np.cumsum(indptr, out=indptr)
        indices = np.tile(
            np.arange(n_elements, dtype=np.int32), row_count
        )
        data = np.repeat(active_vals, n_elements).astype(np.float32)

        return InfluenceData(
            indptr=indptr,
            indices=indices,
            data=data,
            n_voxels=n_voxels,
            n_elements=n_elements,
        )

    # -----------------------------------------------------------------
    # apply_influence
    # -----------------------------------------------------------------

    def apply_influence(
        self,
        influence: InfluenceData,
        weights: np.ndarray,
        grid_shape: tuple,
    ) -> NominalDose:
        if weights.shape != (influence.n_elements,):
            raise EngineParamError(
                f"weights shape {weights.shape} != ({influence.n_elements},)"
            )
        if int(np.prod(grid_shape)) != influence.n_voxels:
            raise EngineParamError(
                f"grid_shape {grid_shape} prod != influence.n_voxels "
                f"{influence.n_voxels}"
            )
        # CSR matvec without bringing in scipy: assemble a (n_voxels,) result.
        from scipy.sparse import csr_matrix  # local — scipy is a project dep
        mat = csr_matrix(
            (influence.data, influence.indices, influence.indptr),
            shape=(influence.n_voxels, influence.n_elements),
        )
        flat = mat @ weights.astype(np.float32)
        dose = np.asarray(flat).reshape(grid_shape).astype(np.float32)
        return NominalDose(dose=dose)

    # -----------------------------------------------------------------
    # compute_grad
    # -----------------------------------------------------------------

    def compute_grad(
        self,
        geometry: GeometryBundle,
        beam_model: BeamModelBundle,
        weights: np.ndarray,
        dL_dDose: np.ndarray,
        scenario: Optional[ScenarioSpec] = None,
        params: Optional[dict] = None,
    ) -> np.ndarray:
        """VJP through the dose op: ∂L/∂w_j = sum_v kernel_v · ∂L/∂dose_v.

        With our kernel-per-element-identical scheme this is the same
        value for every j (which is correct — the gradient should be
        invariant across elements for a fluence model where each
        element contributes the same kernel). Returns a shape
        ``(n_elements,)`` vector.
        """
        if dL_dDose.shape != geometry.density.shape:
            raise EngineParamError(
                f"dL_dDose shape {dL_dDose.shape} != density shape "
                f"{geometry.density.shape}"
            )
        mu = float(_resolve_param(params, "mu_per_cm", 0.05))
        kernel = self._kernel(geometry, scenario, mu)
        contribution = float(np.sum(kernel * dL_dDose.astype(np.float32)))
        n_elements = int(beam_model.result.fluence_elements.total_count)
        return np.full((n_elements,), contribution, dtype=np.float32)


# Side-effect on import: register the engine.
register_engine(AnalyticEngine())


__all__ = ["AnalyticEngine"]
