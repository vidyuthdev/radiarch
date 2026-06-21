"""Gamma analysis for the Evaluation Service (Service 6).

The gamma index (Low et al. 1998) quantifies agreement between an evaluated dose
and a reference, combining a dose-difference (DD) criterion and a
distance-to-agreement (DTA) criterion into one pass/fail per voxel:

    γ(r_e) = min over reference voxels r_r of
             sqrt( (ΔD / ΔD_crit)² + (|r_e − r_r| / DTA)² )

A voxel passes if ``γ ≤ 1``. The pass rate is the fraction of evaluated voxels
(above a low-dose threshold) that pass. ``ΔD_crit`` is ``dose_percent`` of the
reference's global max (global gamma) or of the local reference value (local
gamma).

This is a straightforward search-window implementation: for each evaluated voxel
it searches reference voxels within a box of radius ``ceil(DTA·search_factor /
spacing)``. That is O(N · window³) — fine for the bundled fantom and test grids;
for full clinical grids it should be swapped for a kd-tree / GPU kernel (noted in
the ADR). The search radius is capped so a pathological spacing can't explode the
window.
"""

from __future__ import annotations

import itertools
from typing import Tuple

import numpy as np

from ..models.evaluation import GammaResult

# Search out to this multiple of the DTA — far enough that the DTA term alone
# would already exceed γ=1, so widening further can't lower the minimum.
_SEARCH_FACTOR = 2.0
_MAX_RADIUS_VOX = 6  # hard cap per axis, keeps the window tractable


def gamma_index(
    evaluated: np.ndarray,
    reference: np.ndarray,
    spacing_mm: Tuple[float, float, float],
    dose_percent: float = 3.0,
    distance_mm: float = 3.0,
    threshold_pct: float = 10.0,
    local: bool = False,
) -> GammaResult:
    """Compute the gamma pass-rate of ``evaluated`` against ``reference``.

    ``spacing_mm`` is ``(sz, sy, sx)`` matching the array axis order. Only voxels
    whose reference dose exceeds ``threshold_pct`` of the reference max are
    evaluated (low-dose noise is excluded).
    """
    ev = np.asarray(evaluated, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    if ev.shape != ref.shape:
        raise ValueError(f"evaluated shape {ev.shape} != reference shape {ref.shape}")

    ref_max = float(ref.max()) if ref.size else 0.0
    if ref_max <= 0:
        return GammaResult(dose_percent=dose_percent, distance_mm=distance_mm,
                           threshold_pct=threshold_pct, local=local,
                           pass_rate_pct=100.0, mean_gamma=0.0, evaluated_voxels=0)

    dd_crit_global = (dose_percent / 100.0) * ref_max
    thresh = (threshold_pct / 100.0) * ref_max

    spacing = tuple(float(s) for s in spacing_mm)
    radii = [
        min(_MAX_RADIUS_VOX, max(1, int(np.ceil(_SEARCH_FACTOR * distance_mm / s))))
        for s in spacing
    ]
    offsets = list(itertools.product(
        range(-radii[0], radii[0] + 1),
        range(-radii[1], radii[1] + 1),
        range(-radii[2], radii[2] + 1),
    ))

    eval_idx = np.argwhere(ref > thresh)
    if eval_idx.size == 0:
        return GammaResult(dose_percent=dose_percent, distance_mm=distance_mm,
                           threshold_pct=threshold_pct, local=local,
                           pass_rate_pct=100.0, mean_gamma=0.0, evaluated_voxels=0)

    nz, ny, nx = ev.shape
    gammas = np.empty(eval_idx.shape[0], dtype=np.float64)

    for k, (z, y, x) in enumerate(eval_idx):
        d_eval = ev[z, y, x]
        dd_crit = ((dose_percent / 100.0) * ref[z, y, x]) if local else dd_crit_global
        dd_crit = dd_crit if dd_crit > 1e-12 else dd_crit_global or 1e-12
        best = np.inf
        for dz, dy, dx in offsets:
            zz, yy, xx = z + dz, y + dy, x + dx
            if not (0 <= zz < nz and 0 <= yy < ny and 0 <= xx < nx):
                continue
            dist2 = ((dz * spacing[0]) ** 2 + (dy * spacing[1]) ** 2
                     + (dx * spacing[2]) ** 2)
            dose_diff = d_eval - ref[zz, yy, xx]
            g2 = (dose_diff / dd_crit) ** 2 + dist2 / (distance_mm ** 2)
            if g2 < best:
                best = g2
        gammas[k] = np.sqrt(best)

    pass_rate = float(100.0 * np.mean(gammas <= 1.0))
    return GammaResult(
        dose_percent=dose_percent, distance_mm=distance_mm,
        threshold_pct=threshold_pct, local=local,
        pass_rate_pct=pass_rate, mean_gamma=float(gammas.mean()),
        evaluated_voxels=int(gammas.size),
    )


__all__ = ["gamma_index"]
