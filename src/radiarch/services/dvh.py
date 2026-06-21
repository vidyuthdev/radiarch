"""Dose-volume histogram (DVH) computation for the Evaluation Service (Service 6).

A *cumulative* DVH answers: for each dose level d, what fraction of a structure's
volume receives **at least** d? It is the clinical workhorse — D95 (dose to 95%
of the volume), V20 (volume receiving ≥20 Gy), mean/max/min all read off it.

These are pure-numpy functions operating on a dose array + a boolean structure
mask, so they're trivially unit-testable and independent of how the dose was
produced.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ..models.evaluation import DVHMetrics


def _structure_doses(dose: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Flat array of dose values inside the structure."""
    d = np.asarray(dose, dtype=np.float64)
    m = np.asarray(mask).astype(bool)
    if d.shape != m.shape:
        raise ValueError(f"dose shape {d.shape} != mask shape {m.shape}")
    return d[m]


def cumulative_dvh(dose: np.ndarray, mask: np.ndarray,
                   bins: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """Cumulative DVH: ``(dose_bins, volume_pct)``.

    ``volume_pct[i]`` is the percentage of structure voxels receiving at least
    ``dose_bins[i]``. Bins span ``[0, max_dose]`` so the curve starts at 100%
    and decreases monotonically to 0.
    """
    doses = _structure_doses(dose, mask)
    if doses.size == 0:
        return np.zeros(bins), np.zeros(bins)
    d_max = float(doses.max())
    if d_max <= 0:
        return np.zeros(bins), np.full(bins, 100.0)
    dose_bins = np.linspace(0.0, d_max, bins)
    # For each bin edge, fraction of voxels at or above it.
    vol = np.array([100.0 * np.mean(doses >= b) for b in dose_bins])
    return dose_bins, vol


def dose_at_volume(doses: np.ndarray, pct: float) -> float:
    """``Dx``: the dose received by at least ``pct`` % of the volume.

    Equivalent to the (100−pct) percentile of the structure dose, so D95 is the
    5th percentile (95% of voxels get at least this much).
    """
    if doses.size == 0:
        return 0.0
    return float(np.percentile(doses, 100.0 - pct))


def volume_at_dose(doses: np.ndarray, dose_gy: float) -> float:
    """``Vx``: percentage of the volume receiving at least ``dose_gy``."""
    if doses.size == 0:
        return 0.0
    return float(100.0 * np.mean(doses >= dose_gy))


def dvh_metrics(dose: np.ndarray, mask: np.ndarray, prescription_gy: float,
                voxel_volume_cc: float) -> DVHMetrics:
    """Extract the standard scalar metrics from a structure's dose."""
    doses = _structure_doses(dose, mask)
    n = doses.size
    if n == 0:
        return DVHMetrics(mean_gy=0, max_gy=0, min_gy=0, d2_gy=0, d50_gy=0,
                          d95_gy=0, d98_gy=0, v_prescription_pct=0, volume_cc=0)
    return DVHMetrics(
        mean_gy=float(doses.mean()),
        max_gy=float(doses.max()),
        min_gy=float(doses.min()),
        d2_gy=dose_at_volume(doses, 2.0),
        d50_gy=dose_at_volume(doses, 50.0),
        d95_gy=dose_at_volume(doses, 95.0),
        d98_gy=dose_at_volume(doses, 98.0),
        v_prescription_pct=volume_at_dose(doses, prescription_gy),
        volume_cc=float(n * voxel_volume_cc),
    )


__all__ = [
    "cumulative_dvh",
    "dose_at_volume",
    "volume_at_dose",
    "dvh_metrics",
]
