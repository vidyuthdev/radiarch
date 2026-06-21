"""Plan-quality indices for the Evaluation Service (Service 6).

Three standard scalars summarizing how good a dose distribution is for a target:

* **Homogeneity Index (HI)** — ICRU-83: ``(D2 − D98) / D50``. Lower is a more
  uniform target dose (0 is perfectly flat).
* **Conformity Index (CI)** — Paddick: ``TV_PIV² / (TV · PIV)`` where ``TV`` is
  the target volume, ``PIV`` the prescription-isodose volume *anywhere*, and
  ``TV_PIV`` the target volume covered by the prescription isodose. 1.0 is ideal
  (the prescription isodose exactly fills the target); lower means under-coverage
  or spillage into normal tissue.
* **Coverage** — the fraction of the target receiving at least the prescription.

All pure-numpy, operating on the dose array + the target mask.
"""

from __future__ import annotations

import numpy as np

from ..models.evaluation import DoseIndices
from .dvh import dose_at_volume


def homogeneity_index(target_doses: np.ndarray) -> float:
    """ICRU-83 HI = (D2 − D98) / D50. Returns 0 for an empty/zero target."""
    if target_doses.size == 0:
        return 0.0
    d50 = dose_at_volume(target_doses, 50.0)
    if d50 <= 0:
        return 0.0
    d2 = dose_at_volume(target_doses, 2.0)
    d98 = dose_at_volume(target_doses, 98.0)
    return float((d2 - d98) / d50)


def conformity_index_paddick(dose: np.ndarray, target_mask: np.ndarray,
                             prescription_gy: float) -> float:
    """Paddick CI = TV_PIV² / (TV · PIV). Returns 0 when undefined."""
    d = np.asarray(dose, dtype=np.float64)
    tgt = np.asarray(target_mask).astype(bool)
    piv = int(np.count_nonzero(d >= prescription_gy))           # whole-grid
    tv = int(np.count_nonzero(tgt))                             # target volume
    tv_piv = int(np.count_nonzero((d >= prescription_gy) & tgt))  # covered target
    if tv == 0 or piv == 0:
        return 0.0
    return float((tv_piv * tv_piv) / (tv * piv))


def coverage_pct(target_doses: np.ndarray, prescription_gy: float) -> float:
    """Percentage of the target receiving at least the prescription dose."""
    if target_doses.size == 0:
        return 0.0
    return float(100.0 * np.mean(target_doses >= prescription_gy))


def dose_indices(dose: np.ndarray, target_mask: np.ndarray,
                 target_structure: str, prescription_gy: float) -> DoseIndices:
    """Compute all target indices in one pass."""
    d = np.asarray(dose, dtype=np.float64)
    tgt = np.asarray(target_mask).astype(bool)
    target_doses = d[tgt]
    return DoseIndices(
        target_structure=target_structure,
        prescription_gy=prescription_gy,
        homogeneity_index=homogeneity_index(target_doses),
        conformity_index=conformity_index_paddick(d, tgt, prescription_gy),
        coverage_pct=coverage_pct(target_doses, prescription_gy),
        hotspot_gy=float(d.max()) if d.size else 0.0,
    )


__all__ = [
    "homogeneity_index",
    "conformity_index_paddick",
    "coverage_pct",
    "dose_indices",
]
