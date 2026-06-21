"""Unit tests for the Evaluation metric functions (DVH / indices / gamma)."""

from __future__ import annotations

import numpy as np
import pytest

from radiarch.services.dvh import (
    cumulative_dvh,
    dose_at_volume,
    dvh_metrics,
    volume_at_dose,
)
from radiarch.services.gamma import gamma_index
from radiarch.services.indices import (
    conformity_index_paddick,
    coverage_pct,
    dose_indices,
    homogeneity_index,
)


# ---------------------------------------------------------------------------
# DVH
# ---------------------------------------------------------------------------

def test_dvh_uniform_target_is_flat():
    dose = np.full((4, 4, 4), 10.0)
    mask = np.ones((4, 4, 4), dtype=bool)
    bins, vol = cumulative_dvh(dose, mask, bins=50)
    assert vol[0] == pytest.approx(100.0)   # everyone gets ≥ 0
    # All voxels at 10 Gy → 100% until the very top bin.
    assert vol[:-1].min() == pytest.approx(100.0)


def test_dose_and_volume_at():
    doses = np.arange(1, 101, dtype=float)  # 1..100
    # 95% of volume gets at least D95 → D95 ≈ 5th percentile ≈ 5.95.
    assert dose_at_volume(doses, 95.0) == pytest.approx(np.percentile(doses, 5.0))
    # V50 → fraction at ≥ 50 Gy = 51/100.
    assert volume_at_dose(doses, 50.0) == pytest.approx(51.0)


def test_dvh_metrics_fields():
    dose = np.linspace(0, 60, 64).reshape(4, 4, 4)
    mask = np.ones((4, 4, 4), dtype=bool)
    m = dvh_metrics(dose, mask, prescription_gy=30.0, voxel_volume_cc=0.5)
    assert m.max_gy == pytest.approx(60.0)
    assert m.min_gy == pytest.approx(0.0)
    assert m.d95_gy <= m.d50_gy <= m.d2_gy
    assert m.volume_cc == pytest.approx(64 * 0.5)


# ---------------------------------------------------------------------------
# Indices
# ---------------------------------------------------------------------------

def test_homogeneity_zero_for_uniform():
    assert homogeneity_index(np.full(100, 50.0)) == pytest.approx(0.0)


def test_conformity_perfect_when_isodose_equals_target():
    # Target = a 2x2x2 block; dose ≥ prescription exactly on that block.
    dose = np.zeros((4, 4, 4))
    mask = np.zeros((4, 4, 4), dtype=bool)
    mask[1:3, 1:3, 1:3] = True
    dose[mask] = 60.0
    ci = conformity_index_paddick(dose, mask, prescription_gy=60.0)
    assert ci == pytest.approx(1.0)


def test_conformity_drops_with_spillage():
    dose = np.zeros((4, 4, 4))
    mask = np.zeros((4, 4, 4), dtype=bool)
    mask[1:3, 1:3, 1:3] = True
    dose[mask] = 60.0
    dose[0, 0, 0] = 60.0  # spill outside target → PIV grows, CI drops
    ci = conformity_index_paddick(dose, mask, prescription_gy=60.0)
    assert ci < 1.0


def test_coverage_pct():
    doses = np.array([50.0, 60.0, 70.0, 40.0])
    assert coverage_pct(doses, 60.0) == pytest.approx(50.0)  # 2 of 4 ≥ 60


def test_dose_indices_bundle():
    dose = np.zeros((4, 4, 4))
    mask = np.zeros((4, 4, 4), dtype=bool)
    mask[1:3, 1:3, 1:3] = True
    dose[mask] = 60.0
    idx = dose_indices(dose, mask, "PTV", 60.0)
    assert idx.target_structure == "PTV"
    assert idx.coverage_pct == pytest.approx(100.0)
    assert idx.hotspot_gy == pytest.approx(60.0)


# ---------------------------------------------------------------------------
# Gamma
# ---------------------------------------------------------------------------

def test_gamma_self_comparison_passes_fully():
    dose = np.zeros((6, 6, 6))
    dose[2:4, 2:4, 2:4] = 10.0
    g = gamma_index(dose, dose, spacing_mm=(2, 2, 2))
    assert g.pass_rate_pct == pytest.approx(100.0)
    assert g.mean_gamma == pytest.approx(0.0)
    assert g.evaluated_voxels > 0


def test_gamma_large_difference_fails():
    ref = np.zeros((6, 6, 6))
    ref[2:4, 2:4, 2:4] = 10.0
    ev = ref.copy()
    ev[2:4, 2:4, 2:4] = 20.0  # +100% dose, well beyond 3%/3mm
    g = gamma_index(ev, ref, spacing_mm=(2, 2, 2), dose_percent=3.0, distance_mm=3.0)
    assert g.pass_rate_pct < 50.0


def test_gamma_shape_mismatch_raises():
    with pytest.raises(ValueError):
        gamma_index(np.zeros((4, 4, 4)), np.zeros((4, 4, 3)), spacing_mm=(2, 2, 2))
