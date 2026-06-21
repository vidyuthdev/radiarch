"""Unit tests for point-dose objectives (O3).

Strategy:
* Hand-computed loss + gradient on a tiny 4x4x4 grid with explicit masks.
* Finite-difference check of the analytic gradient at several voxels.
* Boundary cases where each objective should be exactly zero.
* Mask exclusion: out-of-structure voxels contribute nothing to loss/grad.
"""

from __future__ import annotations

import numpy as np
import pytest

from radiarch.services.objectives import (
    ConstraintPenalty,
    DMax,
    DMin,
    DUniform,
    DVHMax,
    DVHMin,
    EUD,
    Objective,
    SmoothnessRegularizer,
    TotalVariationRegularizer,
    layer_neighbor_pairs,
)


GRID = (4, 4, 4)


def _finite_diff_grad(obj, dose, mask, indices, eps=1e-6):
    """Central-difference gradient at the given flat indices."""
    fd = np.zeros(len(indices))
    flat = dose.ravel()
    for k, idx in enumerate(indices):
        plus = flat.copy()
        plus[idx] += eps
        minus = flat.copy()
        minus[idx] -= eps
        loss_plus, _ = obj(plus.reshape(dose.shape), mask)
        loss_minus, _ = obj(minus.reshape(dose.shape), mask)
        fd[k] = (loss_plus - loss_minus) / (2 * eps)
    return fd


def test_protocol_conformance():
    for obj in (DMin("a", 1.0), DMax("a", 1.0), DUniform("a", 1.0)):
        assert isinstance(obj, Objective)
        assert isinstance(obj.name, str) and obj.name


# ---------------------------------------------------------------------------
# Hand-computed values
# ---------------------------------------------------------------------------

def test_dmin_hand_computed():
    dose = np.zeros(GRID)
    mask = np.zeros(GRID, dtype=bool)
    # Two masked voxels: one below target (cold), one above (free).
    mask[0, 0, 0] = True
    dose[0, 0, 0] = 2.0  # target 5 -> deficit 3
    mask[1, 1, 1] = True
    dose[1, 1, 1] = 8.0  # above target -> deficit 0

    obj = DMin("ptv", dose_gy=5.0, weight=2.0)
    loss, grad = obj(dose, mask)

    # loss = w * (max(0,5-2)^2 + max(0,5-8)^2) = 2 * (9 + 0) = 18
    assert np.isclose(loss, 18.0)
    expected_grad = np.zeros(GRID)
    expected_grad[0, 0, 0] = -2 * 2.0 * 3.0  # -12
    assert np.allclose(grad, expected_grad)
    assert grad.shape == dose.shape


def test_dmax_hand_computed():
    dose = np.zeros(GRID)
    mask = np.zeros(GRID, dtype=bool)
    mask[0, 0, 0] = True
    dose[0, 0, 0] = 9.0  # target 5 -> excess 4
    mask[2, 2, 2] = True
    dose[2, 2, 2] = 1.0  # below target -> excess 0

    obj = DMax("oar", dose_gy=5.0, weight=0.5)
    loss, grad = obj(dose, mask)

    # loss = 0.5 * (max(0,9-5)^2 + 0) = 0.5 * 16 = 8
    assert np.isclose(loss, 8.0)
    expected_grad = np.zeros(GRID)
    expected_grad[0, 0, 0] = 2 * 0.5 * 4.0  # +4
    assert np.allclose(grad, expected_grad)


def test_duniform_hand_computed():
    dose = np.zeros(GRID)
    mask = np.zeros(GRID, dtype=bool)
    mask[0, 0, 0] = True
    dose[0, 0, 0] = 7.0  # deviation +2
    mask[3, 3, 3] = True
    dose[3, 3, 3] = 3.0  # deviation -2

    obj = DUniform("ptv", dose_gy=5.0, weight=1.0)
    loss, grad = obj(dose, mask)

    # loss = 1 * (2^2 + (-2)^2) = 8
    assert np.isclose(loss, 8.0)
    expected_grad = np.zeros(GRID)
    expected_grad[0, 0, 0] = 2 * 1.0 * 2.0  # +4
    expected_grad[3, 3, 3] = 2 * 1.0 * -2.0  # -4
    assert np.allclose(grad, expected_grad)


# ---------------------------------------------------------------------------
# Finite-difference gradient checks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "obj",
    [DMin("s", 5.0, 1.7), DMax("s", 5.0, 0.3), DUniform("s", 5.0, 2.1)],
)
def test_gradient_matches_finite_differences(obj):
    rng = np.random.default_rng(42)
    dose = rng.uniform(0.0, 10.0, size=GRID)
    mask = rng.integers(0, 2, size=GRID).astype(bool)

    _, grad = obj(dose, mask)
    # Pick a handful of in-mask voxels to compare.
    in_mask_flat = np.flatnonzero(mask.ravel())
    sample = in_mask_flat[:: max(1, len(in_mask_flat) // 5)][:5]
    fd = _finite_diff_grad(obj, dose, mask, sample)
    assert np.allclose(grad.ravel()[sample], fd, atol=1e-4)


# ---------------------------------------------------------------------------
# Zero-loss boundary cases
# ---------------------------------------------------------------------------

def test_dmin_zero_when_all_above_target():
    dose = np.full(GRID, 7.0)
    mask = np.ones(GRID, dtype=bool)
    loss, grad = DMin("s", dose_gy=5.0, weight=3.0)(dose, mask)
    assert loss == 0.0
    assert np.all(grad == 0.0)


def test_dmax_zero_when_all_below_target():
    dose = np.full(GRID, 2.0)
    mask = np.ones(GRID, dtype=bool)
    loss, grad = DMax("s", dose_gy=5.0, weight=3.0)(dose, mask)
    assert loss == 0.0
    assert np.all(grad == 0.0)


def test_duniform_zero_when_all_equal_target():
    dose = np.full(GRID, 5.0)
    mask = np.ones(GRID, dtype=bool)
    loss, grad = DUniform("s", dose_gy=5.0, weight=3.0)(dose, mask)
    assert loss == 0.0
    assert np.all(grad == 0.0)


def test_dmin_boundary_at_target_is_free():
    dose = np.full(GRID, 5.0)  # exactly at target
    mask = np.ones(GRID, dtype=bool)
    loss, grad = DMin("s", dose_gy=5.0)(dose, mask)
    assert loss == 0.0
    assert np.all(grad == 0.0)


# ---------------------------------------------------------------------------
# Mask exclusion
# ---------------------------------------------------------------------------

def test_mask_excludes_out_of_structure_voxels():
    dose = np.full(GRID, 100.0)  # wildly off target everywhere
    mask = np.zeros(GRID, dtype=bool)
    mask[0, 0, 0] = True  # only one voxel in structure
    dose[0, 0, 0] = 5.0  # ...and it's on target

    for obj in (DMin("s", 5.0), DMax("s", 5.0), DUniform("s", 5.0)):
        loss, grad = obj(dose, mask)
        assert loss == 0.0, obj.name
        # Out-of-structure voxels must be exactly zero in the gradient.
        assert np.all(grad == 0.0), obj.name


def test_mask_dtype_variants_equivalent():
    """Boolean, integer 0/1, and float masks must behave identically."""
    rng = np.random.default_rng(7)
    dose = rng.uniform(0, 10, size=GRID)
    base = rng.integers(0, 2, size=GRID)

    obj = DUniform("s", 5.0, 1.3)
    l_bool, g_bool = obj(dose, base.astype(bool))
    l_int, g_int = obj(dose, base.astype(int))
    l_float, g_float = obj(dose, base.astype(float))
    assert np.isclose(l_bool, l_int) and np.isclose(l_bool, l_float)
    assert np.allclose(g_bool, g_int) and np.allclose(g_bool, g_float)


def test_shape_mismatch_raises():
    dose = np.zeros(GRID)
    mask = np.zeros((4, 4, 3), dtype=bool)
    with pytest.raises(ValueError):
        DMin("s", 5.0)(dose, mask)


# ===========================================================================
# O4 — DVH + EUD objectives
# ===========================================================================

def _full_mask():
    return np.ones(GRID, dtype=bool)


def test_dvhmin_zero_when_coverage_met():
    # All voxels above target → cold fraction 0 → no penalty.
    dose = np.full(GRID, 20.0)
    loss, grad = DVHMin("PTV", dose_gy=10.0, volume_fraction=0.05)(dose, _full_mask())
    assert loss == pytest.approx(0.0, abs=1e-9)
    assert np.allclose(grad, 0.0)


def test_dvhmin_penalizes_undercoverage():
    # Half the volume is cold; allowance is 5% → penalized, grad pushes up.
    dose = np.full(GRID, 20.0)
    flat = dose.ravel()
    flat[: flat.size // 2] = 0.0
    dose = flat.reshape(GRID)
    loss, grad = DVHMin("PTV", dose_gy=10.0, volume_fraction=0.05)(dose, _full_mask())
    assert loss > 0.0
    # Gradient is negative (or zero) — raising cold-voxel dose reduces loss.
    assert grad.min() < 0.0
    assert grad.max() <= 1e-12


def test_dvhmin_finite_difference():
    rng = np.random.default_rng(1)
    dose = rng.uniform(0, 20, size=GRID)
    mask = _full_mask()
    obj = DVHMin("PTV", dose_gy=10.0, volume_fraction=0.3, k=1.0)
    _, grad = obj(dose, mask)
    idxs = [0, 5, 17, 33, 60]
    fd = _finite_diff_grad(obj, dose, mask, idxs, eps=1e-5)
    np.testing.assert_allclose(grad.ravel()[idxs], fd, rtol=1e-2, atol=1e-4)


def test_dvhmax_penalizes_overexposure():
    dose = np.full(GRID, 20.0)  # everything hot
    loss, grad = DVHMax("OAR", dose_gy=10.0, volume_fraction=0.05)(dose, _full_mask())
    assert loss > 0.0
    assert grad.max() > 0.0  # lowering hot dose reduces loss


def test_dvhmax_finite_difference():
    rng = np.random.default_rng(2)
    dose = rng.uniform(0, 20, size=GRID)
    mask = _full_mask()
    obj = DVHMax("OAR", dose_gy=10.0, volume_fraction=0.3, k=1.0)
    _, grad = obj(dose, mask)
    idxs = [1, 8, 20, 40, 63]
    fd = _finite_diff_grad(obj, dose, mask, idxs, eps=1e-5)
    np.testing.assert_allclose(grad.ravel()[idxs], fd, rtol=1e-2, atol=1e-4)


def test_eud_limits():
    dose = np.array([[[1.0, 3.0], [5.0, 7.0]]])
    mask = np.ones_like(dose)
    assert EUD("s", 4.0, a=1.0).eud(dose, mask) == pytest.approx(4.0)      # mean
    # gEUD's 1/N normalization means it approaches max/min slowly for few
    # voxels, so a large |a| is needed to get within 2%.
    assert EUD("s", 7.0, a=200.0).eud(dose, mask) == pytest.approx(7.0, rel=2e-2)   # ~max
    assert EUD("s", 1.0, a=-200.0).eud(dose, mask) == pytest.approx(1.0, rel=2e-2)  # ~min


def test_eud_finite_difference():
    rng = np.random.default_rng(3)
    dose = rng.uniform(1, 20, size=GRID)  # positive doses for stable d^a
    mask = _full_mask()
    obj = EUD("PTV", dose_gy=12.0, a=8.0, weight=1.0)
    _, grad = obj(dose, mask)
    idxs = [2, 11, 25, 44, 61]
    fd = _finite_diff_grad(obj, dose, mask, idxs, eps=1e-6)
    np.testing.assert_allclose(grad.ravel()[idxs], fd, rtol=1e-3, atol=1e-5)


def test_eud_rejects_zero_exponent():
    with pytest.raises(ValueError):
        EUD("s", 5.0, a=0.0)


# ===========================================================================
# O5 — constraints + regularizers
# ===========================================================================

def test_constraint_max_matches_dmax():
    rng = np.random.default_rng(4)
    dose = rng.uniform(0, 20, size=GRID)
    mask = _full_mask()
    con = ConstraintPenalty("OAR", op="<=", value_gy=10.0, weight=2.0)
    dmax = DMax("OAR", 10.0, weight=2.0)
    cl, cg = con(dose, mask)
    dl, dg = dmax(dose, mask)
    assert cl == pytest.approx(dl)
    np.testing.assert_allclose(cg, dg)


def test_constraint_min_matches_dmin():
    rng = np.random.default_rng(5)
    dose = rng.uniform(0, 20, size=GRID)
    mask = _full_mask()
    con = ConstraintPenalty("PTV", op=">=", value_gy=10.0, weight=1.5)
    dmin = DMin("PTV", 10.0, weight=1.5)
    np.testing.assert_allclose(con(dose, mask)[1], dmin(dose, mask)[1])


def test_smoothness_regularizer_value_and_grad():
    w = np.array([1.0, 2.0, 4.0])
    reg = SmoothnessRegularizer(1.0, [(0, 1), (1, 2)])
    loss, grad = reg(w)
    # (1-2)^2 + (2-4)^2 = 1 + 4 = 5
    assert loss == pytest.approx(5.0)
    # d/dw1 = 2(w1-w0)+2(w1-w2) ... check via finite diff
    fd = np.zeros_like(w)
    for i in range(w.size):
        wp = w.copy(); wp[i] += 1e-6
        wm = w.copy(); wm[i] -= 1e-6
        fd[i] = (reg(wp)[0] - reg(wm)[0]) / 2e-6
    np.testing.assert_allclose(grad, fd, rtol=1e-4, atol=1e-6)


def test_total_variation_regularizer():
    w = np.array([1.0, 3.0, 2.0])
    reg = TotalVariationRegularizer(1.0)
    loss, grad = reg(w)
    assert loss == pytest.approx(abs(3 - 1) + abs(2 - 3))  # 2 + 1 = 3
    assert grad.shape == w.shape


def test_layer_neighbor_pairs():
    class _FE:
        def __init__(self):
            self.fluence_elements = self
            self.per_beam = [
                type("PB", (), {"element_count": 4, "spots_per_layer": [2, 2]})(),
                type("PB", (), {"element_count": 3, "spots_per_layer": None})(),
            ]
    pairs = layer_neighbor_pairs(_FE())
    # Beam 0: layer [0,1] -> (0,1); layer [2,3] -> (2,3).
    # Beam 1 (offset 4): one layer of 3 -> (4,5),(5,6).
    assert pairs == [(0, 1), (2, 3), (4, 5), (5, 6)]
