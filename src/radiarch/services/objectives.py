"""Point-dose treatment-planning objectives (Service 4, task O3).

An *objective* turns a candidate dose distribution into a scalar penalty
(``loss``) plus the per-voxel gradient of that penalty with respect to the
dose. We return the gradient — not just the loss — because the optimizer
composes many objectives over the *same* dose volume and needs to **sum**
their gradients to form the total search direction. Returning a dense,
full-volume gradient (same shape as ``dose``) lets the caller add objective
gradients together with a plain ``+`` regardless of which structure each one
targets; the mask zeroes out voxels outside the structure so they contribute
nothing.

All objectives here are *point-dose* objectives: each in-mask voxel is scored
independently, so the loss is a sum of per-voxel terms and every objective is
differentiable in closed form. DVH / EUD objectives (which couple voxels) live
in O4.

Each objective is a small callable class exposing a ``name`` attribute and a
``__call__`` matching the :class:`Objective` protocol, so they can be stored
and dispatched uniformly by the optimizer.
"""

from __future__ import annotations

from typing import Protocol, Tuple, runtime_checkable

import numpy as np
from loguru import logger


@runtime_checkable
class Objective(Protocol):
    """Callable that scores a dose volume for one structure.

    Implementations return ``(loss, grad)`` where ``grad`` has the *same shape*
    as ``dose`` — a dense full-volume gradient with out-of-structure voxels set
    to zero, so the optimizer can sum gradients across objectives elementwise.
    """

    #: Human-readable identifier, surfaced in logs and optimizer reports.
    name: str

    def __call__(
        self, dose: np.ndarray, mask: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """Score ``dose`` against this objective on the voxels in ``mask``."""
        ...


def _prepare(dose: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Coerce inputs to float arrays of a common shape and validate them.

    The mask is cast to ``float`` (so boolean, 0/1 integer, or already-float
    masks all behave identically: in-structure voxels weight 1.0, others 0.0).
    Multiplying by this float mask — rather than indexing — keeps the gradient
    dense and the same shape as ``dose``, which is what the optimizer expects.
    """
    dose_arr = np.asarray(dose, dtype=np.float64)
    mask_arr = np.asarray(mask, dtype=np.float64)
    if dose_arr.shape != mask_arr.shape:
        raise ValueError(
            f"dose shape {dose_arr.shape} != mask shape {mask_arr.shape}"
        )
    return dose_arr, mask_arr


class DMin:
    """Penalize masked voxels that fall *below* a target dose (cold spots).

    ``loss = w * sum_i mask_i * max(0, d_target - d_i)^2``

    The one-sided ``max(0, .)`` means voxels at or above target are free; only
    under-dosed voxels are pushed up. The gradient is negative there (raising
    dose lowers the loss), which is what we want for a minimum-dose constraint.
    """

    def __init__(self, structure_name: str, dose_gy: float, weight: float = 1.0):
        self.structure_name = structure_name
        self.dose_gy = float(dose_gy)
        self.weight = float(weight)
        self.name = f"DMin({structure_name}, {dose_gy} Gy, w={weight})"

    def __call__(
        self, dose: np.ndarray, mask: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        dose_arr, mask_arr = _prepare(dose, mask)
        # Shortfall below target, clamped at zero so over-dosed voxels score 0.
        deficit = np.maximum(0.0, self.dose_gy - dose_arr) * mask_arr
        loss = self.weight * float(np.sum(deficit * deficit))
        # d/d(d_i) of (d_target - d_i)^2 is -2 (d_target - d_i); mask already in deficit.
        grad = -2.0 * self.weight * deficit
        logger.trace("{} -> loss={:.6g}", self.name, loss)
        return loss, grad


class DMax:
    """Penalize masked voxels that exceed a target dose (hot spots).

    ``loss = w * sum_i mask_i * max(0, d_i - d_target)^2``

    Symmetric to :class:`DMin`: only over-dosed voxels are penalized, and the
    gradient there is positive (lowering dose lowers the loss).
    """

    def __init__(self, structure_name: str, dose_gy: float, weight: float = 1.0):
        self.structure_name = structure_name
        self.dose_gy = float(dose_gy)
        self.weight = float(weight)
        self.name = f"DMax({structure_name}, {dose_gy} Gy, w={weight})"

    def __call__(
        self, dose: np.ndarray, mask: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        dose_arr, mask_arr = _prepare(dose, mask)
        # Excess above target, clamped at zero so under-dosed voxels score 0.
        excess = np.maximum(0.0, dose_arr - self.dose_gy) * mask_arr
        loss = self.weight * float(np.sum(excess * excess))
        # d/d(d_i) of (d_i - d_target)^2 is +2 (d_i - d_target); mask already in excess.
        grad = 2.0 * self.weight * excess
        logger.trace("{} -> loss={:.6g}", self.name, loss)
        return loss, grad


class DUniform:
    """Drive masked voxels *toward* a target dose, penalizing any deviation.

    ``loss = w * sum_i mask_i * (d_i - d_target)^2``

    Two-sided least squares: both cold and hot voxels are penalized, so this
    pushes the structure to a flat, uniform dose at the target.
    """

    def __init__(self, structure_name: str, dose_gy: float, weight: float = 1.0):
        self.structure_name = structure_name
        self.dose_gy = float(dose_gy)
        self.weight = float(weight)
        self.name = f"DUniform({structure_name}, {dose_gy} Gy, w={weight})"

    def __call__(
        self, dose: np.ndarray, mask: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        dose_arr, mask_arr = _prepare(dose, mask)
        deviation = (dose_arr - self.dose_gy) * mask_arr
        loss = self.weight * float(np.sum(deviation * deviation))
        # d/d(d_i) of (d_i - d_target)^2 is 2 (d_i - d_target); mask already applied.
        grad = 2.0 * self.weight * deviation
        logger.trace("{} -> loss={:.6g}", self.name, loss)
        return loss, grad


# ===========================================================================
# O4 — DVH and EUD objectives (voxel-coupled, differentiable surrogates)
# ===========================================================================
#
# Unlike the point-dose objectives above, these *couple* voxels: the loss
# depends on an aggregate over the whole structure (a volume fraction, or a
# generalized mean), so the per-voxel gradient carries a normalization that
# depends on every other voxel. They're still returned as dense full-volume
# gradients so the optimizer can sum them with the point-dose ones.


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic, used as a differentiable step surrogate."""
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def _dvh_k(dose_gy: float) -> float:
    """Default sigmoid steepness — sharp relative to the target dose.

    ~10/dose_gy puts the logistic transition within ~10% of the target, sharp
    enough to approximate the DVH step yet smooth enough to give usable
    gradients. Falls back to a fixed slope when the target is ~0 Gy.
    """
    return 10.0 / dose_gy if dose_gy > 1e-9 else 10.0


class DVHMin:
    """DVH minimum-dose goal via a smooth volume-fraction surrogate.

    Clinical intent: no more than ``volume_fraction`` of the structure may
    fall *below* ``dose_gy``. We approximate the (non-differentiable) "fraction
    of voxels at or above target" with ``mean_i sigmoid(k (d_i - d_target))``;
    the cold fraction is ``1 - mean(H)``. We penalize only the amount by which
    the cold fraction exceeds the allowance:

    ``loss = w * max(0, (1 - mean(H)) - volume_fraction)^2``

    The gradient is negative on under-dosed voxels (raising their dose grows
    ``mean(H)`` and shrinks the cold fraction), which is the desired push-up.
    """

    def __init__(self, structure_name: str, dose_gy: float,
                 volume_fraction: float, weight: float = 1.0,
                 k: float | None = None):
        self.structure_name = structure_name
        self.dose_gy = float(dose_gy)
        self.volume_fraction = float(volume_fraction)
        self.weight = float(weight)
        self.k = float(k) if k is not None else _dvh_k(dose_gy)
        self.name = (f"DVHMin({structure_name}, {dose_gy} Gy, "
                     f"v={volume_fraction}, w={weight})")

    def __call__(self, dose: np.ndarray, mask: np.ndarray):
        dose_arr, mask_arr = _prepare(dose, mask)
        n = float(mask_arr.sum())
        if n <= 0:
            return 0.0, np.zeros_like(dose_arr)
        h = _sigmoid(self.k * (dose_arr - self.dose_gy))
        mean_h = float(np.sum(h * mask_arr) / n)
        cold = 1.0 - mean_h
        shortfall = max(0.0, cold - self.volume_fraction)
        loss = self.weight * shortfall * shortfall
        if shortfall <= 0.0:
            return loss, np.zeros_like(dose_arr)
        # dloss/d(cold) = 2 w shortfall; d(cold)/d(mean_h) = -1;
        # d(mean_h)/d(d_i) = (k / n) * H_i (1 - H_i) on masked voxels.
        sig_prime = h * (1.0 - h)
        grad = (-2.0 * self.weight * shortfall) * (self.k / n) * sig_prime * mask_arr
        logger.trace("{} -> loss={:.6g} cold={:.4f}", self.name, loss, cold)
        return loss, grad


class DVHMax:
    """DVH maximum-dose goal — symmetric to :class:`DVHMin`.

    Clinical intent: no more than ``volume_fraction`` of the structure may
    rise *above* ``dose_gy``. The hot fraction is ``mean(H)`` with the same
    sigmoid surrogate; we penalize the excess over the allowance and the
    gradient is positive on hot voxels (lowering their dose helps).
    """

    def __init__(self, structure_name: str, dose_gy: float,
                 volume_fraction: float, weight: float = 1.0,
                 k: float | None = None):
        self.structure_name = structure_name
        self.dose_gy = float(dose_gy)
        self.volume_fraction = float(volume_fraction)
        self.weight = float(weight)
        self.k = float(k) if k is not None else _dvh_k(dose_gy)
        self.name = (f"DVHMax({structure_name}, {dose_gy} Gy, "
                     f"v={volume_fraction}, w={weight})")

    def __call__(self, dose: np.ndarray, mask: np.ndarray):
        dose_arr, mask_arr = _prepare(dose, mask)
        n = float(mask_arr.sum())
        if n <= 0:
            return 0.0, np.zeros_like(dose_arr)
        h = _sigmoid(self.k * (dose_arr - self.dose_gy))
        hot = float(np.sum(h * mask_arr) / n)
        excess = max(0.0, hot - self.volume_fraction)
        loss = self.weight * excess * excess
        if excess <= 0.0:
            return loss, np.zeros_like(dose_arr)
        sig_prime = h * (1.0 - h)
        grad = (2.0 * self.weight * excess) * (self.k / n) * sig_prime * mask_arr
        logger.trace("{} -> loss={:.6g} hot={:.4f}", self.name, loss, hot)
        return loss, grad


class EUD:
    """Generalized equivalent uniform dose objective (Niemierko).

    ``EUD = (mean_i d_i^a)^(1/a)`` over the structure, penalized toward a
    target: ``loss = w * (EUD - dose_gy)^2``. The exponent ``a`` tunes the
    behaviour — ``a -> +inf`` approaches the max dose (serial organs),
    ``a -> -inf`` approaches the min dose (targets), and ``a = 1`` is the
    arithmetic mean. Dose is floored at a small epsilon so ``d^a`` stays finite
    for negative ``a`` (which is exactly the target-coverage regime).
    """

    _EPS = 1e-6

    def __init__(self, structure_name: str, dose_gy: float, a: float,
                 weight: float = 1.0):
        if abs(a) < 1e-9:
            raise ValueError("EUD exponent a must be non-zero")
        self.structure_name = structure_name
        self.dose_gy = float(dose_gy)
        self.a = float(a)
        self.weight = float(weight)
        self.name = f"EUD({structure_name}, {dose_gy} Gy, a={a}, w={weight})"

    def eud(self, dose: np.ndarray, mask: np.ndarray) -> float:
        """Public helper: the gEUD value itself (no penalty), for tests/QA."""
        dose_arr, mask_arr = _prepare(dose, mask)
        n = float(mask_arr.sum())
        if n <= 0:
            return 0.0
        d = np.maximum(dose_arr, self._EPS)
        m = float(np.sum((d ** self.a) * mask_arr) / n)
        return float(m ** (1.0 / self.a))

    def __call__(self, dose: np.ndarray, mask: np.ndarray):
        dose_arr, mask_arr = _prepare(dose, mask)
        n = float(mask_arr.sum())
        if n <= 0:
            return 0.0, np.zeros_like(dose_arr)
        d = np.maximum(dose_arr, self._EPS)
        m = float(np.sum((d ** self.a) * mask_arr) / n)  # mean(d^a)
        if m <= 0:
            return 0.0, np.zeros_like(dose_arr)
        eud = m ** (1.0 / self.a)
        diff = eud - self.dose_gy
        loss = self.weight * diff * diff
        # dEUD/dd_i = m^(1/a - 1) * (1/n) * d_i^(a-1)  on masked voxels.
        deud_dd = (m ** (1.0 / self.a - 1.0)) * (1.0 / n) * (d ** (self.a - 1.0))
        grad = 2.0 * self.weight * diff * deud_dd * mask_arr
        logger.trace("{} -> loss={:.6g} eud={:.4f}", self.name, loss, eud)
        return loss, grad


# ===========================================================================
# O5 — Constraints (penalty form) + fluence regularizers
# ===========================================================================


class ConstraintPenalty:
    """A hard dose limit folded into the composite cost as a soft penalty.

    For ``op == "<="`` the penalty is ``w * sum(max(0, d - limit)^2)`` over the
    structure (a maximum-dose limit); for ``op == ">="`` it is
    ``w * sum(max(0, limit - d)^2)`` (a minimum-dose floor). Identical in form
    to :class:`DMax` / :class:`DMin` but kept distinct so the service can report
    a per-constraint residual (``loss`` value) in ``ConvergenceInfo``.
    """

    def __init__(self, structure_name: str, op: str, value_gy: float,
                 weight: float = 1.0, label: str = "constraint"):
        if op not in (">=", "<="):
            raise ValueError(f'op must be ">=" or "<=", got {op!r}')
        self.structure_name = structure_name
        self.op = op
        self.value_gy = float(value_gy)
        self.weight = float(weight)
        self.name = f"Constraint({label}:{structure_name} {op} {value_gy} Gy)"

    def __call__(self, dose: np.ndarray, mask: np.ndarray):
        dose_arr, mask_arr = _prepare(dose, mask)
        if self.op == "<=":
            viol = np.maximum(0.0, dose_arr - self.value_gy) * mask_arr
            grad = 2.0 * self.weight * viol
        else:  # ">="
            viol = np.maximum(0.0, self.value_gy - dose_arr) * mask_arr
            grad = -2.0 * self.weight * viol
        loss = self.weight * float(np.sum(viol * viol))
        return loss, grad


@runtime_checkable
class WeightRegularizer(Protocol):
    """Callable scoring the *weight vector* directly (not the dose).

    Regularizers act on ``w`` rather than ``D(w)`` so they don't need the
    influence matrix; the service evaluates them separately and adds their
    ``(loss, grad)`` (gradient already in weight space) to the dose-side
    gradient after the ``Dij.T`` projection.
    """

    name: str

    def __call__(self, w: np.ndarray) -> Tuple[float, np.ndarray]:
        ...


class SmoothnessRegularizer:
    """Penalize differences between spatially-neighboring fluence elements.

    ``loss = weight * sum_{(i,j) in pairs} (w_i - w_j)^2``

    ``pairs`` are adjacency tuples derived from the beam model's spot layout
    (adjacent spots within a layer for v0.1 — see
    :func:`layer_neighbor_pairs`). Encourages smooth, deliverable spot maps.
    """

    def __init__(self, weight: float, pairs: "list[tuple[int, int]]"):
        self.weight = float(weight)
        self._pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2) if pairs \
            else np.zeros((0, 2), dtype=np.int64)
        self.name = f"Smoothness(w={weight}, pairs={len(self._pairs)})"

    def __call__(self, w: np.ndarray) -> Tuple[float, np.ndarray]:
        w = np.asarray(w, dtype=np.float64).ravel()
        grad = np.zeros_like(w)
        if self._pairs.size == 0:
            return 0.0, grad
        i = self._pairs[:, 0]
        j = self._pairs[:, 1]
        diff = w[i] - w[j]
        loss = self.weight * float(np.sum(diff * diff))
        # d/dw_i sum (w_i - w_j)^2 = 2 (w_i - w_j); symmetric on j.
        g = 2.0 * self.weight * diff
        np.add.at(grad, i, g)
        np.add.at(grad, j, -g)
        return loss, grad


class TotalVariationRegularizer:
    """1D total-variation penalty in fluence-element order.

    ``loss = weight * sum_i |w_{i+1} - w_i|``

    A cheaper, sparsity-promoting alternative to full smoothness; uses the
    subgradient (``sign`` of consecutive differences) where non-differentiable.
    """

    def __init__(self, weight: float):
        self.weight = float(weight)
        self.name = f"TotalVariation(w={weight})"

    def __call__(self, w: np.ndarray) -> Tuple[float, np.ndarray]:
        w = np.asarray(w, dtype=np.float64).ravel()
        grad = np.zeros_like(w)
        if w.size < 2:
            return 0.0, grad
        diff = np.diff(w)
        loss = self.weight * float(np.sum(np.abs(diff)))
        s = self.weight * np.sign(diff)
        grad[:-1] -= s
        grad[1:] += s
        return loss, grad


def layer_neighbor_pairs(beam_model_result) -> "list[tuple[int, int]]":
    """Adjacency pairs of fluence elements that share a layer.

    Walks ``beam_model_result.fluence_elements.per_beam`` and, within each
    beam, chunks the flat element index range by ``spots_per_layer`` (falling
    back to one layer per beam when proton layer info is absent, e.g. photon
    beamlets). Consecutive elements within a layer are treated as neighbors —
    the v0.1 approximation called for in O5. Returns global (offset-corrected)
    index pairs ready for :class:`SmoothnessRegularizer`.
    """
    pairs: "list[tuple[int, int]]" = []
    offset = 0
    fe = beam_model_result.fluence_elements
    for pb in fe.per_beam:
        count = int(pb.element_count)
        spl = getattr(pb, "spots_per_layer", None)
        if spl:
            layers = [int(s) for s in spl]
            # Guard against metadata drift: clamp the layer sum to the beam's
            # declared element_count so indices never run past the beam.
            if sum(layers) != count:
                layers = [count]
        else:
            layers = [count]
        local = offset
        for layer_size in layers:
            for k in range(layer_size - 1):
                pairs.append((local + k, local + k + 1))
            local += layer_size
        offset += count
    return pairs


__all__ = [
    "Objective",
    "DMin",
    "DMax",
    "DUniform",
    "DVHMin",
    "DVHMax",
    "EUD",
    "ConstraintPenalty",
    "WeightRegularizer",
    "SmoothnessRegularizer",
    "TotalVariationRegularizer",
    "layer_neighbor_pairs",
]
