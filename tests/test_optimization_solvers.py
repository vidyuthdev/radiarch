"""Unit tests for :mod:`radiarch.services.optimization_solvers`.

These tests exercise the L-BFGS-B solver on a small convex quadratic with a
known minimum, decoupled from any real objective / dose engine.

NOTE on the ``ConvergenceInfo`` dependency (task O1):
``radiarch.models.optimization`` is authored in parallel (task O1) and may
not exist when these tests run. The solver module itself imports
``from ..models.optimization import ConvergenceInfo`` (the real contract).
To let this test stand alone we install a *minimal* stand-in module into
``sys.modules`` **before** importing the solver, but only if the real module
is absent. Once O1 lands, the real ``ConvergenceInfo`` is used automatically
and this fallback is skipped. The fallback must be reconciled away once O1
is merged (see report).
"""

from __future__ import annotations

import importlib
import sys
import types
from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Test-only fallback for ConvergenceInfo (O1) if the real module is missing.
# ---------------------------------------------------------------------------
def _ensure_convergence_info() -> None:
    try:  # real module from O1 takes precedence
        importlib.import_module("radiarch.models.optimization")
        return
    except ModuleNotFoundError:
        pass

    @dataclass
    class ConvergenceInfo:  # minimal stand-in matching O1's documented fields
        success: bool
        iterations: int
        final_cost: float
        cost_history: List[float] = field(default_factory=list)
        constraint_violations: Optional[Any] = None

    stub = types.ModuleType("radiarch.models.optimization")
    stub.ConvergenceInfo = ConvergenceInfo  # type: ignore[attr-defined]
    sys.modules["radiarch.models.optimization"] = stub


_ensure_convergence_info()

from radiarch.services.optimization_solvers import (  # noqa: E402
    LBFGSBSolver,
    SolverPlugin,
)


# ---------------------------------------------------------------------------
# Convex toy problem: minimize f(w) = 0.5 * ||A w - b||^2
# Unconstrained minimizer is the least-squares solution w* = A^+ b.
# A and b are chosen so w* is strictly positive, so the non-negativity bound
# is inactive and the constrained solution equals the unconstrained one.
# ---------------------------------------------------------------------------
def _make_problem():
    rng = np.random.default_rng(0)
    n = 5
    # Well-conditioned SPD-ish A (square, full rank) to keep the least-squares
    # solution unique and the analytic check clean.
    A = np.eye(n) + 0.1 * rng.standard_normal((n, n))
    w_star = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # strictly positive target
    b = A @ w_star

    def cost_and_grad(w: np.ndarray):
        r = A @ w - b
        f = 0.5 * float(r @ r)
        g = A.T @ r
        return f, g

    return cost_and_grad, w_star, n


def test_converges_to_analytic_minimum():
    solver = LBFGSBSolver()
    cost_and_grad, w_star, n = _make_problem()
    w0 = np.zeros(n)

    w, info = solver.run(cost_and_grad, w0, max_iter=200, convergence_tol=1e-12)

    assert info.success
    np.testing.assert_allclose(w, w_star, atol=1e-4)
    assert info.final_cost < 1e-8


def test_non_negativity_enforced():
    solver = LBFGSBSolver()
    # Problem whose unconstrained minimum is negative everywhere; the bound
    # must clip it to zero.
    n = 3
    A = np.eye(n)
    b = -np.ones(n)  # unconstrained min would be w = -1

    def cost_and_grad(w):
        r = A @ w - b
        return 0.5 * float(r @ r), A.T @ r

    w, info = solver.run(cost_and_grad, np.ones(n), max_iter=100, convergence_tol=1e-12)
    assert np.all(w >= 0.0)
    np.testing.assert_allclose(w, np.zeros(n), atol=1e-6)


def test_cost_history_populated_and_monotone():
    solver = LBFGSBSolver()
    cost_and_grad, _, n = _make_problem()

    _, info = solver.run(cost_and_grad, np.zeros(n), max_iter=200, convergence_tol=1e-12)

    assert info.iterations > 0
    assert len(info.cost_history) > 0
    # Monotonically non-increasing (L-BFGS-B accepts only descent steps),
    # allowing a tiny numerical wiggle.
    hist = np.asarray(info.cost_history)
    diffs = np.diff(hist)
    assert np.all(diffs <= 1e-9), f"cost history not non-increasing: {hist}"


def test_user_callback_invoked():
    solver = LBFGSBSolver()
    cost_and_grad, _, n = _make_problem()

    calls = []

    def cb(iteration, cost, w):
        calls.append((iteration, cost, np.array(w, copy=True)))

    _, info = solver.run(
        cost_and_grad, np.zeros(n), max_iter=200, convergence_tol=1e-12, callback=cb
    )

    assert len(calls) > 0
    # Iteration counter is 1-based and strictly increasing.
    iters = [c[0] for c in calls]
    assert iters[0] == 1
    assert iters == sorted(iters)
    assert all(b > a for a, b in zip(iters, iters[1:]))


def test_respects_max_iter():
    solver = LBFGSBSolver()
    cost_and_grad, w_star, n = _make_problem()

    calls = []
    _, info = solver.run(
        cost_and_grad,
        np.zeros(n),
        max_iter=1,
        convergence_tol=1e-16,
        callback=lambda i, c, w: calls.append(i),
    )

    # With maxiter=1 the solver must stop early (not reach the optimum) and
    # report at most one iteration.
    assert info.iterations <= 1
    assert len(calls) <= 1
    # It should NOT have converged to the true minimum in a single step.
    assert not np.allclose(_, w_star, atol=1e-4)


def test_solver_satisfies_protocol():
    # Structural typing: LBFGSBSolver is a SolverPlugin.
    assert isinstance(LBFGSBSolver(), SolverPlugin)
    assert LBFGSBSolver().name == "L-BFGS-B"


# ===========================================================================
# O7/O8 — Adam + ProjectedGradient + registry
# ===========================================================================

from radiarch.services.optimization_solvers import (  # noqa: E402
    AdamSolver,
    ProjectedGradientSolver,
    get_solver,
)


@pytest.mark.parametrize("name", ["L-BFGS-B", "Adam", "ProjectedGradient"])
def test_registry_resolves_all_methods(name):
    solver = get_solver(name)
    assert isinstance(solver, SolverPlugin)
    assert solver.name == name


def test_registry_rejects_unknown():
    with pytest.raises(ValueError, match="unknown solver"):
        get_solver("nope")


@pytest.mark.parametrize("solver", [AdamSolver(learning_rate=0.1),
                                    ProjectedGradientSolver()])
def test_first_order_solvers_match_lbfgs(solver):
    """Adam / ProjectedGradient converge to the same optimum as L-BFGS-B (±1%)."""
    cost_and_grad, w_star, n = _make_problem()
    w, info = solver.run(cost_and_grad, np.zeros(n), max_iter=20000,
                         convergence_tol=1e-12)
    np.testing.assert_allclose(w, w_star, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("solver", [AdamSolver(), ProjectedGradientSolver()])
def test_first_order_enforce_non_negativity(solver):
    # Minimizer wants negatives; projection must clamp to 0.
    def cg(w):
        return 0.5 * float(((w + 5.0) ** 2).sum()), (w + 5.0)
    w, _ = solver.run(cg, np.ones(3), max_iter=500, convergence_tol=1e-12)
    assert np.all(w >= 0.0)


@pytest.mark.parametrize("solver", [AdamSolver(), ProjectedGradientSolver()])
def test_first_order_invoke_callback(solver):
    cost_and_grad, _, n = _make_problem()
    seen = []
    solver.run(cost_and_grad, np.zeros(n), max_iter=10, convergence_tol=1e-12,
               callback=lambda it, c, w: seen.append((it, c)))
    assert seen
    assert seen[0][0] == 1
