# ADR 0002 — Optimization solver choice

**Status:** Accepted
**Date:** 2026-06-21
**Context:** Service 4 (Optimization) — inverse planning solver selection.

## Context

The Optimization Service solves `min_w L(w)` subject to `w >= 0`, where
`L(w) = Σ_i obj_i(D(w))`, `D(w) = Dij·w`, and `Dij` is the engine's sparse
dose-influence matrix. The forward pass (a sparse matvec) dominates cost; the
objective is smooth almost everywhere (the one-sided `max(0,·)²` penalties are
C¹, the DVH surrogates are sigmoidal, gEUD is smooth for positive dose). The
weight vector ranges from a few hundred spots (single proton field, demo) to
>100k spots (multi-field IMPT on a clinical grid). The Hessian is mildly
ill-conditioned: spot sensitivities vary by orders of magnitude across a wide
Dij.

We need a default solver plus alternatives, all behind one `SolverPlugin`
protocol so the service can switch without touching call sites.

## Decision

**Default: L-BFGS-B.** Reasons:

1. **Quasi-Newton convergence.** It builds a limited-memory curvature model
   from gradient history, converging in far fewer (expensive) `Dij·w`
   evaluations than a first-order method on the smooth, mildly ill-conditioned
   objectives that dominate IMPT/IMRT.
2. **Native box bounds.** `bounds=[(0, None)]·n` enforces non-negativity of
   fluence directly — no penalty term, no separate projection — so the returned
   weights are always physical.
3. **Single value+gradient oracle.** We pass `jac=True`; `cost_and_grad`
   returns `(cost, grad)` in one call, halving the forward passes scipy would
   otherwise need.

`convergence_tol` maps to scipy's `ftol`, `max_iterations` to `maxiter`.

**Alternatives, selectable per request:**

- **Adam** — pick when the weight vector is very large (>~100k spots) or the
  Hessian is too ill-conditioned for L-BFGS-B's limited-memory model to help.
  Each step is O(n) with no line search, so iterations are cheap even when there
  are hundreds of thousands of them; per-coordinate adaptive moments handle the
  wide dynamic range of spot sensitivities. Non-negativity by projection after
  each step.
- **ProjectedGradient** — the debugging engine. Steepest descent + backtracking
  line search + projection. Slower to converge, but predictable: if a composite
  objective misbehaves under the fancier solvers, run it here to inspect the raw
  descent behaviour.

All three converge to the same optimum (±1%) on a convex toy problem — see
`tests/test_optimization_solvers.py::test_first_order_solvers_match_lbfgs`.

## Consequences

- One protocol, three engines; adding a fourth (e.g. FISTA, OSQP) is a localized
  change in `services/optimization_solvers.py` plus a registry entry.
- The gradient is assembled once in the service (`Dijᵀ · Σ ∂obj/∂D`), so solvers
  stay ignorant of the objective library and are unit-testable on synthetic
  convex problems decoupled from any dose engine.
- Robust optimization aggregates per-scenario cost+gradient (WORST_CASE /
  EXPECTED / CVaR) *before* handing a single `(cost, grad)` to the solver, so
  the solver choice is orthogonal to the robustness setting.
