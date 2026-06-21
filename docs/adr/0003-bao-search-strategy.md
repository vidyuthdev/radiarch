# ADR 0003 — BAO search strategy

**Status:** Accepted
**Date:** 2026-06-21
**Context:** Service 5 (BAO) — how to select beam directions.

## Context

Beam Angle Optimization picks which gantry/couch directions a plan should use,
*before* fluence optimization. The quality of an angle set is only knowable by
how good a plan it produces, so the natural score for a set is the achieved
composite objective after a (short) fluence optimization. The candidate space is
a discretized sweep (e.g. gantry every 20–45°, optionally a few couch angles);
the full subset search is combinatorial (`C(n_candidates, n_beams)`), so we need
a tractable strategy.

## Decision

**Score = build a beam model for the angle set → run Service 4 with a small
iteration budget → take `convergence.final_cost`.** This makes BAO engine- and
objective-agnostic for free: it reuses the exact same objectives, engine, and
solver machinery as standalone optimization, so a beam set is judged by the
plan it actually enables.

**Two pluggable search strategies (`SolverPlugin`-style protocol):**

- **`greedy` (default)** — forward selection: start empty, repeatedly add the
  candidate whose addition most lowers the *combined*-set score. Cost is
  `O(n_beams · n_candidates)` scoring runs. It captures beam-to-beam
  complementarity (it won't pick two near-parallel fields that add little), which
  a per-angle ranking misses, while staying far cheaper than exhaustive search.
- **`top_k`** — score every candidate individually and keep the `n_beams`
  lowest. One score per candidate; ignores interplay, but a useful fast baseline
  and a sanity check against greedy.

The strategies receive a black-box `score_fn(angle_set) -> float`, so they're
unit-tested with a cheap synthetic scorer decoupled from any dose engine.

## Consequences

- BAO sits cleanly on top of Optimization; no duplicate objective/solver code.
- Scoring is the cost driver — each score is a full (if short) optimization, so
  BAO is the heaviest task in the system (its own long Celery time budget and a
  conservative rate limit). `scoring_iterations` trades selection fidelity for
  speed.
- Exhaustive and metaheuristic strategies (simulated annealing, column
  generation) can be added later as new registry entries without touching the
  service or the API contract.
- The selected beam set is materialized as a real beam model (`beam_model_id` in
  the result) so a downstream `/optimize/run` can plan on it directly.
