"""Scenario expansion — turn a :class:`ScenarioSetSpec` into a concrete
list of :class:`ScenarioSpec` to feed engine plugins.

Two modes, both deterministic (no RNG state leaks across requests):

1. **Explicit** — pass-through, with the empty/nominal scenario prepended
   if it wasn't already present.

2. **Generator** — given ``setup_sigma_mm`` and/or ``range_sigma`` plus
   ``count``, produce a *stratified* sample that always includes:

     * the nominal scenario,
     * axis-aligned worst-case corners along whichever sigma is set
       (8 corners for setup if 3D; 2 for range; 16 combined),
     * additional random samples (deterministic seed) to reach ``count``.

Real clinical robust optimization uses richer sampling (Latin hypercube
+ probability weighting), but this stratified-corner approach is the
right baseline: it always evaluates the cases that dominate the worst-
case dose, and adds enough variance to drive a SAA-style robust loss.
"""

from __future__ import annotations

import itertools
from typing import List

import numpy as np

from ..models.dose import ScenarioSetSpec, ScenarioSpec


def expand_scenarios(spec: ScenarioSetSpec) -> List[ScenarioSpec]:
    """Materialize the scenario list for an engine to iterate over.

    Returns at least one scenario (the nominal). Never raises — the
    request validator already guaranteed that either ``scenarios`` is
    populated or generator inputs are.
    """
    if spec.scenarios is not None:
        out = list(spec.scenarios)
        if not any(s.is_nominal() for s in out):
            out.insert(0, ScenarioSpec(name="nominal"))
        return out

    # Generator mode.
    count = int(spec.count or 1)
    setup_sigma = spec.setup_sigma_mm
    range_sigma = spec.range_sigma

    scenarios: List[ScenarioSpec] = [ScenarioSpec(name="nominal")]

    if setup_sigma and setup_sigma > 0:
        for i, signs in enumerate(itertools.product((-1, 1), repeat=3)):
            shift = tuple(float(s) * float(setup_sigma) for s in signs)
            scenarios.append(ScenarioSpec(
                name=f"setup_corner_{i:02d}",
                setup_shift_mm=shift,
            ))

    if range_sigma and range_sigma > 0:
        for sign in (-1, 1):
            scenarios.append(ScenarioSpec(
                name=f"range_{'pos' if sign > 0 else 'neg'}",
                range_scale=1.0 + sign * float(range_sigma),
            ))

    # If asked for more samples than the corners, deterministically fill
    # with low-discrepancy random samples (numpy default_rng with a fixed
    # seed derived from the spec hash → identical inputs, identical output).
    if len(scenarios) < count:
        seed_int = int(spec.hash()[:8], 16)
        rng = np.random.default_rng(seed_int)
        while len(scenarios) < count:
            shift = None
            if setup_sigma and setup_sigma > 0:
                shift = tuple(float(rng.normal(0, setup_sigma)) for _ in range(3))
            rscale = None
            if range_sigma and range_sigma > 0:
                rscale = max(0.8, min(1.2, float(1.0 + rng.normal(0, range_sigma))))
            scenarios.append(ScenarioSpec(
                name=f"sample_{len(scenarios):02d}",
                setup_shift_mm=shift,
                range_scale=rscale,
            ))
    elif len(scenarios) > count:
        # Always keep nominal at index 0, then truncate.
        nominal = scenarios[0]
        rest = scenarios[1:]
        scenarios = [nominal] + rest[: max(0, count - 1)]

    return scenarios


__all__ = ["expand_scenarios"]
