"""Dose-engine plugin layer.

The Dose Service is engine-agnostic. Concrete engines (MCsquare for
protons, CCC for photons, future engines for VMAT / brachy / Monte Carlo
photons) implement a small :class:`DoseEnginePlugin` protocol and
register themselves with this module's registry.

Engines live in submodules:

* ``mcsquare`` — MCsquare-backed proton dose + influence + scenario.
* ``ccc`` — Collapsed-cone-convolution-based photon dose + influence.

A test-only ``analytic`` engine in :mod:`.analytic` is registered by
default so the service can be exercised end-to-end without OpenTPS /
MCsquare. It implements all five protocol methods using a deterministic
toy depth-falloff model — wrong physics, right shape.
"""

from .protocol import (
    DoseEnginePlugin,
    EngineParamError,
    EngineRuntimeError,
    EngineUnavailableError,
    InfluenceData,
    NominalDose,
)
from .registry import (
    EngineRegistryError,
    engine_health,
    get_all_engines,
    get_engine,
    list_engines,
    register_engine,
    reset_registry,
)
from . import analytic  # noqa: F401 — side-effect: registers the analytic engine
from . import mcsquare  # noqa: F401 — registers the MCsquare proton engine
from . import ccc       # noqa: F401 — registers the CCC photon engine

# Re-export the concrete engine classes for convenience (tests / direct use).
from .analytic import AnalyticEngine
from .mcsquare import MCsquareEngine
from .ccc import CCCEngine

__all__ = [
    "DoseEnginePlugin",
    "EngineParamError",
    "EngineRuntimeError",
    "EngineUnavailableError",
    "InfluenceData",
    "NominalDose",
    "EngineRegistryError",
    "engine_health",
    "get_all_engines",
    "get_engine",
    "list_engines",
    "register_engine",
    "reset_registry",
    "AnalyticEngine",
    "MCsquareEngine",
    "CCCEngine",
]
