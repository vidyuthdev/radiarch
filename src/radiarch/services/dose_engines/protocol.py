"""``DoseEnginePlugin`` protocol — the contract every engine implements.

The Dose Service dispatches all dose / influence / scenario work through
this five-method interface. Each engine is responsible for:

1. Validating its own params against geometry + beam-model (``validate``).
2. Computing a single nominal-or-scenario dose grid (``compute_dose``).
3. Building a sparse dose-influence matrix Dij (``build_influence``).
4. Applying an existing Dij to a fresh weight vector (``apply_influence``)
   — cheap inner product, no engine call.
5. Computing the gradient of a downstream loss wrt weights
   (``compute_grad``) — used by gradient-based optimizers in Service 4.

Engines that don't yet implement (3)–(5) raise
:class:`EngineUnavailableError`; the service surfaces this to the API
as 501 Not Implemented. (1) and (2) are mandatory.

Engine plugins receive *bundles* loaded by the service rather than raw
ids — this isolates engines from persistence layout details. The bundle
dataclasses live here so engines don't depend on ``services.dose``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Protocol, runtime_checkable

import numpy as np

from ...models.beam_model import BeamModelResult
from ...models.dose import ScenarioSpec
from ...models.geometry import GeometryResult


# ---------------------------------------------------------------------------
# Bundles passed to engines
# ---------------------------------------------------------------------------

@dataclass
class GeometryBundle:
    """What the service hands to an engine for the geometry.

    The engine treats this as immutable — it never writes to the arrays.
    ``density`` and ``masks`` are loaded eagerly so engines don't depend
    on SimpleITK / NIfTI loading.

    ``ct_hu`` and ``ct_image`` carry the original CT in Hounsfield Units
    on the same grid as ``density``. They're optional for two reasons:

    1. Geometries cached before D6.1 don't have a CT NIfTI on disk —
       ``DoseService._load_geometry`` returns those bundles with
       ``ct_hu=None``.
    2. Tests construct bundles by hand without going through persistence;
       leaving CT optional keeps them terse.

    Engines that need a CT (e.g. MCsquare) must check for ``None`` and
    surface an :class:`EngineUnavailableError` if absent rather than
    crashing on a NoneType attribute.
    """

    result: GeometryResult
    density: np.ndarray         # (nz, ny, nx) float32, g/cm³
    masks: np.ndarray           # (nz, ny, nx) uint16, structure labels
    spacing_mm: tuple           # (sx, sy, sz)
    ct_hu: Optional[np.ndarray] = None    # (nz, ny, nx) int16, Hounsfield Units
    ct_image: Optional[Any] = None        # OpenTPS CTImage wrapping ct_hu, when OpenTPS is importable


@dataclass
class BeamModelBundle:
    """What the service hands to an engine for the beam model."""

    result: BeamModelResult
    plan: Any                   # unpickled OpenTPS plan (or test double)


# ---------------------------------------------------------------------------
# Engine results
# ---------------------------------------------------------------------------

@dataclass
class NominalDose:
    """Engine output for one dose compute.

    ``dose`` has the same shape as ``GeometryBundle.density`` and is in
    absolute dose units (Gy per unit weight × weight). The service is
    responsible for persistence / cache-keying.
    """

    dose: np.ndarray            # (nz, ny, nx) float32, Gy


@dataclass
class InfluenceData:
    """Sparse-matrix representation of Dij.

    Stored CSR. Rows are flattened voxels (row-major), cols are
    fluence-elements in the same order as the beam-model's
    ``FluenceElementSet`` flattens them.
    """

    indptr: np.ndarray          # int64
    indices: np.ndarray         # int32
    data: np.ndarray            # float32
    n_voxels: int
    n_elements: int

    @property
    def nnz(self) -> int:
        return int(self.data.size)


# ---------------------------------------------------------------------------
# Engine exceptions
# ---------------------------------------------------------------------------

class EngineParamError(ValueError):
    """Engine params or inputs failed validation. Surfaced as 422."""


class EngineRuntimeError(RuntimeError):
    """Engine crashed during compute. Surfaced as failed job, 500 if sync."""


class EngineUnavailableError(NotImplementedError):
    """Engine doesn't support this operation. Surfaced as 501 / failed job."""


# ---------------------------------------------------------------------------
# The protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class DoseEnginePlugin(Protocol):
    """Minimum surface every dose engine must expose.

    Implementations are typically dataclasses (or simple classes) that
    hold engine config — they're cheap to instantiate per request. The
    registry creates one instance per engine name and reuses it.
    """

    #: Unique engine name (matches the value used in EngineSpec.name).
    name: str

    #: Engine version pinned by this build. Folded into cache keys.
    version: str

    #: Modalities this engine claims to support. Dose Service rejects
    #: builds whose beam-model modality is not in this list.
    modalities: List[str]

    def validate(
        self,
        geometry: GeometryBundle,
        beam_model: BeamModelBundle,
        params: dict,
    ) -> List[str]:
        """Return a list of validation issues (empty = valid)."""
        ...

    def compute_dose(
        self,
        geometry: GeometryBundle,
        beam_model: BeamModelBundle,
        weights: np.ndarray,
        scenario: Optional[ScenarioSpec] = None,
        params: Optional[dict] = None,
    ) -> NominalDose:
        """Compute dose for a single weight vector + scenario."""
        ...

    def build_influence(
        self,
        geometry: GeometryBundle,
        beam_model: BeamModelBundle,
        scenario: Optional[ScenarioSpec] = None,
        params: Optional[dict] = None,
    ) -> InfluenceData:
        """Build the sparse Dij matrix for the given scenario."""
        ...

    def apply_influence(
        self,
        influence: InfluenceData,
        weights: np.ndarray,
        grid_shape: tuple,
    ) -> NominalDose:
        """Apply an existing Dij to a fresh weight vector → dose.

        Default implementation: ``(D @ w).reshape(grid_shape)``. Engines
        can override only if they need a smarter route.
        """
        ...

    def compute_grad(
        self,
        geometry: GeometryBundle,
        beam_model: BeamModelBundle,
        weights: np.ndarray,
        dL_dDose: np.ndarray,
        scenario: Optional[ScenarioSpec] = None,
        params: Optional[dict] = None,
    ) -> np.ndarray:
        """Return ∂L/∂w given ∂L/∂dose — VJP through the dose op."""
        ...


__all__ = [
    "GeometryBundle",
    "BeamModelBundle",
    "NominalDose",
    "InfluenceData",
    "DoseEnginePlugin",
    "EngineParamError",
    "EngineRuntimeError",
    "EngineUnavailableError",
]
